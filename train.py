"""
Обучение модели с выбором режима и поддержкой краулера Википедии
Добавлено продолжение обучения с последнего сохранения
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import re
import random
import time
import argparse
from collections import deque
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import pickle

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig
from tokenizer import CharTokenizer, BPETokenizer
from model import LLM


def beep(count=1, frequency=1000, duration=200):
    """Издаёт звуковой сигнал"""
    import platform
    if platform.system() == "Windows":
        import winsound
        for i in range(count):
            winsound.Beep(frequency, duration)
            if i < count - 1:
                time.sleep(0.1)
    else:
        for i in range(count):
            print('\a', end='', flush=True)
            time.sleep(0.1)


def check_gpu():
    """Проверяет доступность GPU"""
    print("\n" + "="*60)
    print("🖥️  ПРОВЕРКА ОБОРУДОВАНИЯ")
    print("="*60)

    if torch.cuda.is_available():
        print(f"✅ GPU ДОСТУПЕН")
        print(f"   Название: {torch.cuda.get_device_name(0)}")
        print(f"   Видеопамять: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print(f"⚠️ GPU НЕ ОБНАРУЖЕН, используется CPU")
        return False


class TrainingState:
    """Состояние обучения для продолжения"""

    def __init__(self, save_path="data/models/training_state.pkl"):
        self.save_path = save_path
        self.epoch = 0
        self.batch = 0
        self.best_loss = float('inf')
        self.losses_history = []
        self.optimizer_state = None
        self.scheduler_state = None
        self.config_snapshot = None

    def save(self, epoch, batch, best_loss, losses_history, optimizer, scheduler=None, config=None):
        """Сохраняет состояние обучения"""
        self.epoch = epoch
        self.batch = batch
        self.best_loss = best_loss
        self.losses_history = losses_history
        self.optimizer_state = optimizer.state_dict()
        if scheduler:
            self.scheduler_state = scheduler.state_dict()
        if config:
            self.config_snapshot = {
                'vocab_size': config.vocab_size,
                'hidden_size': config.hidden_size,
                'num_layers': config.num_layers,
                'num_heads': config.num_heads,
                'max_seq_len': config.max_seq_len,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate
            }

        with open(self.save_path, 'wb') as f:
            pickle.dump({
                'epoch': epoch,
                'batch': batch,
                'best_loss': best_loss,
                'losses_history': losses_history,
                'optimizer_state': self.optimizer_state,
                'scheduler_state': self.scheduler_state,
                'config_snapshot': self.config_snapshot
            }, f)
        print(f"💾 Состояние обучения сохранено (эпоха {epoch}, батч {batch})")

    def load(self):
        """Загружает состояние обучения"""
        if not os.path.exists(self.save_path):
            return None

        try:
            with open(self.save_path, 'rb') as f:
                data = pickle.load(f)

            self.epoch = data.get('epoch', 0)
            self.batch = data.get('batch', 0)
            self.best_loss = data.get('best_loss', float('inf'))
            self.losses_history = data.get('losses_history', [])
            self.optimizer_state = data.get('optimizer_state')
            self.scheduler_state = data.get('scheduler_state')
            self.config_snapshot = data.get('config_snapshot')

            print(f"📂 Загружено состояние обучения (эпоха {self.epoch}, батч {self.batch}, loss: {self.best_loss:.4f})")
            return self
        except Exception as e:
            print(f"⚠️ Не удалось загрузить состояние обучения: {e}")
            return None

    def check_compatibility(self, config):
        """Проверяет совместимость конфигурации с сохранённой"""
        if not self.config_snapshot:
            return True

        issues = []
        if self.config_snapshot.get('vocab_size') != config.vocab_size:
            issues.append(f"vocab_size: {self.config_snapshot['vocab_size']} -> {config.vocab_size}")
        if self.config_snapshot.get('hidden_size') != config.hidden_size:
            issues.append(f"hidden_size: {self.config_snapshot['hidden_size']} -> {config.hidden_size}")
        if self.config_snapshot.get('num_layers') != config.num_layers:
            issues.append(f"num_layers: {self.config_snapshot['num_layers']} -> {config.num_layers}")
        if self.config_snapshot.get('num_heads') != config.num_heads:
            issues.append(f"num_heads: {self.config_snapshot['num_heads']} -> {config.num_heads}")

        if issues:
            print("⚠️ Конфигурация модели изменилась! Продолжение обучения может быть небезопасно:")
            for issue in issues:
                print(f"   • {issue}")
            return False

        return True


class WikipediaCrawler:
    """Веб-краулер для сбора данных с русской Википедии"""

    def __init__(self, max_depth=3, max_pages=200, data_dir="data"):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.base_url = "https://ru.wikipedia.org"
        self.visited = set()
        self.pages_data = []
        self.data_dir = data_dir

    def clean_text(self, text):
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\[[^\]]+\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_content(self, soup):
        content_parts = []
        content_div = soup.find('div', {'id': 'mw-content-text'})

        if content_div:
            paragraphs = content_div.find_all('p')
            for p in paragraphs:
                text = self.clean_text(p.get_text())
                if len(text) > 80:
                    content_parts.append(text)

            headings = content_div.find_all(['h2', 'h3'])
            for h in headings:
                text = self.clean_text(h.get_text())
                if text and len(text) > 10:
                    content_parts.append(f"=== {text} ===")

        return content_parts

    def get_links(self, soup):
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('/wiki/') and ':' not in href and '#' not in href:
                if not any(prefix in href for prefix in ['/wiki/Служебная', '/wiki/Обсуждение', '/wiki/Файл', '/wiki/Категория']):
                    full_url = urljoin(self.base_url, href)
                    if full_url not in self.visited:
                        links.append(full_url)
        return links[:15]

    def crawl(self, start_url="/wiki/Заглавная_страница"):
        print(f"🕷️ Запуск веб-краулера для русской Википедии")
        print(f"   Глубина поиска: {self.max_depth}")
        print(f"   Максимум страниц: {self.max_pages}")
        print("-"*40)

        queue = deque()
        start_full_url = urljoin(self.base_url, start_url)
        queue.append((start_full_url, 0))

        while queue and len(self.pages_data) < self.max_pages:
            url, depth = queue.popleft()

            if url in self.visited:
                continue

            if depth > self.max_depth:
                continue

            print(f"📄 Загрузка ({depth}/{self.max_depth}): {url[:80]}...")

            try:
                response = requests.get(url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')

                title_tag = soup.find('h1', {'id': 'firstHeading'})
                title = title_tag.get_text() if title_tag else "Без названия"

                content_parts = self.extract_content(soup)

                if content_parts:
                    full_text = f"[{title}]\n" + "\n\n".join(content_parts)
                    self.pages_data.append(full_text)
                    print(f"   ✅ Добавлено {len(content_parts)} блоков (всего: {len(self.pages_data)})")
                else:
                    print(f"   ⚠️ Контент не найден")

                self.visited.add(url)

                new_links = self.get_links(soup)
                for link in new_links:
                    if link not in self.visited:
                        queue.append((link, depth + 1))

                time.sleep(0.3)

            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                self.visited.add(url)
                continue

        print(f"\n✅ Краулер завершил работу")
        print(f"   Загружено страниц: {len(self.pages_data)}")

        total_blocks = sum(len(p.split('\n\n')) for p in self.pages_data)
        print(f"   Всего блоков текста: {total_blocks}")

        return self.pages_data

    def save_to_file(self, filename="data/wiki_data.txt"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            for page in self.pages_data:
                f.write(page + "\n\n" + "="*80 + "\n\n")
        print(f"💾 Данные сохранены в {filename}")
        return filename


class TextDataset(Dataset):
    """Датасет для обучения"""

    def __init__(self, texts, tokenizer, max_seq_len: int, pad_id: int = 0, max_samples: int = None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.data = []

        print("📊 Создание датасета...")

        total_tokens = 0
        skipped_short = 0

        for text in texts:
            if not text or len(text.strip()) < 10:
                skipped_short += 1
                continue

            tokens = tokenizer.encode_with_bos_eos(text)
            total_tokens += len(tokens)

            if len(tokens) < 2:
                skipped_short += 1
                continue

            stride = max_seq_len // 2
            for i in range(0, len(tokens) - 1, stride):
                if max_samples and len(self.data) >= max_samples:
                    break

                seq = tokens[i:i + max_seq_len + 1]
                if len(seq) < 2:
                    continue

                x = seq[:-1]
                y = seq[1:]

                if len(x) < max_seq_len:
                    x = x + [pad_id] * (max_seq_len - len(x))
                    y = y + [pad_id] * (max_seq_len - len(y))

                self.data.append((x[:max_seq_len], y[:max_seq_len]))

            if max_samples and len(self.data) >= max_samples:
                break

        print(f"✅ Создано {len(self.data)} примеров для обучения")
        if skipped_short > 0:
            print(f"   Пропущено коротких текстов: {skipped_short}")
        if len(self.data) > 0 and len(texts) > 0:
            print(f"   Средняя длина последовательности: {total_tokens / max(1, len(texts)):.1f} токенов")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def load_priority_book(book_file: str = "У меня нету рта.txt", sample_ratio: float = 1.0) -> list:
    """Загружает приоритетную книгу с высоким приоритетом"""
    texts = []

    # Поиск файла в разных местах
    possible_paths = [
        book_file,
        os.path.join(os.path.dirname(__file__), book_file),
        os.path.join("data", "books", book_file),
        os.path.join("data", book_file),
        os.path.join("books", book_file),
    ]

    found_path = None
    for path in possible_paths:
        if os.path.exists(path):
            found_path = path
            break

    if found_path:
        print(f"📖 ЗАГРУЗКА ПРИОРИТЕТНОЙ КНИГИ: {found_path}")

        try:
            with open(found_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Разбиваем на абзацы
            paragraphs = content.split('\n\n')

            # Фильтруем пустые и слишком короткие
            for para in paragraphs:
                para = para.strip()
                if para and len(para) > 100:
                    # Добавляем маркер приоритетного источника
                    texts.append(f"[ПРИОРИТЕТНАЯ КНИГА] {para}")

            print(f"   ✅ Загружено {len(texts)} абзацев из приоритетной книги")
            print(f"   📊 Общий объём: {len(content):,} символов")

        except Exception as e:
            print(f"   ❌ Ошибка загрузки: {e}")
    else:
        print(f"⚠️ Приоритетная книга не найдена: {book_file}")
        print(f"   Искал в: {', '.join(possible_paths)}")

    return texts


def load_base_knowledge(base_file: str = "data/base.txt", sample_ratio: float = 1.0) -> list:
    """Загружает базу знаний с возможностью выборки"""
    texts = []

    if os.path.exists(base_file):
        print(f"📚 Загрузка базы знаний из {base_file}...")

        try:
            with open(base_file, 'r', encoding='utf-8') as f:
                content = f.read()

            words = [line.strip() for line in content.split('\n') if line.strip()]

            print(f"   Всего слов: {len(words):,}")

            if sample_ratio < 1.0:
                sample_size = int(len(words) * sample_ratio)
                words = random.sample(words, sample_size)
                print(f"   Используется {sample_ratio*100:.0f}%: {len(words):,} слов")

            chunk_size = 20 if sample_ratio >= 0.8 else 15
            for i in range(0, len(words), chunk_size):
                chunk = words[i:i+chunk_size]
                if chunk:
                    text = " ".join(chunk)
                    texts.append(text)

            print(f"✅ Загружено {len(texts):,} блоков")

        except Exception as e:
            print(f"   ❌ Ошибка: {e}")

    return texts


def load_dialogues(dialogues_file: str = "data/dialogues.txt") -> list:
    """Загружает диалоги"""
    texts = []

    if os.path.exists(dialogues_file):
        print(f"📚 Загрузка диалогов из {dialogues_file}...")

        with open(dialogues_file, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and ': ' in line:
                texts.append(line)

        print(f"✅ Загружено {len(texts)} реплик")

        full_dialogues = content.split('\n\n')
        for dialogue in full_dialogues:
            dialogue = dialogue.strip()
            if dialogue and len(dialogue) > 50:
                texts.append(dialogue)

        print(f"   Всего текстов после обработки: {len(texts)}")

    return texts


def load_wiki_data(wiki_file: str = "data/wiki_data.txt", sample_ratio: float = 1.0, use_crawler: bool = False, crawl_depth: int = 3) -> list:
    """Загружает данные из Википедии с возможностью краулера"""
    texts = []

    if use_crawler and not os.path.exists(wiki_file):
        print(f"🕷️ Запуск краулера Википедии...")
        crawler = WikipediaCrawler(max_depth=crawl_depth, max_pages=300)
        wiki_data = crawler.crawl()
        crawler.save_to_file(wiki_file)

    if os.path.exists(wiki_file):
        print(f"📚 Загрузка Википедии из {wiki_file}...")

        with open(wiki_file, 'r', encoding='utf-8') as f:
            content = f.read()

        pages = content.split("\n\n" + "="*80 + "\n\n")
        pages = [p for p in pages if p.strip()]

        print(f"   Всего страниц: {len(pages)}")

        if sample_ratio < 1.0:
            sample_size = int(len(pages) * sample_ratio)
            pages = random.sample(pages, sample_size)
            print(f"   Используется {sample_ratio*100:.0f}%: {len(pages)} страниц")

        for page in pages:
            page = page.strip()
            if page:
                parts = page.split('\n\n')
                max_parts = 10 if sample_ratio >= 0.8 else 6
                for part in parts[:max_parts]:
                    part = part.strip()
                    if len(part) > 100:
                        max_len = 1000 if sample_ratio >= 0.8 else 600
                        texts.append(part[:max_len])

        print(f"✅ Загружено {len(texts)} блоков из Википедии")

    return texts


def get_training_mode(use_crawler=False, crawl_depth=3, continue_training=False):
    """Запрашивает режим обучения у пользователя"""
    if use_crawler:
        print(f"🕷️ КРАУЛЕР ВИКИПЕДИИ ВКЛЮЧЕН (глубина {crawl_depth})")
        print()

    if continue_training:
        print("🔄 ПРОДОЛЖЕНИЕ ОБУЧЕНИЯ С ПОСЛЕДНЕГО СОХРАНЕНИЯ")
        print()

    print("\n" + "="*60)
    print("🎯 ВЫБЕРИТЕ РЕЖИМ ОБУЧЕНИЯ")
    print("="*60 + "\n")
    print("A. 🚀 ГЛУБОКОЕ ОБУЧЕНИЕ (максимум данных) - РЕКОМЕНДУЕТСЯ")
    print("   - Все данные (диалоги + Википедия + база знаний)")
    print("   - 50 эпох, увеличенная модель (512 hidden, 8 слоёв)")
    print("   - BPE токенизатор с 10000 токенов")
    print("   - Время: 3-5 часов на GPU, 40-60 часов на CPU\n")
    print("B. ⚡ СРЕДНЕЕ ОБУЧЕНИЕ (~65% данных)")
    print("   - Выборка 65% данных")
    print("   - 30 эпох, стандартная модель (384 hidden, 6 слоёв)")
    print("   - BPE токенизатор с 5000 токенов")
    print("   - Время: 1-2 часа на GPU, 15-20 часов на CPU\n")
    print("C. 🏃 МИНИМАЛЬНОЕ ОБУЧЕНИЕ (быстрое)")
    print("   - Выборка 20% данных")
    print("   - 20 эпох, маленькая модель (256 hidden, 4 слоя)")
    print("   - BPE токенизатор с 3000 токенов")
    print("   - Время: 30-40 минут на GPU, 3-4 часа на CPU\n")

    if continue_training:
        print("D. 🔄 ПРОДОЛЖИТЬ С ПОСЛЕДНЕГО СОХРАНЕНИЯ")
        print()

    while True:
        choice = input("Ваш выбор (A/B/C" + ("/D" if continue_training else "") + "): ").strip()
        if choice.upper() == 'A':
            return 'deep'
        elif choice.upper() == 'B':
            return 'medium'
        elif choice.upper() == 'C':
            return 'minimal'
        elif continue_training and choice.upper() == 'D':
            return 'continue'
        else:
            print("Пожалуйста, введите A, B или C" + ("/D" if continue_training else ""))


def train_model(use_crawler=False, crawl_depth=3, mode=None, continue_training=False):
    """Основная функция обучения с поддержкой продолжения и звуковыми сигналами"""

    # Проверяем наличие сохранённого состояния
    training_state = TrainingState()
    saved_state = training_state.load() if continue_training else None

    if saved_state and continue_training:
        print("\n🔄 Найдено сохранённое состояние обучения!")
        print(f"   Эпоха: {saved_state.epoch}")
        print(f"   Батч: {saved_state.batch}")
        print(f"   Лучший loss: {saved_state.best_loss:.4f}")
        print()

        cont = input("Продолжить обучение? (y/n): ").strip().lower()
        if cont not in ['y', 'да', 'yes']:
            saved_state = None
            print("Продолжение отменено, начинаем новое обучение.")

    # Если режим не передан, запрашиваем
    if mode is None:
        mode = get_training_mode(use_crawler, crawl_depth, saved_state is not None)

    if mode == 'continue' and saved_state:
        # Загружаем конфигурацию из сохранённого состояния
        mode = 'deep'  # Базовый режим, но будем использовать сохранённые параметры
        print("🔄 Продолжение обучения с последнего сохранения")
    elif mode == 'continue':
        print("⚠️ Нет сохранённого состояния для продолжения, начинаем новое обучение")
        mode = get_training_mode(use_crawler, crawl_depth, False)

    has_gpu = check_gpu()

    print("\n" + "=" * 60)
    print(f"🚀 ОБУЧЕНИЕ LLM - РЕЖИМ: {mode.upper()}")
    if use_crawler:
        print(f"🕷️ С КРАУЛЕРОМ ВИКИПЕДИИ (глубина {crawl_depth})")
    if saved_state:
        print(f"🔄 ПРОДОЛЖЕНИЕ С ЭПОХИ {saved_state.epoch}")
    print("=" * 60)

    config = ModelConfig()

    if mode == 'deep':
        config.num_epochs = 50
        config.batch_size = 32 if has_gpu else 12
        config.hidden_size = 512
        config.num_layers = 8
        config.num_heads = 8
        config.max_seq_len = 512
        config.vocab_size = 10000
        base_sample_ratio = 1.0
        wiki_sample_ratio = 1.0
        dataset_max_samples = None
        max_batches_per_epoch = None

    elif mode == 'medium':
        config.num_epochs = 30
        config.batch_size = 24 if has_gpu else 10
        config.hidden_size = 384
        config.num_layers = 6
        config.num_heads = 6
        config.max_seq_len = 384
        config.vocab_size = 5000
        base_sample_ratio = 0.65
        wiki_sample_ratio = 0.65
        dataset_max_samples = 80000
        max_batches_per_epoch = 800

    else:  # minimal
        config.num_epochs = 20
        config.batch_size = 16 if has_gpu else 8
        config.hidden_size = 256
        config.num_layers = 4
        config.num_heads = 4
        config.max_seq_len = 256
        config.vocab_size = 3000
        base_sample_ratio = 0.2
        wiki_sample_ratio = 0.2
        dataset_max_samples = 20000
        max_batches_per_epoch = 300

    # Если продолжаем обучение, проверяем совместимость
    if saved_state:
        if not saved_state.check_compatibility(config):
            print("\n⚠️ Конфигурация изменилась! Продолжение обучения невозможно.")
            print("Начинаем обучение с нуля.")
            saved_state = None
        else:
            # Используем сохранённые параметры
            start_epoch = saved_state.epoch
            best_loss = saved_state.best_loss
            losses_history = saved_state.losses_history
    else:
        start_epoch = 0
        best_loss = float('inf')
        losses_history = []

    print(f"\n📋 КОНФИГУРАЦИЯ ДЛЯ РЕЖИМА {mode.upper()}:")
    print(f"   vocab_size: {config.vocab_size}")
    print(f"   hidden_size: {config.hidden_size}")
    print(f"   num_layers: {config.num_layers}")
    print(f"   num_heads: {config.num_heads}")
    print(f"   max_seq_len: {config.max_seq_len}")
    print(f"   num_epochs: {config.num_epochs}")
    print(f"   batch_size: {config.batch_size}")
    print(f"   device: {config.device}")
    print(f"   tokenizer: BPE")

    # 1. Загружаем данные
    print("\n" + "-" * 60)
    print("📖 ЗАГРУЗКА ДАННЫХ")
    print("-" * 60)

    start_time = time.time()

    dialogue_texts = load_dialogues("data/dialogues.txt")
    base_texts = load_base_knowledge("data/base.txt", sample_ratio=base_sample_ratio)
    wiki_texts = load_wiki_data("data/wiki_data.txt", sample_ratio=wiki_sample_ratio,
                                use_crawler=use_crawler, crawl_depth=crawl_depth)

    # ЗАГРУЗКА ПРИОРИТЕТНОЙ КНИГИ
    priority_texts = load_priority_book("У меня нету рта.txt", sample_ratio=1.0)

    # Приоритетные тексты добавляются с дублированием для усиления
    all_texts = priority_texts * 3 + dialogue_texts + base_texts + wiki_texts

    print(f"\n📊 ИТОГО ТЕКСТОВ: {len(all_texts):,}")
    print(f"   - ПРИОРИТЕТНАЯ КНИГА: {len(priority_texts):,} (x3 = {len(priority_texts)*3})")
    print(f"   - Диалогов: {len(dialogue_texts):,}")
    print(f"   - База знаний: {len(base_texts):,}")
    print(f"   - Википедия: {len(wiki_texts):,}")
    print(f"   Время загрузки: {time.time() - start_time:.1f} сек")

    if len(all_texts) < 50:
        print("⚠️ Мало данных для качественного обучения!")
        return

    # 2. Обучаем токенизатор
    print("\n" + "-" * 60)
    print("🔤 ОБУЧЕНИЕ ТОКЕНИЗАТОРА")
    print("-" * 60)

    tokenizer = BPETokenizer(config.vocab_size)

    actual_vocab_size = tokenizer.train(all_texts, model_prefix="data/models/tokenizer")
    config.vocab_size = actual_vocab_size
    print(f"📊 Итоговый размер словаря: {actual_vocab_size}")

    # 3. Создаём датасет
    print("\n" + "-" * 60)
    print("📊 СОЗДАНИЕ ДАТАСЕТА")
    print("-" * 60)

    dataset = TextDataset(all_texts, tokenizer, config.max_seq_len, tokenizer.pad_id,
                          max_samples=dataset_max_samples)

    if len(dataset) == 0:
        print("❌ Нет данных для обучения!")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2 if has_gpu else 0,
        pin_memory=has_gpu
    )

    # 4. Создаём модель
    print("\n" + "-" * 60)
    print("🤖 СОЗДАНИЕ МОДЕЛИ")
    print("-" * 60)

    model = LLM(config)
    model.to(config.device)

    total_params = model.get_num_params()
    print(f"✅ Модель создана")
    print(f"   Параметров: {total_params:,}")
    print(f"   Размер: {total_params * 4 / (1024 * 1024):.1f} MB")
    print(f"   Устройство: {config.device}")

    # 5. Оптимизатор
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    if mode == 'deep':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    # Загружаем сохранённое состояние оптимизатора
    if saved_state and saved_state.optimizer_state:
        try:
            optimizer.load_state_dict(saved_state.optimizer_state)
            print("✅ Состояние оптимизатора загружено")
        except Exception as e:
            print(f"⚠️ Не удалось загрузить состояние оптимизатора: {e}")

    if saved_state and saved_state.scheduler_state and mode == 'deep':
        try:
            scheduler.load_state_dict(saved_state.scheduler_state)
            print("✅ Состояние планировщика загружено")
        except Exception as e:
            print(f"⚠️ Не удалось загрузить состояние планировщика: {e}")

    # 6. Обучение
    print("\n" + "-" * 60)
    print(f"🏋️ НАЧАЛО ОБУЧЕНИЯ")
    print(f"   Эпох: {config.num_epochs}")
    if start_epoch > 0:
        print(f"   Начало с эпохи: {start_epoch}")
    if max_batches_per_epoch:
        print(f"   Батчей на эпоху: {max_batches_per_epoch}")
    print("-" * 60)

    model.train()
    best_loss = best_loss
    losses_history = losses_history

    beep(1, 800, 300)  # Начальный сигнал
    total_start = time.time()

    # Переменная для отслеживания последнего сигнала
    last_beep_batch = 0

    try:
        for epoch in range(start_epoch, config.num_epochs):
            total_loss = 0
            num_batches = 0
            epoch_start = time.time()

            if epoch > 0:
                beep(1, 600+(epoch*10), 200)

            for batch_idx, (x, y) in enumerate(dataloader):
                if max_batches_per_epoch and batch_idx >= max_batches_per_epoch:
                    break

                if saved_state and epoch == start_epoch and batch_idx < saved_state.batch:
                    continue

                x, y = x.to(config.device), y.to(config.device)

                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits.view(-1, config.vocab_size), y.view(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if batch_idx % 100 == 0 and batch_idx > 0:
                    training_state.save(epoch, batch_idx, best_loss, losses_history, optimizer,
                                        scheduler if mode == 'deep' else None, config)

                if batch_idx % 50 == 0 and batch_idx > 0 and batch_idx != last_beep_batch:
                    beep(1, 500, 100)
                    last_beep_batch = batch_idx

                if batch_idx % 50 == 0 or batch_idx == (max_batches_per_epoch or 0) - 1:
                    avg_loss = total_loss / num_batches
                    print(
                        f"Epoch {epoch + 1}/{config.num_epochs} | Batch {batch_idx} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")

            avg_loss = total_loss / num_batches
            losses_history.append(avg_loss)
            epoch_time = time.time() - epoch_start

            if mode == 'deep':
                scheduler.step()
                lr = scheduler.get_last_lr()[0]
                print(f"\n✅ Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Время: {epoch_time:.0f}с | LR: {lr:.6f}")
            else:
                print(f"\n✅ Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Время: {epoch_time:.0f}с")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), config.model_path)
                print(f"   💾 Сохранена (loss: {best_loss:.4f})")
                beep(2, 1200, 150)  # Двойной сигнал при сохранении

            # Сохраняем состояние после каждой эпохи
            training_state.save(epoch + 1, 0, best_loss, losses_history, optimizer,
                               scheduler if mode == 'deep' else None, config)

            print()

    except KeyboardInterrupt:
        print("\n⚠️ Обучение прервано")
        torch.save(model.state_dict(), config.model_path)
        training_state.save(epoch, batch_idx, best_loss, losses_history, optimizer,
                           scheduler if mode == 'deep' else None, config)
        beep(3, 500, 200)  # Тройной сигнал при прерывании

    total_time = time.time() - total_start

    print("\n🔔 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    beep(5, 1500, 300)  # Пять сигналов в конце

    model_info = {
        'mode': mode,
        'crawler_used': use_crawler,
        'crawler_depth': crawl_depth if use_crawler else None,
        'config': {
            'vocab_size': config.vocab_size,
            'hidden_size': config.hidden_size,
            'num_layers': config.num_layers,
            'num_heads': config.num_heads,
            'max_seq_len': config.max_seq_len,
            'num_epochs': config.num_epochs,
            'batch_size': config.batch_size,
            'use_bpe': True
        },
        'total_params': total_params,
        'trained_on_dialogues': len(dialogue_texts),
        'trained_on_knowledge': len(base_texts),
        'trained_on_wiki': len(wiki_texts),
        'trained_on_priority': len(priority_texts),
        'total_texts': len(all_texts),
        'total_examples': len(dataset),
        'vocab_size_actual': config.vocab_size,
        'best_loss': best_loss if best_loss != float('inf') else None,
        'losses_history': losses_history,
        'gpu_used': has_gpu,
        'training_time_hours': total_time / 3600,
        'epochs_completed': config.num_epochs
    }

    with open("data/models/model_info.json", 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    print(f"\n📊 СТАТИСТИКА ОБУЧЕНИЯ:")
    print(f"   Режим: {mode.upper()}")
    print(f"   Краулер: {'Включен' if use_crawler else 'Выключен'}")
    print(f"   Токенизатор: BPE")
    print(f"   Реальный размер словаря: {config.vocab_size}")
    print(f"   Модель: {config.model_path}")
    print(f"   Всего текстов: {len(all_texts):,}")
    print(f"   Из них приоритетная книга: {len(priority_texts):,} (x3)")
    print(f"   Примеров обучения: {len(dataset):,}")
    print(f"   Лучший loss: {best_loss:.4f}")
    print(f"   Время обучения: {total_time / 3600:.1f} часов")
    print(f"\n🚀 Запуск ассистента: python main.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение LLM с поддержкой краулера")
    parser.add_argument("--crawler", action="store_true", help="Включить краулер Википедии")
    parser.add_argument("--depth", type=int, default=3, help="Глубина краулера (по умолчанию 3)")
    parser.add_argument("--mode", type=str, choices=['deep', 'medium', 'minimal', 'continue'],
                        help="Режим обучения (deep/medium/minimal/continue)")
    parser.add_argument("--continue", dest="continue_training", action="store_true",
                        help="Продолжить обучение с последнего сохранения")
    args = parser.parse_args()

    # Если указан режим continue, автоматически включаем continue_training
    if args.mode == 'continue':
        args.continue_training = True

    train_model(use_crawler=args.crawler, crawl_depth=args.depth,
                mode=args.mode, continue_training=args.continue_training)