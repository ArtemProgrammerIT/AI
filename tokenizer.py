"""
Токенизатор на основе SentencePiece (BPE) и символьный токенизатор
"""

import os
import pickle
from typing import List, Dict
import sentencepiece as spm


class BPETokenizer:
    """
    BPE токенизатор на основе SentencePiece
    Поддерживает русский и английский языки
    """

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.sp = None
        self.model_path = None
        self.vocab_path = None

        # Специальные токены
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'

        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

    def train(self, texts: List[str], model_prefix: str = "tokenizer"):
        """
        Обучает BPE токенизатор на текстах
        """
        if not texts:
            raise ValueError("Нет текстов для обучения токенизатора")

        print(f"📝 Обучение BPE токенизатора на {len(texts)} текстах...")
        print(f"   Целевой размер словаря: {self.vocab_size}")

        # Сохраняем тексты во временный файл
        temp_file = "temp_corpus.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            for text in texts:
                # Очищаем текст, но сохраняем структуру
                clean_text = ' '.join(text.split())
                if clean_text:
                    f.write(clean_text + '\n')

        # Получаем размер файла для информации
        if os.path.exists(temp_file):
            file_size = os.path.getsize(temp_file) / (1024 * 1024)
            print(f"   Размер корпуса: {file_size:.1f} MB")

        # Обучаем модель
        # Важно: не добавляем <UNK> в user_defined_symbols, так как он уже определён как unk_piece
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type='bpe',
            character_coverage=0.9995,
            input_sentence_size=1000000,
            shuffle_input_sentence=True,
            normalization_rule_name='nmt_nfkc',
            remove_extra_whitespaces=True,
            add_dummy_prefix=True,
            allow_whitespace_only_pieces=True,
            unk_piece=self.unk_token,
            bos_piece=self.bos_token,
            eos_piece=self.eos_token,
            pad_piece=self.pad_token,
            user_defined_symbols=[self.pad_token, self.bos_token, self.eos_token]  # <UNK> не добавляем
        )

        # Загружаем модель
        self.model_path = f"{model_prefix}.model"
        self.vocab_path = f"{model_prefix}.vocab"
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model_path)

        # Обновляем индексы специальных токенов
        self._update_special_tokens()

        # Удаляем временный файл
        if os.path.exists(temp_file):
            os.remove(temp_file)

        print(f"✅ BPE токенизатор обучен. Реальный размер словаря: {len(self.sp)}")
        return len(self.sp)

    def _update_special_tokens(self):
        """Обновляет индексы специальных токенов"""
        if self.sp:
            self.pad_id = self.sp.piece_to_id(self.pad_token)
            self.unk_id = self.sp.piece_to_id(self.unk_token)
            self.bos_id = self.sp.piece_to_id(self.bos_token)
            self.eos_id = self.sp.piece_to_id(self.eos_token)

    def load(self, model_path: str):
        """Загружает обученную модель"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.model_path = model_path
        self.vocab_path = model_path.replace('.model', '.vocab')
        self._update_special_tokens()

        print(f"✅ BPE токенизатор загружен. Размер словаря: {len(self.sp)}")

    def encode(self, text: str) -> List[int]:
        """Кодирует текст в последовательность токенов"""
        if not self.sp:
            raise RuntimeError("Токенизатор не загружен")
        if not text:
            return []
        text = ' '.join(text.split())
        return self.sp.encode(text, out_type=int)

    def encode_with_bos_eos(self, text: str) -> List[int]:
        """Кодирует текст с добавлением BOS и EOS токенов"""
        tokens = self.encode(text)
        return [self.bos_id] + tokens + [self.eos_id]

    def decode(self, tokens: List[int]) -> str:
        """Декодирует последовательность токенов в текст"""
        if not self.sp:
            raise RuntimeError("Токенизатор не загружен")
        if not tokens:
            return ""
        text = self.sp.decode(tokens)
        return ' '.join(text.split())

    def get_vocab_size(self) -> int:
        """Возвращает реальный размер словаря"""
        return len(self.sp) if self.sp else self.vocab_size

    def __len__(self):
        return self.get_vocab_size()

    @property
    def pad_id(self):
        return self._pad_id

    @pad_id.setter
    def pad_id(self, value):
        self._pad_id = value

    @property
    def unk_id(self):
        return self._unk_id

    @unk_id.setter
    def unk_id(self, value):
        self._unk_id = value

    @property
    def bos_id(self):
        return self._bos_id

    @bos_id.setter
    def bos_id(self, value):
        self._bos_id = value

    @property
    def eos_id(self):
        return self._eos_id

    @eos_id.setter
    def eos_id(self, value):
        self._eos_id = value


class CharTokenizer:
    """
    Простой символьный токенизатор
    """

    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        self.model_path = None

        # Специальные токены
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'

        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

    def train(self, texts: List[str], model_prefix: str = "tokenizer"):
        """
        Собирает все символы из текстов и создаёт словарь
        """
        if not texts:
            raise ValueError("Нет текстов для обучения токенизатора")

        print(f"📝 Обучение символьного токенизатора на {len(texts)} текстах...")

        # Собираем все уникальные символы
        chars = set()
        for text in texts:
            chars.update(text)

        print(f"   Найдено уникальных символов: {len(chars)}")

        # Специальные токены
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]

        # Сортируем символы для детерминированности
        all_tokens = special_tokens + sorted(chars)

        # Ограничиваем до vocab_size
        if len(all_tokens) > self.vocab_size:
            print(f"   ⚠️ Количество токенов ({len(all_tokens)}) превышает vocab_size ({self.vocab_size})")
            all_tokens = all_tokens[:self.vocab_size]
            print(f"   Использовано {len(all_tokens)} токенов")

        # Создаём словари
        self.char_to_id = {ch: i for i, ch in enumerate(all_tokens)}
        self.id_to_char = {i: ch for i, ch in enumerate(all_tokens)}

        # Обновляем специальные ID
        self.pad_id = self.char_to_id.get(self.pad_token, 0)
        self.unk_id = self.char_to_id.get(self.unk_token, 1)
        self.bos_id = self.char_to_id.get(self.bos_token, 2)
        self.eos_id = self.char_to_id.get(self.eos_token, 3)

        # Сохраняем модель
        self.model_path = f"{model_prefix}.model"
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'char_to_id': self.char_to_id,
                'id_to_char': self.id_to_char,
                'vocab_size': len(self.char_to_id),
                'pad_id': self.pad_id,
                'unk_id': self.unk_id,
                'bos_id': self.bos_id,
                'eos_id': self.eos_id
            }, f)

        actual_vocab_size = len(self.char_to_id)
        print(f"✅ Символьный токенизатор обучен. Размер словаря: {actual_vocab_size}")

        return actual_vocab_size

    def load(self, model_path: str):
        """Загружает обученную модель"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        self.char_to_id = data['char_to_id']
        self.id_to_char = data['id_to_char']
        self.vocab_size = data['vocab_size']
        self.pad_id = data.get('pad_id', 0)
        self.unk_id = data.get('unk_id', 1)
        self.bos_id = data.get('bos_id', 2)
        self.eos_id = data.get('eos_id', 3)
        self.model_path = model_path

        print(f"✅ Символьный токенизатор загружен. Размер словаря: {self.vocab_size}")

    def encode(self, text: str) -> List[int]:
        """Кодирует текст в последовательность ID символов"""
        return [self.char_to_id.get(ch, self.unk_id) for ch in text]

    def encode_with_bos_eos(self, text: str) -> List[int]:
        """Кодирует текст с добавлением BOS и EOS токенов"""
        return [self.bos_id] + self.encode(text) + [self.eos_id]

    def decode(self, tokens: List[int]) -> str:
        """Декодирует последовательность ID в текст"""
        chars = []
        for i in tokens:
            if i in self.id_to_char:
                ch = self.id_to_char[i]
                if ch not in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
                    chars.append(ch)
            else:
                chars.append('?')
        return ''.join(chars)

    def get_vocab_size(self) -> int:
        """Возвращает размер словаря"""
        return self.vocab_size

    def get_vocab(self) -> Dict[str, int]:
        """Возвращает словарь токен -> ID"""
        return self.char_to_id.copy()

    def __len__(self):
        return self.vocab_size