"""
Конфигурация модели — обновлена для работы с BPE и большим словарём
"""

import torch
import os


class ModelConfig:
    """Конфигурация архитектуры LLM"""

    def __init__(self):
        # Размеры модели (увеличены для лучшего качества)
        self.vocab_size = 10000         # Увеличен с 5000 до 10000 для BPE
        self.hidden_size = 512          # Увеличен с 256 для лучшей ёмкости
        self.num_layers = 8             # Увеличен с 4 для глубины
        self.num_heads = 8              # Увеличен с 4
        self.max_seq_len = 512          # Увеличен с 256 для длинного контекста
        self.dropout = 0.1              # Dropout для регуляризации

        # Обучение
        self.batch_size = 16             # Увеличен для GPU (на CPU уменьшить до 8)
        self.learning_rate = 3e-4        # Скорость обучения
        self.num_epochs = 50             # Увеличен для лучшего обучения
        self.warmup_steps = 500          # Шаги разогрева

        # Генерация
        self.temperature = 0.8           # Температура (креативность)
        self.top_k = 50                  # Top-k фильтрация
        self.top_p = 0.9                 # Top-p (nucleus) фильтрация

        # Пути
        self.model_path = "data/models/llm_model.pt"
        self.tokenizer_path = "data/models/tokenizer.model"
        self.vocab_path = "data/models/tokenizer.vocab"
        self.dialogues_path = "data/dialogues.txt"

        # Токенизация — ВКЛЮЧАЕМ BPE для нормального словаря
        self.use_bpe_tokenizer = True     # True = использовать BPE, False = CharTokenizer

        # Создаём директории
        os.makedirs("data/models", exist_ok=True)
        os.makedirs("data/knowledge", exist_ok=True)

        # EOS токен ID (будет установлен после загрузки токенизатора)
        self.eos_id = 3

        # Определяем устройство
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Автоматическая настройка batch_size под устройство
        if torch.cuda.is_available():
            self.batch_size = 32          # На GPU можно больше
        else:
            self.batch_size = 8           # На CPU меньше

    def __repr__(self):
        return f"""ModelConfig(
    vocab_size={self.vocab_size},
    hidden_size={self.hidden_size},
    num_layers={self.num_layers},
    num_heads={self.num_heads},
    max_seq_len={self.max_seq_len},
    num_epochs={self.num_epochs},
    device={self.device}
)"""