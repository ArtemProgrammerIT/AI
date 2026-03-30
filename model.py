"""
Реализация Decoder-only Transformer (как GPT) с нуля на PyTorch
Добавлена поддержка сохранения и загрузки
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadAttention(nn.Module):
    """Многоголовое внимание"""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Проецируем
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Вычисляем внимание
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Применяем внимание к значениям
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        return self.out_proj(context)


class FeedForward(nn.Module):
    """Feed-forward слой с GELU активацией"""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Один блок трансформера"""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.feed_forward = FeedForward(hidden_size, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        # Feed-forward with residual
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)

        return x


class LLM(nn.Module):
    """Полная LLM модель (Decoder-only Transformer)"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Эмбеддинги
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_size)

        # Трансформерные блоки
        self.blocks = nn.ModuleList([
            TransformerBlock(config.hidden_size, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        ])

        # Выходной слой
        self.norm = nn.LayerNorm(config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.vocab_size)

        self.dropout = nn.Dropout(config.dropout)

        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов (Xavier/Glorot для линейных слоёв)"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход

        Args:
            input_ids: [batch_size, seq_len]

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Создаём causal mask (нижний треугольник)
        mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
        mask = mask.to(input_ids.device)

        # Эмбеддинги
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)

        x = self.dropout(token_emb + pos_emb)

        # Проходим через блоки
        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)
        logits = self.output_proj(x)

        return logits

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 0.8, top_k: int = 50) -> torch.Tensor:
        """
        Генерация текста

        Args:
            input_ids: Начальная последовательность [batch_size, seq_len]
            max_new_tokens: Максимальное количество новых токенов
            temperature: Температура для семплирования
            top_k: Top-k фильтрация

        Returns:
            torch.Tensor: Полная последовательность с сгенерированными токенами
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Обрезаем до максимальной длины
            if input_ids.size(1) > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]

            # Получаем логиты
            logits = self(input_ids)
            logits = logits[:, -1, :] / temperature

            # Top-k фильтрация
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
                logits[indices_to_remove] = float('-inf')

            # Softmax и семплирование
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Добавляем токен
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Проверяем на EOS
            if hasattr(self.config, 'eos_id') and next_token.item() == self.config.eos_id:
                break

        return input_ids

    def get_num_params(self) -> int:
        """Возвращает количество параметров модели"""
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str):
        """Сохраняет модель"""
        torch.save(self.state_dict(), path)
        print(f"💾 Модель сохранена в {path}")

    def load(self, path: str):
        """Загружает модель"""
        self.load_state_dict(torch.load(path, map_location='cpu'))
        print(f"📂 Модель загружена из {path}")