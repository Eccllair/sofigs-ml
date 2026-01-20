import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class ComplexEventEmbedding(nn.Module):
    """Embedding для сложных событий с несколькими признаками"""

    def __init__(
        self,
        n_actions: int,
        n_params: int,
        n_values: int,
        n_categories: int,
        n_seasons: int = 4,
        embedding_dim: int = 64,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        # Embedding слои для категориальных признаков
        # Увеличиваем на 1 для padding (индекс 0)
        self.action_embedding = nn.Embedding(
            n_actions + 1, embedding_dim, padding_idx=0
        )
        self.param_embedding = nn.Embedding(n_params + 1, embedding_dim, padding_idx=0)
        self.value_embedding = nn.Embedding(n_values + 1, embedding_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(
            n_categories + 1, embedding_dim, padding_idx=0
        )
        self.season_embedding = nn.Embedding(
            n_seasons + 1, embedding_dim, padding_idx=0
        )

        # Для числового признака (дни с предыдущего действия)
        self.days_projection = nn.Linear(1, embedding_dim)

        # Слой для объединения всех эмбеддингов
        self.total_embedding_dim = embedding_dim * 6
        self.combine_projection = nn.Linear(self.total_embedding_dim, embedding_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        actions: torch.Tensor,
        params: torch.Tensor,
        values: torch.Tensor,
        categories: torch.Tensor,
        days_since_prev: torch.Tensor,
        seasons: torch.Tensor,
    ) -> torch.Tensor:
        # Получаем эмбеддинги
        action_emb = self.action_embedding(actions)
        param_emb = self.param_embedding(params)
        value_emb = self.value_embedding(values)
        category_emb = self.category_embedding(categories)
        season_emb = self.season_embedding(seasons)

        # Обрабатываем числовой признак
        days_emb = self.days_projection(days_since_prev.unsqueeze(-1).float())

        # Объединяем все эмбеддинги
        combined = torch.cat(
            [action_emb, param_emb, value_emb, category_emb, days_emb, season_emb],
            dim=-1,
        )

        # Проекция и нормализация
        projected = self.combine_projection(combined)
        projected = self.dropout(projected)
        projected = self.layer_norm(projected)

        return projected


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention слой"""

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_rate)
        self.output_linear = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Линейные преобразования
        Q = (
            self.query(x)
            .view(batch_size, seq_len, self.num_heads, self.head_size)
            .transpose(1, 2)
        )
        K = (
            self.key(x)
            .view(batch_size, seq_len, self.num_heads, self.head_size)
            .transpose(1, 2)
        )
        V = (
            self.value(x)
            .view(batch_size, seq_len, self.num_heads, self.head_size)
            .transpose(1, 2)
        )

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_size**0.5)

        # Применяем маску
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Применяем attention weights
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Выходной слой
        output = self.output_linear(context)
        output = self.dropout(output)

        # Add & Norm
        output = self.layer_norm(output + x)

        return output


class FeedForward(nn.Module):
    """Feed Forward Network"""

    def __init__(self, hidden_size: int, dropout_rate: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x


class SASRecBlock(nn.Module):
    """Блок трансформера для SASRec"""

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout_rate)
        self.ffn = FeedForward(hidden_size, dropout_rate)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.attention(x, mask)
        x = self.ffn(x)
        return x


class MultiTaskPredictionHead(nn.Module):
    """Multi-task слой"""

    def __init__(
        self,
        hidden_size: int,
        n_actions: int,
        n_params: int,
        n_values: int,
        n_categories: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        # Общие слои
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_size),
        )

        # Головы для каждого типа предсказания
        # +1 для учета padding (индекс 0)
        self.action_head = nn.Linear(hidden_size, n_actions + 1)
        self.param_head = nn.Linear(hidden_size, n_params + 1)
        self.value_head = nn.Linear(hidden_size, n_values + 1)
        self.category_head = nn.Linear(hidden_size, n_categories + 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_features = self.shared_layer(x)

        action_logits = self.action_head(shared_features)
        param_logits = self.param_head(shared_features)
        value_logits = self.value_head(shared_features)
        category_logits = self.category_head(shared_features)

        return {
            "action": action_logits,
            "param": param_logits,
            "value": value_logits,
            "category": category_logits,
        }


class ComplexSASRec(nn.Module):
    """SASRec модель для пердсказания последовательностей событий"""

    def __init__(
        self,
        n_actions: int,
        n_params: int,
        n_values: int,
        n_categories: int,
        n_seasons: int = 4,
        max_seq_length: int = 50,
        embedding_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 4,
        dropout_rate: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 32,
        num_epochs: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.n_actions = n_actions
        self.n_params = n_params
        self.n_values = n_values
        self.n_categories = n_categories
        self.n_seasons = n_seasons
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device

        # Позиционные эмбеддинги
        self.positional_embedding = nn.Embedding(max_seq_length, embedding_dim)

        # Эмбеддинг событий
        self.event_embedding = ComplexEventEmbedding(
            n_actions=n_actions,
            n_params=n_params,
            n_values=n_values,
            n_categories=n_categories,
            n_seasons=n_seasons,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
        )

        # Блоки трансформера
        self.blocks = nn.ModuleList(
            [
                SASRecBlock(embedding_dim, num_heads, dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        # Головы для предсказания
        self.prediction_head = MultiTaskPredictionHead(
            hidden_size=embedding_dim,
            n_actions=n_actions,
            n_params=n_params,
            n_values=n_values,
            n_categories=n_categories,
            dropout_rate=dropout_rate,
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=0
        )  # Игнорируем padding (индекс 0)

        # Инициализация весов
        self._init_weights()

        # Перенос модели на устройство
        self.to(self.device)

    def _init_weights(self):
        """Инициализация весов"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def create_mask(self, sequence_lengths: torch.Tensor) -> torch.Tensor:
        """Создание маски для последовательности"""
        batch_size = sequence_lengths.shape[0]
        max_len = self.max_seq_length

        mask = (
            torch.arange(max_len)
            .expand(batch_size, max_len)
            .to(sequence_lengths.device)
        )
        mask = mask < sequence_lengths.unsqueeze(1)

        return mask

    def forward(
        self,
        actions: torch.Tensor,
        params: torch.Tensor,
        values: torch.Tensor,
        categories: torch.Tensor,
        days_since_prev: torch.Tensor,
        seasons: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = actions.shape

        # Получаем эмбеддинги событий
        event_emb = self.event_embedding(
            actions, params, values, categories, days_since_prev, seasons
        )

        # Добавляем позиционные эмбеддинги
        positions = (
            torch.arange(seq_len, device=actions.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        pos_emb = self.positional_embedding(positions)
        x = event_emb + pos_emb
        x = self.dropout(x)

        # Создаем маску
        mask = self.create_mask(sequence_lengths)

        # Применяем блоки трансформера
        for block in self.blocks:
            x = block(x, mask)

        # Берем последний ненулевой элемент каждой последовательности
        last_idx = sequence_lengths - 1
        batch_indices = torch.arange(batch_size, device=x.device)
        last_hidden = x[batch_indices, last_idx]

        # Получаем предсказания
        predictions = self.prediction_head(last_hidden)

        return predictions

    def compute_loss(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Вычисление multi-task loss"""
        losses = []

        for key in ["action", "param", "value", "category"]:
            pred_logits = predictions[key]
            target = targets[key]

            # Применяем cross entropy loss
            # Таргеты уже должны быть в правильном формате (0 для padding, 1...n для классов)
            loss = self.loss_fn(pred_logits, target)
            if not torch.isnan(loss):
                losses.append(loss)

        # Среднее по всем задачам
        total_loss = sum(losses) / len(losses)
        return total_loss

    def prepare_sequences(
        self, user_sequences: List[Dict[str, List]]
    ) -> Dict[str, torch.Tensor]:
        """
        Подготовка последовательностей событий для обучения.

        Args:
            user_sequences: Список словарей с последовательностями для каждого пользователя

        Returns:
            Словарь с тензорами для обучения
        """
        batch_data = {
            "actions": [],
            "params": [],
            "values": [],
            "categories": [],
            "days_since_prev": [],
            "seasons": [],
            "sequence_lengths": [],
            "next_action": [],
            "next_param": [],
            "next_value": [],
            "next_category": [],
        }

        for seq in user_sequences:
            if len(seq["actions"]) < 2:
                continue

            # Все кроме последнего события - контекст
            seq_len = len(seq["actions"])
            context_len = min(seq_len - 1, self.max_seq_length)

            # Берем последние context_len событий в качестве контекста
            start_idx = seq_len - context_len - 1

            # Заполняем данные (убеждаемся, что индексы начинаются с 1)
            batch_data["actions"].append([x for x in seq["actions"][start_idx:-1]])
            batch_data["params"].append([x for x in seq["params"][start_idx:-1]])
            batch_data["values"].append([x for x in seq["values"][start_idx:-1]])
            batch_data["categories"].append(
                [x for x in seq["categories"][start_idx:-1]]
            )
            batch_data["days_since_prev"].append(
                [x for x in seq["days_since_prev"][start_idx:-1]]
            )
            batch_data["seasons"].append([x for x in seq["seasons"][start_idx:-1]])
            batch_data["sequence_lengths"].append(context_len)

            # Цели (следующее событие)
            batch_data["next_action"].append(seq["actions"][-1])
            batch_data["next_param"].append(seq["params"][-1])
            batch_data["next_value"].append(seq["values"][-1])
            batch_data["next_category"].append(seq["categories"][-1])

        # Конвертируем в тензоры и паддим до максимальной длины
        processed_data = {}

        for key in batch_data:
            if key == "sequence_lengths":
                processed_data[key] = torch.tensor(
                    batch_data[key], dtype=torch.long
                ).to(self.device)
            elif key in ["next_action", "next_param", "next_value", "next_category"]:
                # Это целые числа, преобразуем в тензор
                # Не вычитаем 1, так как у нас padding индекс 0
                processed_data[key] = torch.tensor(
                    batch_data[key], dtype=torch.long
                ).to(self.device)
            else:
                # Паддим последовательности
                padded_sequences = []
                for seq in batch_data[key]:
                    if len(seq) < self.max_seq_length:
                        pad_len = self.max_seq_length - len(seq)
                        if key == "days_since_prev":
                            padded_seq = list(seq) + [0.0] * pad_len
                        else:
                            padded_seq = list(seq) + [0] * pad_len  # padding индекс = 0
                    else:
                        padded_seq = seq[: self.max_seq_length]
                    padded_sequences.append(padded_seq)

                # Конвертируем в тензор
                if key == "days_since_prev":
                    processed_data[key] = torch.tensor(
                        padded_sequences, dtype=torch.float32
                    ).to(self.device)
                else:
                    processed_data[key] = torch.tensor(
                        padded_sequences, dtype=torch.long
                    ).to(self.device)

        return processed_data

    def train_on_sequences(
        self,
        train_sequences: List[Dict[str, List]],
        val_sequences: Optional[List[Dict[str, List]]] = None,
    ):
        """
        Обучение модели на подготовленных последовательностях.

        Args:
            train_sequences: Обучающие последовательности
            val_sequences: Валидационные последовательности (опционально)
        """
        from torch.utils.data import DataLoader, TensorDataset

        # Подготавливаем данные
        train_data = self.prepare_sequences(train_sequences)

        # Создаем DataLoader
        dataset = TensorDataset(
            train_data["actions"],
            train_data["params"],
            train_data["values"],
            train_data["categories"],
            train_data["days_since_prev"],
            train_data["seasons"],
            train_data["sequence_lengths"],
            train_data["next_action"],
            train_data["next_param"],
            train_data["next_value"],
            train_data["next_category"],
        )

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Оптимизатор
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # Обучение
        self.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()

                # Разбираем батч
                (
                    actions,
                    params,
                    values,
                    categories,
                    days_since_prev,
                    seasons,
                    seq_lens,
                    next_action,
                    next_param,
                    next_value,
                    next_category,
                ) = batch

                # Прямой проход
                predictions = self(
                    actions,
                    params,
                    values,
                    categories,
                    days_since_prev,
                    seasons,
                    seq_lens,
                )

                # Цели (уже в правильном формате: 1...n)
                targets = {
                    "action": next_action,
                    "param": next_param,
                    "value": next_value,
                    "category": next_category,
                }

                # Вычисляем loss
                loss = self.compute_loss(predictions, targets)

                # Обратный проход
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

            print(
                f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {total_loss / len(dataloader):.4f}"
            )

            # Валидация
            if val_sequences is not None:
                val_loss = self.evaluate_on_sequences(val_sequences)
                print(f"Validation Loss: {val_loss:.4f}")

    def evaluate_on_sequences(self, sequences: List[Dict[str, List]]) -> float:
        """Оценка модели на последовательностях"""
        from torch.utils.data import DataLoader, TensorDataset

        self.eval()
        val_data = self.prepare_sequences(sequences)

        dataset = TensorDataset(
            val_data["actions"],
            val_data["params"],
            val_data["values"],
            val_data["categories"],
            val_data["days_since_prev"],
            val_data["seasons"],
            val_data["sequence_lengths"],
            val_data["next_action"],
            val_data["next_param"],
            val_data["next_value"],
            val_data["next_category"],
        )

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                (
                    actions,
                    params,
                    values,
                    categories,
                    days_since_prev,
                    seasons,
                    seq_lens,
                    next_action,
                    next_param,
                    next_value,
                    next_category,
                ) = batch

                predictions = self(
                    actions,
                    params,
                    values,
                    categories,
                    days_since_prev,
                    seasons,
                    seq_lens,
                )

                targets = {
                    "action": next_action,
                    "param": next_param,
                    "value": next_value,
                    "category": next_category,
                }

                loss = self.compute_loss(predictions, targets)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def predict_next_event(self, user_sequence: Dict[str, List]) -> Dict[str, Any]:
        """
        Предсказание следующего события для пользователя.

        Args:
            user_sequence: Последовательность событий пользователя

        Returns:
            Словарь с предсказанными значениями и их вероятностями
        """
        self.eval()

        # Подготавливаем последовательность
        seq_len = len(user_sequence["actions"])

        # Если последовательность пустая или слишком короткая, возвращаем случайное предсказание
        if seq_len == 0:
            return self._random_prediction()

        # Если в последовательности только одно событие, используем его как контекст
        if seq_len == 1:
            # Дублируем событие, чтобы было что предсказывать
            context_len = 1
            # Создаем последовательность с одним событием
            sequences = {
                "actions": torch.tensor([[user_sequence["actions"][0]]])
                .long()
                .to(self.device),
                "params": torch.tensor([[user_sequence["params"][0]]])
                .long()
                .to(self.device),
                "values": torch.tensor([[user_sequence["values"][0]]])
                .long()
                .to(self.device),
                "categories": torch.tensor([[user_sequence["categories"][0]]])
                .long()
                .to(self.device),
                "days_since_prev": torch.tensor([[user_sequence["days_since_prev"][0]]])
                .float()
                .to(self.device),
                "seasons": torch.tensor([[user_sequence["seasons"][0]]])
                .long()
                .to(self.device),
                "sequence_lengths": torch.tensor([1]).long().to(self.device),
            }
        else:
            # Используем последние события как контекст
            context_len = min(seq_len - 1, self.max_seq_length)
            start_idx = seq_len - context_len - 1

            sequences = {
                "actions": torch.tensor([user_sequence["actions"][start_idx:-1]])
                .long()
                .to(self.device),
                "params": torch.tensor([user_sequence["params"][start_idx:-1]])
                .long()
                .to(self.device),
                "values": torch.tensor([user_sequence["values"][start_idx:-1]])
                .long()
                .to(self.device),
                "categories": torch.tensor([user_sequence["categories"][start_idx:-1]])
                .long()
                .to(self.device),
                "days_since_prev": torch.tensor(
                    [user_sequence["days_since_prev"][start_idx:-1]]
                )
                .float()
                .to(self.device),
                "seasons": torch.tensor([user_sequence["seasons"][start_idx:-1]])
                .long()
                .to(self.device),
                "sequence_lengths": torch.tensor([context_len]).long().to(self.device),
            }

        # Паддим если нужно
        for key in ["actions", "params", "values", "categories", "seasons"]:
            if sequences[key].shape[1] < self.max_seq_length:
                pad_len = self.max_seq_length - sequences[key].shape[1]
                sequences[key] = F.pad(sequences[key], (0, pad_len), value=0)

        if sequences["days_since_prev"].shape[1] < self.max_seq_length:
            pad_len = self.max_seq_length - sequences["days_since_prev"].shape[1]
            sequences["days_since_prev"] = F.pad(
                sequences["days_since_prev"], (0, pad_len), value=0.0
            )

        with torch.no_grad():
            predictions = self(
                sequences["actions"],
                sequences["params"],
                sequences["values"],
                sequences["categories"],
                sequences["days_since_prev"],
                sequences["seasons"],
                sequences["sequence_lengths"],
            )

        # Получаем наиболее вероятные значения
        result = {}
        for key in predictions:
            probs = F.softmax(predictions[key], dim=-1).cpu().numpy()[0]
            # Индекс 0 - padding, поэтому берем срез начиная с 1
            valid_probs = probs[1:]  # пропускаем padding
            predicted_idx = (
                np.argmax(valid_probs) + 1
            )  # +1 чтобы вернуться к оригинальной индексации
            result[f"next_{key}"] = int(predicted_idx)
            result[f"next_{key}_prob"] = float(valid_probs[predicted_idx - 1])

        return result

    def _random_prediction(self) -> Dict[str, Any]:
        """Случайное предсказание для пустой последовательности"""
        return {
            "next_action": np.random.randint(1, self.n_actions + 1),
            "next_action_prob": 1.0 / self.n_actions,
            "next_param": np.random.randint(1, self.n_params + 1),
            "next_param_prob": 1.0 / self.n_params,
            "next_value": np.random.randint(1, self.n_values + 1),
            "next_value_prob": 1.0 / self.n_values,
            "next_category": np.random.randint(1, self.n_categories + 1),
            "next_category_prob": 1.0 / self.n_categories,
        }
