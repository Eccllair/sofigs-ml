import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import (
    Input,
    GRU,
    Dense,
    Embedding,
    Concatenate,
    Dropout,
    BatchNormalization,
)

from app.common.schemas import TrainingEvent
from app.common.enums import SeasonEnum
from app.common.validators import event_to_int

SEQ_LENGTH = 10


def create_gru_model(
    num_propeties=100,
    num_propet_values=100,
    num_categories=100,
    num_actions=50,
    embedding_dim=32,
    gru_units=128,
    dropout_rate=0.2,
):
    """
    Создает GRU модель для предсказания следующего действия
    на основе последовательности состояний и действий
    """

    # 1. ВХОДНЫЕ СЛОИ
    prev_action_input = Input(shape=(SEQ_LENGTH,), name="prev_action_input")
    prev_object_property_input = Input(
        shape=(SEQ_LENGTH,), name="object_property_input"
    )
    prev_object_value_input = Input(shape=(SEQ_LENGTH,), name="object_value_input")
    prev_object_category_input = Input(
        shape=(SEQ_LENGTH,), name="object_category_input"
    )
    date_delta_input = Input(shape=(SEQ_LENGTH, 1), name="time_delta_input")
    prev_season_input = Input(shape=(SEQ_LENGTH,), name="season_input")

    # 2. СЛОИ ВЕКТОРНОГО ПРЕДСТАВЛЕНИЯ (EMBEDDING)
    action_embedding = Embedding(
        input_dim=num_actions, output_dim=embedding_dim, name="action_embedding"
    )(prev_action_input)
    propetry_embedding = Embedding(
        input_dim=num_propeties,
        output_dim=embedding_dim,
        name="property_embedding",
    )(prev_object_property_input)
    value_embedding = Embedding(
        input_dim=num_propet_values,
        output_dim=embedding_dim,
        name="value_embedding",
    )(prev_object_value_input)
    category_embedding = Embedding(
        input_dim=num_categories,
        output_dim=embedding_dim,
        name="category_embedding",
    )(prev_object_category_input)
    season_embedding = Embedding(
        input_dim=len(SeasonEnum), output_dim=embedding_dim, name="season_embedding"
    )(prev_season_input)

    # 3. ОБЪЕДИНЕНИЕ ПРИЗНАКОВ
    combined = Concatenate(axis=-1, name="concat_embeddings")(
        [
            action_embedding,
            propetry_embedding,
            value_embedding,
            category_embedding,
            date_delta_input,
            season_embedding,
        ]
    )

    # 4. GRU СЛОЙ
    gru_layer = GRU(
        units=gru_units,
        return_sequences=False,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        name="gru_layer",
    )(combined)

    # 5. БАТЧ-НОРМАЛИЗАЦИЯ
    batch_norm = BatchNormalization(name="batch_norm")(gru_layer)

    # 6. ДРОПАУТ ДЛЯ РЕГУЛЯРИЗАЦИИ
    dropout_layer = Dropout(dropout_rate, name="dropout")(batch_norm)

    # 7. ВЫХОДНЫЕ СЛОИ (МУЛЬТИ-ВЫХОДНАЯ МОДЕЛЬ)
    # Каждый выход предсказывает свою категорию
    action_output = Dense(num_actions, activation="softmax", name="action_output")(
        dropout_layer
    )
    property_output = Dense(
        num_propeties, activation="softmax", name="property_output"
    )(dropout_layer)
    value_output = Dense(num_propet_values, activation="softmax", name="value_output")(
        dropout_layer
    )
    category_output = Dense(
        num_categories, activation="softmax", name="category_output"
    )(dropout_layer)

    # Создаем модель с несколькими выходами
    model = Model(
        inputs=[
            prev_action_input,
            prev_object_property_input,
            prev_object_value_input,
            prev_object_category_input,
            date_delta_input,
            prev_season_input,
        ],
        outputs=[action_output, property_output, value_output, category_output],
        name="gru_action_predictor",
    )

    # Компилируем модель с несколькими потерями
    model.compile(
        optimizer="adam",
        loss={
            "action_output": "categorical_crossentropy",
            "property_output": "categorical_crossentropy",
            "value_output": "categorical_crossentropy",
            "category_output": "categorical_crossentropy",
        },
        metrics=["accuracy"],
    )

    return model


# Подготовка данных для обучения
def prepare_data(raw_data: list[TrainingEvent]) -> tuple | None:
    """
    Подготавливает последовательности для обучения
    """
    if len(raw_data) == 0:
        return None

    # Списки для входных данных
    actions = []
    properties = []
    values = []
    categories = []
    deltas = []
    seasons = []

    # Списки для целевых данных (на 1 шаг вперед)
    target_actions = []
    target_properties = []
    target_values = []
    target_categories = []

    # Создаем последовательности
    for i in range(len(raw_data) - 1):
        # Входная последовательность
        seq_actions = []
        seq_properties = []
        seq_values = []
        seq_categories = []
        seq_deltas = []
        seq_seasons = []

        seq_target_actions = []
        seq_target_properties = []
        seq_target_values = []
        seq_target_categories = []

        if i < SEQ_LENGTH:
            for k in range(SEQ_LENGTH - i):
                seq_actions.append(0)
                seq_properties.append(0)
                seq_values.append("")
                seq_categories.append(0)
                seq_deltas.append(0)
                seq_seasons.append(0)

                if k != SEQ_LENGTH - i - 1:
                    seq_target_actions.append(0)
                    seq_target_properties.append(0)
                    seq_target_values.append("")
                    seq_target_categories.append(0)
                else:
                    target_event = raw_data[0]
                    seq_target_actions.append(event_to_int(target_event.event))
                    seq_target_properties.append(target_event.item_property)
                    seq_target_values.append(target_event.property_value)
                    seq_target_categories.append(target_event.item_category)

            for j in range(i):
                event = raw_data[j]
                seq_actions.append(event_to_int(event.event))
                seq_properties.append(event.item_property)
                seq_values.append(event.property_value)
                seq_categories.append(event.item_category)
                seq_deltas.append(event.timedelta_days)
                seq_seasons.append(event.season)

                target_event = raw_data[j + 1]
                seq_target_actions.append(event_to_int(target_event.event))
                seq_target_properties.append(target_event.item_property)
                seq_target_values.append(target_event.property_value)
                seq_target_categories.append(target_event.item_category)
        else:
            for j in range(SEQ_LENGTH):
                event = raw_data[i - SEQ_LENGTH + j]
                seq_actions.append(event_to_int(event.event))
                seq_properties.append(event.item_property)
                seq_values.append(event.property_value)
                seq_categories.append(event.item_category)
                seq_deltas.append(event.timedelta_days)
                seq_seasons.append(event.season)

                target_event = raw_data[i - SEQ_LENGTH + j + 1]
                seq_target_actions.append(event_to_int(target_event.event))
                seq_target_properties.append(target_event.item_property)
                seq_target_values.append(target_event.property_value)
                seq_target_categories.append(target_event.item_category)

        # Добавляем в общие списки
        actions.append(seq_actions)
        properties.append(seq_properties)
        values.append(seq_values)
        categories.append(seq_categories)
        deltas.append(seq_deltas)
        seasons.append(seq_seasons)

        target_actions.append(seq_target_actions)
        target_properties.append(seq_target_properties)
        target_values.append(seq_target_values)
        target_categories.append(seq_target_categories)

    if len(actions) == 0:
        return None

    return (
        np.array(actions),
        np.array(properties),
        np.array(values),
        np.array(categories),
        np.array(deltas),
        np.array(seasons),
        np.array(target_actions),
        np.array(target_properties),
        np.array(target_values),
        np.array(target_categories),
    )


# Разделяем данные на тренировочные и валидационные наборы
def train_val_split(inputs, targets, val_split=0.2):
    """Разделение данных на тренировочные и валидационные"""
    train_inputs = []
    val_inputs = []
    train_targets = []
    val_targets = []

    # Для каждого массива входных данных
    for i in range(len(inputs)):
        train_val_split_result = train_test_split(
            inputs[i], test_size=val_split, random_state=42
        )
        train_inputs.append(train_val_split_result[0])
        val_inputs.append(train_val_split_result[1])

    # Для каждого массива целевых данных
    for i in range(len(targets)):
        train_val_split_result = train_test_split(
            targets[i], test_size=val_split, random_state=42
        )
        train_targets.append(train_val_split_result[0])
        val_targets.append(train_val_split_result[1])

    return train_inputs, val_inputs, train_targets, val_targets
