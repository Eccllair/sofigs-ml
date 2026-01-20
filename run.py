import json
import torch
from pathlib import Path
from app.common.schemas import Event
from app.common.schemas import TrainingEvent
from app.common.enums import SeasonEnum, EventEnum
from app.modules.ml.model import prepare_data
from app.modules.ml.sasrec_model import ComplexSASRec

with Path("training-data/combined/short_users_events.json").open("r") as f:
    short_users_events = dict(json.loads(f.read()))
users_events: dict[int, list[Event]] = {
    int(visitor_id): [Event.model_validate(event) for event in user_events]
    for visitor_id, user_events in short_users_events.items()
}

with Path("training-data/combined/short_item_properties.json").open("r") as f:
    item_properties = dict(json.loads(f.read()))
with Path("training-data/combined/short_item_category.json").open("r") as f:
    item_category = dict(json.loads(f.read()))
with Path("training-data/combined/short_formalizable_properties.json").open("r") as f:
    short_formalizable_properties = dict(json.loads(f.read()))


training_events: list[list[TrainingEvent]] = []
for user_events in users_events.values():
    prev_event: Event | None = None
    training_user_events = []
    for user_event in sorted(user_events, key=lambda event: event.timestamp):
        training_event = TrainingEvent(
            event=user_event.event,
            season=SeasonEnum((user_event.timestamp.month % 12) // 3),
            timedelta_days=(user_event.timestamp - prev_event.timestamp).days
            if prev_event is not None
            else 0,
        )

        actual_timestamp: int = 0
        actual_category: int = 0
        if item_category.get(str(user_event.item_id)) is not None:
            for timestamp, category in item_category[str(user_event.item_id)]:
                if (
                    user_event.timestamp.timestamp() - timestamp
                    < user_event.timestamp.timestamp() - actual_timestamp
                    and user_event.timestamp.timestamp() - timestamp >= 0
                ):
                    actual_timestamp = timestamp
                    actual_category = category
            training_event.item_category = actual_category

        form_factor = 0
        if item_properties.get(str(user_event.item_id)) is not None:
            for item_property, property_values in item_properties[
                str(user_event.item_id)
            ].items():
                if (
                    short_formalizable_properties.get(item_property) is not None
                    and short_formalizable_properties[item_property] > form_factor
                ):
                    actual_timestamp: int = 0
                    actual_value: str = ""
                    for timestamp, values in property_values:
                        if (
                            user_event.timestamp.timestamp() - timestamp / 1000
                            < user_event.timestamp.timestamp() - actual_timestamp
                            and user_event.timestamp.timestamp() - timestamp / 1000 >= 0
                        ):
                            actual_timestamp = int(timestamp / 1000)
                            actual_value = values
                    if actual_value != "":
                        training_event.item_property = item_property
                        training_event.property_value = actual_value
                        form_factor = short_formalizable_properties[item_property]

        training_user_events.append(training_event)
        prev_event = user_event

    training_events.append(training_user_events)


props: list[int] = []
values: list[int | str] = []
categories: list[int | str] = []

for user_training_event in training_events:
    for training_event in user_training_event:
        if training_event.item_property not in props:
            props.append(int(training_event.item_property))
        if training_event.property_value not in values:
            values.append(training_event.property_value)
        if training_event.item_category not in categories:
            categories.append(training_event.item_category)


num_propeties = len(props)
num_propet_values = len(values)
num_categories = len(categories)
num_actions = len(EventEnum)

props_by_id: dict[int, int | str] = {}
values_by_id: dict[int, int | str] = {}

id_by_props: dict[int | str, int] = {}
id_by_values: dict[int | str, int] = {}

for i in range(num_propeties):
    props_by_id[i + 1] = props[i]
    id_by_props[props[i]] = i + 1
for i in range(num_propet_values):
    values_by_id[i + 1] = values[i]
    id_by_values[values[i]] = i + 1


all_actions: list[list[int]] = []
all_properties: list[list[int]] = []
all_values: list[list[str]] = []
all_categories: list[list[int]] = []
all_deltas: list[list[int]] = []
all_seasons: list[list[int]] = []

all_target_actions: list[list[int]] = []
all_target_properties: list[list[int]] = []
all_target_values: list[list[str]] = []
all_target_categories: list[list[int]] = []
for training_event in training_events:
    data = prepare_data(training_event)
    if data is not None:
        (
            actions,
            properties,
            values,
            categories,
            deltas,
            seasons,
            target_actions,
            target_properties,
            target_values,
            target_categories,
        ) = data
        all_actions.extend(actions)
        all_properties.extend(
            [[id_by_props[int(prop)] for prop in props] for props in properties]
        )
        all_values.extend([[id_by_values[val] for val in vals] for vals in values])
        all_categories.extend(categories)
        all_deltas.extend(deltas)
        all_seasons.extend(seasons)
        all_target_actions.extend(target_actions)
        all_target_properties.extend(
            [[id_by_props[int(prop)] for prop in props] for props in target_properties]
        )
        all_target_values.extend(
            [[id_by_values[val] for val in vals] for vals in target_values]
        )
        all_target_categories.extend(target_categories)

all_sequences = []
seq_length = 10

for i in range(len(all_actions)):
    all_sequences.append(
        {
            "actions": all_actions[i],
            "params": all_properties[i],
            "values": all_values[i],
            "categories": all_categories[i],
            "days_since_prev": all_deltas[i],
            "seasons": all_seasons[i],
        }
    )

# Разделяем на train/val
train_size = int(0.8 * len(all_sequences))
train_sequences = all_sequences[:train_size]
val_sequences = all_sequences[train_size:]

# Создаем модель
print("Инициализация модели...")
model = ComplexSASRec(
    n_actions=num_actions,
    n_params=num_propeties,
    n_values=num_propet_values,
    n_categories=num_categories,
    max_seq_length=11,
    embedding_dim=32,
    num_blocks=2,
    num_heads=2,
    dropout_rate=0.1,
    learning_rate=1e-3,
    batch_size=16,
    num_epochs=5,
)

# Обучаем модель
print("Обучение модели...")
model.train_on_sequences(train_sequences, val_sequences)
# torch.save(model, "model")

# torch.load("model")
# Тестируем предсказание
print("\nТестирование предсказаний...")
test_sequence = {
    "actions": [0, 0, 1, 1, 1, 1, 1, 1, 1],  # 1
    "params": [1, 1, 1, 1, 1, 1, 1, 1, 1],  # 1
    "values": [1, 1, 1, 1, 1, 1, 1, 1, 1],  # 1
    "categories": [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
    "days_since_prev": [0, 0, 0, 0, 0, 0, 0, 0, 3],  # 0
    "seasons": [0, 0, 1, 1, 1, 1, 1, 1, 1],  # 1
}

prediction = model.predict_next_event(test_sequence)

print("\nПредсказание следующего события:")
for key, value in prediction.items():
    print(f"{key}: {value}")
