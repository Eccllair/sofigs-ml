from pydantic import BaseModel
from datetime import datetime

from app.common.enums import EventEnum, SeasonEnum


class User(BaseModel):
    id: int


class Event(BaseModel):
    event: EventEnum
    user_id: int
    item_id: int
    transaction_id: int | None = None
    timestamp: datetime


class Category(BaseModel):
    category_id: int
    parent_id: int


class Property(BaseModel):
    name: int | str
    snap_time: datetime
    value: str | float


class Item(BaseModel):
    item_id: int


class Transaction(BaseModel):
    transaction_id: int


class ItemState(BaseModel):
    item_id: int
    properties: list[Property]
    actual_from: datetime
    actual_to: datetime


class CartState(BaseModel):
    user_id: int
    items: list[Item]
    actual_from: datetime
    actual_to: datetime


class TrainingEvent(BaseModel):
    event: EventEnum
    item_category: int = 0
    item_property: int = 0
    property_value: str = ""
    season: SeasonEnum
    timedelta_days: int = 0


class TrainingProperty(BaseModel):
    name: int | str
    value_from: int
