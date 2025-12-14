from enum import StrEnum, IntEnum


class EventEnum(StrEnum):
    VIEW = "view"
    ADD = "addtocart"
    BUY = "transaction"


class SeasonEnum(IntEnum):
    WINTER = 0
    SPRING = 1
    SUMMER = 2
    AUTUMN = 3
