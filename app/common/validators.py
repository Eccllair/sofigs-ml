from app.common.enums import EventEnum


def event_to_int(event: EventEnum) -> int:
    match event:
        case EventEnum.VIEW:
            return 1
        case EventEnum.ADD:
            return 2
        case EventEnum.BUY:
            return 3
