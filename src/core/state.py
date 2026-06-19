from datetime import datetime
from typing import Annotated, TypedDict, List, Optional, Literal


def append_history(existing: List[dict] | None, new: List[dict] | None) -> List[dict]:
    """Reducer для ``history``: дописывает новые ходы, не затирая предыдущие."""
    return (existing or []) + (new or [])


class ReservationState(TypedDict, total=False):
    operation: Literal["book", "cancel", "modify"]  # что делает пользователь
    start_date_time: datetime
    end_date_time: datetime
    vehicle_plate: str
    contact_email: str
    confirmed: bool
    space_id: Optional[int]  # выбранное место после проверки в Postgres
    reservation_id: Optional[str]

class ChatState(TypedDict, total=False):
    user_input: str                    # текущий ввод пользователя
    history: Annotated[List[dict], append_history]  # история диалога (накапливается)
    intent: Optional[Literal[          # классификация роутера
        "info_query", "dynamic_query", "reservation", "out_of_scope"]]
    retrieved_chunks: List[str]        # чанки из Weaviate (RAG)
    reservation_fields: ReservationState           # собранные поля брони
    input_blocked: bool                # флаг входного guardrail
    output_blocked: bool               # флаг выходного guardrail
    response: str                      # финальный ответ
