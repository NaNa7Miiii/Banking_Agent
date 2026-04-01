from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field

TRANSACTION_COLUMNS: List[str] = [
    "transaction_datetime",
    "customer_id_number",
    "merchant_name",
    "merchant_category",
    "transaction_amount",
    "customer_first_name",
    "customer_last_name",
    "customer_gender",
    "customer_street",
    "customer_city",
    "customer_state",
    "customer_zip",
    "customer_latitude",
    "customer_longitude",
    "customer_city_population",
    "customer_job_title",
    "customer_dob",
    "transaction_id",
    "transaction_unix_time",
    "merchant_latitude",
    "merchant_longitude",
    "is_fraud",
]

TRANSACTION_NUMERIC_COLUMNS: List[str] = [
    "customer_id_number",
    "transaction_amount",
    "customer_zip",
    "customer_latitude",
    "customer_longitude",
    "customer_city_population",
    "transaction_unix_time",
    "merchant_latitude",
    "merchant_longitude",
    "is_fraud",
]

class Transaction(BaseModel):
    transaction_datetime: datetime
    customer_id_number: float
    merchant_name: Optional[str] = None
    merchant_category: Optional[str] = None
    transaction_amount: float
    customer_first_name: Optional[str] = None
    customer_last_name: Optional[str] = None
    customer_gender: Optional[str] = None
    customer_street: Optional[str] = None
    customer_city: Optional[str] = None
    customer_state: Optional[str] = None
    customer_zip: Optional[int] = None
    customer_latitude: Optional[float] = None
    customer_longitude: Optional[float] = None
    customer_city_population: Optional[int] = None
    customer_job_title: Optional[str] = None
    customer_dob: Optional[datetime] = None
    transaction_id: str = Field(..., description="Primary key in DB")
    transaction_unix_time: Optional[int] = None
    merchant_latitude: Optional[float] = None
    merchant_longitude: Optional[float] = None
    is_fraud: int

    @property
    def is_fraud_bool(self) -> bool:
        return self.is_fraud == 1