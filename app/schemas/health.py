from pydantic import BaseModel


class HealthData(BaseModel):
    status: str
    app: str
    environment: str
