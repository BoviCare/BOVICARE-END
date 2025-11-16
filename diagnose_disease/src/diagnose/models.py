from typing import List
from pydantic import BaseModel, Field

class Diagnosis(BaseModel):
    name: str = Field(description="The name of the diagnosis")
    probability: float = Field(description="The probability of the diagnosis")
    description: str = Field(description="A description of the diagnosis")
    treatment: str = Field(description="The treatment for the diagnosis")
    prevention: str = Field(description="The prevention for the diagnosis")
    prognosis: str = Field(description="The prognosis for the diagnosis")
    symptoms: List[str] = Field(description="The symptoms of the diagnosis")
    causes: List[str] = Field(description="The causes of the diagnosis")
    treatments: List[str] = Field(description="The treatments for the diagnosis")