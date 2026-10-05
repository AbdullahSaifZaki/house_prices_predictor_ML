"""Public input is measured in real units; categories match the dataset."""
import json
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, field_validator

OPTIONS = json.loads((Path(__file__).resolve().parents[1] / 'Model/input_options.json').read_text())
EXAMPLE = dict(city='adana', net_area=120, rooms=4, bathrooms=2,
               heating='Kombi Doğalgaz', floor='3.Kat', total_floors=10,
               building_age=5, usage_status='Boş')


class HouseInput(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False,
                              json_schema_extra={'example': EXAMPLE})
    city: str
    net_area: float = Field(gt=50, lt=500, description='Net area in square metres; supported training range')
    rooms: float = Field(gt=0, lt=9, description='Total rooms including living rooms (3+1 means 4)')
    bathrooms: int = Field(ge=1, le=4)
    heating: str
    floor: str
    total_floors: int = Field(ge=1, le=100)
    building_age: int = Field(ge=0, le=200, description='Actual years, not the encoded training group')
    usage_status: str

    @field_validator('city', 'heating', 'floor', 'usage_status')
    @classmethod
    def known_category(cls, value, info):
        value = value.strip()
        if info.field_name == 'city':
            value = value.translate(str.maketrans('İıŞşĞğÜüÖöÇç', 'IiSsGgUuOoCc')).lower()
        if value not in OPTIONS[info.field_name]:
            raise ValueError(f'Unsupported {info.field_name}; choose a value from /options')
        return value


class PredictionOutput(BaseModel):
    prediction: float = Field(gt=0, allow_inf_nan=False)
