from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.scoring import score_trend

router = APIRouter(tags=["scoring"])


class ScoreIn(BaseModel):
    persistence: float = Field(ge=0, le=100)
    growth: float = Field(ge=0, le=100)
    diversity: float = Field(ge=0, le=100)
    diffusion: float = Field(ge=0, le=100)
    adoption: float = Field(ge=0, le=100)
    viral_spike: float = Field(0, ge=0, le=1)
    fit: float = Field(ge=0, le=100)
    timing: float = Field(ge=0, le=100)

    model_config = {"json_schema_extra": {"example": {
        "persistence": 85, "growth": 78, "diversity": 80, "diffusion": 70,
        "adoption": 60, "viral_spike": 0, "fit": 88, "timing": 75}}}


@router.post("/score")
def score(body: ScoreIn):
    d = body.model_dump()
    return score_trend({k: d[k] for k in ("persistence", "growth", "diversity",
                                          "diffusion", "adoption")},
                       d["viral_spike"], d["fit"], d["timing"])
