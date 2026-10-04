from __future__ import annotations

from typing import Optional

from .baseline import BaselineModel
from .calibration import ThresholdPolicy
from .inference import HybridExtractor
from .ontology import KnowledgeBase
from .schemas import WorkOrder


def create_app(kb_path: str, model_path: Optional[str] = None, policy_path: Optional[str] = None):
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as error:
        raise RuntimeError("API dependencies missing. Install: pip install -e '.[api]'") from error

    kb = KnowledgeBase.load(kb_path)
    model = BaselineModel.load(model_path) if model_path else None
    policy = ThresholdPolicy.load(policy_path) if policy_path else ThresholdPolicy()
    extractor = HybridExtractor(kb, model, policy)
    app = FastAPI(title="Engine Fault NLP", version="0.1.0")

    class PredictionRequest(BaseModel):
        work_order: dict

    @app.get("/health")
    def health():
        return {"status": "ok", "components": len(kb.components), "dtc": len(kb.dtc)}

    @app.post("/predict")
    def predict(request: PredictionRequest):
        try:
            order = WorkOrder.from_dict(request.work_order)
            return extractor.predict(order).to_dict()
        except (TypeError, ValueError, KeyError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app

