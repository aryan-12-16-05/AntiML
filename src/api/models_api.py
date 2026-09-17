"""
models_api.py — Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class AlertResponse(BaseModel):
    id: str
    tx_id: str
    timestamp: Optional[str]
    from_account: str
    to_account: str
    from_bank: Optional[str]
    to_bank: Optional[str]
    amount_usd: Optional[float]
    payment_format: Optional[str]
    is_laundering: bool
    final_score: Optional[float]
    detected_pattern: Optional[str]
    severity: str
    retroactive: bool
    triggering_tx_id: Optional[str]
    retroactive_reason: Optional[str]
    component_scores: Optional[Dict[str, float]]
    shap_values: Optional[Dict[str, float]]
    triggered_rules: Optional[List[str]]
    all_patterns: Optional[List[str]]
    from_entity_type: Optional[str]
    to_entity_type: Optional[str]
    involved_accounts: Optional[List[str]]
    status: str
    reviewed_at: Optional[str]
    reviewer_note: Optional[str]
    created_at: Optional[str]


class AlertListResponse(BaseModel):
    alerts: List[AlertResponse]
    total: int
    page: int
    page_size: int


class ReviewRequest(BaseModel):
    alert_id: str
    action: str = Field(..., pattern="^(APPROVED|REJECTED)$")
    note: Optional[str] = ""
    whitelist_accounts: Optional[List[str]] = []
    whitelist_ttl_days: Optional[int] = 30


class ReviewResponse(BaseModel):
    success: bool
    alert: Optional[AlertResponse]
    whitelist_added: int
    message: str


class StatsResponse(BaseModel):
    total: int
    pending: int
    approved: int
    rejected: int
    retroactive: int
    critical: int


class CustomerProfileResponse(BaseModel):
    account: str
    entity_name: Optional[str]
    entity_type: Optional[str]
    tx_count: Optional[int]
    total_sent_usd: Optional[float]
    avg_sent_usd: Optional[float]
    max_sent_usd: Optional[float]
    unique_recipients: Optional[int]
    historical_laundering_rate: Optional[float]
    risk_modifier: Optional[float]
    is_benign_keyword: Optional[bool]
    whitelist_active: Optional[bool]
    whitelist_expires_at: Optional[str]


class WhitelistEntry(BaseModel):
    id: str
    account: str
    alert_id: Optional[str]
    reason: Optional[str]
    approved_at: str
    expires_at: str
    days_remaining: int


class ModelWeightsRequest(BaseModel):
    rule_weight: float = Field(0.20, ge=0.0, le=1.0)
    xgb_weight: float = Field(0.35, ge=0.0, le=1.0)
    gnn_weight: float = Field(0.45, ge=0.0, le=1.0)
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


class TransactionIn(BaseModel):
    timestamp: str
    from_bank: str
    from_account: str
    to_bank: str
    to_account: str
    amount_received: float
    receiving_currency: str
    amount_paid: float
    payment_currency: str
    payment_format: str
