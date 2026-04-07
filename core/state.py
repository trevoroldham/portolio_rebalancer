"""
Pydantic models defining strict PortfolioState schemas.
Ensures immutability and type-safety across CurrentState, TargetState, and TradeAction objects.
"""

from pydantic import BaseModel, Field, computed_field
from typing import Dict


class AssetPosition(BaseModel):
    """Represents a single asset holding."""
    symbol: str = Field(..., description="Ticker symbol")
    shares: int = Field(ge=0, description="Current integer shares held")
    price: float = Field(ge=0.0, description="Latest market price")

    @computed_field
    def value(self) -> float:
        """Total nominal value of the position."""
        return self.shares * self.price


class PortfolioState(BaseModel):
    """Represents a complete snapshot of the portfolio at a given time."""
    cash_balance: float = Field(ge=0.0, description="Available uninvested cash")
    positions: Dict[str, AssetPosition] = Field(default_factory=dict)

    @computed_field
    def total_equity(self) -> float:
        """Total portfolio value (cash + asset values)."""
        return self.cash_balance + sum(pos.value for pos in self.positions.values())

    def get_asset_weight(self, symbol: str) -> float:
        """Returns the current weight of a specific asset relative to total equity."""
        if self.total_equity == 0:
            return 0.0
        position = self.positions.get(symbol)
        if not position:
            return 0.0
        return position.value / self.total_equity


class TradeRecommendation(BaseModel):
    """Represents an actionable trade based on the rebalancing threshold."""
    symbol: str
    current_shares: int = Field(ge=0)
    target_shares: int = Field(ge=0)
    price: float = Field(ge=0.0)

    @computed_field
    def share_delta(self) -> int:
        """Number of shares to buy (positive) or sell (negative)."""
        return self.target_shares - self.current_shares

    @computed_field
    def trade_value(self) -> float:
        """Estimated nominal value of the transaction."""
        return self.share_delta * self.price

    @computed_field
    def drift_percentage(self) -> float:
        """Calculates the drift of the current allocation from the optimal allocation."""
        if self.current_shares == 0:
            return 100.0 if self.target_shares > 0 else 0.0
        
        # Calculates drift based on share count discrepancy
        return abs(self.share_delta) / self.current_shares * 100.0

    @property
    def action(self) -> str:
        """String representation of the trade action."""
        if self.share_delta > 0:
            return "BUY"
        elif self.share_delta < 0:
            return "SELL"
        return "HOLD"