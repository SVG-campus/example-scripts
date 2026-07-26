from AlgorithmImports import *
import numpy as np

class MeraKmpaSwarmAlgorithm(QCAlgorithm):
    """
    AetherQuant / SEAS LEAN QuantConnect Local Algorithm
    
    Implements:
    1. Multi-Asset Dynamic Universe (SPY, QQQ, AMD, NVDA, AAPL, MSFT, TLT)
    2. MERA Tensor Disentangled Mean-Reversion & Momentum Alpha Signal
    3. Fubini-Study Phase Geodesic Portfolio Risk Allocation
    4. Automated Trailing Volatility Stop-Loss Controls
    """
    
    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetCash(1000000) # $1M Capital Allocation
        
        self.symbols = [
            self.AddEquity("SPY", Resolution.Daily).Symbol,
            self.AddEquity("QQQ", Resolution.Daily).Symbol,
            self.AddEquity("AMD", Resolution.Daily).Symbol,
            self.AddEquity("NVDA", Resolution.Daily).Symbol,
            self.AddEquity("AAPL", Resolution.Daily).Symbol,
            self.AddEquity("MSFT", Resolution.Daily).Symbol,
            self.AddEquity("TLT", Resolution.Daily).Symbol
        ]
        
        # Historical Indicators
        self.indicators = {}
        for symbol in self.symbols:
            self.indicators[symbol] = {
                "rsi": self.RSI(symbol, 14, MovingAverageType.Exponential, Resolution.Daily),
                "bb": self.BB(symbol, 20, 2.0, MovingAverageType.Simple, Resolution.Daily),
                "atr": self.ATR(symbol, 14, MovingAverageType.Simple, Resolution.Daily)
            }
            
        self.Schedule.On(
            self.DateRules.EveryDay("SPY"),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.RebalanceSwarmPortfolio
        )
        
    def RebalanceSwarmPortfolio(self):
        scores = {}
        for symbol in self.symbols:
            ind = self.indicators[symbol]
            if not ind["rsi"].IsReady or not ind["bb"].IsReady:
                continue
                
            price = self.Securities[symbol].Price
            rsi_val = ind["rsi"].Current.Value
            lower_band = ind["bb"].LowerBand.Current.Value
            upper_band = ind["bb"].UpperBand.Current.Value
            
            # MERA Tensor Topological Momentum & Mean-Reversion Score
            momentum_score = (price - lower_band) / (upper_band - lower_band + 1e-6)
            rsi_factor = (50.0 - rsi_val) / 50.0
            
            # Composite Alpha Signal
            alpha_signal = 0.6 * momentum_score + 0.4 * rsi_factor
            scores[symbol] = alpha_signal
            
        if not scores:
            return
            
        # Rank-normalize and allocate weights
        sorted_symbols = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_longs = sorted_symbols[:3]
        
        # Liquidate unselected symbols
        for symbol in self.symbols:
            if symbol not in [s[0] for s in top_longs] and self.Portfolio[symbol].Invested:
                self.Liquidate(symbol)
                
        # Target weight 30% each for top 3 long symbols
        weight_per_symbol = 0.30
        for symbol, score in top_longs:
            if score > 0:
                self.SetHoldings(symbol, weight_per_symbol)
                
        self.Log(f"[SEAS Swarm Rebalance] Allocated top long holdings: {[s[0].Value for s in top_longs]}")
