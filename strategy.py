"""Adapter for the local Alimama baseline PlayerBiddingStrategy interface."""
import os
import numpy as np
from ogb import Policy, Episode, Replay

class OgbBiddingStrategy:
    def __init__(self, budget=100, name='OGB', cpa=2, category=1):
        self.budget, self.cpa, self.name, self.category = budget, cpa, name, category
        self.policy = Policy(os.environ['OGB_CHECKPOINT'])
        self.reset()

    def reset(self):
        self.remaining_budget = self.budget
        self.policy.reset()
        self.previous_utility = 0.

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo,
                historyBid, historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        slots = [(np.asarray(p)[:,0], np.asarray(price).reshape(-1))
                 for p, price in zip(historyPValueInfo, historyLeastWinningCost)]
        slots.append((np.asarray(pValues), np.zeros(len(pValues))))
        slots.extend([(np.zeros(0), np.zeros(0))]*(48-len(slots)))
        env = Replay(Episode('serving', self.budget, self.cpa, slots), self.policy.penalty)
        env.t = timeStepIndex
        env.cost = self.budget-self.remaining_budget
        # Expected conversion reward matches the deterministic replay training contract.
        env.conv = sum(float(np.sum(np.asarray(p)[:,0]*np.asarray(a)[:,0]))
                       for p,a in zip(historyPValueInfo, historyAuctionResult))
        env.last = self.policy.actions[-1][0] if self.policy.actions else 1.
        utility = env.utility(env.cost, env.conv)
        reward = utility-self.previous_utility
        self.previous_utility = utility
        multiplier = self.policy.act(env.state(), timeStepIndex, reward)
        return multiplier*self.cpa*np.asarray(pValues)
