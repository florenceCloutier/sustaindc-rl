"""Dual (price-coordinated routing, geo_dual.py) planning for FORECAST next-hour arrivals instead of the past hour's.

Everything is dual's (prices, site LPs, routing, execution) except the bundles: every hour, the arrival forecaster of
geo_arrivals.py (trained on the real 7 weeks of the trace, with the evaluation days held out) predicts the coming
hour's job work per urgency class and turns it into forecast jobs (job mix of the same hour on previous days, each
job released at its minute in the hour). If the forecaster has no prediction (not enough history), dual's past-hour
bundles are used.
"""
from geo_arrivals import ArrivalForecaster
from geo_dual import DualPolicy

DEFAULT_MODEL = '/network/scratch/f/florence.cloutier/sustaindc-rl_results/arrival_forecaster/v2/arrival_model.pt'


class DualForecastPolicy(DualPolicy):
    def __init__(self, model_path=DEFAULT_MODEL, **kw):
        super().__init__(**kw)
        self.forecaster = ArrivalForecaster(model_path)

    def _bundles(self, geo):
        groups = self.forecaster.bundles(geo, geo.t)
        return groups if groups else super()._bundles(geo)
