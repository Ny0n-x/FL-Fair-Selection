from .guarantee import GuaranteeSelector
from .round_robin import RoundRobinSelector
from .deficit import DeficitSelector

REGISTRY = {
    "guarantee": GuaranteeSelector,
    "round_robin": RoundRobinSelector,
    "deficit": DeficitSelector,
}
