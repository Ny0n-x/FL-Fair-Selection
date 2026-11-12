from .random_sel import RandomSelector
from .power_of_choice import PowerOfChoiceSelector
from .divfl import DivFLSelector
from .subtrunc import SubTruncSelector

REGISTRY = {
    "random": RandomSelector,
    "power_of_choice": PowerOfChoiceSelector,
    "divfl": DivFLSelector,
    "subtrunc": SubTruncSelector
}