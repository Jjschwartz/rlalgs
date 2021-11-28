# Algorithms
from rlalgs.algos.a2c.a2c import a2c    # noqa
from rlalgs.algos.dqn.dqn import dqn    # noqa
from rlalgs.algos.simplepg.simplepg import simplepg     # noqa
from rlalgs.algos.vpg.vpg import vpg    # noqa

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
