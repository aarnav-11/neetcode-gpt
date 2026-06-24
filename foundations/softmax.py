import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_val = np.max(z)
        z = z - max_val
        z = np.exp(z)
        summation = np.sum(z)
        softmax = z / summation

        return np.round(softmax, 4)
        
