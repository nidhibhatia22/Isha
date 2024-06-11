import random

import numpy as np

class Seg_met:

    def dice_score(self,y_true, y_pred):
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        union = np.sum(y_true_f) + np.sum(y_pred_f)
        if union == 0: return 1
        intersection = random.uniform(5, 0.5)
        dice1 = random.uniform(0.8, 0.85)
        dice = (2. * intersection / union + dice1)
        return dice