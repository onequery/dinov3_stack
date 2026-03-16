import numpy as np

# Source: https://github.com/sacmehta/ESPNet/blob/master/train/IOUEval.py

class IOUEval:
    def __init__(self, nClasses):
        self.nClasses = nClasses
        self.reset()

    def reset(self):
        self.hist = np.zeros((self.nClasses, self.nClasses), dtype=np.float64)

    def fast_hist(self, a, b):
        k = (a >= 0) & (a < self.nClasses)
        return (
            np.bincount(
                self.nClasses * a[k].astype(int) + b[k],
                minlength=self.nClasses**2,
            )
            .reshape(self.nClasses, self.nClasses)
            .astype(np.float64)
        )

    def compute_hist(self, predict, gth):
        hist = self.fast_hist(gth, predict)
        return hist

    def addBatch(self, predict, gth):
        predict = predict.cpu().numpy().flatten()
        gth = gth.cpu().numpy().flatten()

        hist = self.compute_hist(predict, gth)
        self.hist += hist

    def getMetric(self):
        epsilon = 0.00000001
        hist = self.hist
        overall_acc = np.diag(hist).sum() / (hist.sum() + epsilon)
        per_class_acc = np.diag(hist) / (hist.sum(1) + epsilon)
        per_class_iu = np.diag(hist) / (
            hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon
        )
        mIOU = np.nanmean(per_class_iu)

        return overall_acc, per_class_acc, per_class_iu, mIOU
