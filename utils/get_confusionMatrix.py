import torch


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            # 创建混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            # 寻找GT中为目标的像素索引
            k = (a >= 0) & (a < n)
            # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        # h = self.mat.float()
        # # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        # acc_global = torch.diag(h).sum() / h.sum()
        # # 计算每个类别的准确率
        # acc = torch.diag(h) / h.sum(1)
        # # 计算每个类别预测与真实目标的iou
        # iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        # return acc_global, acc, iu
        conf_mat = self.mat.float().cpu().numpy()
        TP, FP, FN, TN = conf_mat[0, 0], conf_mat[0, 1], conf_mat[1, 0], conf_mat[1, 1]
        mIoU = TP / (TP + FP + FN)
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * (precision * recall) / (precision + recall)
        return mIoU, accuracy, precision, recall, F1

    # def __str__(self):
    #     acc_global, acc, iu = self.compute()
    #     return (
    #         'global correct: {:.1f}\n'
    #         'average row correct: {}\n'
    #         'IoU: {}\n'
    #         'mean IoU: {:.1f}').format(
    #             acc_global.item() * 100,
    #             ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
    #             ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
    #             iu.mean().item() * 100)
