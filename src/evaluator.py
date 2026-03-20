from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np

class Evaluator:
    def __init__(self, args):
        self.args = args

    def evaluate(self, model, dataset, split, train_step):
        self.args.logger.write('\nEvaluating on split = ' + split)
        eval_ind = dataset.splits[split]
        num_samples = len(eval_ind)
        model.eval()

        pbar = tqdm(range(0, num_samples, self.args.eval_batch_size),
                    desc='running forward pass')
        true, pred = [], []
        for start in pbar:
            batch_ind = eval_ind[start:min(num_samples,
                                           start + self.args.eval_batch_size)]
            batch = dataset.get_batch(batch_ind)
            true.append(batch['labels'])
            del batch['labels']
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            with torch.no_grad():
                pred.append(model(**batch).cpu())

        true = torch.cat(true).numpy()   # shape: (N, K)
        pred = torch.cat(pred).numpy()   # shape: (N, K)

        per_target_auroc = []
        per_target_auprc = []
        per_target_minrp = []

        for j, col in enumerate(dataset.target_columns):
            y_true = true[:, j]
            y_pred = pred[:, j]

            # skip degenerate columns in this split
            if len(np.unique(y_true)) < 2:
                self.args.logger.write(
                    f"Skipping target {col} on split {split} because it has only one class",
                    show_time=False
                )
                continue

            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            pr_auc = auc(recall, precision)
            minrp = np.minimum(precision, recall).max()
            roc_auc = roc_auc_score(y_true, y_pred)

            per_target_auroc.append(roc_auc)
            per_target_auprc.append(pr_auc)
            per_target_minrp.append(minrp)

        if len(per_target_auroc) == 0:
            result = {'auroc': 0.0, 'auprc': 0.0, 'minrp': 0.0}
        else:
            result = {
                'auroc': float(np.mean(per_target_auroc)),
                'auprc': float(np.mean(per_target_auprc)),
                'minrp': float(np.mean(per_target_minrp)),
            }

        if train_step is not None:
            self.args.logger.write(
                'Result on ' + split + ' split at train step ' +
                str(train_step) + ': ' + str(result)
            )
        return result