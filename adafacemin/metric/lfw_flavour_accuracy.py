from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
import torch
from ..evaluate_utils import evaluate

__all__: list[str] = ['LFWFlavourAccuracy']

class LFWFlavourAccuracy(Metric):
    def __init__(self, nrof_folds=10, pca=0):
        super().__init__(dist_sync_on_step=True)
        self.n_folds = nrof_folds
        self.pca = pca
        self.add_state("embeddings", default=[], dist_reduce_fx="cat")
        self.add_state("indices", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, embeddings, indices, target):
        self.embeddings.append(embeddings)
        self.indices.append(indices)
        self.target.append(target)
    
    def compute(self):
        embedding = dim_zero_cat(self.embeddings)
        indicies = dim_zero_cat(self.indices)
        target = dim_zero_cat(self.target)
        
        self.embeddings = []
        self.indices = []
        self.target = []
        
        indicies_argsort = torch.argsort(indicies)
        indicies = indicies[indicies_argsort].cpu().numpy()
        target = target[indicies_argsort][::2].cpu().numpy()
        embedding = embedding[indicies_argsort].cpu().numpy()
        
        tpr, fpr, accuracy, best_thresholds = evaluate(embedding, target, nrof_folds=self.n_folds, pca=self.pca)
        return torch.tensor(accuracy.mean())
        
        
        
    

