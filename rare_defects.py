"""Rare-defect experiment using digit 9 as the rare faulty class."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, classification_report

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

class DefectNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Linear(32,2))
    def forward(self,x): return self.net(x)

def main():
    out = Path("artifacts"); out.mkdir(exist_ok=True)
    ds = load_digits()
    X = ds.data.astype(np.float32)/16.0
    y = (ds.target == 9).astype(np.int64)  # 1=faulty, 0=OK
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.30,random_state=SEED,stratify=y)
    # Keep exactly 20 faulty examples in the training subset, as specified in the slides.
    rng = np.random.default_rng(SEED)
    faulty = np.where(ytr == 1)[0]
    normal = np.where(ytr == 0)[0]
    keep_faulty = rng.choice(faulty, size=20, replace=False)
    keep = np.concatenate([normal, keep_faulty])
    rng.shuffle(keep)
    Xtr, ytr = Xtr[keep], ytr[keep]
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    Xv, yv = torch.tensor(Xte), torch.tensor(yte)

    def fit(weighted=False):
        torch.manual_seed(SEED)
        model = DefectNet()
        counts = np.bincount(ytr, minlength=2)
        weights = torch.tensor(len(ytr)/(2*counts),dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=weights) if weighted else nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(),lr=.003)
        for _ in range(100):
            model.train(); opt.zero_grad()
            loss = loss_fn(model(Xt),yt); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): pred = model(Xv).argmax(1).numpy()
        return pred, weights.numpy()

    baseline = np.zeros_like(yte)
    plain, weights = fit(False)
    weighted, _ = fit(True)
    rows=[]
    for name,pred in [("Always OK baseline",baseline),("Plain model",plain),("Class-weighted model",weighted)]:
        cm=confusion_matrix(yte,pred,labels=[0,1])
        rows.append({"model":name,"accuracy":accuracy_score(yte,pred),
                     "faulty_recall":recall_score(yte,pred,zero_division=0),
                     "faulty_precision":precision_score(yte,pred,zero_division=0),
                     "true_OK":cm[0,0],"false_alarms":cm[0,1],
                     "missed_defects":cm[1,0],"defects_found":cm[1,1]})
    frame=pd.DataFrame(rows)
    frame.to_csv(out/"results.csv",index=False)
    with open(out/"REPORT.md","w",encoding="utf-8") as f:
        f.write("# Rare defects – experiment report\\n\\n")
        f.write("Digits dataset: class 9 is treated as faulty; only 20 faulty samples are retained in training.\\n\\n")
        f.write(frame.to_markdown(index=False))
        f.write(f"\\n\\nClass weights (OK, faulty): `{weights.round(3).tolist()}`.\\n")
        f.write("\\n## Interpretation\\nAccuracy alone is misleading when defects are rare. Compare faulty-class recall, precision and the confusion matrix. The class-weighted model is the experiment intended to reduce missed defects.\\n")
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,(name,pred) in zip(axes,[("Plain model",plain),("Class-weighted model",weighted)]):
        cm=confusion_matrix(yte,pred,labels=[0,1])
        ax.imshow(cm)
        ax.set_title(name); ax.set_xlabel("Predicted class"); ax.set_ylabel("Actual class")
        ax.set_xticks([0,1],["OK","Faulty"]); ax.set_yticks([0,1],["OK","Faulty"])
        for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha="center",va="center")
    fig.tight_layout(); fig.savefig(out/"confusion_matrices.png",dpi=160)
    print("Training class counts:",np.bincount(ytr).tolist())
    print("Class weights (OK, faulty):",weights.round(3).tolist())
    print(frame.to_string(index=False))
    print("Saved artifacts/results.csv, REPORT.md, confusion_matrices.png")

if __name__=="__main__": main()
