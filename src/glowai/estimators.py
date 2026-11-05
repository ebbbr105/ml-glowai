"""Seven-category probabilities, ordinal logistic and comparator classifiers."""
import warnings
import numpy as np
from scipy.special import expit, softmax
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

CLASSES = np.arange(-3, 4)


class OrdinalLogistic:
    def __init__(self, features):
        self.features=list(features)
    def fit(self, df):
        y = df.tsv.to_numpy(dtype=int) + 3
        if set(y) != set(range(7)):
            raise ValueError("All seven labels required in population training")
        self.imp=SimpleImputer(strategy="median")
        self.scaler=StandardScaler()
        x=self.scaler.fit_transform(self.imp.fit_transform(df[self.features]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl=OrderedModel(y, x, distr="logit")
            res=mdl.fit(method="bfgs",disp=False,maxiter=600)
            if not res.mle_retvals.get("converged",False):
                res=mdl.fit(start_params=res.params,method="lbfgs",disp=False,maxiter=1300)
        if not res.mle_retvals.get("converged",False):
            raise RuntimeError("Ordinal logistic failed to converge")
        self.beta=np.asarray(res.params[:x.shape[1]])
        self.thresholds=np.array([-np.inf,*mdl.transform_threshold_params(res.params)[1:-1],np.inf])
        return self
    def score_raw(self, df):
        x=self.scaler.transform(self.imp.transform(df[self.features]))
        return x@self.beta
    def predict_shift(self,df,shift=0.0):
        score=self.score_raw(df)+float(shift)
        p=np.diff(expit(self.thresholds[None,:]-score[:,None]),axis=1)
        assert p.shape==(len(df),7) and np.allclose(p.sum(axis=1),1,atol=1e-8)
        return np.clip(p,1e-14,1)/np.clip(p,1e-14,1).sum(axis=1,keepdims=True)
    def predict_proba(self,df):return self.predict_shift(df,0)


def fit_boosting(train,features,kind):
    if kind=="xgboost":
        clf=XGBClassifier(objective="multi:softprob",num_class=7,n_estimators=100,max_depth=2,
            learning_rate=.05,min_child_weight=5,reg_lambda=10,subsample=.9,
            tree_method="hist",eval_metric="mlogloss",random_state=42,n_jobs=2)
    elif kind=="catboost":
        clf=CatBoostClassifier(loss_function="MultiClass",iterations=160,depth=3,learning_rate=.05,
            l2_leaf_reg=10,random_seed=42,thread_count=2,verbose=False,allow_writing_files=False)
    else:raise ValueError(kind)
    imp=SimpleImputer(strategy="median")
    xx=imp.fit_transform(train[features])
    clf.fit(xx,train.tsv.to_numpy(dtype=int)+3)
    return (imp,clf)


def multiclass_probs(model,df,features):
    imp,clf=model
    p=clf.predict_proba(imp.transform(df[features])); assert p.shape[1]==7
    return p


def evaluate(y,p):
    y=np.asarray(y,dtype=int)
    p=np.asarray(p,dtype=float)
    assert len(y)==len(p) and np.isfinite(p).all()
    pred=CLASSES[np.argmax(p,axis=1)]
    med=CLASSES[np.argmax(np.cumsum(p,axis=1)>=.5,axis=1)]
    truth=(y[:,None]<=CLASSES[:-1][None,:]).astype(float)
    rps=np.mean((np.cumsum(p,axis=1)[:,:-1]-truth)**2)
    return {"n":len(y),"accuracy7":float(np.mean(pred==y)),
        "mae_argmax":float(np.mean(abs(pred-y))),"mae_median":float(np.mean(abs(med-y))),
        "rps":float(rps),"logloss":float(-np.log(np.clip(p[np.arange(len(y)),y+3],1e-12,1)).mean()),
        "macro_f1":float(f1_score(y,pred,labels=CLASSES,average="macro",zero_division=0))}
