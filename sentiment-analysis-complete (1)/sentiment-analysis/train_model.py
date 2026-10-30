from pathlib import Path
import json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, roc_auc_score, roc_curve, auc)
from sklearn.base import clone
import joblib
from sentiment_utils import clean_text

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'reviews.csv'
OUT = ROOT / 'outputs'; MODELS = ROOT / 'models'; REPORTS = ROOT / 'reports'
for p in (OUT, MODELS, REPORTS): p.mkdir(exist_ok=True)
RANDOM_STATE = 42

# 1. Load and validate
raw = pd.read_csv(DATA)
required = {'review_id','review_text','sentiment','category','source'}
missing_cols = required - set(raw.columns)
if missing_cols: raise ValueError(f'Missing required columns: {missing_cols}')
df = raw.copy()
df['review_text'] = df['review_text'].fillna('').astype(str)
df['sentiment'] = df['sentiment'].astype(str).str.strip().str.title()
df = df[df['review_text'].str.strip().ne('')].drop_duplicates().reset_index(drop=True)
classes = ['Negative','Neutral','Positive']
if set(df.sentiment.unique()) != set(classes): raise ValueError(f'Unexpected labels: {df.sentiment.unique()}')

# EDA tables
summary = {
 'rows_original': int(len(raw)), 'rows_after_cleaning': int(len(df)),
 'columns': list(df.columns), 'missing_values_original': {k:int(v) for k,v in raw.isna().sum().items()},
 'duplicate_rows_original': int(raw.duplicated().sum()),
 'sentiment_counts': {k:int(v) for k,v in df.sentiment.value_counts().items()},
 'mean_review_words': float(df.review_text.str.split().str.len().mean()),
 'median_review_words': float(df.review_text.str.split().str.len().median()),
 'unique_reviews': int(df.review_text.nunique()),
 'repeated_text_rows': int(len(df) - df.review_text.nunique())
}
(OUT/'dataset_summary.json').write_text(json.dumps(summary, indent=2))
df['word_count'] = df.review_text.str.split().str.len()

sns.set_theme(style="whitegrid", palette="deep")
plt.figure(figsize=(7,5)); order=df.sentiment.value_counts().index
sns.countplot(data=df, x='sentiment', order=order, color='steelblue'); plt.title('Sentiment Class Distribution'); plt.xlabel('Sentiment'); plt.ylabel('Number of reviews'); plt.tight_layout(); plt.savefig(OUT/'sentiment_distribution.png', dpi=180); plt.close()
plt.figure(figsize=(8,5)); sns.histplot(data=df, x='word_count', hue='sentiment', bins=25, element='step', stat='count', common_norm=False); plt.title('Review Length Distribution by Sentiment'); plt.xlabel('Words per review'); plt.tight_layout(); plt.savefig(OUT/'review_length_distribution.png', dpi=180); plt.close()
plt.figure(figsize=(8,5)); sns.countplot(data=df, y='category', order=df.category.value_counts().index, color='steelblue'); plt.title('Reviews by Category'); plt.xlabel('Number of reviews'); plt.ylabel('Category'); plt.tight_layout(); plt.savefig(OUT/'category_distribution.png', dpi=180); plt.close()
plt.figure(figsize=(8,5)); sns.countplot(data=df, y='source', order=df.source.value_counts().index, color='steelblue'); plt.title('Reviews by Source'); plt.xlabel('Number of reviews'); plt.ylabel('Source'); plt.tight_layout(); plt.savefig(OUT/'source_distribution.png', dpi=180); plt.close()

# 2. Group split: identical review texts must never cross train/test boundary.
# The dataset contains repeated text templates, so grouping avoids optimistic leakage.
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
train_idx, test_idx = next(splitter.split(df.review_text, df.sentiment, groups=df.review_text))
X_train, X_test = df.review_text.iloc[train_idx], df.review_text.iloc[test_idx]
y_train, y_test = df.sentiment.iloc[train_idx], df.sentiment.iloc[test_idx]
train_groups = X_train.values

# 3. Baseline model comparison. Vectorization is inside each pipeline.
models = {
 'MultinomialNB (TF-IDF)': Pipeline([('tfidf', TfidfVectorizer(preprocessor=clean_text, ngram_range=(1,2), min_df=2)), ('clf', MultinomialNB())]),
 'Logistic Regression (TF-IDF)': Pipeline([('tfidf', TfidfVectorizer(preprocessor=clean_text, ngram_range=(1,2), min_df=2)), ('clf', LogisticRegression(max_iter=2000, class_weight=None, random_state=RANDOM_STATE))]),
 'Linear SVM (TF-IDF)': Pipeline([('tfidf', TfidfVectorizer(preprocessor=clean_text, ngram_range=(1,2), min_df=2)), ('clf', LinearSVC(random_state=RANDOM_STATE))]),
 'Logistic Regression (BoW)': Pipeline([('bow', CountVectorizer(preprocessor=clean_text, ngram_range=(1,2), min_df=2)), ('clf', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
}
rows=[]; fitted={}
for name, pipe in models.items():
    start=time.time(); pipe.fit(X_train,y_train); pred=pipe.predict(X_test); elapsed=time.time()-start
    fitted[name]=pipe
    rows.append({'Model':name,'Accuracy':accuracy_score(y_test,pred),
                 'Precision_macro':precision_score(y_test,pred,average='macro',zero_division=0),
                 'Recall_macro':recall_score(y_test,pred,average='macro',zero_division=0),
                 'F1_macro':f1_score(y_test,pred,average='macro',zero_division=0),
                 'Training_seconds':elapsed})
comparison=pd.DataFrame(rows).sort_values('F1_macro',ascending=False)
comparison.to_csv(OUT/'model_comparison.csv',index=False)

# 4. Grid search on training partition only; stratified 5-fold CV.
base = Pipeline([('tfidf', TfidfVectorizer(preprocessor=clean_text)),
                 ('clf', LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))])
param_grid = {
 'tfidf__ngram_range': [(1,1),(1,2)],
 'tfidf__min_df': [1,2],
 'clf__C': [0.5,1.0,2.0]
}
cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=RANDOM_STATE)
grid=GridSearchCV(base,param_grid,scoring='f1_macro',cv=cv,n_jobs=-1,refit=True,return_train_score=True)
grid.fit(X_train,y_train,groups=train_groups)
best=grid.best_estimator_

# CV score for selected estimator on training data
cv_result=cross_validate(best,X_train,y_train,groups=train_groups,cv=cv,scoring={'accuracy':'accuracy','precision_macro':'precision_macro','recall_macro':'recall_macro','f1_macro':'f1_macro'},n_jobs=-1)
cv_metrics={k:float(np.mean(v)) for k,v in cv_result.items() if k.startswith('test_')}
cv_metrics={k.replace('test_',''):v for k,v in cv_metrics.items()}

# 5. One-time untouched test evaluation
pred=best.predict(X_test)
metrics={
 'accuracy':float(accuracy_score(y_test,pred)),
 'precision_macro':float(precision_score(y_test,pred,average='macro',zero_division=0)),
 'recall_macro':float(recall_score(y_test,pred,average='macro',zero_division=0)),
 'f1_macro':float(f1_score(y_test,pred,average='macro',zero_division=0)),
 'precision_weighted':float(precision_score(y_test,pred,average='weighted',zero_division=0)),
 'recall_weighted':float(recall_score(y_test,pred,average='weighted',zero_division=0)),
 'f1_weighted':float(f1_score(y_test,pred,average='weighted',zero_division=0)),
 'roc_auc_ovr_macro':float(roc_auc_score(y_test,best.predict_proba(X_test),multi_class='ovr',average='macro')), 
 'train_size':int(len(X_train)), 'test_size':int(len(X_test)),
 'best_params':grid.best_params_, 'best_cv_macro_f1':float(grid.best_score_),
 'cross_validation_mean':cv_metrics,
 'classification_report':classification_report(y_test,pred,labels=classes,output_dict=True,zero_division=0)
}
(OUT/'evaluation_metrics.json').write_text(json.dumps(metrics,indent=2))
(OUT/'classification_report.txt').write_text(classification_report(y_test,pred,labels=classes,zero_division=0))
comparison.to_csv(REPORTS/'model_comparison.csv',index=False)

# One-vs-rest ROC curves (multiclass probability estimates from logistic regression)
from sklearn.preprocessing import label_binarize
Y_test_bin = label_binarize(y_test, classes=classes)
prob_matrix = best.predict_proba(X_test)
plt.figure(figsize=(7,6))
for i, label in enumerate(classes):
    fpr, tpr, _ = roc_curve(Y_test_bin[:, i], prob_matrix[:, i])
    plt.plot(fpr, tpr, label=f'{label} (AUC={auc(fpr,tpr):.3f})')
plt.plot([0,1],[0,1], linestyle='--', label='Chance')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('One-vs-Rest ROC Curves'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'roc_curves.png',dpi=200); plt.close()

cm=confusion_matrix(y_test,pred,labels=classes)
plt.figure(figsize=(7,6)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=classes,yticklabels=classes); plt.title('Confusion Matrix – Tuned Logistic Regression'); plt.xlabel('Predicted label'); plt.ylabel('True label'); plt.tight_layout(); plt.savefig(OUT/'confusion_matrix.png',dpi=200); plt.close()

# Model comparison chart
plt.figure(figsize=(10,5)); plotdf=comparison.melt(id_vars='Model',value_vars=['Accuracy','Precision_macro','Recall_macro','F1_macro'],var_name='Metric',value_name='Score'); sns.barplot(data=plotdf,x='Model',y='Score',hue='Metric'); plt.ylim(0,1.05); plt.xticks(rotation=18,ha='right'); plt.title('Baseline Model Comparison'); plt.tight_layout(); plt.savefig(OUT/'model_comparison.png',dpi=200); plt.close()

# Train-vs-validation learning curve diagnostic using grid search's best candidate
best_idx=grid.best_index_; cvres=pd.DataFrame(grid.cv_results_)
train_score=float(cvres.loc[best_idx,'mean_train_score']); val_score=float(cvres.loc[best_idx,'mean_test_score'])
overfit={'best_candidate_mean_train_macro_f1':train_score,'best_candidate_mean_cv_macro_f1':val_score,'train_cv_gap':train_score-val_score,
         'interpretation':'A small train/CV gap suggests limited overfitting; a larger gap suggests the model fits training folds better than unseen folds. This is a diagnostic, not proof of generalization.'}
(OUT/'fit_diagnostic.json').write_text(json.dumps(overfit,indent=2))

# Persist complete pipeline. Import sentiment_utils before joblib.load in app.
joblib.dump(best,MODELS/'sentiment_pipeline.joblib')
joblib.dump(best,MODELS/'sentiment_pipeline.pkl')

# Prediction examples for report
examples=pd.DataFrame({'review_text':['I absolutely love this product, it works perfectly.','This is disappointing and does not work as expected.','The package contains a charger and a user manual.']})
examples['predicted_sentiment']=best.predict(examples.review_text)
if hasattr(best,'predict_proba'):
    probs=best.predict_proba(examples.review_text)
    examples['confidence']=probs.max(axis=1)
examples.to_csv(OUT/'example_predictions.csv',index=False)

# Store split/test labels and predictions for reproducibility
pd.DataFrame({'review_text':X_test.values,'actual':y_test.values,'predicted':pred}).to_csv(OUT/'test_predictions.csv',index=False)
print('SUMMARY',json.dumps(summary))
print('COMPARISON\n',comparison.to_string(index=False))
print('BEST',grid.best_params_,'CV',grid.best_score_)
print('METRICS',json.dumps({k:v for k,v in metrics.items() if k!='classification_report'},indent=2))
print('FIT',overfit)
