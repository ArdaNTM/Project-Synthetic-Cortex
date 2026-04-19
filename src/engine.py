from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from mne.decoding import CSP

def create_bci_pipeline():
    """Creates a professional CSP + SVM pipeline."""
    csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)
    # Optimized hyperparameters from Phase 12
    svc = SVC(kernel='linear', C=100, probability=True)
    return Pipeline([('CSP', csp), ('SVM', svc)])

def train_model(pipeline, X_train, y_train):
    """Trains the model with provided calibration data."""
    return pipeline.fit(X_train, y_train)

def get_prediction(pipeline, single_epoch):
    """Predicts intent and returns (label, confidence)."""
    prediction = pipeline.predict(single_epoch)[0]
    probabilities = pipeline.predict_proba(single_epoch)[0]
    confidence = max(probabilities) * 100
    return prediction, confidence