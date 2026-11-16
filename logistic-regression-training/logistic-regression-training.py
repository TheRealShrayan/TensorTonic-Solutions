import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    m,n=X.shape
    weight=np.zeros(n)
    bias=0
    for step in range(steps):
        wighted_sum=np.dot(X,weight)+bias
        y_predictions=_sigmoid(wighted_sum)
        epsilon=1e-15
        log_loss=-np.mean(y*np.log(y_predictions+epsilon)+(1-y)*np.log((1-y_predictions)+epsilon))
        weight_gradient=(1/m)*np.dot(X.T,(y_predictions-y))
        bias_gradient=np.mean(y_predictions-y)

        updated_weight=weight-lr*(weight_gradient)
        weight=updated_weight
        updated_bias=bias-lr*(bias_gradient)
        bias=updated_bias
        if step%100==0:
            print(f"Loss at step{step} is {log_loss:.4f}")
    return weight,bias

    