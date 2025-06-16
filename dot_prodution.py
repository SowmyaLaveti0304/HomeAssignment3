import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Compute scaled dot-product attention.
    Args:
        Q: Query matrix of shape (n_queries, d)
        K: Key   matrix of shape (n_keys,    d)
        V: Value matrix of shape (n_keys,  d_v)
    Returns:
        weights: Attention weights matrix of shape (n_queries, n_keys)
        output:  Output matrix of shape (n_queries, d_v)
    """
    # 1. Dot product of Q and Kᵀ
    scores = Q @ K.T

    # 2. Scale by √d
    d = Q.shape[1]
    scores_scaled = scores / np.sqrt(d)

    # 3. Softmax to get attention weights
    #    (subtract max for numerical stability)
    exp_scores = np.exp(scores_scaled - np.max(scores_scaled, axis=1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # 4. Multiply weights by V to get the final output
    output = weights @ V

    return weights, output

# --- Test with provided inputs ---
Q = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1]])
K = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1]])
V = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8]])

weights, output = scaled_dot_product_attention(Q, K, V)

print("Attention weights matrix (after softmax):")
print(weights)
print("\nFinal output matrix:")
print(output)
