import numpy as np

def divide_and_conquer_pseudoinverse(matrix, divide_and_conquer_threshold=64):
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    n = matrix.shape[0]

    if n <= divide_and_conquer_threshold:
        # For small matrices, directly compute the Moore-Penrose pseudoinverse
        inv_matrix = np.linalg.pinv(matrix)
    else:
        # Divide the matrix into submatrices using divide and conquer
        submatrix_size = n // 2
        A = matrix[:submatrix_size, :submatrix_size]
        B = matrix[:submatrix_size, submatrix_size:]
        C = matrix[submatrix_size:, :submatrix_size]
        D = matrix[submatrix_size:, submatrix_size:]

        # Compute the Moore-Penrose pseudoinverse of each submatrix recursively
        A_inv = divide_and_conquer_pseudoinverse(A)
        B_inv = divide_and_conquer_pseudoinverse(B)
        C_inv = divide_and_conquer_pseudoinverse(C)
        D_inv = divide_and_conquer_pseudoinverse(D)

        # Recombine the pseudoinverses of the submatrices
        inv_matrix = np.zeros((n, n))
        inv_matrix[:submatrix_size, :submatrix_size] = A_inv
        inv_matrix[:submatrix_size, submatrix_size:] = B_inv
        inv_matrix[submatrix_size:, :submatrix_size] = C_inv
        inv_matrix[submatrix_size:, submatrix_size:] = D_inv

    return inv_matrix

# Example with a larger matrix (size > divide_and_conquer_threshold)
A = np.random.rand(128, 128)  # Replace with your larger matrix
A_inv = divide_and_conquer_pseudoinverse(A)
print("Inverse of A:")
print(A_inv)
