import numpy as np
def simple_regression(X, y, learning_rate=0.001, epochs=1000, error_function=None):
    a, b = 0, 0  # Inicialización de parámetros
    n = len(X)
    
    for _ in range(epochs):
        y_pred = a * X + b
        if error_function == "MSE":
            error = y_pred - y
            a_gradient = (2 / n) * np.sum(X * error)
            b_gradient = (2 / n) * np.sum(error)
        elif error_function == "MAE":
            error = np.sign(y_pred - y)
            a_gradient = (1 / n) * np.sum(X * error)
            b_gradient = (1 / n) * np.sum(error)
        else:
            raise ValueError("error_function must be 'MSE' or 'MAE'")
        
        a -= learning_rate * a_gradient
        b -= learning_rate * b_gradient

    return a, b 


def linear_regression(X: np.ndarray, y: np.ndarray,
                      regression_error: callable, 
                      error_function: callable,
                      learning_rate: float, 
                      epochs: int) -> tuple:
    
    model = np.random.rand(X.shape[1])  # Inicialización aleatoria de parámetros
    bias = np.random.rand(X.shape[0])  # Inicialización aleatoria del bias
    prev_loss = float('inf')
    for _ in range(epochs):
        gradient, bias_gradient = regression_error(X, y, model, bias)

        model -= learning_rate * gradient
        bias -= learning_rate * bias_gradient
        loss = error_function(X, y, model, bias)

        if abs(prev_loss - loss) < 1e-6:
            break
        prev_loss = loss
    return model, np.mean(bias)
                      
# ...existing code...

def batched_linear_regression(X: np.ndarray, y: np.ndarray,
                      regression_error: callable, 
                      error_function: callable,
                      learning_rate: float, 
                      epochs: int, batch_size: float = 1.0) -> tuple:
    """
    Realiza la regresión lineal con descenso de gradiente estocástico.
    Args:
        X (np.ndarray): Matriz de características (normalizada y con bias).
        y (np.ndarray): Vector de etiquetas.
        regression_error (callable): Función para calcular el error.
        error_function (callable): Función para calcular el error.
        learning_rate (float): Tasa de aprendizaje.
        epochs (int): Número de épocas.
        batch_size (float): Porcentaje del dataset usado en cada batch (0-1).

    Returns:    
        np.ndarray: Vector de parámetros ajustados (model).
    """
    assert 0 < batch_size <= 1, "batch_size debe estar en el rango (0, 1]"
    assert X.shape[0] == y.shape[0], "X e y deben tener la misma cantidad de muestras"
    dataset_size, features_size = X.shape
    batch = max(1, int(batch_size * dataset_size))  # Asegura al menos 1 muestra por batch
    model = np.random.rand(features_size)  # Inicialización aleatoria de parámetros
    bias = np.random.rand()  # Inicialización aleatoria del bias (scalar)
    prev_loss = float('inf')

    for _ in range(epochs):
        indices = np.random.permutation(dataset_size)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for i in range(0, dataset_size, batch):
            X_batch = X_shuffled[i:i + batch]
            y_batch = y_shuffled[i:i + batch]

            gradient, bias_gradient = regression_error(X_batch, y_batch, model, bias)
            model -= learning_rate * gradient
            bias -= learning_rate * bias_gradient
        loss = error_function(X, y, model, bias)
        if abs(prev_loss - loss) < 1e-6:
            break
        prev_loss = loss
    return model, bias

def multi_linear_regression(X: np.ndarray, y: np.ndarray, 
                      error_function: callable, regularization_derivative: callable,
                      regression_error: callable, 
                      lambda_reg: float, learning_rate: float, 
                      epochs: int, batch_size: float = 1.0) -> np.ndarray:
    """
    Realiza la regresión lineal múltiple con descenso de gradiente y regularización.

    Args:
        X (np.ndarray): Matriz de características (normalizada y con bias).
        y (np.ndarray): Vector de etiquetas.
        error_function (callable): Función para calcular el error.
        regularization_derivative (callable): Derivada de la función de regularización.
        epsilon (float): Umbral de convergencia.
        lambda_reg (float): Peso de la regularización.
        learning_rate (float): Tasa de aprendizaje.
        epochs (int): Número de épocas.
        batch_size (float): Porcentaje del dataset usado en cada batch (0-1).

    Returns:
        np.ndarray: Vector de parámetros ajustados (model).
    """
    assert 0 < batch_size <= 1, "batch_size debe estar en el rango (0, 1]"
    assert X.shape[0] == y.shape[0], "X e y deben tener la misma cantidad de muestras"
    
    dataset_size, features_size = X.shape
    batch = max(1, int(batch_size * dataset_size))  # Asegura al menos 1 muestra por batch
    model = np.random.rand(features_size)  
    bias = np.random.rand()  # Inicialización aleatoria del bias (scalar)

    prev_loss = float('inf')

    for _ in range(epochs):
        indices = np.random.permutation(dataset_size)  # Barajar datos en cada época
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, dataset_size, batch):
            X_batch = X_shuffled[i:i + batch]
            y_batch = y_shuffled[i:i + batch]

            gradient, bias = regression_error(X_batch, y_batch, model, bias)  # w_i = w_i * error
            gradient += + lambda_reg * regularization_derivative(model)
            model -= learning_rate * gradient  # Actualización de pesos
            bias -= learning_rate * bias
        
        loss = error_function(X, y, model, bias)
        if abs(prev_loss - loss) < 1e-6:
            break
        prev_loss = loss

    return model, bias

def predict(X: np.ndarray, model: np.ndarray) -> np.ndarray:
    """
    Realiza la predicción de un conjunto de datos.

    Args:
        X (np.ndarray): Matriz de características.
        model (np.ndarray): Vector de parámetros ajustados.

    Returns:
        np.ndarray: Vector de etiquetas predichas.
    """
    return X @ model