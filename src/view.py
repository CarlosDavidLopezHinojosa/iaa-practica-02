import streamlit as st
import numpy as np
from algorithms import (simple_regression, linear_regression, batched_linear_regression,
                        multi_linear_regression, predict)
from utils.functions import mse, mae, rmae, rmse, l1, l2
from utils.stats import statistical_test
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

def participants():
    st.header("Participantes")
    st.markdown("""
    - **Javier Gómez Aparicio, Daniel Grande Rubio, Carlos David López Hinojosa, Carlos de la Torre Frías**
    - **Grupo: 6**
    """)

def run_simple_regression():
    st.header("1. Regresión lineal con gradiente descendente")
    st.subheader("1.1. Gradiente descendente: Explicación y ejemplo")
    
    st.markdown("""
    El gradiente descendente es un algoritmo de optimización iterativo utilizado para minimizar funciones de costo.
    Se utiliza para ajustar los parámetros del modelo (pendiente y sesgo) para minimizar el error.
    
    Por ejemplo, para una ecuación:
    
    $$y = ax + b$$
    
    Donde:
    - $a$: pendiente
    - $b$: sesgo o intercepto
    
    **Pasos del Algoritmo:**
    1. Inicialización.
    2. Cálculo del error (por ejemplo, MSE).
    3. Cálculo del gradiente.
    4. Actualización de parámetros.
    5. Repetir hasta convergencia.
    """)

    X_simple = np.array([1, 2, 3])
    y_simple = np.array([2, 3, 5])
    st.dataframe({"X": X_simple, "y": y_simple})

    error_func_simple = st.selectbox("Selecciona el error de regresión", ("MSE", "MAE"))
    learning_rate_simple = st.slider("Tasa de aprendizaje (α)", 0.01, 0.1, 0.01)
    epochs_simple = st.slider("Número de iteraciones", 1, 10000, 100)

    if st.button("Ejecutar gradiente descendente (Ejemplo)"):
        a, b = simple_regression(X_simple, y_simple, error_function=error_func_simple, 
                                 learning_rate=learning_rate_simple, epochs=epochs_simple)
        st.markdown(f"### Resultados:\n- $a = {a}$\n- $b = {b}$")
        st.markdown("### Predicciones:")
        st.scatter_chart({"Valores reales": y_simple, "Predicciones": a * X_simple + b})

def run_dataset_regression():
    st.header("2. Aplicación sobre un conjunto de datos proporcionado")
    
    X_example = np.array([
        [1, 2],
        [1, 3],
        [2, 3],
        [2, 4],
        [3, 2],
        [3, 5],
        [4, 1]
    ])
    y_example = np.array([1.03, -1.44, 4.53, 2.24, 13.27, 5.62, 21.53])
    st.markdown("Aplicamos el gradiente descendente al conjunto de datos proporcionado:")
    st.dataframe({"X1": X_example[:, 0], "X2": X_example[:, 1], "y": y_example})
    
    learning_rate_example = st.slider("Tasa de aprendizaje (α)", 0.01, 0.1, 0.01, key="learning_rate_example")
    epochs_example = st.slider("Número de iteraciones", 1, 10000, 1000, key="epochs_example")

    if st.button("Ejecutar gradiente descendente (Datos Proporcionados)"):
        # Versión usando MSE
        model_example_mse, bias_example_mse = linear_regression(
            X_example, y_example, regression_error=rmse, 
            learning_rate=learning_rate_example, epochs=epochs_example, error_function=mse
        )
        # Versión usando MAE
        model_example_mae, bias_example_mae = linear_regression(
            X_example, y_example, regression_error=rmae, 
            learning_rate=learning_rate_example, epochs=epochs_example, error_function=mae
        )
        st.markdown("#### Resultados:")
        st.markdown("**Usando MSE:**")
        st.markdown(f"- Coeficientes: $a = {model_example_mse[0]}$, $b = {model_example_mse[1]}$")
        st.markdown(f"- Sesgo: $c = {bias_example_mse}$")
        st.markdown("**Usando MAE:**")
        st.markdown(f"- Coeficientes: $a = {model_example_mae[0]}$, $b = {model_example_mae[1]}$")
        st.markdown(f"- Sesgo: $c = {bias_example_mae}$")

        st.subheader("Predicciones:")
        st.scatter_chart({
            "Valores reales": y_example, 
            "Predicciones MSE": X_example @ model_example_mse + bias_example_mse, 
            "Predicciones MAE": X_example @ model_example_mae + bias_example_mae
        })
        
        st.subheader("Comparación de errores")
        st.write("Comparamos estadísticamente los errores de ambos modelos.")
        stat_result = statistical_test(np.array([
            predict(X_example, model_example_mse) + bias_example_mse, 
            predict(X_example, model_example_mae) + bias_example_mae, 
            y_example
        ]))
        st.markdown(f"""
        Prueba realizada: **{stat_result['stat-test']}**
        Conclusión: Los errores son **{'significativamente diferentes' if stat_result['reject'] else 'no significativamente diferentes'}**
        """)
        st.write("Observación: Con pocas iteraciones, el modelo con MSE se aproxima más a los valores reales.")

def run_batch_and_minibatch():
    st.header("3. Modo Batch y Mini-Batch")
    st.markdown("""
    Se presentan dos enfoques:
    - **Batch:** Utiliza todo el dataset en cada iteración.
    - **Mini-Batch:** Divide el dataset en pequeños subconjuntos.
    """)
    
    learning_rate_batch = st.slider("Tasa de aprendizaje (α)", 0.01, 0.1, 0.01, key="learning_rate_batch")
    epochs_batch = st.slider("Número de iteraciones", 1, 10000, 1000, key="epochs_batch")
    error_func_choice = st.selectbox("Selecciona la función de error", ("MSE", "MAE"), key="error_func_batch")
    regression_error_choice = st.selectbox("Selecciona el error de regresión", ("RMSE", "RMAE"), key="regression_error_batch")

    # Mapeo de funciones de error
    error_func = mse if error_func_choice == "MSE" else mae
    regression_error = rmse if regression_error_choice == "RMSE" else rmae

    gradient_mode = st.selectbox("Modo de gradiente descendente", ("Batch", "Mini-Batch"))
    batch_size = None
    if gradient_mode == "Mini-Batch":
        batch_size = st.slider("Proporción del mini-batch", 0.1, 1.0, 0.1)

    if st.button("Ejecutar gradiente descendente (Modo Seleccionado)"):
        if gradient_mode == "Batch":
            model_batch, bias_batch = batched_linear_regression(
                X_example, y_example, regression_error=regression_error, 
                learning_rate=learning_rate_batch, epochs=epochs_batch, 
                error_function=error_func
            )
            st.markdown("#### Resultados (Batch):")
            st.markdown(f"- Coeficientes: $a = {model_batch[0]}$, $b = {model_batch[1]}$")
            st.markdown(f"- Sesgo: $c = {bias_batch}$")
            st.scatter_chart({"Valores reales": y_example, "Predicciones": X_example @ model_batch + bias_batch})
        else:
            model_mini, bias_mini = batched_linear_regression(
                X_example, y_example, regression_error=regression_error, 
                learning_rate=learning_rate_batch, epochs=epochs_batch, 
                error_function=error_func, batch_size=batch_size
            )
            st.markdown("#### Resultados (Mini-Batch):")
            st.markdown(f"- Coeficientes: $a = {model_mini[0]}$, $b = {model_mini[1]}$")
            st.markdown(f"- Sesgo: $c = {bias_mini}$")
            st.scatter_chart({"Valores reales": y_example, "Predicciones": X_example @ model_mini + bias_mini})

def run_regularization_and_sklearn():
    st.header("4. Aplicación de regularizaciones sobre el conjunto de datos de diabetes")
    # Cargar datos
    diabetes = load_diabetes()
    X_diabetes = diabetes.data
    y_diabetes = diabetes.target
    st.dataframe({"Características": diabetes.feature_names, "Valores": X_diabetes[0]})
    
    st.markdown("""
    Se aplicarán técnicas de regularización:
    - **L1 (Lasso)**
    - **L2 (Ridge)**
    """)
    
    learning_rate_reg = st.slider("Tasa de aprendizaje (α)", 0.01, 0.1, 0.01, key="learning_rate_reg_final")
    epochs_reg = st.slider("Número de iteraciones", 1, 10000, 1000, key="epochs_reg")
    error_func_final_choice = st.selectbox("Selecciona la función de error", ("MSE", "MAE"), key="error_func_final")
    regression_error_final_choice = st.selectbox("Selecciona el error de regresión", ("RMSE", "RMAE"), key="regression_error_final")
    lambda_reg = st.slider("Coeficiente de regularización (λ)", 0.01, 1.0, 0.1, key="lambda_reg_final")
    regularization_type = st.selectbox("Tipo de regularización", ("L1 (Lasso)", "L2 (Ridge)"))
    
    error_func_final = mse if error_func_final_choice == "MSE" else mae
    regression_error_final = rmse if regression_error_final_choice == "RMSE" else rmae
    regularization = l1 if regularization_type == "L1 (Lasso)" else l2

    if st.button("Ejecutar regresión con regularización"):
        model_reg, bias_reg = multi_linear_regression(
            X_diabetes, y_diabetes, learning_rate=learning_rate_reg, epochs=epochs_reg, 
            error_function=error_func_final, regression_error=regression_error_final, 
            regularization_derivative=regularization, lambda_reg=lambda_reg
        )
        st.markdown(f"### Resultados ({regularization_type}):")
        st.markdown(f"- Coeficientes: {model_reg}")
        st.markdown(f"- Sesgo: {bias_reg}")
        st.markdown("### Predicciones:")
        y_pred_reg = X_diabetes @ model_reg
        st.scatter_chart({"Valores reales": y_diabetes, "Predicciones": y_pred_reg})
    
    st.header("5. Comparación con Scikit-learn")
    st.markdown("Se comparan las implementaciones propias con las de Scikit-learn.")
    
    alpha_sklearn = st.slider("Coeficiente de regularización (α)", 0.01, 1.0, 0.1, key="alpha_sklearn")
    max_iter_sklearn = st.slider("Número máximo de iteraciones", 100, 10000, 1000, key="max_iter_sklearn")
    regularization_type_sklearn = st.selectbox("Tipo de regularización (Scikit-learn)", 
                                               ("Lasso", "Ridge", "Elastic Net"))
    
    if st.button("Ejecutar regresión con Scikit-learn"):
        start_time = time.time()
        if regularization_type_sklearn == "Lasso":
            model = Lasso(alpha=alpha_sklearn, max_iter=max_iter_sklearn)
        elif regularization_type_sklearn == "Ridge":
            model = Ridge(alpha=alpha_sklearn, max_iter=max_iter_sklearn)
        else:  # Elastic Net
            l1_ratio = st.slider("Proporción L1 (Elastic Net)", 0.0, 1.0, 0.5, key="l1_ratio_sklearn")
            model = ElasticNet(alpha=alpha_sklearn, l1_ratio=l1_ratio, max_iter=max_iter_sklearn)
        
        model.fit(X_diabetes, y_diabetes)
        sklearn_time = time.time() - start_time
        y_pred_sklearn = model.predict(X_diabetes)
        mse_sklearn = mean_squared_error(y_diabetes, y_pred_sklearn)
        mae_sklearn = mean_absolute_error(y_diabetes, y_pred_sklearn)
        
        st.markdown(f"### Resultados ({regularization_type_sklearn} - Scikit-learn):")
        st.markdown(f"- Coeficientes: {model.coef_}")
        st.markdown(f"- Sesgo: {model.intercept_}")
        st.markdown(f"- MSE: {mse_sklearn:.4f}")
        st.markdown(f"- MAE: {mae_sklearn:.4f}")
        st.markdown(f"- Tiempo de ejecución: {sklearn_time:.4f} segundos")
        st.markdown("### Predicciones:")
        st.scatter_chart({"Valores reales": y_diabetes, "Predicciones (Scikit-learn)": y_pred_sklearn})

def main():
    st.title("IAA: Práctica 2 - Regresión lineal")
    participants()
    run_simple_regression()
    run_dataset_regression()
    run_batch_and_minibatch()
    run_regularization_and_sklearn()
    
    st.header("6. Conclusión")
    st.markdown("""
    En esta práctica se han explorado diversos enfoques para la regresión lineal:
    
    1. **Gradiente Descendente:** Optimización de parámetros para minimizar el error.
    2. **Modos Batch y Mini-Batch:** Impacto en la eficiencia y estabilidad.
    3. **Regularización:** Uso de L1 (Lasso) y L2 (Ridge) para evitar el sobreajuste.
    4. **Comparación con Scikit-learn:** Validación de la implementación propia.
    
    Implementar estos métodos desde cero refuerza la comprensión profunda de los algoritmos, mientras que el uso de librerías como Scikit-learn facilita la aplicación práctica en entornos reales.
    """)

if __name__ == "__main__":
    # Definición global de X_example e y_example para reutilizarlos en funciones
    X_example = np.array([
        [1, 2],
        [1, 3],
        [2, 3],
        [2, 4],
        [3, 2],
        [3, 5],
        [4, 1]
    ])
    y_example = np.array([1.03, -1.44, 4.53, 2.24, 13.27, 5.62, 21.53])
    main()
