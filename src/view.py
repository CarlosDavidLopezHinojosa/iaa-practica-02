import streamlit as st
import numpy as np
from algorithms import simple_regression, linear_regression, batched_linear_regression, multi_linear_regression, predict
from utils.functions import mse, mae,rmae, rmse, l1, l2
from utils.stats import statistical_test

st.title("IAA: Práctica 2 - Regresión lineal")

st.header("1. Regresión lineal con gradiente descendente")
st.subheader("1.1. Gradiente descendente: Explicación y ejemplo")

st.markdown("""
El gradiente descendente es un algoritmo de optimización iterativo utilizado para minimizar funciones de costo. 
En el contexto de la regresión lineal, se utiliza para ajustar los parámetros del modelo (pendiente y sesgo) 
de manera que se minimice el error entre las predicciones del modelo y los valores reales.

Por ejemplo, consideremos una ecuación lineal simple de la forma:


$$y = ax + b$$


Donde:
- ($a$) es la pendiente.
- ($b$) es el sesgo o intercepto.

El objetivo es encontrar los valores de ($a$) y ($b$) que minimicen una función de costo, como el error cuadrático medio (MSE).

### Pasos del Algoritmo de Gradiente Descendente
1. **Inicialización**: Se asignan valores iniciales a los parámetros ($a = 0$) y ($b = 0$).
2. **Cálculo del error**: Se mide la diferencia entre las predicciones y los valores reales con MSE.
3. **Cálculo del gradiente**: Se determinan las derivadas parciales con respecto a ($a$) y ($b$).
4. **Actualización de parámetros**: Se ajustan ($a$) y ($b$) restando una fracción de las derivadas.
5. **Repetir** hasta convergencia o alcanzar el número máximo de iteraciones.

### Ejemplo:
Supongamos que tenemos los siguientes datos:
""")

X_simple = np.array([1, 2, 3])
y_simple = np.array([2, 3, 5])

st.dataframe({"X": X_simple, "y": y_simple})

st.markdown("Inicializamos ($a = 0$) y ($b = 0$), y aplicamos gradiente descendente para minimizar el error.")

error_func_simple = st.selectbox("Seleccina el error de regresión", ("MSE", "MAE"))
learning_rate_simple = st.slider("Tasa de aprendizaje (α)", 0.01, 0.1, 0.01)
epochs_simple = st.slider("Número de iteraciones", 1, 10000, 100)

if st.button("Ejecutar gradiente descendente (Ejemplo)"):
    a, b = simple_regression(X_simple, y_simple, error_function=error_func_simple, 
                         learning_rate=learning_rate_simple, epochs=epochs_simple)

    st.markdown(f"""
### Resultados:
Después de aplicar el gradiente descendente:
- ($a = {a}$)
- ($b = {b}$)
""")

    st.markdown("### Predicciones:")
    st.scatter_chart({"Valores reales": y_simple, "Predicciones": a * X_simple + b})

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

st.markdown("""
Ahora vamos a aplicar el algoritmo anterior al conjunto de datos que se nos proporciona, vamos a observar como aprende el algoritmo.
""")

st.dataframe({"X1": X_example[:, 0], "X2": X_example[:, 1], "y": y_example})

learning_rate_example = st.slider("Tasa de aprendizaje (α)", 0.01, 0.1, 0.01,key="learning_rate_example")
epochs_example = st.slider("Número de iteraciones", 1, 10000, 1000,key="epochs_example")

if st.button("Ejecutar gradiente descendente (Datos Proporcionados)"):
    model_example_mse, bias_example_mse = linear_regression(X_example, y_example, regression_error=rmse, 
                         learning_rate=learning_rate_example, epochs=epochs_example, error_function=mse )
    
    model_example_mae, bias_example_mae = linear_regression(X_example, y_example, regression_error=rmae, 
                         learning_rate=learning_rate_example, epochs=epochs_example, error_function=mae)
    
    st.markdown("### 2.1. Resultados:")
    st.markdown("#### 2.1.1. Usando MSE:")
    st.markdown(f"- Coeficientes: $a = {model_example_mse[0]}$, $b =  {model_example_mse[1]}$")
    st.markdown(f"- Sesgo: $c = {bias_example_mse}$")
    st.markdown("#### 2.1.2. Usando MAE:")
    st.markdown(f"- Coeficientes: $a =  {model_example_mae[0]}$, $b = {model_example_mae[1]}$")
    st.markdown(f"- Sesgo: $c = {bias_example_mae}$")


    st.subheader("Predicciones:")
    st.scatter_chart({"Valores reales": y_example, "Predicciones MSE": X_example @ model_example_mse + bias_example_mse, 
                      "Predicciones MAE": X_example @ model_example_mae + bias_example_mae})
    

    st.subheader("2.2. Comparación de errores")

    st.write(""" Ahora vamos a comparar de manera estadística los errores de ambos modelos, para ver si hay a priori una diferencia significativa entre ellos.""")

    stat_result = statistical_test(np.array([predict(X_example, model_example_mse) + bias_example_mse, predict(X_example, model_example_mae) + bias_example_mae, y_example]))

    st.markdown(f"""
    Prueba realizada: **{stat_result['stat-test']}**\n
    Conclusión: Los errores son **{'significativamente diferentes' if stat_result['reject'] else 'no significativamente diferentes'}**
    """)

    st.write("Lo único que si podemos notar es que el modelo que utilizo MSE como error de regresión con pocas iteraciones ya se aproxima a los valores reales, mientras que el modelo que utilizo MAE no se aproxima tanto.")

st.header("3. Modo bacth y mini-bacth")

st.markdown("""
El modo **batch** y **mini-batch** son dos enfoques diferentes para entrenar un modelo de regresión lineal utilizando gradiente descendente. Ambos afectan la eficiencia y el rendimiento del modelo de manera distinta.

### 3.1. Gradiente Descendente en Modo Batch
En el modo batch, el gradiente descendente utiliza **todo el conjunto de datos** para calcular el gradiente y actualizar los parámetros en cada iteración. 

#### Ventajas:
- Convergencia más estable.
- Resultados más precisos en cada actualización.

#### Desventajas:
- Computacionalmente costoso para conjuntos de datos grandes.
- Requiere más memoria, ya que se procesan todos los datos a la vez.

### 3.2. Gradiente Descendente en Modo Mini-Batch
En el modo mini-batch, el conjunto de datos se divide en pequeños subconjuntos (mini-batches). El gradiente se calcula y los parámetros se actualizan utilizando solo un mini-batch en cada iteración.

#### Ventajas:
- Más eficiente en términos de tiempo y memoria.
- Puede aprovechar la paralelización en hardware como GPUs.
- Introduce cierta aleatoriedad que puede ayudar a escapar de mínimos locales.

#### Desventajas:
- Convergencia menos estable en comparación con el modo batch.
- Puede requerir más iteraciones para alcanzar un resultado óptimo.

### Comparación con las versiones anteriores
En comparación con los enfoques anteriores (donde se procesaban todos los datos directamente), el modo mini-batch puede acelerar el entrenamiento significativamente, especialmente en conjuntos de datos grandes. Sin embargo, puede introducir más ruido en las actualizaciones de los parámetros, lo que podría afectar la precisión final del modelo.

### Ejemplo de Implementación
Selecciona el modo de gradiente descendente:
""")

learning_rate_batch = st.slider("Tasa de aprendizaje (α)", 0.01, 0.1, 0.01,key="learning_rate_batch")
epochs_example_bacth = st.slider("Número de iteraciones", 1, 10000, 1000,key="epochs_batch")
error_func_batch = st.selectbox("Selecciona la función de error", ("MSE", "MAE"), key="error_func_batch")
regression_error_bacth = st.selectbox("Seleccina el error de regresión", ("RMSE", "RMAE"), key="regression_error_batch")

if error_func_batch == "MSE":
    error_func_batch = mse
elif error_func_batch == "MAE":
    error_func_batch = mae

if regression_error_bacth == "RMSE":
    regression_error_bacth = rmse
elif regression_error_bacth == "RMAE":
    regression_error_bacth = rmae

gradient_mode = st.selectbox("Modo de gradiente descendente (Modo Batch o Mini-Batch)", ("Batch", "Mini-Batch"))

batch_size = None
if gradient_mode == "Mini-Batch":
    batch_size = st.slider("Proporción del mini-batch", 0.1, 1.0, 0.1)

if st.button("Ejecutar gradiente descendente (Modo Seleccionado)"):
    if gradient_mode == "Batch":
        model_batch, bias_batch = batched_linear_regression(X_example, y_example, regression_error=regression_error_bacth, 
                                                    learning_rate=learning_rate_example, epochs=epochs_example, 
                                                    error_function=error_func_batch)
        st.markdown(f"Resultados (Batch):")
        st.markdown(f"- Coeficientes: $a = {model_batch[0]}$, $b =  {model_batch[1]}$")
        st.markdown(f"- Sesgo: $c = {bias_batch}$")
        st.scatter_chart({"Valores reales": y_example, "Predicciones": X_example @ model_batch + bias_batch})

    elif gradient_mode == "Mini-Batch":
        model_mini_batch, bias_mini_batch = batched_linear_regression(X_example, y_example, regression_error=regression_error_bacth, 
                                                              learning_rate=learning_rate_example, epochs=epochs_example, 
                                                              error_function=error_func_batch, batch_size=batch_size)
        st.markdown(f"Resultados (Mini-Batch)")
        st.markdown(f"- Coeficientes: $a = {model_mini_batch[0]}$, $b =  {model_mini_batch[1]}$")
        st.markdown(f"- Sesgo: $c = {bias_mini_batch}$")
        st.scatter_chart({"Valores reales": y_example, "Predicciones": X_example @ model_mini_batch + bias_mini_batch})

from sklearn.datasets import load_diabetes

st.header("4. Aplicación de regularizaciones sobre el conjunto de datos de diabetes")

# Cargar el conjunto de datos de diabetes
diabetes = load_diabetes()
X_diabetes = diabetes.data
y_diabetes = diabetes.target

st.markdown("""
En esta sección, aplicaremos los métodos de regresión Lasso (L1) y Ridge (L2) sobre el conjunto de datos de diabetes.
Estos métodos incluyen un término de regularización que penaliza los coeficientes grandes para evitar el sobreajuste.
""")

st.dataframe({"Características": diabetes.feature_names, "Valores": X_diabetes[0]})

# Selección de hiperparámetros
learning_rate_reg_final = st.slider("Tasa de aprendizaje (α)", 0.01, 0.1, 0.01, key="learning_rate_reg_final")
epochs_reg = st.slider("Número de iteraciones", 1, 10000, 1000, key="epochs_reg")
error_func_final = st.selectbox("Selecciona la función de error", ("MSE", "MAE"), key="error_func_final")
regression_error_final = st.selectbox("Seleccina el error de regresión", ("RMSE", "RMAE"), key="regression_error_final")
lambda_reg_final = st.slider("Coeficiente de regularización (λ)", 0.01, 1.0, 0.1, key="lambda_reg_final")
regularization_type = st.selectbox("Tipo de regularización", ("L1 (Lasso)", "L2 (Ridge)"))

if error_func_final == "MSE":
    error_func_final = mse
elif error_func_final == "MAE":
    error_func_final = mae

if regression_error_final == "RMSE":
    regression_error_final = rmse
elif regression_error_final == "RMAE":
    regression_error_final = rmae

# Mapear el tipo de regularización
if regularization_type == "L1 (Lasso)":
    regularization = l1
elif regularization_type == "L2 (Ridge)":
    regularization = l2

if st.button("Ejecutar regresión con regularización"):
    model_reg, bias_reg = multi_linear_regression(X_diabetes, y_diabetes, learning_rate=learning_rate_reg_final, 
                                                  epochs=epochs_reg, error_function=error_func_final,
                                                  regression_error=regression_error_final, 
                                                  regularization_derivative=regularization, lambda_reg=lambda_reg_final)

    st.markdown(f"### Resultados ({regularization_type}):")
    st.markdown(f"- Coeficientes: {model_reg}")
    st.markdown(f"- Sesgo: {bias_reg}")

    st.markdown("### Predicciones:")
    y_pred_reg = X_diabetes @ model_reg
    st.scatter_chart({"Valores reales": y_diabetes, "Predicciones": y_pred_reg})



from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

st.header("5. Comparación con Scikit-learn")

st.markdown("""
En esta sección, compararemos los resultados y la eficiencia de nuestras implementaciones de regresión Lasso, Ridge y Elastic Net con las implementaciones de Scikit-learn.
""")

# Selección de hiperparámetros
alpha_sklearn = st.slider("Coeficiente de regularización (α)", 0.01, 1.0, 0.1, key="alpha_sklearn")
max_iter_sklearn = st.slider("Número máximo de iteraciones", 100, 10000, 1000, key="max_iter_sklearn")
regularization_type_sklearn = st.selectbox("Tipo de regularización (Scikit-learn)", ("Lasso", "Ridge", "Elastic Net"))

if st.button("Ejecutar regresión con Scikit-learn"):
    start_time = time.time()

    # Selección del modelo de Scikit-learn
    if regularization_type_sklearn == "Lasso":
        model = Lasso(alpha=alpha_sklearn, max_iter=max_iter_sklearn)
    elif regularization_type_sklearn == "Ridge":
        model = Ridge(alpha=alpha_sklearn, max_iter=max_iter_sklearn)
    elif regularization_type_sklearn == "Elastic Net":
        l1_ratio = st.slider("Proporción L1 (Elastic Net)", 0.0, 1.0, 0.5, key="l1_ratio_sklearn")
        model = ElasticNet(alpha=alpha_sklearn, l1_ratio=l1_ratio, max_iter=max_iter_sklearn)

    # Entrenamiento del modelo
    model.fit(X_diabetes, y_diabetes)
    sklearn_time = time.time() - start_time

    # Predicciones
    y_pred_sklearn = model.predict(X_diabetes)

    # Cálculo de métricas
    mse_sklearn = mean_squared_error(y_diabetes, y_pred_sklearn)
    mae_sklearn = mean_absolute_error(y_diabetes, y_pred_sklearn)

    # Resultados
    st.markdown(f"### Resultados ({regularization_type_sklearn} - Scikit-learn):")
    st.markdown(f"- Coeficientes: {model.coef_}")
    st.markdown(f"- Sesgo: {model.intercept_}")
    st.markdown(f"- MSE: {mse_sklearn:.4f}")
    st.markdown(f"- MAE: {mae_sklearn:.4f}")
    st.markdown(f"- Tiempo de ejecución: {sklearn_time:.4f} segundos")

    # Visualización de predicciones
    st.markdown("### Predicciones:")
    st.scatter_chart({"Valores reales": y_diabetes, "Predicciones (Scikit-learn)": y_pred_sklearn})


st.header("6. Conclusión")

st.markdown("""
    En esta práctica, hemos explorado diferentes enfoques para resolver problemas de regresión lineal, desde la implementación manual de gradiente descendente hasta el uso de técnicas avanzadas como regularización y comparación con bibliotecas como Scikit-learn.

    ### Resumen de los aprendizajes:
    1. **Gradiente Descendente**:
        - Es un método iterativo para optimizar funciones de costo.
        - La elección de la tasa de aprendizaje y el número de iteraciones afecta significativamente los resultados.

    2. **Modos de Gradiente Descendente**:
        - El modo batch es más estable pero computacionalmente costoso.
        - El modo mini-batch es más eficiente pero introduce ruido en las actualizaciones.

    3. **Regularización**:
        - Lasso (L1) y Ridge (L2) ayudan a evitar el sobreajuste penalizando coeficientes grandes.
        - La elección del coeficiente de regularización ($\lambda$) es crucial para equilibrar sesgo y varianza.

    4. **Comparación con Scikit-learn**:
        - Las implementaciones de Scikit-learn son más rápidas y optimizadas.
        - Los resultados obtenidos con nuestras implementaciones son comparables, lo que valida su correcto funcionamiento.

    ### Reflexión:
    El uso de herramientas como Scikit-learn simplifica el desarrollo de modelos, pero implementar los algoritmos desde cero proporciona un entendimiento profundo de los conceptos subyacentes. Este conocimiento es esencial para diagnosticar problemas y personalizar soluciones en proyectos reales.
    """)