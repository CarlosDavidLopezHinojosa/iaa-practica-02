# Introducción al Aprendizaje Automático: Práctica 2

En esta práctica se implementan y comparan distintas variaciones del algoritmo de regresión lineal múltiple, incluyendo regularización y funciones de error personalizadas. Además, se realiza una comparación con las implementaciones de la librería `scikit-learn`.

## Objetivos

1. Implementar un modelo de regresión lineal múltiple con descenso de gradiente y regularización.
2. Definir funciones de error personalizadas:
   - Error medio cuadrático (_MSE_).
   - Error medio absoluto (_MAE_).
3. Implementar regularizaciones basadas en normas:
   - Norma $L_1$ (Lasso).
   - Norma $L_2$ (Ridge).
4. Comparar el modelo implementado con las soluciones de `scikit-learn` utilizando el conjunto de datos de diabetes.
5. Visualizar las predicciones y errores de los modelos.

## Contenido

### Implementaciones Personalizadas

1. **Regresión Lineal Múltiple**:
   - Función `linear_regression`: Implementa el modelo con soporte para regularización y funciones de error personalizadas.
   - Función `predict`: Realiza predicciones basadas en el modelo ajustado.

2. **Funciones de Error**:
   - `mse`: Calcula el error medio cuadrático.
   - `mae`: Calcula el error medio absoluto.

3. **Regularización**:
   - `l1`: Derivada de la norma $L_1$.
   - `l2`: Derivada de la norma $L_2$.

4. **Visualización**:
   - `plot_predictions`: Genera gráficos comparando valores reales y predicciones.

### Comparación con `scikit-learn`

Se utilizan los modelos de `scikit-learn` para comparar el rendimiento con el modelo implementado:
- `LinearRegression`
- `Lasso`
- `Ridge`
- `ElasticNet`

### Conjunto de Datos

- **Datos de prueba**: Un conjunto de datos pequeño para validar la implementación.
- **Conjunto de datos de diabetes**: Proporcionado por `scikit-learn` para evaluar el rendimiento en un caso práctico.

## Ejecución

1. **Requisitos**:
   - Python 3.x
   - Librerías: `numpy`, `matplotlib`, `scikit-learn`

2. **Pasos**:
   - Ejecutar las celdas del archivo `linear_regression.ipynb` en orden.
   - Analizar los resultados y gráficos generados.

## Resultados Esperados

- Ajuste del modelo implementado a los datos de prueba y al conjunto de datos de diabetes.
- Comparación de errores entre el modelo personalizado y los modelos de `scikit-learn`.
- Visualización de las predicciones y errores.

## Notas

- El modelo implementado permite variar la regularización y las funciones de error, lo que lo hace flexible para diferentes escenarios.
- Los gráficos generados facilitan la interpretación de los resultados.

¡Explora y experimenta con los parámetros para observar cómo afectan al rendimiento del modelo!
