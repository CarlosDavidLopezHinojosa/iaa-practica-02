# Introducción al Aprendizaje Automático: Práctica 2

En esta práctica se implementan y comparan distintas variaciones del algoritmo de regresión lineal múltiple, incluyendo regularización y funciones de error personalizadas. Además, se desarrolla una aplicación interactiva utilizando `Streamlit` para visualizar y analizar los resultados.

## Objetivos

1. Implementar un modelo de regresión lineal múltiple con descenso de gradiente y regularización.
2. Definir funciones de error personalizadas:
   - Error medio cuadrático (_MSE_).
   - Error medio absoluto (_MAE_).
3. Implementar regularizaciones basadas en normas:
   - Norma $L_1$ (Lasso).
   - Norma $L_2$ (Ridge).
4. Comparar el modelo implementado con las soluciones de `scikit-learn` utilizando el conjunto de datos de diabetes.
5. Crear una aplicación interactiva para visualizar y analizar los resultados.

## Estructura del Proyecto

El proyecto está organizado en los siguientes archivos y directorios:

- **`start.sh`**: Script para configurar el entorno virtual, instalar dependencias y ejecutar la aplicación.
- **`requirements.txt`**: Archivo con las dependencias necesarias para el proyecto.
- **`src/`**: Carpeta principal del código fuente.
  - **`view.py`**: Archivo principal de la aplicación `Streamlit`.
  - **`algorithms.py`**: Contiene las implementaciones de los algoritmos de regresión lineal.
  - **`utils/`**: Carpeta con utilidades auxiliares.
    - **`functions.py`**: Funciones de error y regularización.
    - **`stats.py`**: Funciones para realizar pruebas estadísticas.

## Funcionalidades

### Implementaciones Personalizadas

1. **Regresión Lineal**:
   - `simple_regression`: Implementación básica de regresión lineal con gradiente descendente.
   - `linear_regression`: Implementación avanzada con soporte para funciones de error personalizadas.
   - `batched_linear_regression`: Implementación con soporte para mini-batches.
   - `multi_linear_regression`: Implementación con regularización y funciones de error personalizadas.

2. **Funciones de Error**:
   - `mse`: Error medio cuadrático.
   - `mae`: Error medio absoluto.
   - `rmse`: Gradiente para el error cuadrático medio.
   - `rmae`: Gradiente para el error absoluto medio.

3. **Regularización**:
   - `l1`: Derivada de la norma $L_1$ (Lasso).
   - `l2`: Derivada de la norma $L_2$ (Ridge).

4. **Pruebas Estadísticas**:
   - `wilcoxon`: Prueba de Wilcoxon para muestras emparejadas.
   - `friedman`: Prueba de Friedman para muestras relacionadas.
   - `nemenyi`: Prueba post hoc de Nemenyi.
   - `statistical_test`: Determina y ejecuta la prueba estadística adecuada.

### Comparación con `scikit-learn`

Se utilizan los modelos de `scikit-learn` para comparar el rendimiento con el modelo implementado:
- `LinearRegression`
- `Lasso`
- `Ridge`
- `ElasticNet`

### Aplicación Interactiva

La aplicación permite:
- Configurar parámetros como tasa de aprendizaje, número de iteraciones, tipo de regularización, etc.
- Visualizar los resultados de los modelos implementados.
- Comparar los resultados con las implementaciones de `scikit-learn`.
- Realizar análisis estadísticos sobre los errores de los modelos.

## Ejecución

### Opción 1: Accede a la aplicación alojada en el servidor de `streamlit`
- Te permite ejecutar la práctica sin instalar nada, pero puede ir más lenta que si la ejecutas en tu ordenador.
- (enlace)[ssss]


### Opción 2: Usar el script `start.sh`

1. Abre una terminal y navega al directorio del proyecto.
2. Ejecuta el siguiente comando:
   ```bash
   ./start.sh
   ```
3. Esto configurará el entorno virtual, instalará las dependencias y abrirá la aplicación en tu navegador.

### Opción 3: Ejecutar manualmente

1. Abre una terminal y navega al directorio del proyecto.
2. Configura el entorno virtual:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip3 install -r requirements.txt
   ```
3. Ejecuta la aplicación:
   ```bash
   streamlit run src/view.py
   ```
4. Accede a la aplicación en tu navegador en la dirección: [http://localhost:8501](http://localhost:8501).

## Resultados Esperados

- Ajuste del modelo implementado a los datos de prueba y al conjunto de datos de diabetes.
- Comparación de errores entre el modelo personalizado y los modelos de `scikit-learn`.
- Visualización interactiva de las predicciones y errores.
- Análisis estadístico de los resultados.
