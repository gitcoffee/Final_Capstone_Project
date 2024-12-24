# Professional Certificate in Machine Learning and Artificial Intelligence
##  University of California, Berkeley's

#### June to December, 2024

### Project Title [Final Capstone Project]:

## Clasificación de Tickets de Soporte Técnico Informático

**Author:** [JOSE LUIS ALVAREZ GONZALEZ]

#### Executive summary
Este proyecto se basa en el modelo CRISP-DM para abordar la clasificación automática de tickets de soporte. Se aplicarán una de las técnicas fundamentales  como Clasificador Naive Bayes para modelos multinomiales., regresión logística, SVC, para comparar distintos modelos, tambien el uso de: Cross-validation of models, Grid Search hyperparameters. Finalmente se tomo la decisión de utilizar Clasificador Naive Bayes para modelos multinomiales por rendimiento, pese a que los resultados entre los modelos fueron bastante similares, esto debido a que la Data se preparó adecuadamente para su uso. Ya que de un dataSet correctamente limpio y adaptado, es clave para un buen entrenamiento de los modelos de machine Learning.

El clasificador multinomial de Naive Bayes es adecuado para la clasificación con características discretas (por ejemplo, recuentos de palabras para la clasificación de textos). La distribución multinomial normalmente requiere recuentos enteros de características. [Fuente](https://qu4nt.github.io/sklearn-doc-es/modules/generated/sklearn.naive_bayes.MultinomialNB.html) 

La estructura de CRISP-DM garantiza un enfoque ordenado, desde la comprensión del problema hasta el despliegue del modelo, permitiendo la integración eficiente de procesos de preprocesamiento, modelado y evaluación para obtener resultados precisos y aplicables.

De Capstone Project 11.1: Initial Question and Data:


	Idea de la pregunta general:
	¿Cómo podemos predecir la categoría de importancia (primaria, actualización, spam, alta, media, baja) de los tickets en un sistema de soporte al cliente, basándonos en factores como el remitente, el contexto del ticket, el tiempo transcurrido desde su creación y las imágenes adjuntas, mientras sugerimos una priorización independientemente de la prioridad inicial establecida por el remitente, de manera similar a cómo los sistemas de correo electrónico categorizan y priorizan los mensajes?

	Suposición inicial sobre los datos:
	Se requerirá información sobre:

	El remitente del ticket.
	El contenido textual del ticket (contexto).
	El tiempo transcurrido desde que se creó el ticket.
	Cualquier imagen adjunta al ticket.

	Suposición inicial sobre cómo obtener los datos:
	Los datos se generarán hipotéticamente, utilizando herramientas de generación de datos sintéticos o simulaciones basadas en ejemplos realistas de tickets.

	El único cambio es en relación a los datos que se utilizarón, inicialmente se penso en utilizar datos sinteticos y si hizo asi pero los resultados con los modelos fueron bajos y los ajustes de clases e hiperparametros se estaba haciendo muy tedioso y complejo, por lo que se decidio útilizar una base de datos real, y con este dataset se empezo a enfocar más tiempo en los modelos que al final es el objeto de aprendiza de este Certificado.

#### Rationale
El proyecto busca mejorar la eficiencia operativa y tiempos de respuesta del equipo al automatizar la clasificación de tickets de soporte en categorías predefinidas. Esto permitirá optimizar recursos y priorizar tareas críticas.

#### Research Question
- ¿Cuáles son las categorías principales de tickets?
- ¿Qué métricas evaluarán el éxito del modelo (precisión, recall, F1-score)?
- ¿Cómo abordarás el problema de clases desbalanceadas?

#### Data Sources
- Fuente: Dataset público de Kaggle.
- Estructura esperada: ID del ticket, descripción del problema, categoría etiquetada.

#### Methodology
**Fase 1: Comprensión del Negocio**
- Definir el objetivo del proyecto.
- Formular preguntas clave para guiar el análisis.

**Fase 2: Comprensión de los Datos**
- Exploración de datos (EDA): distribución de clases, análisis de texto.
- Limpieza de datos: eliminación de duplicados, corrección de errores tipográficos.

**Fase 3: Preparación de los Datos**
- Preprocesamiento de texto: tokenización, eliminación de stopwords, lematización.
- Vectorización: TF-IDF, CountVectorizer, embeddings preentrenados (eg. word2vec, GloVe).

**Fase 4: Modelado**
- Selección de un modelo de los modelos: regresión logística, SVC, Naive Bayes.
- Entrenamiento del modelo y ajuste de hiperparámetros (GridSearchCV o RandomizedSearchCV).

**Fase 5: Evaluación**
- Manejo de clases desbalanceadas: sobremuestreo (SMOTE), submuestreo, ajuste de pesos.
- Métricas de evaluación: precisión, recall, F1-score, matriz de confusión.
- Validación cruzada para evaluar estabilidad del modelo.


**Fase 6: Despliegue**
- Pipeline para preprocesamiento o clasificación.

**Fase 7: Documentación y Presentación**
- Informe técnico con discusiones tomadas y resultados.
- Visualización de métricas y ejemplos de clasificación.

#### Results
El análisis proporcionará un modelo capaz de clasificar automáticamente tickets de soporte con alta precisión. Se incluirán resultados gráficos y métricas clave para evaluar el rendimiento del modelo.

#### Next steps
- Refinar el modelo mediante optimización adicional de hiperparámetros.
- Integrar el modelo en un entorno de producción.
- Explorar nuevas técnicas de procesamiento de lenguaje natural (NLP).

#### Outline of project
- [Link to notebook 1  Capstone Project_module 20.1](https://github.com/gitcoffee/Final_Capstone_Project/blob/main/Capstone_Project_20_1.ipynb)
- [Link to notebook 2 Capstone Project_module 24.1](https://github.com/gitcoffee/Final_Capstone_Project/blob/main/Capstone_Project_24_1.ipynb)

##### Contact and Further Information
Para más información, contacta a [1800joseluis@gmail.com] o visita el repositorio del proyecto en GitHub.
