# Cargo la librerias necesarias
library(class)  # Para KNN
library(caret)  # Para scaling

# Cargo mi dataset que se llama moon_landings (es mi dataset limpio solo con la informacion que necesito usar)
# Las columnas que incluye mi dataset son "Carrier.Rocket," "Launch.Failure.Occurrences," "Outcome," and "Outcome.Text"
moon_landings <- read_delim("moon_landings.csv", 
                            +     delim = ";", escape_double = FALSE, trim_ws = TRUE)

# Divido mi dataset en features y labels (características y etiquetas)
features <- moon_landings[, -c(1, 4)]  # Y excluyo del análisis las columnas "Carrier.Rocket" y "Outcome.Text" porque éstas son para dar contexto
labels <- moon_landings$Outcome

# Analisis de KNN con un valor default de K (K=5)
knn_model_default <- knn(train = features, test = features, cl = labels)

# Se calcula el precisión con el K default
accuracy_default <- mean(knn_model_default == labels)

# Define un vector de valores K para testear
k_values <- c(3, 5, 7)

# Aqui creo un marco de datos para poder almacenar los resultados
results <- data.frame(K = numeric(0), Accuracy = numeric(0))

# En este loop lo que se hago es recorrer cada valor de K y calculo la precisión
for (k in k_values) {
  knn_model <- knn(train = features, test = features, cl = labels, k = k)
  accuracy <- mean(knn_model == labels)
  results <- rbind(results, data.frame(K = k, Accuracy = accuracy))
}

# Aqui lo que hago es basicamente encontrar la K con la precisión más alta
best_k <- results[which.max(results$Accuracy), "K"]
best_accuracy <- max(results$Accuracy)

# Escala las variables
scaled_features <- scale(features)

# Se realiza el KNN con las variables escaladas
knn_model_scaled <- knn(train = scaled_features, test = scaled_features, cl = labels, k = best_k)

# Se calcula la precisión con las variables escaladas
accuracy_scaled <- mean(knn_model_scaled == labels)

# Imprimir resultados
print(results)

# Imprime el mejor K y su precisión
cat("El mejor K es", best_k, "con una precisión de", best_accuracy, "\n")

# Imprime la precisión con los valores escalados
cat("Precisión con los valores escalados:", accuracy_scaled, "\n")


# Testeo para ver que mi modelo KNN esta funcionando correctamente que se corre después de correr lo anterior


library(caret)
set.seed(123)
train_index <- createDataPartition(labels, p = 0.7, list = FALSE)
train_features <- features[train_index, ]
test_features <- features[-train_index, ]
train_labels <- labels[train_index]
test_labels <- labels[-train_index]

# Entreno mi modelo KNN con respecto al training que tengo en mi dataset con el mejor valor de K
best_k <- 5
knn_model <- knn(train = train_features, test = test_features, cl = train_labels, k = best_k)

# Aqui lo que hago es calcular la precisión y realizo un análisis de las métricas en el dataset de testeo
library(caret)

knn_model <- factor(knn_model, levels = c("0", "1"))
test_labels <- factor(test_labels, levels = c("0", "1"))

confusion <- confusionMatrix(data = knn_model, reference = test_labels)
print(confusion)

accuracy <- confusion$overall["Precisión"]
sensitivity <- confusion$byClass["Sensibilidad"]
specificity <- confusion$byClass["Especificidad"]

cat("Precisión:", accuracy, "\n")
cat("Sensibilidad:", sensitivity, "\n")
cat("Especificidad:", specificity, "\n")
