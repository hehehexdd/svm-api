# SVM API

This project provides a REST API for training and using an SVM classifier on the [diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) from sklearn.

## Requirements

- Python 3.12
- Docker

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd project-root
   ```

2. Build the Docker image:
   ```bash
   docker build -t svm-api .
   ```

3. Run tests:
   ```bash
   docker run -it --rm svm-api pytest
   ```

4. Run the Docker container:
   ```bash
   docker run -d -p 8000:8000 svm-api
   ```

5. Access the API at `http://localhost:8000`


## Endpoints

### Train the model

```http
  POST /train/
```

Trains the SVM model using the diabetes dataset.

### Predict

```http
  POST /predict/
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `data`    | `array`  | **Required**. List of lists containing the feature data for prediction. Each inner list should have 10 float values. Example: `[[0.038075906, 0.05068012, 0.061696206, 0.021872354, -0.044223498, -0.03482076, -0.043400846, -0.002592261, 0.01990749, -0.017646125]]` |

Predicts the class and probability for the given data points using the trained SVM model.

### Evaluate the model

```http
  GET /evaluate/
```

Evaluates the trained SVM model and returns the accuracy.

### Get model metadata

```http
  GET /metadata/
```

Returns metadata about the trained SVM model, including the number of samples, number of features, model type, kernel type, regularization parameter (C), and overall accuracy.

### Predict from CSV

```http
  POST /predict/csv/
```

Allows prediction from a CSV file uploaded as form data.

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `file`    | `file`   | **Required**. Upload a CSV file containing the feature data for prediction. |

## Error Handling

The API returns appropriate error messages for incorrect data inputs, prediction issues, and other errors.


## Example Requests

### Train

```bash
curl -X POST "http://localhost:8000/train/"
```

### Predict

```bash
curl -X POST "http://localhost:8000/predict/" -H "Content-Type: application/json" -d '{"data": [[0.038075906, 0.05068012, 0.061696206, 0.021872354, -0.044223498, -0.03482076, -0.043400846, -0.002592261, 0.01990749, -0.017646125]]}'
```

### Evaluate

```bash
curl -X GET "http://localhost:8000/evaluate/"
```

### Metadata

```bash
curl -X GET "http://localhost:8000/metadata/"
```

### Predict from CSV

```bash
curl -X POST "http://localhost:8000/predict/csv/" -F "file=@/path/to/your/file.csv"
```

## Files

- `main.py`: Entry point of the application
- `api.py`: Contains the API endpoints
- `exception_handlers.py`: Custom exception handlers
- `Dockerfile`: Instructions for building the Docker image
- `requirements.txt`: Python dependencies
