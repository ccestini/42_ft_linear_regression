## Project Overview

The goal of this project is to implement gradient descent to train a linear regression model that predicts car prices based on mileage. The trained model will use the formula:

estimatePrice(mileage) = θ₀ + (θ₁ × mileage)

Where:
- θ₀ is the intercept.
- θ₁ is the slope.

### Gradient Descent Implementation

The model is trained using the following gradient descent formulas to iteratively update the parameters θ₀ and θ₁:

tmp_theta_0 = (learningRate / m) * Σ(i=0 to m-1) [(estimatePrice(mileage[i]) - price[i])] 
tmp_theta_1 = (learningRate / m) * Σ(i=0 to m-1) [(estimatePrice(mileage[i]) - price[i]) * mileage[i]]


Where:
- m is the total number of data points.
- `learningRate` is the step size for parameter updates.

---

## Programs Overview

### 1. **train.py**

The `train.py` script handles the following tasks:

- **Dataset Loading & Validation:** Reads the dataset (CSV file with `km` and `price` columns) and ensures the data is valid.
- **Data Normalization:** Applies min-max normalization to the mileage data to improve gradient descent performance.
- **Gradient Descent Training:** Performs linear regression by iteratively updating θ₀ and θ₁.
- **Model Saving:** Saves the trained parameters θ₀ and θ₁ to a `.txt` file for later use.
- **Visualization:** Plots a graph comparing actual prices with predicted prices to evaluate the model’s performance.

### 2. **predict.py**

The `predict.py` script allows the user to input mileage and predicts the corresponding car price:

- **User Input:** Prompts the user to input mileage (ensuring valid input).
- **Model Loading:** Loads the trained model parameters (θ₀ and θ₁) from the `trained_model.txt` file.
  - If no trained model is found, it defaults to θ₀ = 0 and θ₁ = 0.
- **Price Prediction:** Uses the model to predict the price based on the given mileage.
- **Output:** Displays the predicted price.

### 3. **precision.py**

The `precision.py` script calculates the precision of the model’s predictions by comparing them with actual prices:

- **Dataset Loading:** Loads both the actual prices and the predicted prices.
- **Precision Calculation:** Computes the Mean Absolute Error (MAE), which measures the average magnitude of errors in the predictions:
MAE = (1 / n) * Σ(i=1 to n) |predictedPrice[i] - actualPrice[i]|

Where:
- n is the number of data points.

---

## Concepts & Techniques

### Linear Regression

Linear regression is a statistical method for predicting the value of a dependent variable (e.g., car price) based on one or more independent variables (e.g., mileage). It finds the best-fit line that minimizes the error between predicted and actual values.

- **Linear Regression Formula:** 
y = θ₀ + (θ₁ × x)

Where:
- y is the predicted value.
- θ₀ is the intercept.
- θ₁ is the slope.
- x is the independent variable.

For further understanding, see [IBM's Linear Regression Explanation](https://www.ibm.com/think/topics/linear-regression) and the [Wikipedia article on Linear Regression](https://en.wikipedia.org/wiki/Linear_regression).

### Gradient Descent

Gradient descent is an optimization algorithm used to minimize the cost function by iteratively adjusting the model parameters. The goal is to find the optimal values for θ₀ and θ₁ that minimize the error between predicted and actual prices.

For more information, visit [IBM's Gradient Descent Explanation](https://www.ibm.com/think/topics/gradient-descent) and [GeeksforGeeks Gradient Descent in Linear Regression](https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/).

---

### Data Normalization

Normalization is essential for improving the efficiency of gradient descent. By scaling the input features to a similar range (e.g., [0, 1] using min-max normalization), the convergence of gradient descent is faster and more stable.

- **Normalization Formula:** 

Normalized Value = (Value - Min) / (Max - Min)

  
This ensures that features like mileage are on a similar scale, allowing the gradient descent algorithm to converge more quickly.

---

### Cost Function

In this project, the cost function is based on the Mean Squared Error (MSE) but simplifies the gradient calculation by using an un-squared error. The standard cost function is:

MSE = (1 / m) * Σ(i=1 to m) [(hθ(xᵢ) - yᵢ)²]


Where:
- hθ(xᵢ) is the predicted price.
- yᵢ is the actual price.

However, in our implementation, the error function used for gradient descent is not squared.

---

## Types of Machine Learning

1. **Supervised Learning:** The algorithm learns from labeled data (input-output pairs).
   - Common Algorithms: Linear Regression, Logistic Regression, Decision Trees, Neural Networks.
   - Applications: Predicting prices, spam detection, medical diagnosis.

2. **Unsupervised Learning:** The algorithm finds patterns in data without labeled outputs.
   - Common Algorithms: K-Means Clustering, PCA, Anomaly Detection.
   - Applications: Customer segmentation, market basket analysis.

3. **Reinforcement Learning:** The algorithm learns by interacting with the environment and receiving rewards/penalties.
   - Common Algorithms: Q-Learning, Deep Q-Networks (DQN).
   - Applications: Game playing, robotics.

4. **Semi-Supervised Learning:** Combines a small amount of labeled data with a large amount of unlabeled data.
   - Applications: Text classification, image classification with limited labeled data.

5. **Self-Supervised Learning:** The data provides the supervision, often used in natural language processing (NLP).
   - Applications: Language modeling, generative models.

---

## Conclusion

Linear regression remains a powerful and interpretable tool in machine learning, especially for predicting continuous outcomes like car prices. While it is simple to implement and fast for smaller datasets, it is typically used for problems where a linear relationship exists between variables. More advanced algorithms may be used for capturing more complex relationships or when the problem requires higher accuracy.


