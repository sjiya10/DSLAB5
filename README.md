### README for Linear Regression with Gradient Descent

#### Overview

This repository implements **Linear Regression** using **Gradient Descent** to fit a line to a dataset, where the goal is to predict `y` values given `X` values. The code uses **Pandas**, **NumPy**, and **Matplotlib** to load the dataset, perform calculations, and visualize the results. The key objective of this implementation is to understand and demonstrate the concept of cost function minimization and gradient descent for linear regression.

### Objective

The objective of this project is to:
1. Implement linear regression using gradient descent.
2. Visualize the convergence of the cost function during training.
3. Use gradient descent to find the optimal parameters (θ₀ and θ₁) for the linear regression model.
4. Predict new values based on the fitted model and visualize the results.

---

### Key Components

1. **Data Loading and Preprocessing**:
   - The code reads two CSV files: `linearX.csv` and `linearY.csv` containing the features `X` and labels `y`, respectively.
   - The features (`X`) are normalized using standardization (subtracting the mean and dividing by the standard deviation).

2. **Hypothesis Function**:
   - The hypothesis function represents the predicted value of `y` given `X`, `θ₀`, and `θ₁`:
     \[
     h_{\theta}(x) = \theta_0 + \theta_1 x
     \]

3. **Cost Function**:
   - The cost function (mean squared error) is used to evaluate how well the model's predictions match the actual data:
     \[
     J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2
     \]
   - `m` is the number of data points.

4. **Gradient Descent**:
   - The gradient descent algorithm updates the parameters `θ₀` and `θ₁` iteratively to minimize the cost function.
   - The gradients of the cost function with respect to `θ₀` and `θ₁` are calculated and used to update the parameters:
     \[
     \theta_0 = \theta_0 - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})
     \]
     \[
     \theta_1 = \theta_1 - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}) \cdot x^{(i)}
     \]
   - The learning rate `α` controls how big the steps are during optimization.

5. **Visualization**:
   - The cost function convergence is visualized by plotting the cost value against iterations.
   - The regression line and the dataset are plotted to show how well the linear model fits the data.
   - A zoomed-in plot of the first 50 iterations of the gradient descent shows how the cost decreases over time.

---

### Files

- `linearX.csv`: A CSV file containing the input features `X` (independent variable).
- `linearY.csv`: A CSV file containing the target values `y` (dependent variable).
- `linear_regression.py`: The Python script that implements the linear regression algorithm, gradient descent, and visualizations.
  
---

### Dependencies

To run the code, the following libraries are required:
- **Pandas**: For data handling and CSV file loading.
- **NumPy**: For mathematical operations and array manipulation.
- **Matplotlib**: For data visualization and plotting graphs.

You can install the required libraries using `pip`:
```bash
pip install pandas numpy matplotlib
```

---

### Usage

1. Ensure that the CSV files `linearX.csv` and `linearY.csv` are in the same directory as the script or adjust the file paths accordingly.
2. Run the script to train the linear regression model using gradient descent and visualize the results:
   ```bash
   python linear_regression.py
   ```
3. The script will output the optimal values for `θ₀` and `θ₁` and display the following plots:
   - **Cost Function Convergence**: A graph showing the reduction in the cost function over iterations.
   - **Linear Regression Line**: A plot showing the dataset and the fitted regression line.
   - **Cost Function vs Iterations**: A zoomed-in view of the first 50 iterations of gradient descent.

---

### Code Explanation

1. **Data Preprocessing**:
   The `X` and `y` values are loaded from CSV files and flattened into 1D arrays. The input `X` is standardized to have a mean of 0 and a standard deviation of 1.

2. **Hypothesis Function**:
   The `hypothesis` function calculates the predicted `y` values based on the model parameters `θ₀` and `θ₁`.

3. **Cost Function**:
   The `cost_function` computes the cost, which is used to evaluate the performance of the model.

4. **Gradient Descent**:
   The `gradient_descent` function iteratively updates `θ₀` and `θ₁` using the gradients of the cost function. It also tracks the cost history to monitor convergence.

5. **Prediction**:
   The `predict` function uses the learned parameters `θ₀` and `θ₁` to predict new values of `y` for a given `x`.


Additionally, plots showing:
- Cost Function Convergence over iterations.
- The regression line fitted to the dataset.


### Contributing

If you would like to contribute to this project, feel free to open an issue or submit a pull request. Contributions are always welcome!
