# Dose-Response Curve Fitting Task

## Task Description

Given this CSV of drug concentrations and viability, fit a 4-parameter logistic curve, estimate IC50, and show the curve.

## Input Data

**File**: `dose_response.csv`

**Columns**:

- `conc_uM`: Drug concentration in micromolar (µM)
- `viability_percent`: Cell viability as a percentage

**Data Description**: Synthetic dose-response data generated from a known 4-parameter logistic (4PL) function with IC50 approximately 9 µM. The data represents typical cell viability measurements across a range of drug concentrations.

## Expected Output

The agent should:

1. **Execute code** to analyze the dose-response data
2. **Generate a plot**: `dose_response_curve.png`
   - Show original data points
   - Display the fitted 4PL curve
   - Include proper axis labels and legend
3. **Provide a summary** containing:
   - **IC50 estimate**: Should be between 8.5 and 9.5 µM
   - **Goodness-of-fit metric**: R² or similar statistical measure
   - **4PL parameters**: Top, bottom, slope, and IC50 values
   - **Interpretation**: Brief explanation of the results

## 4-Parameter Logistic Model

The 4PL equation is commonly used for dose-response curves:

$$
y = d + \frac{a - d}{1 + \left(\frac{x}{c}\right)^b}
$$

Where:

- `a` = minimum response (bottom asymptote)
- `d` = maximum response (top asymptote)
- `c` = IC50 (inflection point)
- `b` = Hill slope (steepness of the curve)
