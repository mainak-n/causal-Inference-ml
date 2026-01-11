# Causal Inference Portal

**A No-Code Tool for Estimating True Causal Impact using Double Machine Learning (DML) & Difference-in-Differences (DiD).**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![EconML](https://img.shields.io/badge/EconML-Microsoft-green)
![Statsmodels](https://img.shields.io/badge/Statsmodels-v0.14%2B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

The **Causal Inference Portal** is a web-based application designed to bridge the gap between rigorous econometric theory and practical business decision-making.
Portal URL: https://causal-inference-ml.streamlit.app/

In the world of observational data (e.g., historical sales, marketing campaigns, user behavior), simple comparisons are often misleading. **Correlation is not causation.** This tool allows Product Managers, Data Scientists, and Economists to upload raw data and rigorously measure the **true causal impact** of interventions while automatically controlling for confounding variables and time trends.

It seamlessly switches between **Double Machine Learning (DML)** for cross-sectional data and **Difference-in-Differences (DiD)** for time-series data to provide the most accurate estimate possible without writing code.

---

## Key Features

* **Observation-Ready:** Works with data where treatment was *not* randomly assigned (e.g., measuring the impact of a subscription feature on users who *chose* to buy it).
* **Smart Logic Switching:**
    * **Time Logic ON:** Uses **Difference-in-Differences (DiD)** with OLS to measure the "lift" in trajectory over time.
    * **Time Logic OFF:** Uses **Double Machine Learning (DML)** with Causal Forests to control for high-dimensional confounders.
* **Robust Date Handling:** Automatically engineers features like *Month*, *Day of Week*, and *Is_Weekend* to control for seasonality.
* **Visual Sanity Checks:** Includes a **Treatment vs. Control** tab that visualizes parallel trends (for Time Series) or distributional differences (for Cross-Sectional).
* **Automated Reporting:** Generates a downloadable **PDF Report** containing executive summaries, confidence intervals, impact distribution charts, and logic flowcharts.

---

## The Theory: Under the Hood

### 1. The Problem: Selection Bias & Confounding
In randomized controlled trials (A/B tests), the treated and control groups are statistically identical. In real-world data, they are not.
* **Example:** Users who buy "Premium" might already be richer than those who don't.
* **Result:** A simple average comparison overestimates the impact because it captures both the *membership effect* and the *user's wealth*.

### 2. Solution A: Double Machine Learning (DML)
*Used when **Time Logic** is OFF.*

We use **Robinson's Two-Stage Procedure** powered by Microsoft's `EconML`.
1.  **Treatment Model ($M_T$):** Predicts *who* gets the treatment based on user traits ($X$).
2.  **Outcome Model ($M_Y$):** Predicts the *outcome* based on user traits ($X$).
3.  **Residual Analysis:** We regress the *unexplained outcome* ($Y_{res}$) on the *unexplained treatment* ($T_{res}$). This strips away the bias from observed characteristics ($X$), leaving only the true causal effect.

### 3. Solution B: Difference-in-Differences (DiD)
*Used when **Time Logic** is ON.*

When analyzing a policy change or marketing launch over time, the model shifts to a DiD regression.
* **The Logic:** It compares the **change** in the Treatment Group to the **change** in the Control Group after the intervention date.
* **Formula:** $Y = \beta_0 + \beta_1(Treat) + \beta_2(Post) + \delta(Treat \times Post) + \gamma X + \epsilon$
    * $\delta$ (The Interaction Term) represents the **True Causal Impact**.
* **Why it works:** It subtracts out both the baseline differences between groups (Group Effect) and the natural trends affecting everyone (Time Effect).