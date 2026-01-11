# 🔍 Causal Inference Portal

**A No-Code Tool for Estimating True Causal Impact using Double Machine Learning (DML).**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![EconML](https://img.shields.io/badge/EconML-Microsoft-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

Page: https://causal-inference-ml.streamlit.app/

The **Causal Inference Portal** is a web-based application designed to bridge the gap between rigorous econometric theory and practical business decision-making.

In the world of observational data (e.g., historical sales, marketing campaigns, user behavior), simple comparisons are often misleading. **Correlation is not causation.** This tool allows Product Managers, Data Scientists, and Economists to upload raw data and rigorously measure the **true causal impact** of interventions while automatically controlling for confounding variables and time trends.

It leverages Microsoft's **EconML** library to perform **Double Machine Learning (DML)**, providing statistical confidence intervals, p-values, and heterogeneous treatment effect distributions without writing a single line of code.

---

## Key Features

* **Observation-Ready:** Works with data where treatment was *not* randomly assigned (e.g., measuring the impact of a subscription feature on users who *chose* to buy it).
* **Time-Series Intelligence:** Built-in "Time Logic" automatically switches the statistical backend to a **Difference-in-Differences (DiD)** design to account for seasonality and market trends.
* **Double Machine Learning (DML):** Uses `CausalForestDML` to isolate the treatment effect from hundreds of potential confounders (covariates) non-parametrically.
* **Automated Reporting:** Generates a downloadable **PDF Report** containing executive summaries, confidence intervals, impact distribution charts, and logic flowcharts.
* **Interactive Visualizations:** Explore how impact varies across different user segments (Heterogeneous Treatment Effects).

---

## The Theory: Under the Hood

### 1. The Problem: Selection Bias
In randomized controlled trials (A/B tests), the treated and control groups are statistically identical. In real-world data, they are not.

* **Example:** You want to measure if a "Premium Membership" increases "User Spend."
* **The Trap:** Users who buy Premium are likely *already* richer and more engaged than those who don't.
* **Result:** A simple comparison ($Avg(Premium) - Avg(Free)$) overestimates the impact because it captures both the membership effect *and* the user's inherent wealth. This is **Selection Bias**.

### 2. The Solution: Double Machine Learning (DML)
This portal uses **Robinson's Two-Stage Procedure** to strip away this bias. It trains two separate Machine Learning models (Random Forests):

1.  **Treatment Model ($M_T$):** Predicts *who* gets the treatment based on user traits ($X$).
    * *Residual $T_{res} = T_{actual} - M_T(X)$* (The part of treatment that *couldn't* be predicted by traits).
2.  **Outcome Model ($M_Y$):** Predicts the *outcome* based on user traits ($X$).
    * *Residual $Y_{res} = Y_{actual} - M_Y(X)$* (The part of the outcome that *couldn't* be predicted by traits).

**The Final Step:** It regresses $Y_{res}$ on $T_{res}$. By analyzing the residuals, we compare variations in treatment that are *independent* of the user's characteristics, effectively simulating a randomized experiment.

### 3. Time Series Logic (Difference-in-Differences)
When you enable **Time Logic** in the portal, the definition of "Treatment" changes to handle time trends (like inflation or seasonality).

It effectively runs a **Difference-in-Differences (DiD)** analysis:
* Instead of asking *"Do Treated users spend more?"*
* It asks: *"Did the Treated users' spending **increase faster** than the Control users' spending after the intervention date?"*

**Formula:**
$$Impact = (Y_{Treat, Post} - Y_{Treat, Pre}) - (Y_{Control, Post} - Y_{Control, Pre})$$

This removes two massive biases:
1.  **Group Bias:** Treated users were essentially different to start with.
2.  **Time Bias:** Everyone's spending was going up anyway (e.g., holiday season).