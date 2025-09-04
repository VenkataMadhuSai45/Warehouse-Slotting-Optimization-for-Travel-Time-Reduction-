# Warehouse Slotting Optimization for Travel Time Reduction

## Overview

This project implements a comprehensive warehouse slotting and picker routing optimization system to reduce picker travel distance and order fulfillment latency. It integrates demand forecasting, mathematical optimization, and operational research techniques to assign SKUs and optimize picker routes for maximum efficiency.

## Features

1. **Warehouse Slotting Optimization**  
   Assigns SKUs to slots based on forecasted demand and affinity, minimizing picker travel distance.

2. **Demand Forecasting**  
   Uses a weighted model (ARIMA + historical sales) to prioritize SKU placement by demand stability and frequency.

3. **Mathematical Optimization**  
   Formulates the slotting problem as a Mixed Integer Linear Program (MILP) and solves it using Gurobi, minimizing priority-weighted distance to dispatch zones.

4. **Picker Routing Optimization**  
   Utilizes OR-Tools (VRPTW) to optimize picker routes with capacity, speed, service time, and time-window constraints.

5. **Automated Reporting & Visualization**  
   Generates route visualizations and CSV summaries using Matplotlib and Pandas for operational analysis and managerial decision support.

6. **Efficiency Gains**  
   Reduces picker travel time and order fulfillment latency through integrated slotting and routing optimization.

## Requirements

- Python 3.8+
- Gurobi (with license)
- OR-Tools
- Pandas, NumPy, Matplotlib, Seaborn, Statsmodels, scikit-learn, PyYAML, openpyxl

Install dependencies:
```powershell
pip install -r requirements.txt
```

## Usage

1. **Configure warehouse and optimization settings** in `config.yaml`.
2. **Prepare input data** (`Sales.XLSX`, `order_batch.csv`).
3. **Run slotting optimization:**
   ```powershell
   python slotting.Gurobi.py
   ```
4. **Run picker routing optimization:**
   ```powershell
   python picker.py
   ```
5. **Review outputs:**  
   - `slotting_assignments_output.csv`  
   - `picker_summary_by_route.csv`  
   - `picker_routes_analysis.png`

## How It Works

- **Data Loading & Validation:** Cleans and validates order and sales data.
- **Forecasting:** Computes SKU priority scores using ARIMA and historical metrics.
- **Slotting Optimization:** Assigns SKUs to slots via MILP, minimizing weighted travel distance.
- **Routing Optimization:** Solves VRPTW for picker routes, considering operational constraints.
- **Reporting:** Outputs assignment summaries and route visualizations for analysis.

