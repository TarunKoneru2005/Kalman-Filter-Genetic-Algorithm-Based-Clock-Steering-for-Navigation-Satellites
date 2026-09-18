# Kalman Filter & Genetic Algorithm-Based Clock Steering for Navigation Satellites

A simulation-based framework for satellite clock error estimation and clock-steering parameter optimization using Kalman Filtering and Genetic Algorithms.

## Project Overview

Accurate timing is fundamental to satellite navigation systems because errors in satellite clocks directly affect the accuracy of navigation and positioning. Satellite oscillator imperfections can introduce clock bias, frequency offset, and frequency drift over time.

This project presents a software-based approach for estimating and reducing satellite clock errors using a combination of a Kalman Filter and a Genetic Algorithm. The Kalman Filter estimates the stochastic clock states from noisy measurements, while the Genetic Algorithm optimizes the clock-steering parameters to minimize the resulting timing error.

The approach is developed and evaluated through numerical simulations in the context of satellite navigation systems such as NavIC/GNSS.

## Objectives

* Model stochastic satellite clock errors.
* Represent clock bias, frequency offset, and frequency drift as system states.
* Estimate clock states using a Kalman Filter.
* Optimize clock-steering parameters using a Genetic Algorithm.
* Reduce clock estimation and steering error.
* Compare optimized steering with a conventional/manual steering approach.
* Evaluate performance using RMSE and convergence analysis.
* Validate the optimization through Monte Carlo simulations.

## Methodology

The overall methodology consists of the following stages:

```text
Stochastic Clock Model
        ↓
Clock Error Generation
        ↓
Noisy Measurement Generation
        ↓
Kalman Filter
        ↓
Clock-State Estimation
        ↓
Genetic Algorithm
        ↓
Steering Parameter Optimization
        ↓
Clock Steering
        ↓
RMSE & Convergence Analysis
        ↓
Monte Carlo Validation
```

## Satellite Clock Model

The satellite clock is modeled using a three-state representation:

```text
x(k) = [ b(k)  f(k)  d(k) ]ᵀ
```

where:

* `b(k)` represents clock bias.
* `f(k)` represents frequency offset.
* `d(k)` represents frequency drift.

The state evolution is represented by:

```text
x(k+1) = F x(k) + w(k)
```

where `F` is the state-transition matrix and `w(k)` represents process noise.

For the implemented simulation, the state-transition matrix is:

```text
F = [ 1   1   0.5
      0   1   1
      0   0   1 ]
```

## Measurement Model

The clock measurement is represented as:

```text
z(k) = Hx(k) + v(k)
```

where:

* `z(k)` is the measured clock error.
* `H` is the measurement matrix.
* `v(k)` represents measurement noise.

The measurement noise is modeled using the selected measurement-noise covariance.

## Kalman Filter

The Kalman Filter is used to estimate the underlying satellite clock states from noisy measurements.

### Prediction Step

```text
x̂(k|k-1) = F x̂(k-1|k-1)

P(k|k-1) = F P(k-1|k-1) Fᵀ + Q
```

### Update Step

```text
K(k) = P(k|k-1)Hᵀ
       [H P(k|k-1)Hᵀ + R]⁻¹

x̂(k|k) = x̂(k|k-1)
         + K(k)[z(k) - Hx̂(k|k-1)]
```

The resulting estimates of clock bias, frequency offset, and frequency drift are used in the subsequent clock-steering optimization.

## Genetic Algorithm

A Genetic Algorithm is used to search for optimized clock-steering parameters.

Each individual in the population represents a candidate set of steering parameters. The fitness function evaluates the clock error produced by each candidate solution, with the optimization objective being to minimize the selected error metric.

The optimization process consists of:

1. Population initialization
2. Fitness evaluation
3. Selection
4. Crossover
5. Mutation
6. Population update
7. Convergence evaluation

The optimized parameters are subsequently applied to the clock-steering process.

## Simulation Parameters

| Parameter                            |     Value |
| ------------------------------------ | --------: |
| Simulation duration                  |     500 s |
| Sampling interval                    |       1 s |
| Random seed                          |        42 |
| TCXO `h₀`                            | 2 × 10⁻¹⁹ |
| TCXO `h₋₂`                           | 2 × 10⁻²⁰ |
| Measurement noise standard deviation |     10 ns |
| GA population size                   |        40 |
| GA generations                       |        60 |
| Monte Carlo trials                   |       100 |

## Results

The proposed approach was evaluated using clock-error estimation, RMSE, Genetic Algorithm convergence, and Monte Carlo simulations.

### Clock Error

The simulated clock-error characteristics include variations in clock bias, frequency offset, and frequency drift.

![Clock Error](results/figures/clock_error.png)

### Kalman Filter Estimation

The Kalman Filter estimates the underlying clock states from the noisy measurements.

![Kalman Filter Estimation](results/figures/kalman_estimation.png)

### Genetic Algorithm Convergence

The convergence of the Genetic Algorithm is analyzed based on the fitness/error obtained during optimization.

![GA Convergence](results/figures/ga_convergence.png)

### RMSE Comparison

The estimation and steering performance is compared using Root Mean Square Error (RMSE).

![RMSE Comparison](results/figures/rmse_comparison.png)

### Monte Carlo Validation

Multiple simulation trials are performed to evaluate the robustness of the proposed optimization approach under varying stochastic conditions.

![Monte Carlo Results](results/figures/monte_carlo.png)

## Key Results

Under the selected simulation conditions:

* Maximum simulated clock bias was approximately **879.57 ns**.
* Maximum frequency offset was approximately **3.6990 ns/s**.
* Kalman Filter RMSE was reduced from approximately **9.852 ns to 3.778 ns**.
* GA optimization achieved approximately **2.69 ns steady-state RMSE**.
* Monte Carlo evaluation resulted in approximately **4.44 ns RMSE** with GA compared with approximately **60.32 ns** for the manual/reference steering approach.
* The optimized approach demonstrated substantially faster convergence under the evaluated simulation conditions.

## Project Structure

```text
kalman-ga-clock-steering/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── src/
│   ├── main.m
│   ├── clock_model.m
│   ├── kalman_filter.m
│   ├── genetic_algorithm.m
│   ├── fitness_function.m
│   └── monte_carlo.m
│
├── results/
│   └── figures/
│       ├── clock_error.png
│       ├── kalman_estimation.png
│       ├── ga_convergence.png
│       ├── rmse_comparison.png
│       └── monte_carlo.png
│
├── docs/
│   ├── project_report.pdf
│   └── presentation.pdf
│
└── data/
    └── simulation_parameters.csv
```

## How to Run

### Requirements

* MATLAB
* Required MATLAB toolboxes, if applicable to the implementation

### Steps

1. Clone this repository.
2. Open the project directory in MATLAB.
3. Add the `src` directory and its subdirectories to the MATLAB path.
4. Open `main.m`.
5. Run the main script.
6. The simulation will generate the clock-error, Kalman Filter, Genetic Algorithm, and performance-analysis results.

## Technologies and Concepts

* MATLAB
* Kalman Filtering
* Genetic Algorithms
* Stochastic Processes
* State Estimation
* Numerical Simulation
* Monte Carlo Simulation
* Digital Signal Processing
* Satellite Navigation
* GNSS / NavIC

## Applications

The concepts developed in this project are relevant to:

* Satellite navigation timing
* GNSS clock-error estimation
* Satellite oscillator characterization
* Navigation-system synchronization
* Precision timing systems
* Clock steering and reference generation

## Limitations

* The current implementation is simulation-based.
* Clock measurements are generated using a modeled stochastic process.
* Performance is evaluated under the selected simulation conditions.
* Real satellite clock datasets are not currently integrated.
* Real-time hardware implementation is not included in the current version.
* The implementation does not represent a deployment or modification of an operational navigation satellite system.

## Future Work

Future development can include:

* Validation using real GNSS or satellite clock datasets.
* Real-time clock-error acquisition.
* Adaptive estimation of Kalman Filter noise parameters.
* Real-time online clock steering.
* Hardware-in-the-loop validation.
* Fixed-point implementation.
* FPGA-based real-time implementation.
* Integration with real-time GNSS receiver measurements.
* Investigation of alternative satellite clock models and optimization techniques.

## Team

This project was developed as a three-member academic project.

Contributions included clock modeling, Kalman Filter implementation, Genetic Algorithm optimization, simulation, performance analysis, and documentation.

## References

1. R. E. Kalman, "A New Approach to Linear Filtering and Prediction Problems," Journal of Basic Engineering, 1960.
2. Relevant literature on satellite clock modeling and GNSS timing.
3. Relevant literature on Kalman Filtering and state estimation.
4. Relevant literature on Genetic Algorithms and optimization.
5. Technical references related to GNSS/NavIC timing and navigation systems.

## Disclaimer

This project is an academic simulation and research implementation. It is not an implementation or modification of operational NavIC, ISRO, or other satellite-navigation infrastructure. References to NavIC and GNSS describe the intended application context of the work.

## License

This project is intended for academic and educational purposes. Refer to the LICENSE file for applicable usage and redistribution terms.
