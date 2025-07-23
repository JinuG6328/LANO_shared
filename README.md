<h1 align="center">
  Sequential infinite-dimensional Bayesian optimal experimental design with derivative-informed latent attention neural operator
  <br>
</h1>

## Summary

This repository contains a partial differential equation (PDE) based predictive model for tumor growth with application to glioblastoma (GBM) modeling. The repository integrates derivative-informed latent attention neural operators with Bayesian inference to estimate tumor parameters from PDE data. We demonstrate how the adaptive terminal formulation combined with LANO enables computationally efficient sequential Bayesian optimal experimental design (SBOED) for infinite-dimensional PDE-constrained problems.

## Key Features

* **Latent-Attention Neural Operators**: Attention mechanism to capture temporal relationships in PDE solutions
* **Sequential Bayesian Optimal Experimental Design**: Adaptive MRI measurement timing for optimal information acquisition
* **Infinite-Dimensional Bayesian Inference**: Parameter estimation with uncertainty quantification in infinite-dimensional spaces
* **Efficient Computation**: Reduced-order modeling using Active Subspace (AS) and Principal Component Analysis (PCA)
* **Adaptive terminal formulation**: Novel formulation improve convergence to global optima

## Methodology

### Mathematical Framework
- **Tumor Growth Model**: Reaction-diffusion PDE with logistic growth and anisotropic diffusion in heterogeneous brain tissue
- **Dimension reduction**: AS and PCA methods for computational efficiency in Jacobian computations

### Neural Network Architecture
- **Attention Mechanism**: Causal self-attention for temporal dependencies in tumor evolution (includes simplified form for computational efficiency)
- **Dual Output**: Simultaneous prediction of tumor states and pseudo state to compute Jacobians

### Inverse Problem Solving
- **Gradient-Based Optimization**: L-BFGS optimizer integrated with neural operators
- **Uncertainty Quantification**: Posterior sampling using eigenvalue decomposition of posterior covariance matrices

### Sequential Bayesian Optimal Experimental Design

- **Information-Theoretic Criteria**: Expected information gain for optimal observation time
- **Adaptive Terminal Formulation**: Consider global optimality as we update the model parameter estimation

## Installation

### Environment Setup

1. **Create conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate tumor-prediction
   ```
2. **Install FEniCS and hIPPYlib**:
   ```bash
   # Using conda 
   conda install -c conda-forge fenics
   pip install hippylib
   ```
   
## Usage

### 1. Generate Reduced-Order Subspaces
```bash
jupyter notebook TumorModel_generate_subspace.ipynb
```
This generates the Active Subspace (AS) and Principal Component Analysis (PCA) basis for dimensionality reduction.

### 2. Generate Training Data
```bash
python TumorModel_generate_data.py 
```
Generates PDE solution snapshots, parameters, and Jacobian data for neural network training.

### 3. Train the LANO Model and Evaluation
```bash
jupyter notebook LANO_training_map_verification.ipynb
```
Trains the Latent Attention Neural Operator with dual loss (forward prediction + Jacobian matching).

Evaluates trained model performance on test data and compares with ground truth solutions.

Solves inverse problems using the trained neural network surrogate for efficient parameter estimation.

### 4. Run sequential Bayesian optimal experimental design

```bash
jupyter notebook sequential_BOED_with_adaptive_terminal_formulation.ipynb
```
First runs static BOED to find the initial observation point

Then applies adaptive terminal formulation to update parameters as observations are acquired

Demonstrates optimal MRI scheduling for tumor monitoring


### Directory Structure
```
├── mesh/                                 
├── checkpoints/             
├── TumorModel_generate_subspace.ipynb
├── TumorModel_generate_data.py
├── LANO_training_map_verification.ipynb
├── sequential_BOED_with_adaptive_terminal_formulation.ipynb
└── environment.yml
```

## Credits

This software uses the following open source packages:

**FEniCS** is a popular open-source computing platform for solving partial differential equations (PDEs). FEniCS enables users to quickly translate scientific models into efficient finite element code. With high-level Python and C++ interfaces, FEniCS is easy to get started with but offers powerful capabilities for experienced programmers. FEniCS runs on platforms ranging from laptops to high-performance clusters.

**hIPPYlib** implements state-of-the-art scalable algorithms for deterministic and Bayesian inverse problems governed by partial differential equations (PDEs). It builds on FEniCS for PDE discretization and PETSc for scalable and efficient linear algebra operations and solvers.

## References

The tumor growth problem is built from [1]. We extend the inverse problem to sequential optimal experimental design.
The LANO architecture is built from [2] and developed further to achieve accuracy for time-dependent PDE problems.

[1] Liang, Baoshan, et al. "Bayesian inference of tissue heterogeneity for individualized prediction of glioma growth." IEEE Transactions on Medical Imaging 42.10 (2023): 2865-2875.

[2] O'Leary-Roseberry, Thomas, et al. "Derivative-informed neural operator: an efficient framework for high-dimensional parametric derivative learning." Journal of Computational Physics 496 (2024): 112555.


## Publications
If you use this code in your research, please cite:
```bibtex
@article{go2025sequential,
  title={Sequential infinite-dimensional {B}ayesian optimal experimental design with derivative-informed latent attention neural operator},
  author={Go, Jinwoo and Chen, Peng},
  journal={Journal of Computational Physics},
  volume={532},
  pages={113976},
  year={2025},
  publisher={Elsevier}
}
```
## Contact Information

Jinwoo Go [email](mailto:harrison1381a@gmail.com) 
GitHub [@JinuG6328](https://github.com/JinuG6328)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
This work builds upon the foundation established by Liang et al. [1] for tumor growth modeling and O'Leary-Roseberry et al. [2] for derivative-informed neural operators. We thank the FEniCS and hIPPYlib communities for their excellent software tools that made this research possible.&nbsp;&middot;&nbsp;
