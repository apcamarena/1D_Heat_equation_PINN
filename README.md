# Physics-Informed Neural Networks (PINNs) for 1D Heat Diffusion

This repository contains my implementation of Physics-Informed Neural Networks (PINNs)
for solving and analyzing partial differential equations. The main model focuses on the
**1D heat equation** and demonstrates both forward simulation and **inverse parameter
estimation** (learning the thermal diffusivity from noisy data).  
A smaller introductory PINN for a quadratic ODE is also included.

This work was completed as part of my graduate research in Applied Mathematics at UC Irvine. The overall project structure was inspired by example PINN implementations previously shared with me by my advisor. All code and model development in this repository are my own original work.

---

## Project Overview

PINNs combine noisy data with physical laws by embedding the governing PDE directly into
the loss function. For the 1D heat equation, the total loss is:

$$L_{total}= L_{data}+ \eta_{phys} L_{PDE}+ L_{IC}+ L_{BC}$$
  
where the PDE residual is

$$R(x,t) = T_t - \alpha\,T_{xx} - Q(x,t).$$

The model is also used to solve an **inverse problem** by learning the unknown thermal
diffusivity parameter $\alpha$ directly from data.

---

## Training Procedure

The PINN is trained for 5000 epochs using the Adam optimizer:

- Sample interior collocation points across the domain  
- Minimize the PDE residual at those points  
- Fit noisy temperature measurements  
- Enforce the initial condition $T(x,0)=\sin(\pi x)$ 
- Enforce boundary conditions $T(0,t)=T(L,t)=0$ 
- Jointly optimize:
  - The set of all trainable neural-network weights and biases, denoted $\theta$
  - trainable physical parameter $\alpha_{param}$
  

---

## Understanding the Parameters: $\alpha_{learned}$ vs. $\widehat{\alpha}(x,t)$

A PINN contains two types of quantities:

### **1. Global learned parameter — $\alpha_{\text{learned}}$**  
- A **single trainable scalar**  
- Updated every epoch during training  
- Final value after training:

$$\alpha_{learned} = \alpha_{param}^{(N_{epochs})}$$

This is the model’s best estimate of the true diffusivity.

### **2. Local diagnostic parameter — $\widehat{\alpha}(x,t)$**  
After training, the model is **frozen** (no further updates).  
At each grid point $(x_i,t_j)$, we compute:

$$\widehat{\alpha}(x,t) = \frac{T_t(x,t) - Q(x,t)}{T_{xx}(x,t)}.$$

This quantity:

- is not a learned parameter  
- varies across the domain  
- measures how well the PINN satisfies the PDE locally
- forms a distribution whose mean and variance quantify model accuracy  

If training succeeds:

$$\widehat{\alpha}(x,t) \approx \alpha_{true}, \quad \alpha_{learned} \approx \alpha_{true}.$$

---

## Summary of Results

- The PINN accurately reconstructed the temperature field $T(x,t)$
  from sparse, noisy data.
- The learned diffusivity satisfied:
  $\alpha_{learned} \approx \alpha_{true}$  with error < 1%
- Local estimates $\widehat{\alpha}(x,t)$ formed a tight distribution around the truth.
- PDE residuals were centered near zero across the entire domain.
- Compared to a baseline neural network trained only on data:
  - the PINN reduced prediction error by ≈ 99%
  - the baseline produced unstable, noisy estimates of $\widehat{\alpha}(x,t)$
  - the PINN produced a smooth, physically consistent solution

**Conclusion:**  
Enforcing physical laws enables the model to infer both the full temperature field and the
hidden diffusivity parameter, even with noisy data, dramatically outperforming
purely data-driven models.


---

## Repository Structure

- 1D_heat_equation_PINN.ipynb # Main experiment (PINN + inverse problem)
- README.md
- quadratic_PINN.ipynb # Introductory PINN example

---

## How to run
download notebook: 1D_heat-equation_PINN.ipynb 
Then run all cells.
Plots, diagnostics, and learned parameters will be generated automatically.


