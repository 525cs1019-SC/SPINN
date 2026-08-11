# SPINN: Physics-Informed Spiking Neural Network for Tropical Cyclone-Aware Wind Speed Forecasting

Official anonymous implementation of the **SPINN** framework for resilient spatiotemporal forecasting under out-of-distribution (OOD) weather events.

## Key Features
* **Spiking Encoder**: Energy-efficient LIF SNN temporal feature extractor.
* **Adjoint Neural ODE**: $O(1)$ memory consumption backpropagation across multi-horizon prediction spans.
* **Physics Loss Regularization**: Atmospheric state consistency constraints embedded directly into the training dynamics.

## Setup Instructions

```bash
# Clone the repository
git clone [https://github.com/ANONYMOUS/SPINN.git](https://github.com/ANONYMOUS/SPINN.git)
cd SPINN

# Install dependencies
pip install -r requirements.txt

# Run full pipeline and baseline benchmarks
python main.py
