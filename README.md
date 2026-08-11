# SPINN: A Physics-Informed Spiking Neural Network for Tropical Cyclone-Aware Wind Speed Forecasting

Official anonymous implementation of the **SPINN** framework for resilient spatiotemporal forecasting under out-of-distribution (OOD) weather events.

## Key Features
* **Spiking Encoder**: Energy-efficient LIF SNN temporal feature extractor.
* **Adjoint Neural ODE**: $O(1)$ memory consumption backpropagation across multi-horizon prediction spans.
* **Physics Loss Regularization**: Atmospheric state consistency constraints embedded directly into the training dynamics.
## Abstract
Tropical cyclones may cause severe human and economic damage, yet forecasting their intensity mostly remains erratic, especially dur- ing critical and rapid-intensification events. Therefore, enhancing the robustness of detection systems under these extreme condi- tions becomes paramount. To address this challenge, we induce Neural Ordinary Differential Equations (NODEs) in our Spiking Neural Network (SNN) architecture. The use of NODEs is based on a Physics-Informed Neural Network (PINN). The proposed model, known as SPINN, effectively combines physical laws with flexible temporal modeling.
## Setup Instructions

```bash
# Clone the repository
git clone [https://github.com/ANONYMOUS/SPINN.git](https://github.com/ANONYMOUS/SPINN.git)
cd SPINN

# Install dependencies
pip install -r requirements.txt

# Run full pipeline and baseline benchmarks
python main.py
