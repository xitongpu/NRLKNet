# Introduction

This repository contains the implementation code for **NRLKNet**.

We have developed **<ModEval.apk>** for evaluating the actual deployment latency on Android platforms.

## Usage Instructions

Follow the steps below to use **ModEval**:

### Step 1: Download and Install ModEval
Download the installation package **<ModEval.apk>** to your local Android device and install the app.

### Step 2: Export the Model for Deployment
Run the following code to export the model in deployment mode using PyTorch:
```python
from torch.utils.mobile_optimizer import optimize_for_mobile

# Initialize the model
net = NRLKNet(...)
net.eval()

# Define the input sample size
x = torch.rand(1, 1, 2, 128)

# Trace and optimize the model
traced_script_module = torch.jit.trace(net, x)
optimized_traced_model = optimize_for_mobile(traced_script_module)

# Save the optimized model
optimized_traced_model._save_for_lite_interpreter("./nrlknet.pt")
```
After running the code, a deployment model named **`nrlknet.pt`** will be generated in the current directory.

### Step 3: Transfer the Model to Your Android Device
Move the exported deployment model (**`nrlknet.pt`**) to the local storage of your Android device.

### Step 4: Load the Model in ModEval
Launch **ModEval** on your Android device and select the **"Use Custom Model"** option. Click the **"Select Custom Model"** button and choose the deployment model (e.g., **`nrlknet.pt`**) from your local storage.

### Step 5: Generate Input Data
In the input field next to the **"Generate Data"** button, enter the shape of the model input signal. **Note**: This shape must match the input size defined in Step 2 when exporting the model. Then, click the **"Generate Data"** button. A message saying **"Input Success"** will appear, indicating that the input data has been successfully generated.

### Step 6: Run Inference
In the input field next to the **"Start Inference"** button, enter the number of inference runs (e.g., 100). Click the **"Start Inference"** button to measure the model's classification results, average latency, and per-inference latency on the actual mobile device.

## Notes
• **Purpose of ModEval**: This app is designed solely for evaluating the deployment and inference latency of models on actual AIoT devices. It does not assess classification accuracy (a device-independent metric). To simplify the evaluation process, we use random numbers with the same shape as the input signal as model inputs.
• **Flexibility**: By allowing users to select different models and input sizes, this app can evaluate the deployment latency of various models, providing a practical tool for deployment optimization.
• **Future Updates**: We will continue to update this app to enhance its functionality.

