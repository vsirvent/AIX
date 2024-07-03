# AIX (C++)

AIX is a single-header C++ machine learning framework inspired by PyTorch, designed by Arkin Terli for research and AI/ML model development.

### Overview

AIX is designed for high readability while leveraging device acceleration for efficient computations. It supports a wide range of ML and DL tasks, focusing on performance and scalability for both research and production environments.

Key features include:
- Single-header framework with no external dependencies
- Auto differentiation
- Dynamic computation graphs
- Multi-dimension tensor support
- Auto tensor shape alignment (broadcasting)
- Auto data type conversion and promotion
- Plug-and-play hardware acceleration
- Extensible API for custom operations and layers

### Examples

Examples demonstrating AIX usage can be found in the `Targets/AIXExamples` folder.

| Example          | Description                                     |
|------------------|-------------------------------------------------|
| XORApp           | Create a custom Module.                         |
| XORLayerApp      | Use a module within another module.             |
| XORMetalApp      | Use Metal acceleration on Apple Silicon.        |       
| XORSequentialApp | Use the Sequential module.                      |

### Quick Start

Below is an example to train a model for the XOR problem with Metal acceleration for Apple Silicon.

```C++
#include <aix.hpp>
#include <aixDevices.hpp>   // Optional: For acceleration/device support only.

int main()
{
    constexpr int   kNumSamples    = 4;
    constexpr int   kNumInputs     = 2;
    constexpr int   kNumTargets    = 1;
    constexpr int   kNumEpochs     = 1000;
    constexpr float kLearningRate  = 0.02f;
    constexpr float kLossThreshold = 1e-5f;

    // Create a device that uses Apple Metal for GPU computations.
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);

    // Create a model.
    aix::nn::Sequential  model;
    model.add(new aix::nn::Linear(kNumInputs, 8));
    model.add(new aix::nn::Tanh());
    model.add(new aix::nn::Linear(8, 4));
    model.add(new aix::nn::Tanh());
    model.add(new aix::nn::Linear(4, kNumTargets));

    model.to(*device);      // Move the model to the device.

    // Example inputs and targets for demonstration purposes.
    auto inputs  = aix::tensor({0.0, 0.0,
                                0.0, 1.0,
                                1.0, 0.0,
                                1.0, 1.0}, {kNumSamples, kNumInputs}).to(*device);

    auto targets = aix::tensor({0.0,
                                1.0,
                                1.0,
                                0.0}, {kNumSamples, kNumTargets}).to(*device);

    // Create an optimizer.
    aix::optim::Adam optimizer(model.parameters(), kLearningRate);

    // Create a loss function.
    auto lossFunc = aix::nn::MSELoss();

    // Training loop.
    for (size_t epoch = 0; epoch < kNumEpochs; ++epoch)
    {
        auto predictions = model.forward(inputs);
        auto loss = lossFunc(predictions, targets);

        optimizer.zeroGrad();       // Zero the gradients before backward pass.
        loss.backward();            // Compute all gradients in the graph.
        optimizer.step();           // Update neural net's learnable parameters.

        device->commitAndWait();    // Finalize compute batch.

        // Stop training process when loss is lower than the threshold.
        if (loss.value().item() <= kLossThreshold)
            break;
    }

    // Use the trained model for prediction.
    auto predictions = model.forward(inputs);
    device->commitAndWait();

    std::cout << predictions << std::endl;
    // ...
}
```

# Guidance

AIX balances readability and optimization by using a reference device in aix.hpp and supporting plug-and-play devices for 
high-performance acceleration. By default, AIX uses the reference device.

The following core classes form the foundation of AIX:

- `Tensor:` A multi-dimensional array that supports dynamic computation graphs for all operations.
- `TensorValue:` A non-graph version of Tensor, used for computations on the device.
- `Device:` Reference device that allows hardware acceleration for computations.

All other functionalities in AIX are built upon these three primary classes.

### How to build a custom device for hardware acceleration

Derive a new device from the reference device in `aix.hpp` and implement it for immediate use.

```C++
#include <aix.hpp>

class MyDevice : public aix::Device
{
    // Implement your new acceleration device here.
};

int main()
{
    MyDevice device;

    aix::nn::Sequential  model;
    // Add your model layers here

    model.to(device);      // Move the model to the device.
    // ...
}
```

Each custom device should be tested against the reference device implemented in `aix.hpp`. This allows developers to create highly optimized devices without modifying the framework. 
In the future, we plan to publish a device leaderboard to showcase performance.

# Project Build Instructions

Follow the following steps to build the project and make it deployment ready.

Currently, it has been built and tested on macOS Sonoma with no issues.

---

### Step 1: Build Externals

This step will build external libraries.

```bash
cd Externals
./build_all.sh
```

### Step 2: Build Targets

This step will build all binaries and deploy into a specific folder. Assuming you are in the root folder of the project.

```bash
./build.sh release product-rel
```

After the successful build, all target binaries will be deployed into the product-rel folder.

Note: Run the build.sh file without parameters to see all options.

### Build Options

Use the following CMake build options to turn ON or OFF in production:

- AIX_BUILD_STATIC
- AIX_BUILD_EXAMPLES
- AIX_BUILD_TESTS

All options are OFF by default. build.sh enables tests and examples for development purposes only.

# Citation

If you find the library useful in your research, please consider citing it and use the following BibTex entry:

```
@software{AIX2024,
   author = {Arkin Terli},
   title = {{AIX}: Single-header Machine Learning Library},
   url = {https://gitlab.com/arkin/aix},
   version = {0.0.0},
   year = {2024},
}
```

# License

Copyright © 2024 - Present, Arkin Terli. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the name of Arkin Terli nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
