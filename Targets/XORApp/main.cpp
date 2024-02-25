//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.


// Project includes
#include <aix.hpp>
// External includes
// System includes
#include <iostream>


class NeuralNet : public aix::nn::Module
{
public:
    // Constructor
    NeuralNet(size_t numInputs, size_t numOutputs, size_t numSamples)
    {
        constexpr size_t hlSize = 8;      // Hidden layer size.
        m_w1 = aix::tensor(aix::randn({numInputs, hlSize}), {numInputs, hlSize}, true);
        m_b1 = aix::tensor(aix::randn({numSamples, hlSize}), {numSamples, hlSize}, true);
        m_w2 = aix::tensor(aix::randn({hlSize, numOutputs}), {hlSize, numOutputs}, true);
        m_b2 = aix::tensor(aix::randn({numSamples, numOutputs}), {numSamples, numOutputs}, true);

        // Register learnable parameters.
        registerParameter(m_w1);
        registerParameter(m_b1);
        registerParameter(m_w2);
        registerParameter(m_b2);
    }

    // Forward
    aix::Tensor forward(aix::Tensor x) const
    {
        x = aix::Tensor::tanh(aix::Tensor::matmul(x, m_w1) + m_b1);
        x = aix::Tensor::matmul(x, m_w2) + m_b2;
        return x;
    }

    aix::Tensor m_w1, m_b1;
    aix::Tensor m_w2, m_b2;
};


int main()
{
    constexpr int kNumSamples  = 4;
    constexpr int kNumInputs   = 2;
    constexpr int kNumTargets  = 1;
    constexpr int kNumEpochs   = 1000;
    constexpr int kLogInterval = 10;
    constexpr float kLossThreshold = 1e-6;

    // Example inputs and targets for demonstration purposes.
    auto inputs  = aix::tensor({0.0, 0.0,
                                1.0, 0.0,
                                0.0, 1.0,
                                1.0, 1.0}, {kNumSamples, kNumInputs});

    auto targets = aix::tensor({0.0,
                                1.0,
                                1.0,
                                0.0}, {kNumSamples, kNumTargets});

    // Create a model with a single hidden layer.
    NeuralNet model(kNumInputs, kNumTargets, kNumSamples);

    // Define a loss function and an optimizer.
    aix::SGDOptimizer optimizer(model.parameters(), 0.3f);

    auto lossFunc = aix::nn::MSELoss();

    // Training loop.
    for (size_t epoch = 0; epoch < kNumEpochs; ++epoch)
    {
        optimizer.zeroGrad();       // Zero the gradients before backward pass.

        auto predictions = model.forward(inputs);

        auto loss = lossFunc(predictions, targets);         // Loss calculations are still part of computation graph.

        loss.evaluate();                                    // Compute all values in the graph.
        loss.backward({1, {kNumSamples, kNumTargets}});     // Compute all gradients in the graph.

        optimizer.step();           // Update parameters.

        if (epoch % kLogInterval == 0 || loss.value().data()[0] <= kLossThreshold)
            std::cout << "Epoch: " << epoch << " Loss = " << loss.value().data()[0] << std::endl << std::flush;

        if (loss.value().data()[0] <= kLossThreshold)
            break;
    }
    std::cout << std::endl;

    // Final predictions after training the neural network model.
    auto finalPredictions = model.forward(inputs);
    finalPredictions.evaluate();

    std::cout << "Final Predictions: " << std::endl;
    std::cout << finalPredictions.value().data()[0] << std::endl;
    std::cout << finalPredictions.value().data()[1] << std::endl;
    std::cout << finalPredictions.value().data()[2] << std::endl;
    std::cout << finalPredictions.value().data()[3] << std::endl;

    return 0;
}
