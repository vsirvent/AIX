//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

#pragma once

// Project includes
#include <aix.hpp>
// External includes
// System includes


namespace aix::nn
{


class Linear : public Module
{
public:
    Linear() = default;

    // Constructor
    Linear(size_t numInputs, size_t numOutputs, size_t numSamples)
    {
        m_w1 = randn({numInputs, numOutputs},  true);  // A tensor filled with random numbers in [-1, 1].
        m_b1 = randn({numSamples, numOutputs}, true);

        // Register learnable parameters.
        registerParameter(m_w1);
        registerParameter(m_b1);
    }

    // Forward
    Tensor forward(Tensor x) const override
    {
        // TODO: m_b1 needs to support broadcasting to remove numSamples params from constructor.
        return Tensor::matmul(x, m_w1) + m_b1;
    }

    Tensor  m_w1;
    Tensor  m_b1;
};


class Tanh : public Module
{
public:
    // Forward
    Tensor forward(Tensor x) const override
    {
        return Tensor::tanh(x);
    }
};


}   // namespace
