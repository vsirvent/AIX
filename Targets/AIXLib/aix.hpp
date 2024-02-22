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
// External includes
// System includes
#include <iostream>
#include <numeric>
#include <cmath>
#include <cassert>


namespace aix
{

class TensorValue
{
public:
    // Constructor
    TensorValue() = default;

    // Constructor
    TensorValue(const std::vector<float>& data, const std::vector<size_t>& shape)
            : m_data(data), m_shape(shape)
    {
        // Compute the strides for indexing multi-dimensional data.
        computeStrides();
    }

    // Constructor
    TensorValue(const std::vector<size_t>& shape) : m_shape(shape)
    {
        size_t totalSize = std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<size_t>());
        m_data = std::vector<float>(totalSize, 0);
        computeStrides();
    }

    // Access element at a specific index.
    float& operator()(const std::vector<size_t>& indices) { return m_data[getIndex(indices)]; }

    // const float& operator()(const std::vector<size_t>& indices) const { return m_data[getIndex(indices)]; }

    // Get the shape of the tensor
    const std::vector<size_t>& shape() const    { return m_shape; }

    // Get the strides of the tensor
    const std::vector<size_t>& strides() const  { return m_strides; }

    // Get the raw data of the tensor
    const std::vector<float>& data() const      { return m_data; }


    // Operators

    // Overload the + operator
    TensorValue operator+(const TensorValue& other) const
    {
        // Check if the shapes of the two tensors are the same
        if (m_shape != other.m_shape)
        {
            throw std::invalid_argument("Shapes of the tensors must be the same.");
        }

        // Create a new TensorValue to store the result
        TensorValue result(m_shape);

        // Perform element-wise addition
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            result.m_data[i] = m_data[i] + other.m_data[i];
        }

        return result;
    }

    // Overload the - operator
    TensorValue operator-(const TensorValue& other) const
    {
        // Check if the shapes of the two tensors are the same
        if (m_shape != other.m_shape)
        {
            throw std::invalid_argument("Shapes of the tensors must be the same.");
        }

        // Create a new TensorValue to store the result
        TensorValue result(m_shape);

        // Perform element-wise addition
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            result.m_data[i] = m_data[i] - other.m_data[i];
        }

        return result;
    }

    // Overload the * operator
    TensorValue operator*(const TensorValue& other) const
    {
        // Check if the shapes of the two tensors are the same
        if (m_shape != other.m_shape)
        {
            throw std::invalid_argument("Shapes of the tensors must be the same.");
        }

        // Create a new TensorValue to store the result
        TensorValue result(m_shape);

        // Perform element-wise addition
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            result.m_data[i] = m_data[i] * other.m_data[i];
        }

        return result;
    }


    // Overload the / operator
    TensorValue operator/(const TensorValue& other) const
    {
        // Check if the shapes of the two tensors are the same
        if (m_shape != other.m_shape)
        {
            throw std::invalid_argument("Shapes of the tensors must be the same.");
        }

        // Create a new TensorValue to store the result
        TensorValue result(m_shape);

        // Perform element-wise addition
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            result.m_data[i] = m_data[i] / other.m_data[i];
        }

        return result;
    }


private:
    // Compute the strides based on the shape of the tensor
    void computeStrides()
    {
        m_strides.resize(m_shape.size());
        size_t stride = 1;
        for (int i = m_shape.size() - 1; i >= 0; --i)
        {
            m_strides[i] = stride;
            stride *= m_shape[i];
        }
    }

    // Get the flat index from a vector of indices
    size_t getIndex(const std::vector<size_t>& indices) const
    {
        assert(indices.size() == m_shape.size());
        return std::inner_product(indices.begin(), indices.end(), m_strides.begin(), 0);
    }

private:
    std::vector<float>  m_data;      // The flat array of tensor elements
    std::vector<size_t> m_shape;     // The shape of the tensor
    std::vector<size_t> m_strides;   // The strides for indexing the tensor
};


class Tensor
{
public:
    // Constructor
    Tensor() = default;

    // Constructor for a simple tensor with an optional gradient requirement.
    Tensor(float value, bool requireGrad = false)
    {
        m_value = value;
        m_requireGrad = requireGrad;
    }

    // Virtual destructor to allow derived classes to clean up resources.
    virtual ~Tensor() = default;

    virtual void evaluate() { }

    // Perform backpropagation to calculate gradients.
    virtual void backward(float seed)
    {
        if (m_requireGrad)
            m_grad += seed;
    }

    // Getters and setters for the tensor's value.
    float value() const         { return m_value; }
    void setValue(float value)  { m_value = value; }

    // Gradient-related methods.
    float grad() const          { return m_grad; }
    void zeroGrad()             { m_grad = 0; }
    bool isRequireGrad() const  { return m_requireGrad; }

protected:
    float m_value{0};
    float m_grad{0};
    bool  m_requireGrad{false};

    // Pointers to operands, if this tensor is the result of an operation.
    Tensor * m_a{nullptr};
    Tensor * m_b{nullptr};
};


class Add: public Tensor
{
public:
    Add(Tensor & a, Tensor & b) { m_a = &a; m_b = &b; }

    void evaluate() final
    {
        m_a->evaluate();
        m_b->evaluate();
        m_value = m_a->value() + m_b->value();
    }

    void backward(float seed) final
    {
        // Calculate gradients.
        m_a->backward(seed);
        m_b->backward(seed);
    }
};


class Sub: public Tensor
{
public:
    Sub(Tensor & a, Tensor & b) { m_a = &a; m_b = &b; }

    void evaluate() final
    {
        m_a->evaluate();
        m_b->evaluate();
        m_value = m_a->value() - m_b->value();
    }

    void backward(float seed) final
    {
        // Calculate gradients.
        m_a->backward(seed);
        m_b->backward(-seed);
    }
};


class Mul: public Tensor
{
public:
    Mul(Tensor & a, Tensor & b) { m_a = &a; m_b = &b; }

    void evaluate() final
    {
        m_a->evaluate();
        m_b->evaluate();
        m_value = m_a->value() * m_b->value();
    }

    void backward(float seed) final
    {
        // Calculate gradients.
        m_a->backward(m_b->value() * seed);
        m_b->backward(m_a->value() * seed);
    }
};


class Div: public Tensor
{
public:
    Div(Tensor & a, Tensor & b) { m_a = &a; m_b = &b; }

    void evaluate() final
    {
        m_a->evaluate();
        m_b->evaluate();
        m_value = m_a->value() / m_b->value();
    }

    void backward(float seed) final
    {
        // Calculate gradients.
        m_a->backward(seed / m_b->value());                                     // ∂f/∂a = 1 / b
        m_b->backward(-m_a->value() * seed / (m_b->value() * m_b->value()));    // ∂f/∂b = -a / b^2
    }
};


class Sin : public Tensor
{
public:
    explicit Sin(Tensor & a) { m_a = &a; }

    void evaluate() final
    {
        m_a->evaluate();
        m_value = std::sin(m_a->value());
    }

    void backward(float seed) final
    {
        // The derivative of sin(a) with respect to 'a' is cos(a).
        // Therefore, the gradient of the input is multiplied by cos(a).
        m_a->backward(std::cos(m_a->value()) * seed);   // ∂f/∂a = cos(a)
    }
};


class SGDOptimizer
{
public:
    SGDOptimizer(const std::vector<Tensor*> & parameters, float lr = 0.01f) : m_parameters(parameters), m_lr(lr) {}

    void step()
    {
        for (const auto & param : m_parameters)
        {
            if (param->isRequireGrad())
            {
                param->setValue(param->value() - m_lr * param->grad());   // w' = w - lr * w_gradient.
            }
        }
    }

    void zeroGrad()
    {
        for (const auto & param : m_parameters)
        {
            param->zeroGrad();
        }
    }

private:
    std::vector<Tensor*> m_parameters;
    float m_lr;     // Learning rate
};


class Module
{
public:

    virtual ~Module()
    {
        for (auto tensor : m_recycle)
        {
            delete tensor;
        }
    }

    void registerParameter(Tensor & tensor)
    {
        m_parameters.emplace_back(&tensor);
    }

    auto * recycle(auto * tensor)
    {
        m_recycle.emplace_back(tensor);
        return tensor;
    }

    std::vector<Tensor*> parameters()
    {
        return m_parameters;
    }

private:
    std::vector<Tensor*> m_parameters;
    std::vector<Tensor*> m_recycle;
};


}   // namespace
