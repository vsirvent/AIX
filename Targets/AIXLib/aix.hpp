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
#include <utility>
#include <random>


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
    TensorValue(float value, const std::vector<size_t>& shape) : m_shape(shape)
    {
        size_t totalSize = std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<>());
        m_data = std::vector<float>(totalSize, value);
        computeStrides();
    }

    // Constructor
    TensorValue(float value) : m_shape(std::vector<size_t>{1, 1})
    {
        m_data = std::vector<float>(1, value);
        computeStrides();
    }

    // Access element at a specific index (non-const version).
    float & operator()(const std::vector<size_t>& indices) { return m_data[getIndex(indices)]; }

    // Access element at a specific index (const version).
    float operator()(const std::vector<size_t>& indices) const { return m_data[getIndex(indices)]; }

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
        // Check if the shapes of the two tensors are the same.
        validateShapes(m_shape, other.m_shape);

        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(0, m_shape);
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            result.m_data[i] = m_data[i] + other.m_data[i];
        }

        return result;
    }

    // Overload the - operator
    TensorValue operator-(const TensorValue& other) const
    {
        // Check if the shapes of the two tensors are the same.
        validateShapes(m_shape, other.m_shape);

        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(0, m_shape);
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            result.m_data[i] = m_data[i] - other.m_data[i];
        }

        return result;
    }

    // Overload the * operator
    TensorValue operator*(const TensorValue& other) const
    {
        // Check if the shapes of the two tensors are the same.
        validateShapes(m_shape, other.m_shape);

        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(0, m_shape);
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            result.m_data[i] = m_data[i] * other.m_data[i];
        }

        return result;
    }

    // Overload the / operator
    TensorValue operator/(const TensorValue& other) const
    {
        // Check if the shapes of the two tensors are the same.
        validateShapes(m_shape, other.m_shape);

        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(0, m_shape);
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            result.m_data[i] = m_data[i] / other.m_data[i];
        }

        return result;
    }

    // Overload the unary - operator
    TensorValue operator-() const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(0, m_shape);
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            result.m_data[i] = -m_data[i];
        }

        return result;
    }

    TensorValue operator+(float scalar) const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(0, m_shape);
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            result.m_data[i] = m_data[i] + scalar;
        }

        return result;
    }

    TensorValue operator-(float scalar) const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(0, m_shape);
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            result.m_data[i] = m_data[i] - scalar;
        }

        return result;
    }

    TensorValue operator+=(float scalar) const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(0, m_shape);
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            result.m_data[i] += scalar;
        }

        return result;
    }

    TensorValue operator-=(float scalar) const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(0, m_shape);
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            result.m_data[i] -= scalar;
        }

        return result;
    }

    TensorValue operator*(float scalar) const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(0, m_shape);
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            result.m_data[i] = m_data[i] * scalar;
        }

        return result;
    }

    TensorValue operator/(float scalar) const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(0, m_shape);
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            result.m_data[i] = m_data[i] / scalar;
        }

        return result;
    }

    friend TensorValue operator*(float scalar, const TensorValue& tensor)
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(0, tensor.m_shape);
        for (size_t i = 0; i < tensor.m_data.size(); ++i)
        {
            result.m_data[i] = scalar * tensor.m_data[i];
        }

        return result;
    }

    friend TensorValue operator/(float scalar, const TensorValue& tensor)
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(0, tensor.m_shape);
        for (size_t i = 0; i < tensor.m_data.size(); ++i)
        {
            result.m_data[i] = scalar / tensor.m_data[i];
        }

        return result;
    }

    friend TensorValue operator+(float scalar, const TensorValue& tensor)
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(0, tensor.m_shape);
        for (size_t i = 0; i < tensor.m_data.size(); ++i)
        {
            result.m_data[i] = scalar + tensor.m_data[i];
        }

        return result;
    }

    friend TensorValue operator-(float scalar, const TensorValue& tensor)
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(0, tensor.m_shape);
        for (size_t i = 0; i < tensor.m_data.size(); ++i)
        {
            result.m_data[i] = scalar - tensor.m_data[i];
        }

        return result;
    }

    void fill(float value)
    {
        for (float & i : m_data)
        {
            i = value;
        }
    }

    float mean() const
    {
        if (m_data.empty()) return 0.0f; // Guard against division by zero for empty tensors

        float sum = std::accumulate(m_data.begin(), m_data.end(), 0.0f);
        return sum / static_cast<float>(m_data.size());
    }

    static TensorValue sqrt(const TensorValue & value)
    {
        // Perform element-wise sin.
        TensorValue result(0, value.shape());
        for (size_t i = 0; i < value.data().size(); ++i)
        {
            result.m_data[i] = std::sqrt(value.m_data[i]);
        }

        return result;
    }

    static TensorValue sin(const TensorValue & value)
    {
        // Perform element-wise sin.
        TensorValue result(0, value.shape());
        for (size_t i = 0; i < value.data().size(); ++i)
        {
            result.m_data[i] = std::sin(value.m_data[i]);
        }

        return result;
    }

    static TensorValue cos(const TensorValue & value)
    {
        // Perform element-wise cos.
        TensorValue result(0, value.shape());
        for (size_t i = 0; i < value.data().size(); ++i)
        {
            result.m_data[i] = std::cos(value.m_data[i]);
        }

        return result;
    }

    static TensorValue tanh(const TensorValue & value)
    {
        // Perform element-wise tanh.
        TensorValue result(0, value.shape());
        for (size_t i = 0; i < value.data().size(); ++i)
        {
            result.m_data[i] = std::tanh(value.m_data[i]);
        }

        return result;
    }

    // Matrix multiplication for 2D tensors.
    static TensorValue matmul(const TensorValue & a, const TensorValue & b)
    {
        // Ensure both tensors are 2D or can be treated as such.
        if (a.shape().size() != 2 || b.shape().size() != 2)
        {
            throw std::invalid_argument("Both tensors must be 2D for matrix multiplication.");
        }

        // Check if the inner dimensions match.
        if (a.shape()[1] != b.shape()[0])
        {
            throw std::invalid_argument("The inner dimensions of the tensors do not match.");
        }

        size_t m = a.shape()[0];        // Rows of the first matrix
        size_t n = b.shape()[1];        // Columns of the second matrix
        size_t inner = a.shape()[1];    // Inner dimension

        // Resultant tensor shape
        std::vector<size_t> resultShape = {m, n};
        TensorValue result(0.0f, resultShape);

        // Perform matrix multiplication
        for (size_t i = 0; i < m; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                float sum = 0.0f;
                for (size_t k = 0; k < inner; ++k)
                {
                    sum += a({i, k}) * b({k, j});
                }
                result({i, j}) = sum;
            }
        }

        return result;
    }

    // Transpose of the tensor for 2D tensors.
    TensorValue transpose() const
    {
        // Ensure the tensor is 2D.
        if (m_shape.size() != 2)
        {
            throw std::invalid_argument("Transpose operation is currently implemented for 2D tensors only.");
        }

        // The shape of the transposed tensor will have swapped dimensions.
        TensorValue transposed(0.0f, {m_shape[1], m_shape[0]});

        // Perform the transpose operation.
        for (size_t i = 0; i < m_shape[0]; ++i)
        {
            for (size_t j = 0; j < m_shape[1]; ++j)
            {
                // Swap the indices for the transposition.
                transposed({j, i}) = (*this)({i, j});
            }
        }

        return transposed;
    }

private:
    // Compute the strides based on the shape of the tensor
    void computeStrides()
    {
        m_strides.resize(m_shape.size());
        size_t stride = 1;
        for (int64_t i = m_shape.size() - 1; i >= 0; --i)
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

    inline void validateShapes(const auto & shape1, const auto & shape2) const
    {
        if (shape1 != shape2)
        {
            throw std::invalid_argument("Shapes of the tensors must be the same.");
        }
    }

private:
    std::vector<float>  m_data;      // The flat array of tensor elements
    std::vector<size_t> m_shape;     // The shape of the tensor
    std::vector<size_t> m_strides;   // The strides for indexing the tensor
};


class Tensor;

class TensorNode
{
public:
    explicit TensorNode(const TensorValue & value, bool requireGrad = false)
            :  m_value{value}, m_grad{0, value.shape()}, m_requireGrad{requireGrad}
    {
    }

    // Calculate all values in the graph recursively.
    void evaluate()  { m_evaluateFunc(this); }

    // Perform backpropagation to calculate gradients recursively.
    void backward(const TensorValue & seed)  { m_backwardFunc(this, seed); }

    TensorValue  m_value;
    TensorValue  m_grad;
    bool  m_requireGrad;
    std::shared_ptr<TensorNode>  m_a{nullptr};
    std::shared_ptr<TensorNode>  m_b{nullptr};
    std::function<void(TensorNode * tensor)>                            m_evaluateFunc{nullptr};
    std::function<void(TensorNode * tensor, const TensorValue & seed)>  m_backwardFunc{nullptr};
};


class Tensor
{
public:
    // Constructor.
    Tensor() = default;

    // Constructor.
    explicit Tensor(const TensorValue & value, bool requireGrad = false)
    {
        // Create a new Tensor Graph Node.
        m_data = std::make_shared<TensorNode>(value, requireGrad);
        m_data->m_evaluateFunc = defaultEvaluation;
        m_data->m_backwardFunc = defaultBackward;
    }

    // Calculate all values in the graph recursively.
    void evaluate()  { m_data->evaluate(); }

    // Perform backpropagation to calculate gradients recursively.
    void backward(const TensorValue & seed)  { m_data->backward(seed); }

    // Getters and setters for the tensor's value.
    const TensorValue & value() const        { return m_data->m_value; }
    void setValue(const TensorValue & value) { m_data->m_value = value; }

    // Gradient-related methods.
    const TensorValue & grad() const { return m_data->m_grad; }
    void zeroGrad()                  { m_data->m_grad.fill(0); }
    bool isRequireGrad() const       { return m_data->m_requireGrad; }

    static void defaultEvaluation([[maybe_unused]] TensorNode * node) { }
    static void defaultBackward(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_requireGrad)
        {
            node->m_grad = node->m_grad + seed;
        }
    }

    // Auto gradient methods for add operation.
    static void addEvaluateFunc(TensorNode * node)
    {
        if (!node->m_a || !node->m_b) return;
        node->m_a->evaluate();
        node->m_b->evaluate();
        node->m_value = node->m_a->m_value + node->m_b->m_value;
    }

    static void addBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a || !node->m_b) return;
        // Calculate gradients.
        node->m_a->backward(seed);
        node->m_b->backward(seed);
    }

    // Auto gradient methods for sub operation.
    static void subEvaluateFunc(TensorNode * node)
    {
        if (!node->m_a || !node->m_b) return;
        node->m_a->evaluate();
        node->m_b->evaluate();
        node->m_value = node->m_a->m_value - node->m_b->m_value;
    }

    static void subBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a || !node->m_b) return;
        // Calculate gradients.
        node->m_a->backward(seed);
        node->m_b->backward(-seed);
    }

    // Auto gradient methods for mul operation.
    static void mulEvaluateFunc(TensorNode * node)
    {
        if (!node->m_a || !node->m_b) return;
        node->m_a->evaluate();
        node->m_b->evaluate();
        node->m_value = node->m_a->m_value * node->m_b->m_value;
    }

    static void mulBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a || !node->m_b) return;
        // Calculate gradients.
        node->m_a->backward(node->m_b->m_value * seed);
        node->m_b->backward(node->m_a->m_value * seed);
    }

    // Auto gradient methods for div operation.
    static void divEvaluateFunc(TensorNode * node)
    {
        if (!node->m_a || !node->m_b) return;
        node->m_a->evaluate();
        node->m_b->evaluate();
        node->m_value = node->m_a->m_value / node->m_b->m_value;
    }

    static void divBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a || !node->m_b) return;
        // Calculate gradients.
        node->m_a->backward(seed / node->m_b->m_value);                                               // ∂f/∂a = 1 / b
        node->m_b->backward(-node->m_a->m_value * seed / (node->m_b->m_value * node->m_b->m_value));  // ∂f/∂b = -a / b^2
    }

    // Auto gradient methods for sin operation.
    static void sinEvaluateFunc(TensorNode * node)
    {
        if (!node->m_a) return;
        node->m_a->evaluate();
        node->m_value = TensorValue::sin(node->m_a->m_value);
    }

    static void sinBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // The derivative of sin(a) with respect to 'a' is cos(a).
        // Therefore, the gradient of the input is multiplied by cos(a).
        node->m_a->backward(TensorValue::cos(node->m_a->m_value) * seed);   // ∂f/∂a = cos(a)
    }

    static void tanhEvaluateFunc(TensorNode * node)
    {
        if (!node->m_a) return;
        node->m_a->evaluate();
        node->m_value = TensorValue::tanh(node->m_a->m_value);
    }

    static void tanhBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // The derivative of tanh(a) with respect to 'a' is 1 - tanh^2(a).
        // Therefore, the gradient of the input is multiplied by (1 - tanh^2(a)).
        auto tanhValue = TensorValue::tanh(node->m_a->m_value);
        auto oneTensor = TensorValue(1.0, tanhValue.shape());
        node->m_a->backward((oneTensor - tanhValue * tanhValue) * seed);  // ∂f/∂a = (1 - tanh^2(a))
    }

    static void matmulEvaluateFunc(TensorNode *node)
    {
        if (!node->m_a || !node->m_b) return;
        node->m_a->evaluate();
        node->m_b->evaluate();
        node->m_value = TensorValue::matmul(node->m_a->m_value, node->m_b->m_value);
    }

    static void matmulBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a || !node->m_b) return;
        // Assuming m_a and m_b are the input matrices a and b, respectively,
        // and seed is ∂E/∂c, the gradient of the loss with respect to the output matrix c.
        // Compute gradients with respect to a and b

        // Corrected to use matrix multiplication for backward pass calculations
        node->m_a->backward(TensorValue::matmul(seed, node->m_b->m_value.transpose()));      // ∂E/∂a = ∂E/∂c * b^T
        node->m_b->backward(TensorValue::matmul(node->m_a->m_value.transpose(), seed));      // ∂E/∂b = a^T * ∂E/∂c
    }

    static void meanEvaluateFunc(TensorNode * node)
    {
        if (!node->m_a) return;
        node->m_a->evaluate();
        node->m_value = TensorValue(node->m_a->m_value.mean(), {1, 1});
    }

    static void meanBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // The gradient of the mean operation is distributed evenly across all elements. grad = 1/N
        node->m_a->backward(seed / float(node->m_a->m_value.data().size()));
    }

    // Overload the + operator
    Tensor operator+(const Tensor & rhsTensor) const
    {
        Tensor result({0, {value().shape()}});
        result.m_data->m_a = m_data;
        result.m_data->m_b = rhsTensor.m_data;
        result.m_data->m_evaluateFunc = addEvaluateFunc;
        result.m_data->m_backwardFunc = addBackwardFunc;
        return result;
    }

    // Overload the - operator
    Tensor operator-(const Tensor & rhsTensor) const
    {
        Tensor result({0, {value().shape()}});
        result.m_data->m_a = m_data;
        result.m_data->m_b = rhsTensor.m_data;
        result.m_data->m_evaluateFunc = subEvaluateFunc;
        result.m_data->m_backwardFunc = subBackwardFunc;
        return result;
    }

    // Overload the * operator
    Tensor operator*(const Tensor & rhsTensor) const
    {
        Tensor result({0, {value().shape()}});
        result.m_data->m_a = m_data;
        result.m_data->m_b = rhsTensor.m_data;
        result.m_data->m_evaluateFunc = mulEvaluateFunc;
        result.m_data->m_backwardFunc = mulBackwardFunc;
        return result;
    }

    // Overload the / operator
    Tensor operator/(const Tensor & rhsTensor) const
    {
        Tensor result({0, {value().shape()}});
        result.m_data->m_a = m_data;
        result.m_data->m_b = rhsTensor.m_data;
        result.m_data->m_evaluateFunc = divEvaluateFunc;
        result.m_data->m_backwardFunc = divBackwardFunc;
        return result;
    }

    static Tensor sin(const Tensor & rhsTensor)
    {
        Tensor result({0, {rhsTensor.value().shape()}});
        result.m_data->m_a = rhsTensor.m_data;
        result.m_data->m_b = nullptr;
        result.m_data->m_evaluateFunc = sinEvaluateFunc;
        result.m_data->m_backwardFunc = sinBackwardFunc;
        return result;
    };

    static Tensor tanh(const Tensor & rhsTensor)
    {
        Tensor result({0, {rhsTensor.value().shape()}});
        result.m_data->m_a = rhsTensor.m_data;
        result.m_data->m_b = nullptr;
        result.m_data->m_evaluateFunc = tanhEvaluateFunc;
        result.m_data->m_backwardFunc = tanhBackwardFunc;
        return result;
    };

    static Tensor matmul(const Tensor & a, const Tensor & b)
    {
        Tensor result({0, {a.value().shape()[0], b.value().shape()[1]}});
        result.m_data->m_a = a.m_data;
        result.m_data->m_b = b.m_data;
        result.m_data->m_evaluateFunc = matmulEvaluateFunc;
        result.m_data->m_backwardFunc = matmulBackwardFunc;
        return result;
    }

    Tensor mean() const
    {
        Tensor result({0, {1, 1}});     // Scalar tensor for the mean result.
        result.m_data->m_a = m_data;
        result.m_data->m_b = nullptr;
        result.m_data->m_evaluateFunc = meanEvaluateFunc;
        result.m_data->m_backwardFunc = meanBackwardFunc;
        return result;
    }

protected:
    std::shared_ptr<TensorNode>  m_data{nullptr};
};


namespace optim
{

class SGDOptimizer
{
public:
    explicit SGDOptimizer(const std::vector<Tensor> & parameters, float lr = 0.01f)
        : m_parameters(parameters), m_lr(lr) {}

    void step()
    {
        for (auto & param : m_parameters)
        {
            if (param.isRequireGrad())
            {
                param.setValue(param.value() - param.grad() * m_lr);   // w' = w - lr * w_gradient.
            }
        }
    }

    void zeroGrad()
    {
        for (auto & param : m_parameters)
        {
            param.zeroGrad();
        }
    }

private:
    std::vector<Tensor> m_parameters;
    float m_lr;     // Learning rate
};


class AdamOptimizer
{
public:
    explicit AdamOptimizer(const std::vector<Tensor> & parameters,
                           float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
            : m_parameters(parameters), m_lr(lr), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon)
    {
        for (const auto & param : m_parameters)
        {
            m_m.emplace_back(0, param.value().shape());
            m_v.emplace_back(0, param.value().shape());
        }
    }

    void step()
    {
        ++m_timestep;
        for (size_t i = 0; i < m_parameters.size(); ++i)
        {
            if (m_parameters[i].isRequireGrad())
            {
                // Update biased first moment estimate.
                m_m[i] = m_beta1 * m_m[i] + (1.0f - m_beta1) * m_parameters[i].grad();

                // Update biased second raw moment estimate.
                m_v[i] = m_beta2 * m_v[i] + (1.0f - m_beta2) * m_parameters[i].grad() * m_parameters[i].grad();

                // Compute bias-corrected first moment estimate.
                TensorValue mHat = m_m[i] / float(1.0f - std::pow(m_beta1, m_timestep));

                // Compute bias-corrected second raw moment estimate.
                TensorValue vHat = m_v[i] / float(1.0f - std::pow(m_beta2, m_timestep));

                // Update parameter.
                m_parameters[i].setValue(m_parameters[i].value() -  m_lr * mHat / (TensorValue::sqrt(vHat) + m_epsilon));
            }
        }
    }

    void zeroGrad()
    {
        for (auto & param : m_parameters)
        {
            param.zeroGrad();
        }
    }

private:
    std::vector<Tensor>  m_parameters;     // Neural Net's learnable parameters.
    float m_lr;             // Learning rate.
    float m_beta1;          // Exponential decay rate for the first moment estimates.
    float m_beta2;          // Exponential decay rate for the second moment estimates.
    float m_epsilon;        // Small constant for numerical stability.
    size_t m_timestep{0};   // Time step.
    std::vector<TensorValue>    m_m;    // First moment vector.
    std::vector<TensorValue>    m_v;    // Second moment vector.
};


}   // namespace


namespace nn
{

class Module
{
public:
    virtual ~Module() = default;

    void registerParameter(Tensor & tensor)
    {
        m_parameters.emplace_back(tensor);
    }

    void registerModule(const Module & module)
    {
        for (const auto & param : module.parameters())
        {
            m_parameters.emplace_back(param);
        }
    }

    std::vector<Tensor> parameters() const
    {
        return m_parameters;
    }

private:
    std::vector<Tensor> m_parameters;
};


class MSELoss
{
public:
    Tensor operator()(const Tensor & predictions, const Tensor & targets)
    {
        auto diff = predictions - targets;
        auto loss = (diff * diff).mean();
        return loss;
    }
};

}   // namespace


inline Tensor tensor(const std::vector<float>& data, const std::vector<size_t>& shape, bool requireGrad = false)
{
    return Tensor{TensorValue{data, shape}, requireGrad};
}

inline Tensor tensor(const std::vector<float>& data, bool requireGrad = false)
{
    return Tensor{TensorValue{data, {1, data.size()}}, requireGrad};
}

inline Tensor randn(const std::vector<size_t>& shape, bool requireGrad = false)
{
    static std::random_device randomDevice;
    static std::mt19937 randGen(randomDevice());
    std::uniform_real_distribution<float> distr(-1, 1);

    size_t totalSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<float> rndData(totalSize);

    // Fill rndData with random numbers
    std::generate(rndData.begin(), rndData.end(), [&distr]() -> float { return distr(randGen); });

    return Tensor{TensorValue{rndData, shape}, requireGrad};
}


}   // namespace
