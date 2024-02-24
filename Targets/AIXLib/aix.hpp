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
        size_t totalSize = std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<size_t>());
        m_data = std::vector<float>(totalSize, value);
        computeStrides();
    }

    // Constructor
    TensorValue(float value) : m_shape(std::vector<size_t>{1, 1})
    {
        m_data = std::vector<float>(1, value);
        computeStrides();
    }

    // Access element at a specific index.
    float & operator()(const std::vector<size_t>& indices) { return m_data[getIndex(indices)]; }

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

    // Overload the * operator for scalar multiplication
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

    void fill(float value)
    {
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            m_data[i] = value;
        }
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


class Tensor
{
public:
    // Constructor
    Tensor() : m_requireGrad{false}, m_isRoot{false} { }

    // Constructor for a simple tensor with an optional gradient requirement.
    Tensor(const TensorValue & value, bool requireGrad = false, bool isRoot = false)
            :  m_value{value}, m_requireGrad{requireGrad}, m_isRoot{isRoot}
    {
        m_grad = TensorValue(0, value.shape());
    }

    // Calculate all values in the graph recursively.
    void evaluate()  { m_evaluateFunc(this); }

    // Perform backpropagation to calculate gradients recursively.
    void backward(const TensorValue & seed)  { m_backwardFunc(this, seed); }

    // Getters and setters for the tensor's value.
    const TensorValue & value() const        { return m_value; }
    void setValue(const TensorValue & value) { m_value = value; }

    // Gradient-related methods.
    const TensorValue & grad() const { return m_grad; }
    void zeroGrad()                  { m_grad.fill(0); }
    bool isRequireGrad() const       { return m_requireGrad; }

    static void defaultEvaluation([[maybe_unused]] Tensor * obj) { }
    static void defaultBackward(Tensor * obj, const TensorValue & seed)
    {
        if (obj->m_requireGrad)
            obj->m_grad = obj->m_grad + seed;
    }

    // Auto gradient methods for add operation.
    static void addEvaluateFunc(Tensor * obj)
    {
        if (!obj->m_a) return;
        obj->m_a->evaluate();
        obj->m_b->evaluate();
        obj->m_value = obj->m_a->value() + obj->m_b->value();
    }

    static void addBackwardFunc(Tensor * obj, const TensorValue & seed)
    {
        if (!obj->m_a) return;
        // Calculate gradients.
        obj->m_a->backward(seed);
        obj->m_b->backward(seed);
    }

    // Auto gradient methods for sub operation.
    static void subEvaluateFunc(Tensor * obj)
    {
        if (!obj->m_a) return;
        obj->m_a->evaluate();
        obj->m_b->evaluate();
        obj->m_value = obj->m_a->value() - obj->m_b->value();
    }

    static void subBackwardFunc(Tensor * obj, const TensorValue & seed)
    {
        if (!obj->m_a) return;
        // Calculate gradients.
        obj->m_a->backward(seed);
        obj->m_b->backward(-seed);
    }

    // Auto gradient methods for mul operation.
    static void mulEvaluateFunc(Tensor * obj)
    {
        if (!obj->m_a) return;
        obj->m_a->evaluate();
        obj->m_b->evaluate();
        obj->m_value = obj->m_a->value() * obj->m_b->value();
    }

    static void mulBackwardFunc(Tensor * obj, const TensorValue & seed)
    {
        if (!obj->m_a) return;
        // Calculate gradients.
        obj->m_a->backward(obj->m_b->value() * seed);
        obj->m_b->backward(obj->m_a->value() * seed);
    }

    // Auto gradient methods for div operation.
    static void divEvaluateFunc(Tensor * obj)
    {
        if (!obj->m_a) return;
        obj->m_a->evaluate();
        obj->m_b->evaluate();
        obj->m_value = obj->m_a->value() / obj->m_b->value();
    }

    static void divBackwardFunc(Tensor * obj, const TensorValue & seed)
    {
        if (!obj->m_a) return;
        // Calculate gradients.
        obj->m_a->backward(seed / obj->m_b->value());                                               // ∂f/∂a = 1 / b
        obj->m_b->backward(-obj->m_a->value() * seed / (obj->m_b->value() * obj->m_b->value()));    // ∂f/∂b = -a / b^2
    }

    // Auto gradient methods for sin operation.
    static void sinEvaluateFunc(Tensor * obj)
    {
        if (!obj->m_a) return;
        obj->m_a->evaluate();
        obj->m_value = TensorValue::sin(obj->m_a->value());
    }

    static void sinBackwardFunc(Tensor * obj, const TensorValue & seed)
    {
        if (!obj->m_a) return;
        // The derivative of sin(a) with respect to 'a' is cos(a).
        // Therefore, the gradient of the input is multiplied by cos(a).
        obj->m_a->backward(TensorValue::cos(obj->m_a->value()) * seed);   // ∂f/∂a = cos(a)
    }

    static void tanhEvaluateFunc(Tensor * obj)
    {
        if (!obj->m_a) return;
        obj->m_a->evaluate();
        obj->m_value = TensorValue::tanh(obj->m_a->value());
    }

    static void tanhBackwardFunc(Tensor * obj, const TensorValue & seed)
    {
        if (!obj->m_a) return;
        // The derivative of tanh(a) with respect to 'a' is 1 - tanh^2(a).
        // Therefore, the gradient of the input is multiplied by (1 - tanh^2(a)).
        auto tanhValue = TensorValue::tanh(obj->m_a->value());
        auto oneTensor = TensorValue(1.0, tanhValue.shape());
        obj->m_a->backward((oneTensor - tanhValue * tanhValue) * seed);  // ∂f/∂a = (1 - tanh^2(a))
    }

    // Overload the + operator
    Tensor operator+(const Tensor & other) const
    {
        Tensor result;
        result.m_a = duplicateInstance(this, m_isRoot);
        result.m_b = duplicateInstance(&other, other.m_isRoot);
        result.m_evaluateFunc = addEvaluateFunc;
        result.m_backwardFunc = addBackwardFunc;
        return result;
    }

    // Overload the - operator
    Tensor operator-(const Tensor & other) const
    {
        Tensor result;
        result.m_a = duplicateInstance(this, m_isRoot);
        result.m_b = duplicateInstance(&other, other.m_isRoot);
        result.m_evaluateFunc = subEvaluateFunc;
        result.m_backwardFunc = subBackwardFunc;
        return result;
    }

    // Overload the - operator
    Tensor operator*(const Tensor & other) const
    {
        Tensor result;
        result.m_a = duplicateInstance(this, m_isRoot);
        result.m_b = duplicateInstance(&other, other.m_isRoot);
        result.m_evaluateFunc = mulEvaluateFunc;
        result.m_backwardFunc = mulBackwardFunc;
        return result;
    }

    // Overload the / operator
    Tensor operator/(const Tensor & other) const
    {
        Tensor result;
        result.m_a = duplicateInstance(this, m_isRoot);
        result.m_b = duplicateInstance(&other, other.m_isRoot);
        result.m_evaluateFunc = divEvaluateFunc;
        result.m_backwardFunc = divBackwardFunc;
        return result;
    }

    static Tensor sin(const Tensor & other)
    {
        Tensor result;
        result.m_a = duplicateInstance(&other, other.m_isRoot);
        result.m_b = nullptr;
        result.m_evaluateFunc = sinEvaluateFunc;
        result.m_backwardFunc = sinBackwardFunc;
        return result;
    };

    static Tensor tanh(const Tensor & other)
    {
        Tensor result;
        result.m_a = duplicateInstance(&other, other.m_isRoot);
        result.m_b = nullptr;
        result.m_evaluateFunc = tanhEvaluateFunc;
        result.m_backwardFunc = tanhBackwardFunc;
        return result;
    };

protected:
    inline static std::shared_ptr<Tensor> duplicateInstance(const Tensor * tensor, bool isRoot)
    {
        // Duplicates instances that are not root (temporary) objects or converts root objects (source parameters)
        // to shared pointers with a custom no-deleter to prevent deletion of objects.
        auto noDelete = [](const Tensor*){ };   // Custom deleter not to delete external raw pointers.
        return isRoot ? std::shared_ptr<Tensor>(const_cast<Tensor*>(tensor), noDelete) :
                        std::make_shared<Tensor>(*tensor);
    }

    TensorValue m_value;
    TensorValue m_grad;
    bool  m_requireGrad{false};
    bool  m_isRoot{false};

    // Pointers to operands, if this tensor is the result of an operation.
    std::shared_ptr<Tensor>  m_a{nullptr};
    std::shared_ptr<Tensor>  m_b{nullptr};

    std::function<void(Tensor * tensor)>  m_evaluateFunc = defaultEvaluation;
    std::function<void(Tensor * tensor, const TensorValue & seed)>  m_backwardFunc = defaultBackward;
};


class SGDOptimizer
{
public:
    explicit SGDOptimizer(const std::vector<Tensor*> & parameters, float lr = 0.01f)
        : m_parameters(parameters), m_lr(lr) {}

    void step()
    {
        for (const auto & param : m_parameters)
        {
            if (param->isRequireGrad())
            {
                param->setValue(param->value() - param->grad() * m_lr);   // w' = w - lr * w_gradient.
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


namespace nn
{

class Module
{
public:
    virtual ~Module() = default;

    void registerParameter(Tensor & tensor)
    {
        m_parameters.emplace_back(&tensor);
    }

    std::vector<Tensor*> parameters()
    {
        return m_parameters;
    }

private:
    std::vector<Tensor*> m_parameters;
};

}   // namespace


inline Tensor tensor(const std::vector<float>& data, const std::vector<size_t>& shape, bool requireGrad = false)
{
    return {TensorValue{data, shape}, requireGrad, true};
}

inline Tensor tensor(const std::vector<float>& data, bool requireGrad = false)
{
    return {TensorValue{data, {1, data.size()}}, requireGrad, true};
}

}   // namespace
