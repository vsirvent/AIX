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
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            m_data[i] = value;
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

    // Copy Constructor
    Tensor(const Tensor& other)
            : m_value{other.m_value},
              m_grad{other.m_grad},
              m_requireGrad{other.m_requireGrad},
              m_isRoot{false},  // Copy should not be considered as raw pointer.
              m_a{other.m_a},
              m_b{other.m_b},
              m_evaluateFunc{other.m_evaluateFunc},
              m_backwardFunc{other.m_backwardFunc}
    {
        // For shared_ptr, simply use the copy constructor which increases the reference count.
        // If there's any deep copying mechanism needed for complex objects or manual memory management,
        // it should be handled here. In this case, shared_ptr takes care of the memory management,
        // so no additional steps are necessary.
    }

    // Copy Assignment Operator
    Tensor& operator=(const Tensor& other)
    {
        if (this != &other)     // Protect against self-assignment
        {
            m_value = other.m_value;
            m_grad = other.m_grad;
            m_requireGrad = other.m_requireGrad;
            m_isRoot = other.m_isRoot;
            m_a = other.m_a;
            m_b = other.m_b;
            m_evaluateFunc = other.m_evaluateFunc;
            m_backwardFunc = other.m_backwardFunc;
            // Handle shared_ptr or other deep copy mechanisms here if necessary.
        }

        return *this;
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
        {
            obj->m_grad = obj->m_grad + seed;
        }
    }

    // Auto gradient methods for add operation.
    static void addEvaluateFunc(Tensor * obj)
    {
        if (!obj->m_a || !obj->m_b) return;
        obj->m_a->evaluate();
        obj->m_b->evaluate();
        obj->m_value = obj->m_a->value() + obj->m_b->value();
    }

    static void addBackwardFunc(Tensor * obj, const TensorValue & seed)
    {
        if (!obj->m_a || !obj->m_b) return;
        // Calculate gradients.
        obj->m_a->backward(seed);
        obj->m_b->backward(seed);
    }

    // Auto gradient methods for sub operation.
    static void subEvaluateFunc(Tensor * obj)
    {
        if (!obj->m_a || !obj->m_b) return;
        obj->m_a->evaluate();
        obj->m_b->evaluate();
        obj->m_value = obj->m_a->value() - obj->m_b->value();
    }

    static void subBackwardFunc(Tensor * obj, const TensorValue & seed)
    {
        if (!obj->m_a || !obj->m_b) return;
        // Calculate gradients.
        obj->m_a->backward(seed);
        obj->m_b->backward(-seed);
    }

    // Auto gradient methods for mul operation.
    static void mulEvaluateFunc(Tensor * obj)
    {
        if (!obj->m_a || !obj->m_b) return;
        obj->m_a->evaluate();
        obj->m_b->evaluate();
        obj->m_value = obj->m_a->value() * obj->m_b->value();
    }

    static void mulBackwardFunc(Tensor * obj, const TensorValue & seed)
    {
        if (!obj->m_a || !obj->m_b) return;
        // Calculate gradients.
        obj->m_a->backward(obj->m_b->value() * seed);
        obj->m_b->backward(obj->m_a->value() * seed);
    }

    // Auto gradient methods for div operation.
    static void divEvaluateFunc(Tensor * obj)
    {
        if (!obj->m_a || !obj->m_b) return;
        obj->m_a->evaluate();
        obj->m_b->evaluate();
        obj->m_value = obj->m_a->value() / obj->m_b->value();
    }

    static void divBackwardFunc(Tensor * obj, const TensorValue & seed)
    {
        if (!obj->m_a || !obj->m_b) return;
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

    static void matmulEvaluateFunc(Tensor *obj)
    {
        if (!obj->m_a || !obj->m_b) return;
        obj->m_a->evaluate();
        obj->m_b->evaluate();
        obj->m_value = TensorValue::matmul(obj->m_a->value(), obj->m_b->value());
    }

    static void matmulBackwardFunc(Tensor * obj, const TensorValue & seed)
    {
        if (!obj->m_a || !obj->m_b) return;
        // Assuming m_a and m_b are the input matrices a and b, respectively,
        // and seed is ∂E/∂c, the gradient of the loss with respect to the output matrix c.
        // Compute gradients with respect to a and b

        // Corrected to use matrix multiplication for backward pass calculations
        obj->m_a->backward(TensorValue::matmul(seed, obj->m_b->value().transpose()));      // ∂E/∂a = ∂E/∂c * b^T
        obj->m_b->backward(TensorValue::matmul(obj->m_a->value().transpose(), seed));      // ∂E/∂b = a^T * ∂E/∂c
    }

    static void meanEvaluateFunc(Tensor * obj)
    {
        if (!obj->m_a) return;
        obj->m_a->evaluate();
        obj->m_value = TensorValue(obj->m_a->value().mean(), {1, 1});
    }

    static void meanBackwardFunc(Tensor * obj, const TensorValue & seed)
    {
        if (!obj->m_a) return;
        // The gradient of the mean operation is distributed evenly across all elements.
        size_t totalElements = obj->m_a->value().data().size();
        TensorValue grad = TensorValue(1.0f / totalElements, obj->m_a->value().shape());
        obj->m_a->backward(grad * seed);    // Adjust seed by the gradient of mean operation.
    }

    // Overload the + operator
    Tensor operator+(const Tensor & other) const
    {
        Tensor result({0, {value().shape()}});
        result.m_a = duplicateInstance(this, m_isRoot);
        result.m_b = duplicateInstance(&other, other.m_isRoot);
        result.m_evaluateFunc = addEvaluateFunc;
        result.m_backwardFunc = addBackwardFunc;
        return result;
    }

    // Overload the - operator
    Tensor operator-(const Tensor & other) const
    {
        Tensor result({0, {value().shape()}});
        result.m_a = duplicateInstance(this, m_isRoot);
        result.m_b = duplicateInstance(&other, other.m_isRoot);
        result.m_evaluateFunc = subEvaluateFunc;
        result.m_backwardFunc = subBackwardFunc;
        return result;
    }

    // Overload the - operator
    Tensor operator*(const Tensor & other) const
    {
        Tensor result({0, {value().shape()}});
        result.m_a = duplicateInstance(this, m_isRoot);
        result.m_b = duplicateInstance(&other, other.m_isRoot);
        result.m_evaluateFunc = mulEvaluateFunc;
        result.m_backwardFunc = mulBackwardFunc;
        return result;
    }

    // Overload the / operator
    Tensor operator/(const Tensor & other) const
    {
        Tensor result({0, {value().shape()}});
        result.m_a = duplicateInstance(this, m_isRoot);
        result.m_b = duplicateInstance(&other, other.m_isRoot);
        result.m_evaluateFunc = divEvaluateFunc;
        result.m_backwardFunc = divBackwardFunc;
        return result;
    }

    static Tensor sin(const Tensor & other)
    {
        Tensor result({0, {other.value().shape()}});
        result.m_a = duplicateInstance(&other, other.m_isRoot);
        result.m_b = nullptr;
        result.m_evaluateFunc = sinEvaluateFunc;
        result.m_backwardFunc = sinBackwardFunc;
        return result;
    };

    static Tensor tanh(const Tensor & other)
    {
        Tensor result({0, {other.value().shape()}});
        result.m_a = duplicateInstance(&other, other.m_isRoot);
        result.m_b = nullptr;
        result.m_evaluateFunc = tanhEvaluateFunc;
        result.m_backwardFunc = tanhBackwardFunc;
        return result;
    };

    static Tensor matmul(const Tensor & a, const Tensor & b)
    {
        Tensor result({0, {a.value().shape()[0], b.value().shape()[1]}});
        result.m_a = duplicateInstance(&a, a.m_isRoot);
        result.m_b = duplicateInstance(&b, b.m_isRoot);
        result.m_evaluateFunc = matmulEvaluateFunc;
        result.m_backwardFunc = matmulBackwardFunc;
        return result;
    }

    Tensor mean() const
    {
        Tensor result({0, {1, 1}});     // Assuming a scalar tensor for the mean result.
        result.m_a = duplicateInstance(this, m_isRoot);
        result.m_b = nullptr;
        result.m_evaluateFunc = meanEvaluateFunc;
        result.m_backwardFunc = meanBackwardFunc;
        return result;
    }

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


namespace optim
{

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


class AdamOptimizer
{
public:
    explicit AdamOptimizer(const std::vector<Tensor*> & parameters,
                           float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
            : m_parameters(parameters), m_lr(lr), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon)
    {
        for (const auto & param : m_parameters)
        {
            m_m.emplace_back(0, param->value().shape());
            m_v.emplace_back(0, param->value().shape());
        }
    }

    void step()
    {
        ++m_timestep;
        for (size_t i = 0; i < m_parameters.size(); ++i)
        {
            if (m_parameters[i]->isRequireGrad())
            {
                // Update biased first moment estimate.
                m_m[i] = m_beta1 * m_m[i] + (1.0f - m_beta1) * m_parameters[i]->grad();

                // Update biased second raw moment estimate.
                m_v[i] = m_beta2 * m_v[i] + (1.0f - m_beta2) * m_parameters[i]->grad() * m_parameters[i]->grad();

                // Compute bias-corrected first moment estimate.
                TensorValue mHat = m_m[i] / (1.0f - std::pow(m_beta1, m_timestep));

                // Compute bias-corrected second raw moment estimate.
                TensorValue vHat = m_v[i] / (1.0f - std::pow(m_beta2, m_timestep));

                // Update parameter.
                m_parameters[i]->setValue(m_parameters[i]->value() -  m_lr * mHat / (TensorValue::sqrt(vHat) + m_epsilon));
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
    std::vector<Tensor*>  m_parameters;     // Neural Net's learnable parameters.
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
        m_parameters.emplace_back(&tensor);
    }

    void registerModule(Module & module)
    {
        for (auto param : module.parameters())
        {
            m_parameters.emplace_back(param);
        }
    }

    const std::vector<Tensor*> parameters() const
    {
        return m_parameters;
    }

private:
    std::vector<Tensor*> m_parameters;
};


class MSELoss
{
public:
    //Tensor operator()(const Tensor & predictions, const Tensor & targets)
    Tensor operator()(Tensor predictions, Tensor targets)
    {
        auto diff = predictions - targets;
        auto loss = (diff * diff).mean();
        return loss;
    }
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

inline std::vector<float> randn(const std::vector<size_t>& shape, float min= -1, float max= 1)
{
    static std::random_device randomDevice;
    static std::mt19937 randGen(randomDevice());
    std::uniform_real_distribution<float> distr(min, max); // Directly use float

    size_t totalSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<float> rndData(totalSize);

    // Fill rndData with random numbers
    std::generate(rndData.begin(), rndData.end(), [&distr]() -> float { return distr(randGen); });

    return rndData;
}


}   // namespace
