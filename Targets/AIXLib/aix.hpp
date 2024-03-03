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

#ifdef AIX_DATA_TYPE_FLOAT
    using DataType = float;
#elif  AIX_DATA_TYPE_DOUBLE
    using DataType = double;
#else
    using DataType = float;     // Default data type.
#endif

// Forward declarations
class Tensor;


enum class DeviceType
{
    kCPU,
    kGPU_METAL,
};


class DeviceAllocator
{
public:
    // Destructor
    virtual ~DeviceAllocator() = default;

    virtual void* allocate(std::size_t size)                              { return std::malloc(size); }
    virtual void  deallocate(void* p, [[maybe_unused]] std::size_t size)  { std::free(p); }
};


template<typename T>
class Allocator : public std::allocator<T>
{
public:
    Allocator()     { m_allocator = std::make_shared<DeviceAllocator>(); }

    explicit Allocator(std::shared_ptr<DeviceAllocator> alloc) : m_allocator(std::move(alloc)) {}

    T* allocate(std::size_t n)
    {
        return static_cast<T*>(m_allocator->allocate(n * sizeof(T)));
    }

    void deallocate(T* p, std::size_t n)
    {
        m_allocator->deallocate(p, n * sizeof(T));
    }

private:
    std::shared_ptr<DeviceAllocator>  m_allocator{nullptr};
};


using Array = std::vector<DataType, Allocator<DataType>>;
using Shape = std::vector<size_t>;
using Index = std::vector<size_t>;


class Device
{
public:
    virtual DeviceType type() const { return DeviceType::kCPU; }

    virtual std::shared_ptr<DeviceAllocator> createMemoryAllocator()
    {
        return std::make_shared<DeviceAllocator>();
    }

    virtual void * allocate(size_t size)
    {
        return std::malloc(size);
    }

    virtual void deallocate(void * memory)
    {
        return std::free(memory);
    }

    virtual void add(const Array & a1, const Array & a2, const size_t size, Array & result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = a1[i] + a2[i];
        }
    }

    virtual void sub(const Array & a1, const Array & a2, const size_t size, Array & result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = a1[i] - a2[i];
        }
    }

    virtual void mul(const Array & a1, const Array & a2, const size_t size, Array & result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = a1[i] * a2[i];
        }
    }

    virtual void div(const Array & a1, const Array & a2, const size_t size, Array & result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = a1[i] / a2[i];
        }
    }

    // Scalar operations

    virtual void add(const Array & a, DataType scalar, const size_t size, Array & result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = a[i] + scalar;
        }
    }

    virtual void sub(const Array & a, DataType scalar, const size_t size, Array & result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = a[i] - scalar;
        }
    }

    virtual void sub(DataType scalar, const Array & a, const size_t size, Array & result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = scalar - a[i];
        }
    }

    virtual void mul(const Array & a, DataType scalar, const size_t size, Array & result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = a[i] * scalar;
        }
    }

    virtual void div(const Array & a, DataType scalar, const size_t size, Array & result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = a[i] / scalar;
        }
    }

    virtual void div(DataType scalar, const Array & a, const size_t size, Array & result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = scalar / a[i];
        }
    }

    virtual void unary(const Array & a, const size_t size, Array & result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = -a[i];
        }
    }

    virtual void fill(DataType value, const size_t size, Array & result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = value;
        }
    }

    virtual void mean(const Array & a, const size_t size, DataType & result)
    {
        DataType sum = std::accumulate(a.begin(), a.end(), DataType(0));
        result = sum / static_cast<DataType>(size);
    }

    virtual void sqrt(const Array & a, const size_t size, Array & result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = std::sqrt(a[i]);
        }
    }

    virtual void sin(const Array & a, const size_t size, Array & result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = std::sin(a[i]);
        }
    }

    virtual void cos(const Array & a, const size_t size, Array & result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = std::cos(a[i]);
        }
    }

    virtual void tanh(const Array & a, const size_t size, Array & result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = std::tanh(a[i]);
        }
    }

    virtual void matmul(const Array & a1, const Shape & s1, const Array & a2, const Shape & s2, Array & result)
    {
        // NOTE: Since TensorValue validated the parameters, device method do not validate again.
        size_t m = s1[0];        // Rows of the first matrix
        size_t n = s2[1];        // Columns of the second matrix
        size_t inner = s1[1];    // Inner dimension

        // Perform matrix multiplication
        for (size_t i = 0; i < m; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                DataType sum = 0;
                for (size_t k = 0; k < inner; ++k)
                {
                    sum += a1[i * s1[1] + k] * a2[k * n + j];
                }
                result[i * n + j] = sum;
            }
        }
    }

    virtual void transpose(const Array & a, const Shape & shape, Array & result)
    {
        // Perform the transpose operation.
        for (size_t i = 0; i < shape[0]; ++i)
        {
            for (size_t j = 0; j < shape[1]; ++j)
            {
                // Swap the indices for the transposition.
                result[j * shape[0] + i] = a[i * shape[1] + j];
            }
        }
    };
};


// Default device.
static Device defaultDevice;      // TODO: default CPU device needs to move to a global context.


class TensorValue
{
public:
    // Constructor
    TensorValue(Array data, Shape shape, Device * device) : m_shape(std::move(shape)), m_device(device)
    {
        // Each tensor array must use device specific memory allocator.
        m_data = Array(std::move(data), Allocator<DataType>(device->createMemoryAllocator()));
        // Compute the strides for indexing multi-dimensional data.
        computeStrides();
    }

    // Constructor
    TensorValue(DataType value, Shape shape, Device * device) : m_shape(std::move(shape)), m_device(device)
    {
        size_t totalSize = std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<>());
        // Each tensor array must use device specific memory allocator.
        m_data = Array(totalSize, value, Allocator<DataType>(device->createMemoryAllocator()));
        computeStrides();
    }

    // Constructor
    TensorValue(Shape shape, Device * device) : m_shape(std::move(shape)), m_device(device)
    {
        size_t totalSize = std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<>());
        // Each tensor array must use device specific memory allocator.
        m_data = Array(totalSize, Allocator<DataType>(device->createMemoryAllocator()));
        computeStrides();
    }

    // Constructor
    TensorValue(DataType value, Device * device) : m_shape{Shape{1, 1}}, m_device(device)
    {
        // Each tensor array must use device specific memory allocator.
        m_data = Array(1, value, Allocator<DataType>(device->createMemoryAllocator()));
        computeStrides();
    }

    // Access element at a specific index (non-const version).
    DataType & operator()(const Index & indices)     { return m_data[getIndex(indices)]; }

    // Access element at a specific index (const version).
    DataType operator()(const Index & indices) const { return m_data[getIndex(indices)]; }

    // Get the shape of the tensor
    const Shape & shape() const    { return m_shape; }

    // Get the strides of the tensor
    const Shape & strides() const  { return m_strides; }

    // Get the raw data of the tensor
    const Array & data() const     { return m_data; }

    // Get the device
    Device * device() const        { return m_device; }

    // Set the device
    void device(Device * device)
    {
        // Move data to the new device. Create a new data with new device and copy the data. Deallocate the old data.
        m_data = Array(m_data, Allocator<DataType>(device->createMemoryAllocator()));
        m_device = device;
    }

    // Operators

    // Overload the + operator
    TensorValue operator+(const TensorValue & other) const
    {
        // Check if the shapes of the two tensors are the same.
        validateShapes(m_shape, other.m_shape);

        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device);
        m_device->add(m_data, other.m_data, m_data.size(), result.m_data);
        return result;
    }

    // Overload the - operator
    TensorValue operator-(const TensorValue & other) const
    {
        // Check if the shapes of the two tensors are the same.
        validateShapes(m_shape, other.m_shape);

        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device);
        m_device->sub(m_data, other.m_data, m_data.size(), result.m_data);
        return result;
    }

    // Overload the * operator
    TensorValue operator*(const TensorValue & other) const
    {
        // Check if the shapes of the two tensors are the same.
        validateShapes(m_shape, other.m_shape);

        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device);
        m_device->mul(m_data, other.m_data, m_data.size(), result.m_data);
        return result;
    }

    // Overload the / operator
    TensorValue operator/(const TensorValue & other) const
    {
        // Check if the shapes of the two tensors are the same.
        validateShapes(m_shape, other.m_shape);

        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device);
        m_device->div(m_data, other.m_data, m_data.size(), result.m_data);
        return result;
    }

    // Overload the unary - operator
    TensorValue operator-() const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device);
        m_device->unary(m_data, m_data.size(), result.m_data);
        return result;
    }

    TensorValue operator+(DataType scalar) const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device);
        m_device->add(m_data, scalar, m_data.size(), result.m_data);
        return result;
    }

    TensorValue operator-(DataType scalar) const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device);
        m_device->sub(m_data, scalar, m_data.size(), result.m_data);
        return result;
    }

    TensorValue operator+=(DataType scalar) const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device);
        m_device->add(m_data, scalar, m_data.size(), result.m_data);
        return result;
    }

    TensorValue operator-=(DataType scalar) const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device);
        m_device->sub(m_data, scalar, m_data.size(), result.m_data);
        return result;
    }

    TensorValue operator*(DataType scalar) const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device);
        m_device->mul(m_data, scalar, m_data.size(), result.m_data);
        return result;
    }

    TensorValue operator/(DataType scalar) const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device);
        m_device->div(m_data, scalar, m_data.size(), result.m_data);
        return result;
    }

    friend TensorValue operator*(DataType scalar, const TensorValue & tensor)
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(tensor.m_shape, tensor.m_device);
        tensor.m_device->mul(tensor.m_data, scalar, tensor.m_data.size(), result.m_data);
        return result;
    }

    friend TensorValue operator/(DataType scalar, const TensorValue & tensor)
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(tensor.m_shape, tensor.m_device);
        tensor.m_device->div(scalar, tensor.m_data, tensor.m_data.size(), result.m_data);
        return result;
    }

    friend TensorValue operator+(DataType scalar, const TensorValue & tensor)
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(tensor.m_shape, tensor.m_device);
        tensor.m_device->add(tensor.m_data, scalar, tensor.m_data.size(), result.m_data);
        return result;
    }

    friend TensorValue operator-(DataType scalar, const TensorValue & tensor)
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(tensor.m_shape, tensor.m_device);
        tensor.m_device->sub(scalar, tensor.m_data, tensor.m_data.size(), result.m_data);
        return result;
    }

    void fill(DataType value)
    {
        m_device->fill(value, m_data.size(), m_data);
    }

    DataType mean() const
    {
        if (m_data.empty()) return 0;
        DataType result;
        m_device->mean(m_data, m_data.size(), result);
        return result;
    }

    TensorValue sqrt()
    {
        // Perform element-wise sin.
        TensorValue result(m_shape, m_device);
        m_device->sqrt(m_data, m_data.size(), result.m_data);
        return result;
    }

    TensorValue sin()
    {
        // Perform element-wise sin.
        TensorValue result(m_shape, m_device);
        m_device->sin(m_data, m_data.size(), result.m_data);
        return result;
    }

    TensorValue cos()
    {
        // Perform element-wise cos.
        TensorValue result(m_shape, m_device);
        m_device->cos(m_data, m_data.size(), result.m_data);
        return result;
    }

    TensorValue tanh()
    {
        // Perform element-wise tanh.
        TensorValue result(m_shape, m_device);
        m_device->tanh(m_data, m_data.size(), result.m_data);
        return result;
    }

    // Matrix multiplication for 2D tensors.
    TensorValue matmul(const TensorValue & b) const
    {
        // Ensure both tensors are 2D or can be treated as such.
        if (m_shape.size() != 2 || b.shape().size() != 2)
        {
            throw std::invalid_argument("Both tensors must be 2D for matrix multiplication.");
        }

        // Check if the inner dimensions match.
        if (m_shape[1] != b.shape()[0])
        {
            throw std::invalid_argument("The inner dimensions of the tensors do not match.");
        }

        // Resultant tensor shape
        Shape resultShape = {m_shape[0], b.shape()[1]};
        TensorValue result(resultShape, m_device);
        m_device->matmul(m_data, m_shape, b.m_data, b.m_shape, result.m_data);
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
        TensorValue transposed({m_shape[1], m_shape[0]}, m_device);
        m_device->transpose(m_data, m_shape, transposed.m_data);
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
    size_t getIndex(const Index & indices) const
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
    Array m_data;      // The flat array of tensor elements
    Shape m_shape;     // The shape of the tensor
    Index m_strides;   // The strides for indexing the tensor
    Device *  m_device{nullptr};
};


class TensorNode
{
public:
    explicit TensorNode(const TensorValue & value, bool requireGrad = false)
            :  m_value{value}, m_grad{value.shape(), value.device()}, m_requireGrad{requireGrad}
    {
    }

    // Perform backpropagation to calculate gradients recursively.
    void backward(const TensorValue & seed)  { m_backwardFunc(this, seed); }

    Device * device() const          { return m_value.device(); }
    void device(Device * device)     { m_value.device(device); m_grad.device(device); }

    TensorValue  m_value;
    TensorValue  m_grad;
    bool  m_requireGrad;
    std::shared_ptr<TensorNode>  m_a{nullptr};
    std::shared_ptr<TensorNode>  m_b{nullptr};
    std::function<void(TensorNode * tensor, const TensorValue & seed)>  m_backwardFunc{nullptr};
};


class Tensor
{
public:
    // Constructor.
    Tensor() = default;

    // Constructor.
    explicit Tensor(const Array & data, const Shape & shape, bool requireGrad = false)
    {
        // Create a new Tensor Graph Node.
        m_data = std::make_shared<TensorNode>(TensorValue{data, shape, &defaultDevice}, requireGrad);
        m_data->m_backwardFunc = defaultBackward;
    }

    // Constructor.
    explicit Tensor(DataType value, const Shape & shape, bool requireGrad = false)
    {
        // Create a new Tensor Graph Node.
        m_data = std::make_shared<TensorNode>(TensorValue{value, shape, &defaultDevice}, requireGrad);
        m_data->m_backwardFunc = defaultBackward;
    }

    // Constructor.
    explicit Tensor(const Shape & shape, bool requireGrad = false)
    {
        // Create a new Tensor Graph Node.
        m_data = std::make_shared<TensorNode>(TensorValue{shape, &defaultDevice}, requireGrad);
        m_data->m_backwardFunc = defaultBackward;
    }

    // Perform backpropagation to calculate gradients recursively.
    void backward(DataType value=1)  { m_data->backward(TensorValue{value, m_data->m_a->m_grad.shape(), m_data->device()}); }

    // Getters and setters for the tensor's value.
    const TensorValue & value() const        { return m_data->m_value; }
    void setValue(const TensorValue & value) { m_data->m_value = value; }
    const Shape & shape() const { return m_data->m_value.shape(); }

    // Gradient-related methods.
    const TensorValue & grad() const { return m_data->m_grad; }
    void zeroGrad()                  { m_data->m_grad.fill(0); }
    bool isRequireGrad() const       { return m_data->m_requireGrad; }

    // Set operation device for the tensor.
    Tensor & to(Device & device)     { m_data->device(&device); return *this; }

    static void defaultBackward(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_requireGrad)
        {
            node->m_grad = node->m_grad + seed;
        }
    }

    static void addBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a || !node->m_b) return;
        // Calculate gradients.
        node->m_a->backward(seed);
        node->m_b->backward(seed);
    }

    static void subBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a || !node->m_b) return;
        // Calculate gradients.
        node->m_a->backward(seed);
        node->m_b->backward(-seed);
    }

    static void mulBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a || !node->m_b) return;
        // Calculate gradients.
        node->m_a->backward(node->m_b->m_value * seed);
        node->m_b->backward(node->m_a->m_value * seed);
    }

    static void divBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a || !node->m_b) return;
        // Calculate gradients.
        node->m_a->backward(seed / node->m_b->m_value);                                               // ∂f/∂a = 1 / b
        node->m_b->backward(-node->m_a->m_value * seed / (node->m_b->m_value * node->m_b->m_value));  // ∂f/∂b = -a / b^2
    }

    static void sinBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // The derivative of sin(a) with respect to 'a' is cos(a).
        // Therefore, the gradient of the input is multiplied by cos(a).
        node->m_a->backward(node->m_a->m_value.cos() * seed);   // ∂f/∂a = cos(a)
    }

    static void tanhBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // The derivative of tanh(a) with respect to 'a' is 1 - tanh^2(a).
        // Therefore, the gradient of the input is multiplied by (1 - tanh^2(a)).
        const auto & tanhValue = node->m_a->m_value.tanh();
        node->m_a->backward((DataType(1) - tanhValue * tanhValue) * seed);  // ∂f/∂a = (1 - tanh^2(a))
    }

    static void matmulBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a || !node->m_b) return;
        // Assuming m_a and m_b are the input matrices a and b, respectively,
        // and seed is ∂E/∂c, the gradient of the loss with respect to the output matrix c.
        // Compute gradients with respect to a and b

        // Corrected to use matrix multiplication for backward pass calculations
        node->m_a->backward(seed.matmul(node->m_b->m_value.transpose()));      // ∂E/∂a = ∂E/∂c * b^T
        node->m_b->backward(node->m_a->m_value.transpose().matmul(seed));      // ∂E/∂b = a^T * ∂E/∂c
    }

    static void meanBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // The gradient of the mean operation is distributed evenly across all elements. grad = 1/N
        node->m_a->backward(seed / DataType(node->m_a->m_value.data().size()));
    }

    // Overload the + operator
    Tensor operator+(const Tensor & rhsTensor) const
    {
        Tensor result({shape()}, m_data->m_requireGrad || rhsTensor.m_data->m_requireGrad);
        result.m_data->m_value = m_data->m_value + rhsTensor.m_data->m_value;
        result.m_data->m_a = m_data;
        result.m_data->m_b = rhsTensor.m_data;
        result.m_data->m_backwardFunc = addBackwardFunc;
        return result;
    }

    // Overload the - operator
    Tensor operator-(const Tensor & rhsTensor) const
    {
        Tensor result({shape()}, m_data->m_requireGrad || rhsTensor.m_data->m_requireGrad);
        result.m_data->m_value = m_data->m_value - rhsTensor.m_data->m_value;
        result.m_data->m_a = m_data;
        result.m_data->m_b = rhsTensor.m_data;
        result.m_data->m_backwardFunc = subBackwardFunc;
        return result;
    }

    // Overload the * operator
    Tensor operator*(const Tensor & rhsTensor) const
    {
        Tensor result({shape()}, m_data->m_requireGrad || rhsTensor.m_data->m_requireGrad);
        result.m_data->m_value = m_data->m_value * rhsTensor.m_data->m_value;
        result.m_data->m_a = m_data;
        result.m_data->m_b = rhsTensor.m_data;
        result.m_data->m_backwardFunc = mulBackwardFunc;
        return result;
    }

    // Overload the / operator
    Tensor operator/(const Tensor & rhsTensor) const
    {
        Tensor result({shape()}, m_data->m_requireGrad || rhsTensor.m_data->m_requireGrad);
        result.m_data->m_value = m_data->m_value / rhsTensor.m_data->m_value;
        result.m_data->m_a = m_data;
        result.m_data->m_b = rhsTensor.m_data;
        result.m_data->m_backwardFunc = divBackwardFunc;
        return result;
    }

    static Tensor sin(const Tensor & rhsTensor)
    {
        Tensor result({rhsTensor.shape()}, rhsTensor.m_data->m_requireGrad);
        result.m_data->m_value = rhsTensor.m_data->m_value.sin();
        result.m_data->m_a = rhsTensor.m_data;
        result.m_data->m_b = nullptr;
        result.m_data->m_backwardFunc = sinBackwardFunc;
        return result;
    };

    static Tensor tanh(const Tensor & rhsTensor)
    {
        Tensor result({rhsTensor.shape()}, rhsTensor.m_data->m_requireGrad);
        result.m_data->m_value = rhsTensor.m_data->m_value.tanh();
        result.m_data->m_a = rhsTensor.m_data;
        result.m_data->m_b = nullptr;
        result.m_data->m_backwardFunc = tanhBackwardFunc;
        return result;
    };

    static Tensor matmul(const Tensor & a, const Tensor & b)
    {
        Tensor result({a.shape()[0], b.shape()[1]}, a.m_data->m_requireGrad || b.m_data->m_requireGrad);
        result.m_data->m_value = a.m_data->m_value.matmul(b.m_data->m_value);
        result.m_data->m_a = a.m_data;
        result.m_data->m_b = b.m_data;
        result.m_data->m_backwardFunc = matmulBackwardFunc;
        return result;
    }

    Tensor mean() const
    {
        Tensor result({1, 1}, m_data->m_requireGrad);     // Scalar tensor for the mean result.
        result.m_data->m_value = TensorValue(m_data->m_value.mean(), {1, 1}, m_data->device());
        result.m_data->m_a = m_data;
        result.m_data->m_b = nullptr;
        result.m_data->m_backwardFunc = meanBackwardFunc;
        return result;
    }

protected:
    std::shared_ptr<TensorNode>  m_data{nullptr};
};


inline Tensor tensor(const Array & data, const Shape & shape, bool requireGrad = false)
{
    return Tensor{data, shape, requireGrad};
}

inline Tensor tensor(const Array & data, bool requireGrad = false)
{
    return Tensor{data, {1, data.size()}, requireGrad};
}

inline Tensor randn(const Shape & shape, bool requireGrad = false)
{
    static std::random_device randomDevice;
    static std::mt19937 randGen(randomDevice());
    std::uniform_real_distribution<DataType> distr(-1, 1);

    size_t totalSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    Array rndData(totalSize);

    // Fill rndData with random numbers
    std::generate(rndData.begin(), rndData.end(), [&distr]() -> DataType { return distr(randGen); });

    return Tensor{rndData, shape, requireGrad};
}

}   // aix namespace


// Optimizers Namespace


namespace aix::optim
{

class Optimizer
{
public:
    // Constructor
    explicit Optimizer(const std::vector<Tensor> & parameters) : m_parameters(parameters) { }

    // Destructor
    virtual ~Optimizer() = default;

    virtual void step() = 0;

    virtual void zeroGrad()
    {
        for (auto & param : m_parameters)
        {
            param.zeroGrad();
        }
    }

protected:
    std::vector<Tensor> m_parameters;
};


class SGDOptimizer : public Optimizer
{
public:
    explicit SGDOptimizer(const std::vector<Tensor> & parameters, DataType lr = 0.01f)
        : Optimizer(parameters), m_lr(lr) { }

    void step() final
    {
        for (auto & param : m_parameters)
        {
            if (param.isRequireGrad())
            {
                param.setValue(param.value() - param.grad() * m_lr);   // w' = w - lr * w_gradient.
            }
        }
    }

private:
    DataType m_lr;     // Learning rate
};


class AdamOptimizer : public Optimizer
{
public:
    explicit AdamOptimizer(const std::vector<Tensor> & parameters, DataType lr = 0.001f, DataType beta1 = 0.9f,
                           DataType beta2 = 0.999f, DataType epsilon = 1e-8f)
            : Optimizer(parameters), m_lr(lr), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon)
    {
        for (const auto & param : m_parameters)
        {
            m_m.emplace_back(0, param.shape(), param.value().device());
            m_v.emplace_back(0, param.shape(), param.value().device());
        }
    }

    void step() final
    {
        ++m_timestep;
        for (size_t i = 0; i < m_parameters.size(); ++i)
        {
            if (m_parameters[i].isRequireGrad())
            {
                // Update biased first moment estimate.
                m_m[i] = m_beta1 * m_m[i] + DataType(1.0 - m_beta1) * m_parameters[i].grad();

                // Update biased second raw moment estimate.
                m_v[i] = m_beta2 * m_v[i] + DataType(1.0 - m_beta2) * m_parameters[i].grad() * m_parameters[i].grad();

                // Compute bias-corrected first moment estimate.
                TensorValue mHat = m_m[i] / DataType(1.0 - std::pow(m_beta1, m_timestep));

                // Compute bias-corrected second raw moment estimate.
                TensorValue vHat = m_v[i] / DataType(1.0 - std::pow(m_beta2, m_timestep));

                // Update parameter.
                m_parameters[i].setValue(m_parameters[i].value() -  m_lr * mHat / (vHat.sqrt() + m_epsilon));
            }
        }
    }

private:
    DataType m_lr;             // Learning rate.
    DataType m_beta1;          // Exponential decay rate for the first moment estimates.
    DataType m_beta2;          // Exponential decay rate for the second moment estimates.
    DataType m_epsilon;        // Small constant for numerical stability.
    size_t m_timestep{0};      // Time step.
    std::vector<TensorValue>    m_m;    // First moment vector.
    std::vector<TensorValue>    m_v;    // Second moment vector.
};

}   // optim namespace


// Neural Network Namespace


namespace aix::nn
{

class Module
{
public:
    virtual ~Module() = default;

    virtual Tensor forward(Tensor x) const = 0;

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

    // Returns the total number of elements (learnable parameters) in each Tensor.
    size_t learnableParameters() const
    {
        size_t totalParams = 0;
        for (const auto & param: m_parameters)
        {
            if (param.isRequireGrad())
            {
                totalParams += param.value().data().size();
            }
        }
        return totalParams;
    }

    void to(Device & device)
    {
        for (auto & param : parameters())
        {
            param.to(device);
        }
    }

private:
    std::vector<Tensor> m_parameters;
};


class Sequential : public Module
{
public:
    Sequential() = default;

    // Override the forward function.
    Tensor forward(Tensor x) const override
    {
        for (const auto & module : m_modules)
        {
            x = module->forward(x);
        }
        return x;
    }

    // Function to add modules dynamically if needed.
    void add(Module* module)
    {
        registerModule(*module);
        m_modules.emplace_back(module);     // Use std::unique_ptr to take ownership of the module pointer.
    }

protected:
    // Use std::unique_ptr for polymorphic containment.
    std::vector<std::unique_ptr<Module>>  m_modules;
};


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

}   // nn namespace
