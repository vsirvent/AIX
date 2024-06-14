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
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <numeric>
#include <random>
#include <stack>
#include <utility>


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

// Tensor Shape and Index Types
using Shape = std::vector<size_t>;
using Index = std::vector<size_t>;


class Device
{
public:
    virtual ~Device() = default;

    virtual DeviceType type() const { return DeviceType::kCPU; }

    virtual void * allocate(size_t size)
    {
        return std::malloc(size);
    }

    virtual void deallocate(void * memory)
    {
        return std::free(memory);
    }

    virtual void add(const DataType* a1, const DataType* a2, const size_t size, DataType* result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = a1[i] + a2[i];
        }
    }

    virtual void sub(const DataType* a1, const DataType* a2, const size_t size, DataType* result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = a1[i] - a2[i];
        }
    }

    virtual void mul(const DataType* a1, const DataType* a2, const size_t size, DataType* result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = a1[i] * a2[i];
        }
    }

    virtual void div(const DataType* a1, const DataType* a2, const size_t size, DataType* result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = a1[i] / a2[i];
        }
    }

    // Scalar operations

    virtual void add(const DataType* a, DataType scalar, const size_t size, DataType* result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = a[i] + scalar;
        }
    }

    virtual void sub(const DataType* a, DataType scalar, const size_t size, DataType* result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = a[i] - scalar;
        }
    }

    virtual void sub(DataType scalar, const DataType* a, const size_t size, DataType* result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = scalar - a[i];
        }
    }

    virtual void mul(const DataType* a, DataType scalar, const size_t size, DataType* result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = a[i] * scalar;
        }
    }

    virtual void div(const DataType* a, DataType scalar, const size_t size, DataType* result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = a[i] / scalar;
        }
    }

    virtual void div(DataType scalar, const DataType* a, const size_t size, DataType* result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = scalar / a[i];
        }
    }

    virtual void unary(const DataType* a, const size_t size, DataType* result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = -a[i];
        }
    }

    virtual void fill(DataType scalar, const size_t size, DataType* result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = scalar;
        }
    }

    virtual void sum(const DataType* a, const size_t size, DataType & result)
    {
        DataType sum = 0;
        for (size_t i = 0; i < size; ++i)
        {
            sum += a[i];
        }
        result = sum;
    }

    virtual void mean(const DataType* a, const size_t size, DataType & result)
    {
        DataType sum = 0;
        for (size_t i = 0; i < size; ++i)
        {
            sum += a[i];
        }
        result = sum / static_cast<DataType>(size);
    }

    virtual void sqrt(const DataType* a, const size_t size, DataType* result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = std::sqrt(a[i]);
        }
    }

    virtual void sin(const DataType* a, const size_t size, DataType* result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = std::sin(a[i]);
        }
    }

    virtual void cos(const DataType* a, const size_t size, DataType* result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = std::cos(a[i]);
        }
    }

    virtual void tanh(const DataType* a, const size_t size, DataType* result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = std::tanh(a[i]);
        }
    }

    virtual void log(const DataType* a, const size_t size, DataType* result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = std::log(a[i]);
        }
    }

    virtual void exp(const DataType* a, const size_t size, DataType* result)
    {
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = std::exp(a[i]);
        }
    }

    virtual void matmul(const DataType* a1, const Shape & s1, const DataType* a2, const Shape & s2, DataType* result)
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

    virtual void transpose(const DataType* a, const Shape & shape, DataType* result)
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
    }

    virtual void copy(const DataType* src, DataType* dst, size_t size)
    {
        std::memcpy(dst, src, size * sizeof(DataType));
    }

    virtual void copy_immediate(const DataType* src, DataType* dst, size_t size)
    {
        std::memcpy(dst, src, size * sizeof(DataType));
    }

    virtual void commitAndWait()
    {
    }
};


// Default device.
static Device defaultDevice;      // TODO: default CPU device needs to move to a global context.
static std::random_device randomDevice;
static std::mt19937 randGen(randomDevice());


class TensorValue
{
public:
    // Constructor
    TensorValue(const DataType* data, size_t size, Shape shape, Device * device) : m_shape(std::move(shape)), m_device(device)
    {
        m_data = static_cast<DataType*>(device->allocate(size * sizeof(DeviceType)));
        device->copy_immediate(data, m_data, size);
        m_size = size;
        // Compute the strides for indexing multi-dimensional data.
        computeStrides();
    }

    // Constructor
    TensorValue(const std::vector<DataType> & data, Shape shape, Device * device) : m_shape(std::move(shape)), m_device(device)
    {
        m_data = static_cast<DataType*>(device->allocate(data.size() * sizeof(DeviceType)));
        device->copy_immediate(data.data(), m_data, data.size());
        m_size = data.size();
        // Compute the strides for indexing multi-dimensional data.
        computeStrides();
    }

    // Constructor
    TensorValue(DataType value, Shape shape, Device * device) : m_shape(std::move(shape)), m_device(device)
    {
        m_size = std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<>());
        // Each tensor array must use device specific memory allocator.
        m_data = static_cast<DataType *>(device->allocate(m_size * sizeof(DeviceType)));
        // initialize data.
        for (size_t i=0; i<m_size; ++i) m_data[i] = value;
        computeStrides();
    }

    // Constructor
    TensorValue(Shape shape, Device * device) : m_shape(std::move(shape)), m_device(device)
    {
        m_size = std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<>());
        // Each tensor array must use device specific memory allocator.
        m_data = static_cast<DataType *>(device->allocate(m_size * sizeof(DeviceType)));
        computeStrides();
    }

    // Constructor
    TensorValue(DataType value, Device * device) : m_shape{}, m_device(device)
    {
        // Each tensor array must use device specific memory allocator.
        m_size = 1;
        m_data = static_cast<DataType *>(device->allocate(m_size * sizeof(DeviceType)));
        for(size_t i=0; i<m_size; ++i) m_data[i] = value;
        computeStrides();
    }

    // Destructor
    ~TensorValue()
    {
        if (m_data) m_device->deallocate(m_data);
        m_data = nullptr;
        m_device = nullptr;
    }

    // Copy constructor
    TensorValue(const TensorValue& other) noexcept
    {
        m_shape = other.m_shape;
        m_size = other.m_size;
        m_strides = other.m_strides;
        m_device = other.m_device;
        m_data = static_cast<DataType*>(m_device->allocate(other.m_size * sizeof(DeviceType)));
        m_device->copy_immediate(other.m_data, m_data, other.m_size);
    }

    // Copy assignment operator
    TensorValue& operator=(const TensorValue& other) noexcept
    {
        if (this != &other)     // Protect against self-assignment
        {
            m_device->deallocate(m_data);
            m_shape = other.m_shape;
            m_size = other.m_size;
            m_strides = other.m_strides;
            m_device = other.m_device;
            m_data = static_cast<DataType*>(m_device->allocate(other.m_size * sizeof(DeviceType)));
            m_device->copy_immediate(other.m_data, m_data, other.m_size);
        }

        return *this;
    }

    // Move constructor
    TensorValue(TensorValue&& other) noexcept
    {
        m_data = other.m_data;
        m_shape = other.m_shape;
        m_size = other.m_size;
        m_strides = other.m_strides;
        m_device = other.m_device;

        other.m_size = 0;
        other.m_data = nullptr;             // Avoid double deletion
        other.m_device = nullptr;
    }

    // Move assignment operator
    TensorValue& operator=(TensorValue&& other) noexcept
    {
        if (this != &other)
        {
            m_device->deallocate(m_data);   // Free existing resource
            m_data = other.m_data;
            m_shape = other.m_shape;
            m_size = other.m_size;
            m_strides = other.m_strides;
            m_device = other.m_device;
            other.m_size = 0;
            other.m_data = nullptr;         // Avoid double deletion
            other.m_device = nullptr;
        }

        return *this;
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
    const DataType* data() const   { return m_data; }
    DataType* data()               { return m_data; }

    // Get the size of the data
    size_t size() const             { return m_size; }

    // Get the device
    Device * device() const        { return m_device; }

    // Set the device
    void device(Device * device)
    {
        if (m_device == device) return;
        // Move data to the new device. Create a new data with new device and copy the data. Deallocate the old data.
        // Create a new array from the new device.
        auto newData = static_cast<DataType*>(device->allocate(m_size * sizeof(DeviceType)));
        // Copy old data to the new array.
        device->copy_immediate(m_data, newData, m_size);
        // Delete old data from old device.
        m_device->deallocate(m_data);
        // Set new data and the new device.
        m_data = newData;
        m_device = device;
    }

    DataType item() const
    {
        if (!m_shape.empty())    // Scalar value must have no dimension.
        {
            throw std::invalid_argument("Tensor is not a scalar.");
        }
        return m_data[0];
    }

    // Operators

    // Overload the + operator
    TensorValue operator+(const TensorValue & other) const
    {
        // Check if the shapes of the two tensors are the same.
        validateShapes(m_shape, other.m_shape);

        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device);
        m_device->add(m_data, other.m_data, m_size, result.m_data);
        return result;
    }

    // Overload the - operator
    TensorValue operator-(const TensorValue & other) const
    {
        // Check if the shapes of the two tensors are the same.
        validateShapes(m_shape, other.m_shape);

        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device);
        m_device->sub(m_data, other.m_data, m_size, result.m_data);
        return result;
    }

    // Overload the * operator
    TensorValue operator*(const TensorValue & other) const
    {
        // Check if the shapes of the two tensors are the same.
        validateShapes(m_shape, other.m_shape);

        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device);
        m_device->mul(m_data, other.m_data, m_size, result.m_data);
        return result;
    }

    // Overload the / operator
    TensorValue operator/(const TensorValue & other) const
    {
        // Check if the shapes of the two tensors are the same.
        validateShapes(m_shape, other.m_shape);

        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device);
        m_device->div(m_data, other.m_data, m_size, result.m_data);
        return result;
    }

    // Overload the += operator - In-place operation.
    TensorValue & operator+=(const TensorValue & other)
    {
        // Check if the shapes of the two tensors are the same.
        validateShapes(m_shape, other.m_shape);

        // Perform element-wise.
        m_device->add(m_data, other.m_data, m_size, m_data);
        return *this;
    }

    // Overload the -= operator - In-place operation.
    TensorValue & operator-=(const TensorValue & other)
    {
        // Check if the shapes of the two tensors are the same.
        validateShapes(m_shape, other.m_shape);

        // Perform element-wise.
        m_device->sub(m_data, other.m_data, m_size, m_data);
        return *this;
    }

    // Overload the *= operator - In-place operation.
    TensorValue & operator*=(const TensorValue & other)
    {
        // Check if the shapes of the two tensors are the same.
        validateShapes(m_shape, other.m_shape);

        // Perform element-wise.
        m_device->mul(m_data, other.m_data, m_size, m_data);
        return *this;
    }

    // Overload the /= operator - In-place operation.
    TensorValue & operator/=(const TensorValue & other)
    {
        // Check if the shapes of the two tensors are the same.
        validateShapes(m_shape, other.m_shape);

        // Perform element-wise.
        m_device->div(m_data, other.m_data, m_size, m_data);
        return *this;
    }

    // Overload the unary - operator
    TensorValue operator-() const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device);
        m_device->unary(m_data, m_size, result.m_data);
        return result;
    }

    TensorValue operator+(DataType scalar) const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device);
        m_device->add(m_data, scalar, m_size, result.m_data);
        return result;
    }

    TensorValue operator-(DataType scalar) const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device);
        m_device->sub(m_data, scalar, m_size, result.m_data);
        return result;
    }

    TensorValue& operator+=(DataType scalar)
    {
        // Perform element-wise.
        m_device->add(m_data, scalar, m_size, m_data);
        return *this;
    }

    TensorValue& operator-=(DataType scalar)
    {
        // Perform element-wise.
        m_device->sub(m_data, scalar, m_size, m_data);
        return *this;
    }

    TensorValue& operator*=(DataType scalar)
    {
        // Perform element-wise.
        m_device->mul(m_data, scalar, m_size, m_data);
        return *this;
    }

    TensorValue& operator/=(DataType scalar)
    {
        // Perform element-wise.
        m_device->div(m_data, scalar, m_size, m_data);
        return *this;
    }

    TensorValue operator*(DataType scalar) const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device);
        m_device->mul(m_data, scalar, m_size, result.m_data);
        return result;
    }

    TensorValue operator/(DataType scalar) const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device);
        m_device->div(m_data, scalar, m_size, result.m_data);
        return result;
    }

    friend TensorValue operator*(DataType scalar, const TensorValue & tensor)
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(tensor.m_shape, tensor.m_device);
        tensor.m_device->mul(tensor.m_data, scalar, tensor.m_size, result.m_data);
        return result;
    }

    friend TensorValue operator/(DataType scalar, const TensorValue & tensor)
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(tensor.m_shape, tensor.m_device);
        tensor.m_device->div(scalar, tensor.m_data, tensor.m_size, result.m_data);
        return result;
    }

    friend TensorValue operator+(DataType scalar, const TensorValue & tensor)
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(tensor.m_shape, tensor.m_device);
        tensor.m_device->add(tensor.m_data, scalar, tensor.m_size, result.m_data);
        return result;
    }

    friend TensorValue operator-(DataType scalar, const TensorValue & tensor)
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(tensor.m_shape, tensor.m_device);
        tensor.m_device->sub(scalar, tensor.m_data, tensor.m_size, result.m_data);
        return result;
    }

    void fill(DataType value) const
    {
        m_device->fill(value, m_size, m_data);
    }

    DataType sum() const
    {
        if (m_size == 0) return 0;
        DataType result;
        m_device->sum(m_data, m_size, result);
        return result;
    }

    DataType mean() const
    {
        if (m_size == 0) return 0;
        DataType result;
        m_device->mean(m_data, m_size, result);
        return result;
    }

    TensorValue sqrt() const
    {
        // Perform element-wise sin.
        TensorValue result(m_shape, m_device);
        m_device->sqrt(m_data, m_size, result.m_data);
        return result;
    }

    TensorValue sin() const
    {
        // Perform element-wise sin.
        TensorValue result(m_shape, m_device);
        m_device->sin(m_data, m_size, result.m_data);
        return result;
    }

    TensorValue cos() const
    {
        // Perform element-wise cos.
        TensorValue result(m_shape, m_device);
        m_device->cos(m_data, m_size, result.m_data);
        return result;
    }

    TensorValue tanh() const
    {
        // Perform element-wise tanh.
        TensorValue result(m_shape, m_device);
        m_device->tanh(m_data, m_size, result.m_data);
        return result;
    }

    TensorValue log() const
    {
        // Perform element-wise tanh.
        TensorValue result(m_shape, m_device);
        m_device->log(m_data, m_size, result.m_data);
        return result;
    }

    TensorValue exp() const
    {
        // Perform element-wise exp.
        TensorValue result(m_shape, m_device);
        m_device->exp(m_data, m_size, result.m_data);
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

    // Friend function to overload operator<<
    inline friend std::ostream& operator<<(std::ostream & os, const TensorValue & tensor);

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

    // Print Tensor data
    void print(std::ostream & os) const
    {
        // Print scalar value, a tensor with no dimension.
        if (m_shape.empty())
        {
            os << item() << "\n\n";
        }
        else if (m_shape.size() == 1)
        {
            // Print tensor that has only one dimension.
            for (size_t i = 0; i < m_shape[0]; ++i)
            {
                os << "  " << (*this)({i}) << "\n";
            }
            os << "\n";
        }
        else
        {
            // Print tensor that has at least two dimensions.
            std::stack<std::pair<Index, size_t>> stack;
            stack.push({Index(), 0});

            while (!stack.empty())
            {
                auto [indices, dim] = stack.top();
                stack.pop();

                if (dim == m_shape.size() - 2)
                {
                    bool isOverTwo = m_shape.size() > 2;
                    if (isOverTwo)
                    {
                        os << "(";
                    }

                    for (size_t i = 0; i < indices.size(); ++i)
                    {
                        os << indices[i];
                        if (i < indices.size() - 1)
                        {
                            os << ",";
                        }
                    }

                    if (isOverTwo)
                    {
                        os << ",.,.) =\n";
                    }

                    size_t rows = m_shape[dim];
                    size_t cols = m_shape[dim + 1];

                    for (size_t i = 0; i < rows; ++i)
                    {
                        for (size_t j = 0; j < cols; ++j)
                        {
                            Index subIndices = indices;
                            subIndices.push_back(i);
                            subIndices.push_back(j);
                            os << "  " << (*this)(subIndices);
                        }
                        os << '\n';
                    }

                    os << '\n';
                }
                else
                {
                    for (size_t i = m_shape[dim]; i-- > 0;) // Process in reverse order
                    {
                        Index subIndices = indices;
                        subIndices.push_back(i);
                        stack.push({subIndices, dim + 1});
                    }
                }
            }
        }

        // Print shape
        os << "[ Shape{";
        for (size_t i = 0; i < m_shape.size(); ++i)
        {
            os << m_shape[i];
            if (i < m_shape.size() - 1)
            {
                os << ",";
            }
        }
        os << "} ]\n";
    }

private:
    DataType* m_data{nullptr};  // The flat array of tensor elements.
    size_t    m_size;           // Number of DataType elements.
    Shape     m_shape;          // The shape of the tensor.
    Index     m_strides;        // The strides for indexing the tensor.
    Device *  m_device{nullptr};
};


class TensorNode
{
public:
    // Constructor
    explicit TensorNode(const TensorValue & value, bool requireGrad = false)
            :  m_value{value}, m_grad{value.shape(), value.device()}, m_requireGrad{requireGrad}
    {
    }

    // Constructor
    explicit TensorNode(const Shape & shape, Device * device, bool requireGrad = false)
           :  m_value{shape, device}, m_grad{shape, device}, m_requireGrad{requireGrad}
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
    explicit Tensor(const DataType* data, size_t size, const Shape & shape, bool requireGrad = false, Device * device = &defaultDevice)
    {
        // Create a new Tensor Graph Node.
        m_data = std::make_shared<TensorNode>(TensorValue{data, size, shape, device}, requireGrad);
        m_data->m_backwardFunc = defaultBackward;
    }

    // Constructor.
    explicit Tensor(DataType value, const Shape & shape, bool requireGrad = false, Device * device = &defaultDevice)
    {
        // Create a new Tensor Graph Node.
        m_data = std::make_shared<TensorNode>(TensorValue{value, shape, device}, requireGrad);
        m_data->m_backwardFunc = defaultBackward;
    }

    // Constructor.
    explicit Tensor(const Shape & shape, bool requireGrad = false, Device * device = &defaultDevice)
    {
        // Create a new Tensor Graph Node.
        m_data = std::make_shared<TensorNode>(shape, device, requireGrad);
        m_data->m_backwardFunc = defaultBackward;
    }

    // Perform backpropagation to calculate gradients recursively.
    void backward(DataType value=1)  { m_data->backward(TensorValue{value, m_data->m_a->m_grad.shape(), device()}); }
    void backward(DataType value, const Shape & gradShape)  { m_data->backward(TensorValue{value, gradShape, device()}); }

    // Getters and setters for the tensor's value.
    inline const TensorValue & value() const        { return m_data->m_value; }
    inline TensorValue & value()                    { return m_data->m_value; }
    inline const Shape & shape() const              { return m_data->m_value.shape(); }

    // Gradient-related methods.
    inline const TensorValue & grad() const { return m_data->m_grad; }
    inline void zeroGrad()                  { m_data->m_grad.fill(0); }
    inline bool isRequireGrad() const       { return m_data->m_requireGrad; }

    // Set operation device for the tensor.
    inline Tensor & to(Device & device)     { m_data->device(&device); return *this; }
    inline Device * device() const          { return m_data->device(); }

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

    static void unaryBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // Calculate gradients.
        node->m_a->backward(-seed);
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

    static void logBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // TODO: Handle division by zero case.
        // The derivative of log(a) with respect to 'a' is 1/a.
        node->m_a->backward(seed / node->m_a->m_value);  // ∂f/∂a = 1/a
    }

    static void expBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // The derivative of exp(a) with respect to 'a' is exp(a), itself.
        node->m_a->backward(seed * node->m_a->m_value.exp());  // ∂f/∂a = exp(a)
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

    static void sumBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // For the sum operation, the gradient is simply the seed
        node->m_a->backward(seed);
    }

    static void meanBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // The gradient of the mean operation is distributed evenly across all elements. grad = 1/N
        node->m_a->backward(seed / DataType(node->m_a->m_value.size()));
    }

    // Overload the + operator
    Tensor operator+(const Tensor & rhsTensor) const
    {
        Tensor result(shape(), isRequireGrad() || rhsTensor.isRequireGrad(), device());
        auto bcRHSTensor = broadcast(rhsTensor, shape());      // Broadcast rhsTensor if it's a scalar tensor.
        result.m_data->m_value = m_data->m_value + bcRHSTensor.m_data->m_value;
        result.m_data->m_a = m_data;
        result.m_data->m_b = bcRHSTensor.m_data;
        result.m_data->m_backwardFunc = addBackwardFunc;
        return result;
    }

    // Overload the - operator
    Tensor operator-(const Tensor & rhsTensor) const
    {
        Tensor result(shape(), isRequireGrad() || rhsTensor.isRequireGrad(), device());
        auto bcRHSTensor = broadcast(rhsTensor, shape());      // Broadcast rhsTensor if it's a scalar tensor.
        result.m_data->m_value = m_data->m_value - bcRHSTensor.m_data->m_value;
        result.m_data->m_a = m_data;
        result.m_data->m_b = bcRHSTensor.m_data;
        result.m_data->m_backwardFunc = subBackwardFunc;
        return result;
    }

    // Overload the * operator
    Tensor operator*(const Tensor & rhsTensor) const
    {
        Tensor result(shape(), isRequireGrad() || rhsTensor.isRequireGrad(), device());
        auto bcRHSTensor = broadcast(rhsTensor, shape());      // Broadcast rhsTensor if it's a scalar tensor.
        result.m_data->m_value = m_data->m_value * bcRHSTensor.m_data->m_value;
        result.m_data->m_a = m_data;
        result.m_data->m_b = bcRHSTensor.m_data;
        result.m_data->m_backwardFunc = mulBackwardFunc;
        return result;
    }

    // Overload the / operator
    Tensor operator/(const Tensor & rhsTensor) const
    {
        Tensor result(shape(), isRequireGrad() || rhsTensor.isRequireGrad(), device());
        auto bcRHSTensor = broadcast(rhsTensor, shape());      // Broadcast rhsTensor if it's a scalar tensor.
        result.m_data->m_value = m_data->m_value / bcRHSTensor.m_data->m_value;
        result.m_data->m_a = m_data;
        result.m_data->m_b = bcRHSTensor.m_data;
        result.m_data->m_backwardFunc = divBackwardFunc;
        return result;
    }

    Tensor operator-() const
    {
        Tensor result(shape(), isRequireGrad(), device());
        result.m_data->m_value = -m_data->m_value;
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = unaryBackwardFunc;
        return result;
    }

    Tensor operator+(const DataType & scalar) const
    {
        Tensor tensor(scalar, shape(), isRequireGrad(), device());
        return *this + tensor;
    }

    Tensor operator-(const DataType & scalar) const
    {
        Tensor tensor(scalar, shape(), isRequireGrad(), device());
        return *this - tensor;
    }

    Tensor operator*(const DataType & scalar) const
    {
        Tensor tensor(scalar, shape(), isRequireGrad(), device());
        return *this * tensor;
    }

    Tensor operator/(const DataType & scalar) const
    {
        Tensor tensor(scalar, shape(), isRequireGrad(), device());
        return *this / tensor;
    }

    friend Tensor operator+(DataType scalar, const Tensor & rhsTensor)
    {
        Tensor tensor(scalar, rhsTensor.shape(), rhsTensor.isRequireGrad(), rhsTensor.device());
        return tensor + rhsTensor;
    }

    friend Tensor operator-(DataType scalar, const Tensor & rhsTensor)
    {
        Tensor tensor(scalar, rhsTensor.shape(), rhsTensor.isRequireGrad(), rhsTensor.device());
        return tensor - rhsTensor;
    }

    friend Tensor operator*(DataType scalar, const Tensor & rhsTensor)
    {
        Tensor tensor(scalar, rhsTensor.shape(), rhsTensor.isRequireGrad(), rhsTensor.device());
        return tensor * rhsTensor;
    }

    friend Tensor operator/(DataType scalar, const Tensor & rhsTensor)
    {
        Tensor tensor(scalar, rhsTensor.shape(), rhsTensor.isRequireGrad(), rhsTensor.device());
        return tensor / rhsTensor;
    }

    Tensor sin() const
    {
        Tensor result(shape(), isRequireGrad(), device());
        result.m_data->m_value = m_data->m_value.sin();
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = sinBackwardFunc;
        return result;
    };

    Tensor tanh() const
    {
        Tensor result(shape(), isRequireGrad(), device());
        result.m_data->m_value = m_data->m_value.tanh();
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = tanhBackwardFunc;
        return result;
    };

    Tensor log() const
    {
        Tensor result(shape(), isRequireGrad(), device());
        result.m_data->m_value = m_data->m_value.log();
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = logBackwardFunc;
        return result;
    };

    Tensor exp() const
    {
        Tensor result(shape(), isRequireGrad(), device());
        result.m_data->m_value = m_data->m_value.exp();
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = expBackwardFunc;
        return result;
    };

    Tensor sum() const
    {
        Tensor result({}, isRequireGrad(), device());     // Scalar tensor for the mean result.
        result.m_data->m_value = TensorValue(m_data->m_value.sum(), {}, device());
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = sumBackwardFunc;
        return result;
    }

    Tensor mean() const
    {
        Tensor result({}, isRequireGrad(), device());     // Scalar tensor for the mean result.
        result.m_data->m_value = TensorValue(m_data->m_value.mean(), {}, device());
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = meanBackwardFunc;
        return result;
    }

    Tensor matmul(const Tensor & mat) const
    {
        Tensor result({shape()[0], mat.shape()[1]}, isRequireGrad() || mat.isRequireGrad(), device());
        result.m_data->m_value = m_data->m_value.matmul(mat.m_data->m_value);
        result.m_data->m_a = m_data;
        result.m_data->m_b = mat.m_data;
        result.m_data->m_backwardFunc = matmulBackwardFunc;
        return result;
    }

    // Friend function to overload operator<<
    inline friend std::ostream & operator<<(std::ostream& os, const Tensor& tensor);

protected:
    // Returns a broadcasted tensor if the tensor is a scalar tensor and the other tensor's shape is not.
    // NOTE: Limited tensor broadcast support - Only scalar tensor (a tensor that has no dimension).
    static Tensor broadcast(const Tensor & tensor, const Shape & otherShape)
    {
        if (!otherShape.empty() && tensor.shape().empty())
        {
            // Return broadcasted tensor.
            return Tensor(tensor.value().item(), otherShape, tensor.isRequireGrad(), tensor.device());
        }

        return tensor;
    }

    std::shared_ptr<TensorNode>  m_data{nullptr};
};

// Some convenience method definitions.

inline Tensor tensor(DataType value, bool requireGrad = false)
{
    return Tensor{value, {}, requireGrad};
}

inline Tensor tensor(const std::vector<DataType> & data, const Shape & shape, bool requireGrad = false)
{
    return Tensor{data.data(), data.size(), shape, requireGrad};
}

inline Tensor tensor(const std::vector<DataType> & data, bool requireGrad = false)
{
    return Tensor{data.data(), data.size(), {data.size()}, requireGrad};
}

inline Tensor randn(const Shape & shape, bool requireGrad = false)
{
    std::uniform_real_distribution<DataType> distr(-1, 1);

    size_t totalSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<DataType> rndData(totalSize);

    // Fill rndData with random numbers
    std::generate(rndData.begin(), rndData.end(), [&distr]() -> DataType { return distr(randGen); });

    return Tensor{rndData.data(), rndData.size(), shape, requireGrad};
}

inline Tensor ones(const Shape & shape, bool requireGrad = false)
{
    return Tensor{1, shape, requireGrad};
}

inline Tensor zeros(const Shape & shape, bool requireGrad = false)
{
    return Tensor{0, shape, requireGrad};
}

inline Tensor onesLike(const Tensor & tensor, bool requireGrad = false)
{
    return Tensor{1, tensor.shape(), requireGrad, tensor.value().device()};
}

inline Tensor zerosLike(const Tensor & tensor, bool requireGrad = false)
{
    return Tensor{0, tensor.shape(), requireGrad, tensor.value().device()};
}

inline Tensor sin(const Tensor & A)    { return A.sin();  }
inline Tensor tanh(const Tensor & A)   { return A.tanh(); }
inline Tensor log(const Tensor & A)    { return A.log();  }
inline Tensor exp(const Tensor & A)    { return A.exp();  }
inline Tensor sum(const Tensor & A)    { return A.sum();  }
inline Tensor mean(const Tensor & A)   { return A.mean(); }
inline Tensor matmul(const Tensor & A, const Tensor & B)    { return A.matmul(B); }


// Optimizers Namespace


namespace optim
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
                param.value() -= param.grad() * m_lr;   // w' = w - lr * w_gradient.
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
                m_parameters[i].value() -= m_lr * mHat / (vHat.sqrt() + m_epsilon);
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


namespace nn
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
                totalParams += param.value().size();
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
        return matmul(x, m_w1) + m_b1;
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
        return tanh(x);
    }
};


class Sigmoid : public Module
{
public:
    // Forward
    Tensor forward(Tensor x) const override
    {
        return 1 / (1 + exp(-x));
    }
};


class GeLU : public Module
{
public:
    // Forward
    Tensor forward(Tensor x) const override
    {
        return 0.5 * x * (1.0 + tanh(std::sqrtf(2.0 / std::numbers::pi) * (x + 0.044715 * x * x * x)));
    }
};


class MSELoss
{
public:
    Tensor operator()(const Tensor & predictions, const Tensor & targets)
    {
        auto diff = predictions - targets;
        auto loss = mean(diff * diff);
        return loss;
    }
};


class BinaryCrossEntropyLoss
{
public:
    // Prediction values must be in [0..1] range.
    Tensor operator()(const Tensor & predictions, const Tensor & targets)
    {
        return -mean(targets * log(predictions) + (1 - targets) * log(1 - predictions));
    }
};

}   // nn namespace


// Auxiliary Features


inline void save(const nn::Module & module, const std::string & filename)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
    {
        throw std::ios_base::failure("Failed to open file for writing.");
    }

    const auto params = module.parameters();
    for (auto param : params)
    {
        const auto & value = param.value();
        size_t size = value.size();
        ofs.write(reinterpret_cast<const char*>(&size), sizeof(size));                       // Save parameter size
        ofs.write(reinterpret_cast<const char*>(value.data()), size * sizeof(DataType));     // Save parameter data
    }

    ofs.close();
}

inline void load(nn::Module & module, const std::string & filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
    {
        throw std::ios_base::failure("Failed to open model parameter file for reading.");
    }

    const auto params = module.parameters();    // Get model parameters
    for (auto param : params)
    {
        size_t size;
        ifs.read(reinterpret_cast<char*>(&size), sizeof(size));         // Read size of parameter
        if (size != param.value().size())
        {
            throw std::runtime_error("Invalid parameter size found when loading the model.");
        }
        ifs.read(reinterpret_cast<char*>(param.value().data()), size * sizeof(DataType));   // Read the parameter data
    }

    ifs.close();
}

// Overload the << operator to print TensorValue.
std::ostream & operator<<(std::ostream& os, const TensorValue& tensor)
{
    tensor.print(os);
    return os;
}

// Overload the << operator to print Tensor.
std::ostream & operator<<(std::ostream& os, const Tensor& tensor)
{
    os << tensor.value();
    return os;
}

inline void manualSeed(size_t seed)
{
    randGen.seed(seed);
}

}   // aix namespace
