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
#include "mps.hpp"
// External includes
// System includes
#include <unordered_set>


namespace aix
{

class DeviceMPS : public aix::Device
{
public:
    // Constructor
    DeviceMPS()
    {
        m_mpsDevice = mps::createMPSDevice(0);
    }

    // Destructor
    virtual ~DeviceMPS()
    {
        mps::releaseMPSDevice(m_mpsDevice);
    }

    DeviceType type() const override { return DeviceType(-1); }     // Returns -1 since the device is for example only.

    // Allocate GPU memory and return MTL Buffer contents and keeps MTL Buffer pointers in a hashmap.
    void * allocate(size_t size) override
    {
        return mps::allocate(m_mpsDevice, size);
    }

    // Deallocate GPU memory if it's allocated by current device.
    void deallocate(void * memory) override
    {
        mps::deallocate(m_mpsDevice, memory);
    }

    void add(const DataType * a1, const DataType * a2, const size_t size, DataType * result) override
    {
        mps::add_a_a(m_mpsDevice, a1, a2, size, result);
    }

    void sub(const DataType * a1, const DataType * a2, const size_t size, DataType * result) override
    {
        mps::sub_a_a(m_mpsDevice, a1, a2, size, result);
    }

    void mul(const DataType * a1, const DataType * a2, const size_t size, DataType * result) override
    {
        mps::mul_a_a(m_mpsDevice, a1, a2, size, result);
    }

    void div(const DataType * a1, const DataType * a2, const size_t size, DataType * result) override
    {
        mps::div_a_a(m_mpsDevice, a1, a2, size, result);
    }

    void add(const DataType * a, DataType scalar, const size_t size, DataType * result) override
    {
        mps::add_a_s(m_mpsDevice, a, scalar, size, result);
    }

    void sub(const DataType * a, DataType scalar, const size_t size, DataType * result) override
    {
        mps::add_a_s(m_mpsDevice, a, -scalar, size, result);
    }

    void sub(DataType scalar, const DataType * a, const size_t size, DataType * result) override
    {
        mps::sub_s_a(m_mpsDevice, scalar, a, size, result);
    }

    void mul(const DataType * a, DataType scalar, const size_t size, DataType * result) override
    {
        mps::mul_a_s(m_mpsDevice, a, scalar, size, result);
    }

    void div(const DataType * a, DataType scalar, const size_t size, DataType * result) override
    {
        mps::div_a_s(m_mpsDevice, a, scalar, size, result);
    }

    void div(DataType scalar, const DataType * a, const size_t size, DataType * result) override
    {
        mps::div_s_a(m_mpsDevice, scalar, a, size, result);
    }


    void unary(const DataType * a, const size_t size, DataType * result) override
    {
        mps::mul_a_s(m_mpsDevice, a, DataType(-1), size, result);
    }

    void fill(DataType scalar, const size_t size, DataType * result) override
    {
        mps::copy_s_a(m_mpsDevice, scalar, size, result);
    }

    void sum(const DataType * a, const size_t size, DataType* result) override
    {
        // TODO: Add GPU support for the following device methods.
        Device::sum(a, size, result);
    }

    void mean(const DataType * a, const size_t size, DataType* result) override
    {
        // TODO: Add GPU support for the following device methods.
        Device::mean(a, size, result);
    }

    void sqrt(const DataType * a, const size_t size, DataType * result) override
    {
        mps::sqrt_a(m_mpsDevice, a, size, result);
    }

    void sin(const DataType * a, const size_t size, DataType * result) override
    {
        mps::sin_a(m_mpsDevice, a, size, result);
    }

    void cos(const DataType * a, const size_t size, DataType * result) override
    {
        mps::cos_a(m_mpsDevice, a, size, result);
    }

    void tanh(const DataType * a, const size_t size, DataType * result) override
    {
        mps::tanh_a(m_mpsDevice, a, size, result);
    }

    void log(const DataType * a, const size_t size, DataType * result) override
    {
        mps::log_a(m_mpsDevice, a, size, result);
    }

    void exp(const DataType * a, const size_t size, DataType * result) override
    {
        mps::exp_a(m_mpsDevice, a, size, result);
    }

    void matmul(const DataType * a1, const Shape & s1, const DataType * a2, const Shape & s2, DataType * result) override
    {
        mps::matmul(m_mpsDevice, a1, s1[0], s1[1], a2, s2[0], s2[1], result);
    }

    void copy(const DataType * src, DataType * dst, size_t size) override
    {
        mps::copy_a_a(m_mpsDevice, src, dst, size);
    }

    void copyImmediate(const DataType * src, DataType * dst, size_t size) override
    {
        mps::copy_a_a(m_mpsDevice, src, dst, size);
    }

protected:
    void* m_mpsDevice{nullptr};
};

}   // namespace
