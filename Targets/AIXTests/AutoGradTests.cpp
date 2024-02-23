//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

// Project includes
#include "Utils.hpp"
#include <aix.hpp>
// External includes
#include <doctest/doctest.h>
// System includes

using namespace aix;


struct TestModel : public aix::nn::Module
{
    TestModel(const std::vector<float> & xData,
              const std::vector<float> & yData,
              const std::vector<float> & tData,
              const std::vector<float> & uData,
              const std::vector<size_t> & shape)
    {
        m_x = aix::tensor(xData, shape, true);
        m_y = aix::tensor(yData, shape, true);
        m_t = aix::tensor(tData, shape, true);
        m_u = aix::tensor(uData, shape, true);

        registerParameter(m_x);
        registerParameter(m_y);
        registerParameter(m_t);
        registerParameter(m_u);
    }

    Tensor forward() const
    {
        auto z = m_x * (m_x + m_y) / m_t - m_y * m_y;
        auto m = m_x * z + Tensor::sin(m_u) * m_u;
        return m;
    }

    Tensor  m_x;
    Tensor  m_y;
    Tensor  m_t;
    Tensor  m_u;
};


TEST_CASE("Auto Grad - Module Test - 1x1 Tensor")
{
    auto shape = std::vector<size_t>{1, 1};

    auto tm = TestModel({2},   // x
                        {3},   // y
                        {4},   // t
                        {5},   // u
                        shape);

    auto m = tm.forward();
    // Traverse the graph (starting from the end) to calculate all expression values.
    // This approach is known as lazy evaluation, meaning that values are not calculated
    // until the 'evaluate' function is called.
    m.evaluate();

    // Traverse the graph (starting from the end) to calculate all tensor gradients.
    m.backward(1);      // ∂m/∂m = [1]  1x1 tensor.

    // Check shapes
    CHECK(tm.m_x.grad().shape()  == shape);
    CHECK(tm.m_y.grad().shape()  == shape);
    CHECK(tm.m_t.grad().shape()  == shape);
    CHECK(tm.m_u.grad().shape()  == shape);
    CHECK(tm.m_x.value().shape() == shape);
    CHECK(tm.m_y.value().shape() == shape);
    CHECK(tm.m_t.value().shape() == shape);
    CHECK(tm.m_u.value().shape() == shape);
    CHECK(m.value().shape()      == shape);

    CHECK(tm.m_x.grad().data()[0] == Approx(-3));
    CHECK(tm.m_y.grad().data()[0] == Approx(-11));
    CHECK(tm.m_t.grad().data()[0] == Approx(-1.25));
    CHECK(tm.m_u.grad().data()[0] == Approx(0.459387));
    CHECK(m.value().data()[0]     == Approx(-17.7946));
}


TEST_CASE("Auto Grad - Module Test - 1x2 Tensor")
{
    auto shape = std::vector<size_t>{1, 2};

    auto tm = TestModel({1, 2},   // x
                        {3, 4},   // y
                        {5, 6},   // t
                        {7, 8},   // u
                        shape);

    auto m = tm.forward();
    // Traverse the graph (starting from the end) to calculate all expression values.
    // This approach is known as lazy evaluation, meaning that values are not calculated
    // until the 'evaluate' function is called.
    m.evaluate();

    // Traverse the graph (starting from the end) to calculate all tensor gradients.
    m.backward({{1, 1}, shape});   // ∂m/∂m = [1, 1]  1x2 tensor

    // Check shapes
    CHECK(tm.m_x.grad().shape()  == shape);
    CHECK(tm.m_y.grad().shape()  == shape);
    CHECK(tm.m_t.grad().shape()  == shape);
    CHECK(tm.m_u.grad().shape()  == shape);
    CHECK(tm.m_x.value().shape() == shape);
    CHECK(tm.m_y.value().shape() == shape);
    CHECK(tm.m_t.value().shape() == shape);
    CHECK(tm.m_u.value().shape() == shape);
    CHECK(m.value().shape()      == shape);

    CheckVectorApproxValues(tm.m_x.grad().data(), {-7.2,    -11.3333});
    CheckVectorApproxValues(tm.m_y.grad().data(), {-5.8,    -15.3333});
    CheckVectorApproxValues(tm.m_t.grad().data(), {-0.16,   -0.666667});
    CheckVectorApproxValues(tm.m_u.grad().data(), {5.9343,  -0.174642});
    CheckVectorApproxValues(m.value().data(),     {-3.6011, -20.0851});
}


TEST_CASE("Auto Grad - Module Test - 2x3 Tensor")
{
    auto shape = std::vector<size_t>{2, 3};

    auto tm = TestModel({ 1,  2,  3,  4,  5,  6},   // x
                        { 7,  8,  9, 10, 11, 12},   // y
                        {13, 14, 15, 16, 17, 18},   // t
                        {19, 20, 21, 22, 23, 24},   // u
                        shape);

    auto m = tm.forward();
    // Traverse the graph (starting from the end) to calculate all expression values.
    // This approach is known as lazy evaluation, meaning that values are not calculated
    // until the 'evaluate' function is called.
    m.evaluate();

    // Traverse the graph (starting from the end) to calculate all tensor gradients.
    m.backward({{1, 1, 1, 1, 1, 1}, shape});   // ∂m/∂m = [1,1,1,1,1,1]  2x3 tensor

    // Check shapes
    CHECK(tm.m_x.grad().shape()  == shape);
    CHECK(tm.m_y.grad().shape()  == shape);
    CHECK(tm.m_t.grad().shape()  == shape);
    CHECK(tm.m_u.grad().shape()  == shape);
    CHECK(tm.m_x.value().shape() == shape);
    CHECK(tm.m_y.value().shape() == shape);
    CHECK(tm.m_t.value().shape() == shape);
    CHECK(tm.m_u.value().shape() == shape);
    CHECK(m.value().shape()      == shape);

    CheckVectorApproxValues(tm.m_x.grad().data(), {-47.6923, -60.8571, -75.6, -92, -110.118, -130});
    CheckVectorApproxValues(tm.m_y.grad().data(), {-13.9231, -31.7143, -53.4, -79, -108.529, -142});
    CheckVectorApproxValues(tm.m_t.grad().data(), {-0.0473373, -0.204082, -0.48, -0.875, -1.38408, -2});
    CheckVectorApproxValues(tm.m_u.grad().data(), {18.9353, 9.07459, -10.6657, -22.008, -13.1014, 9.27472});
    CheckVectorApproxValues(m.value().data(),     {-45.5369, -106.8840, -218.2302, -386.1947, -600.9337, -849.7339});
}
