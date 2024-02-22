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


TEST_CASE("Simple auto grad test")
{
    auto x = Tensor(2, true);
    auto y = Tensor(3, true);
    auto t = Tensor(4, true);
    auto u = Tensor(5, true);

    // z = x * (x + y) / t - y * y
    // m = x * z + Sin(u) * u
    // Decompose all operations into atomic calculations to construct the graph.
    // Assume that all operations are scalar and applied element-wise.
    // Execute calculations as though they were Assembly instructions. :)
    Add  p1(x, y);    // p1 = x + y
    Mul  m1(x, p1);   // m1 = x * p1
    Div  d1(m1, t);   // d1 = m1 / t
    Mul  m2(y, y);    // m2 = y * y
    Sub  z(d1, m2);   // z = d1 - m2
    Mul  r(x, z);     // r = x * z
    Sin  s1(u);       // s1 = Sin(u)
    Mul  s2(s1, u);   // s2 = s1 * u
    Add  m(r, s2);    // m = r + s2
    // TODO: Implement MatMull when Tensor can handle multi-dimensions.

    // Traverse the graph (starting from the end) to calculate all expression values.
    // This approach is known as lazy evaluation, meaning that values are not calculated
    // until the 'evaluate' function is called.
    m.evaluate();

    // Traverse the graph (starting from the end) to calculate all tensor gradients.
    m.backward(1);      // ∂m/∂m = 1.

    CHECK(x.grad()  == Approx(-3));
    CHECK(y.grad()  == Approx(-11));
    CHECK(t.grad()  == Approx(-1.25));
    CHECK(u.grad()  == Approx(0.459387));
    CHECK(z.value() == Approx(-6.5));
    CHECK(m.value() == Approx(-17.7946));
    // Note: Results are consistent with those from PyTorch.
}

TEST_CASE("Auto Grad - Module Test")
{
    struct MyModel : public aix::Module
    {
        MyModel()
        {
            m_x = Tensor(2, true);
            m_y = Tensor(3, true);
            m_t = Tensor(4, true);
            m_u = Tensor(5, true);

            registerParameter(m_x);
            registerParameter(m_y);
            registerParameter(m_t);
            registerParameter(m_u);
        }

        auto forward()
        {
            // z = x * (x + y) / t - y * y
            // m = x * z + Sin(u) * u
            // Decompose all operations into atomic calculations to construct the graph.
            // Assume that all operations are scalar and applied element-wise.
            // Execute calculations as though they were Assembly instructions. :)
            auto p1 = recycle(new Add(m_x, m_y));      // p1 = x + y
            auto m1 = recycle(new Mul(m_x, *p1));      // m1 = x * p1
            auto d1 = recycle(new Div(*m1, m_t));      // d1 = m1 / t
            auto m2 = recycle(new Mul(m_y, m_y));      // m2 = y * y
            auto z  = recycle(new Sub(*d1, *m2));      // z = d1 - m2
            auto r  = recycle(new Mul(m_x, *z));       // r = x * z
            auto s1 = recycle(new Sin(m_u));           // s1 = Sin(u)
            auto s2 = recycle(new Mul(*s1, m_u));      // s2 = s1 * u
            auto m  = recycle(new Add(*r, *s2));       // m = r + s2

            return *m;
        }

        Tensor  m_x;
        Tensor  m_y;
        Tensor  m_t;
        Tensor  m_u;
    };

    auto tm = MyModel();

    auto m = tm.forward();

    // Traverse the graph (starting from the end) to calculate all expression values.
    // This approach is known as lazy evaluation, meaning that values are not calculated
    // until the 'evaluate' function is called.
    m.evaluate();

    // Traverse the graph (starting from the end) to calculate all tensor gradients.
    m.backward(1);      // ∂m/∂m = 1.

    CHECK(tm.m_x.grad() == Approx(-3));
    CHECK(tm.m_y.grad() == Approx(-11));
    CHECK(tm.m_t.grad() == Approx(-1.25));
    CHECK(tm.m_u.grad() == Approx(0.459387));
    CHECK(m.value() == Approx(-17.7946));
}
