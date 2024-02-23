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


TEST_CASE("Auto Grad - Module Test")
{
    struct TestModel : public aix::nn::Module
    {
        TestModel()
        {
            m_x = aix::tensor(2, true);
            m_y = aix::tensor(3, true);
            m_t = aix::tensor(4, true);
            m_u = aix::tensor(5, true);

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

    auto tm = TestModel();

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
