/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/**
 * @file CCA/Components/PhaseField/Factory/Base.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_Factory_Base_h
#define Packages_Uintah_CCA_Components_PhaseField_Factory_Base_h

#include <string>

namespace Uintah
{
namespace PhaseField
{

/**
 * @brief Generic factory base class
 *
 * Make possible to create a new instance of a virtual class choosing which
 * derived class implementation to create according to a string
 *
 * @tparam B virtual base class type
 */
template<typename B>
class Base
{
private:
    /// Determine if the class definition is registered
    const bool m_isRegistered;

protected:
    /**
    * @brief Default constructor
    *
    * Virtual class is not registered
    */
    Base() :
        m_isRegistered ( false )
    { };

    /**
    * @brief Constructor for Implementation
    *
    * This constructor is called by the constructor of any Implementation of B
    *
    * @param isRegistered whether the Implementation is registered
    */
    Base ( bool isRegistered ) :
        m_isRegistered ( isRegistered )
    { };

    /**
     * @brief Default destructor
     */
    virtual ~Base() = default;

    /**
     * @brief Get registered status
     *
     * Determine if this is an instance of an Implementation registered to the
     * factory.
     *
     * @return whether the Implementation is registered
     */
    bool isRegistered() const
    {
        return m_isRegistered;
    };

    /**
     * @brief Get Implementation name
     *
     * Provide a way for derived classes to identify themselves
     *
     * @return string identifying a derived class
     */
    virtual std::string getName() = 0;
};

} // namespace PhaseField
} // namespace Uintah

#endif
