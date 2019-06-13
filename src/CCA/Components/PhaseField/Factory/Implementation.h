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
 * @file CCA/Components/PhaseField/Factory/Implementation.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_Factory_Implementation_h
#define Packages_Uintah_CCA_Components_PhaseField_Factory_Implementation_h

#include <Core/Malloc/Allocator.h>
#include <CCA/Components/PhaseField/Factory/Base.h>
#include <CCA/Components/PhaseField/Factory/Factory.h>

namespace Uintah
{
namespace PhaseField
{

/**
 * @brief Generic factory implementation class
 *
 * Make possible to create a new instance of a virtual class choosing which
 * derived class implementation to create according to a string
 *
 * @tparam I derived class type
 * @tparam B virtual base class type
 * @tparam Args types of the arguments for the derived classes constructors
 */
template<typename I, typename B, typename ... Args>
class Implementation : public Base<B>
{
protected:
    /// Determine if the class definition is registered
    static const bool m_isRegistered;

protected:
    /**
     * @brief Factory create
     *
     * Give derived classes the ability to create themselves
     *
     * @param args parameters forwarded to the constructor
     * @return new instance of derived class
     */
    static Base<B> *
    Create (
        Args ... args
    )
    {
        return scinew I ( args... );
    }

    /**
     * @brief Get Implementation name
     *
     * Get the identifier of the derived class
     *
     * @return string identifying a derived class
     */
    virtual std::string
    getName()
    override
    {
        return I::Name;
    }

    /**
    * @brief Default constructor
    *
    * Initialize derived class and register it to the factory
    */
    Implementation()
        : Base<B> ( m_isRegistered )
    {}
};

// attempt to initialize the IsRegistered variable of derived classes
// whilst registering them to the factory
template<typename I, typename B, typename ... Args>
const bool Implementation<I, B, Args ...>::m_isRegistered = Factory<B, Args ...>::Register ( I::Name, &Implementation<I, B, Args ... >::Create );

} // namespace PhaseField
} // namespace Uintah

#endif
