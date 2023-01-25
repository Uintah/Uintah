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
 * @file CCA/Components/PhaseField/DataTypes/BCInfo.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_DataTypes_BCInfo_h
#define Packages_Uintah_CCA_Components_PhaseField_DataTypes_BCInfo_h

#include <CCA/Components/PhaseField/Util/Definitions.h>
#include <CCA/Components/PhaseField/DataTypes/ScalarField.h>
#include <CCA/Components/PhaseField/DataTypes/VectorField.h>

namespace Uintah
{
namespace PhaseField
{

/**
 * @brief Boundary Condition Information
 *
 * Stores the type of boundary conditions, o fine coarse interpolation and the
 * value to impose on a boundary face
 *
 * @tparam Field type of field of the variable on which the condition is applied
 */
template<typename Field> struct BCInfo;

/**
 * @brief Boundary Condition Information (ScalarField implementation)
 *
 * Stores the type of boundary conditions, o fine coarse interpolation and the
 * value to impose on a boundary face
 *
 * @tparam T type of the field value at each point
 */
template<typename T>
struct BCInfo< ScalarField<T> >
{
    /// value to impose on boundary
    typename std::remove_const<T>::type value;

    /// Type of boundary conditions
    BC bc;

    /// type of fine/coarse interface conditions
    FC c2f;
};

/**
 * @brief Boundary Condition Information (VectorField implementation)
 *
 * Stores the type of boundary conditions, o fine coarse interpolation and the
 * value to impose on a boundary face
 *
 * @tparam T type of each component of the field at each point
 * @tparam N number of components
 */
template<typename T, size_t N>
struct BCInfo< VectorField<T, N> >
{
    /// array of values to impose on boundary
    std::array < typename std::remove_const<T>::type, N > value;

    /// Type of boundary conditions
    BC bc;

    /// type of fine/coarse interface conditions
    FC c2f;

    /**
     * @brief Constructor
     *
     * From a list of BCInfo
     * @tparam I0 first BCInfo type
     * @tparam I following BCInfo types
     * @param i0 first BCInfo
     * @param i following BCInfo's
     */
    template<typename I0, typename... I>
    BCInfo ( I0 && i0, I && ... i ) :
        value { std::forward<T> ( i0.value ), std::forward<T> ( i.value )... },
          bc ( i0.bc ),
          c2f ( i0.c2f )
    {};
};

} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_DataTypes_BCInfo_h
