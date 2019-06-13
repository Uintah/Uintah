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
 * @file CCA/Components/PhaseField/DataTypes/Variable.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_DataTypes_Variable_h
#define Packages_Uintah_CCA_Components_PhaseField_DataTypes_Variable_h

#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/PerPatch.h>

#include <CCA/Components/PhaseField/Util/Definitions.h>

namespace Uintah
{
namespace PhaseField
{

/**
 * @brief Grid variable wrapper
 *
 * Templetize the different grid variable implementations and allows to switch
 * between different implementations at compile time
 * @tparam VAR type of variable representation
 */
template<VarType VAR, typename T>
struct Variable
        : public std::conditional < std::is_const<T>::value,
          typename std::conditional < VAR == CC, constCCVariable< typename std::remove_const<T>::type >,
          typename std::conditional < VAR == NC, constNCVariable< typename std::remove_const<T>::type>,
          void >::type >::type,
          typename std::conditional < VAR == CC, CCVariable<T>,
          typename std::conditional < VAR == NC, NCVariable<T>,
          void >::type >::type
          >::type
{
}; // struct Variable

/**
 * @brief Grid variable wrapper (PerPatch implementation)
 *
 * Templetize the different grid variable implementations and allows to switch
 * between different implementations at compile time
 * @implements Variable < VAR, T >
 */
template<typename T>
struct Variable<PP, T>
        : public PerPatch < Handle<T> >
{
};

} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_DataTypes_Variable_h
