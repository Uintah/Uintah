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
 * @file CCA/Components/PhaseField/DataTypes/SubProblemsP.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_DataTypes_SubProblemsP_h
#define Packages_Uintah_CCA_Components_PhaseField_DataTypes_SubProblemsP_h

#include <Core/Util/Handle.h>

namespace Uintah
{
namespace PhaseField
{

template <typename Problem> struct SubProblems;

/**
 * @brief Handle for SubProblems
 *
 * @tparam Problem type of PhaseField problem
 */
template <typename Problem> using SubProblemsP = Handle < SubProblems<Problem> >;

} // namespace PhaseField

/**
 * @brief Fix Endianess
 *
 * Ovverride for preventing compiler errors
 * @remark Should never be called
 *
 * @tparam Problem type of PhaseField problem
 * @param problem unused argument
 */
template <typename Problem>
inline void
swapbytes (
    PhaseField::SubProblemsP<Problem> & _DOXYARG ( problem )
)
{
    SCI_THROW ( InternalError ( "Swap bytes for ProblemsP is not implemented", __FILE__, __LINE__ ) );
};

} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_DataTypes_SubProblemsP_h
