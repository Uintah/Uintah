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
 * @file CCA/Components/PhaseField/AMR/AMRRestrictor.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_AMR_AMRRestrictor_h
#define Packages_Uintah_CCA_Components_PhaseField_AMR_AMRRestrictor_h

#include <CCA/Components/PhaseField/AMR/detail/amr_restrictor.h>

namespace Uintah
{
namespace PhaseField
{

/**
 * @brief Wrapper of grid variables for interpolation from finer to
 *        coarser levels.
 *
 * Adds to view the possibility to compute multi-grid interpolation
 *
 * @tparam Problem type of PhaseField problem
 * @tparam I index of the variable within the Problem
 * @tparam FCI problem dimension
 */
template<typename Problem, size_t I, FCIType FCI> using AMRRestrictor = detail::amr_restrictor < typename Problem::template get_field<I>::type, Problem, index_sequence<I>, FCI, Problem::Var >;

} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_AMR_AMRRestrictor_h
