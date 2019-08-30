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
 * @file CCA/Components/PhaseField/AMR/AMRInterface.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_AMR_AMRInterface_h
#define Packages_Uintah_CCA_Components_PhaseField_AMR_AMRInterface_h

#include <CCA/Components/PhaseField/AMR/detail/amr_interface0.h>
#include <CCA/Components/PhaseField/Util/Definitions.h>

namespace Uintah
{
namespace PhaseField
{

/**
 * @brief Interface for AMR
 *
 * groups together various methods to get info about amr patches and levels which
 * depend on the different types of variable representation and problem
 * dimensions allowing to choose the relevant implementation at compile time
 *
 * @tparam VAR type of variable representation
 * @tparam DIM problem dimension
 */
template <VarType VAR, DimType DIM>
struct AMRInterface
{
    /**
     * @brief get index of the grid element on the coarser level closest to the
     * given index
     *
     * @tparam V type of variable, thus grid element (cell or node)
     * @param l fine level
     * @param i fine level index
     * @return coarse level index
     */
    template <VarType V = VAR>
    static inline IntVector
    get_coarser (
        const Level * l,
        const IntVector & i
    )
    {
        return detail::amr_interface0<V>::get_coarser ( l, i );
    }

    /**
     * @brief get index of the grid element on the finer level closest to the
     * given index
     *
     * @tparam V type of variable, thus grid element (cell or node)
     * @param l coarse level
     * @param i coarse level index
     * @return fine level index
     */
    template <VarType V = VAR>
    static inline IntVector
    get_finer (
        const Level * l,
        const IntVector & i
    )
    {
        return detail::amr_interface0<V>::get_finer ( l, i );
    }

}; // struct AMRInterface

} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_AMR_AMRInterface_h
