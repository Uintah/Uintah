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
 * @file CCA/Components/PhaseField/DataWarehouse/DWInterface.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_DWInterface_h
#define Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_DWInterface_h

#include <CCA/Components/PhaseField/Util/Definitions.h>
#include <CCA/Components/PhaseField/DataWarehouse/detail/dw_interface0.h>
#include <CCA/Components/PhaseField/DataWarehouse/detail/dw_interface1.h>

namespace Uintah
{
namespace PhaseField
{

/**
 * @brief Interface for data-warehouse
 *
 * groups together various methods to get info about patches and levels which
 * depend on the different types of variable representation and problem
 * dimensions allowing to choose the relevant implementation at compile time
 *
 * @tparam VAR type of variable representation
 * @tparam DIM problem dimension
 */
template < VarType VAR, DimType DIM >
struct DWInterface
{
    /**
     * @brief get position within patch
     *
     * @tparam V type of variable, thus grid element (cell or node)
     * @param p grid patch
     * @param i grid index
     * @return coordinates of the grid element
     */
    template <VarType V = VAR>
    static inline Point
    get_position (
        const Patch * p,
        const IntVector & i
    )
    {
        return detail::dw_interface0<V>::get_position ( p, i );
    }

    /**
     * @brief get position within level
     *
     * @tparam V type of variable, thus grid element (cell or node)
     * @param l grid level
     * @param i grid index
     * @return coordinates of the grid element
     */
    template <VarType V = VAR>
    static inline Point
    get_position (
        const Level * l,
        const IntVector & i
    )
    {
        return detail::dw_interface0<V>::get_position ( l, i );
    }

    /**
     * @brief get patch first index
     *
     * @tparam V type of variable, thus grid element (cell or node)
     * @param p grid patch
     * @return lower bound for grid elemment indices
     */
    template <VarType V = VAR>
    static inline IntVector
    get_low (
        const Patch * p
    )
    {
        return detail::dw_interface0<V>::get_low ( p );
    }


    /**
     * @brief get patch past the end index
     *
     * @tparam V type of variable, thus grid element (cell or node)
     * @param p grid patch
     * @return upper bound for grid elemment indices
     */
    template <VarType V = VAR>
    static inline IntVector
    get_high (
        const Patch * p
    )
    {
        return detail::dw_interface0<V>::get_high ( p );
    }

    /**
     * @brief get patch index range
     *
     * @tparam V type of variable, thus grid element (cell or node)
     * @param p grid patch
     * @return range for grid elemment indices
     */
    template <VarType V = VAR>
    static inline BlockRange
    get_range (
        const Patch * p
    )
    {
        return { detail::dw_interface0<V>::get_low ( p ), detail::dw_interface0<V>::get_high ( p ) };
    }

    /**
     * @brief get index from position
     *
     * @tparam V type of variable, thus grid element (cell or node)
     * @param p grid patch
     * @param pt point coordinates
     * @param[out] id grid index
     * @return if point has been found within patch
     */
    template <VarType V = VAR>
    static inline bool
    find_point (
        const Patch * p,
        const Point & pt,
        IntVector & id
    )
    {
        return detail::dw_interface1<V, DIM>::find_point ( p, pt, id );
    }
}; // struct DWInterface

} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_DWInterface_h
