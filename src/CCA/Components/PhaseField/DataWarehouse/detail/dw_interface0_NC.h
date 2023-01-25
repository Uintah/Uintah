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
 * @file CCA/Components/PhaseField/DataWarehouse/detail/dw_interface0_NC.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_detail_dw_interface0_NC_h
#define Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_detail_dw_interface0_NC_h

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Interface for data-warehouse (nc implementations)
 *
 * groups together various methods to get info about patches and levels which
 * depend on the different types of variable representation and problem
 * dimensions allowing to choose the relevant implementation at compile time
 *
 * @implements dw_interface0 < VAR >
 */
template<>
class dw_interface0<NC>
{
protected: // CONSTRUCTOR/DESTRUCTOR

    /// prevent coonstruction
    dw_interface0() = delete;

    /// Prevent copy (and move) constructor
    dw_interface0 ( const dw_interface0 & ) = delete;

    /// Prevent copy (and move) assignment
    /// @return deleted
    dw_interface0 & operator= ( const dw_interface0 & ) = delete;

public:

    /// type of ghost for variable in the datawarehouse
    static constexpr Ghost::GhostType ghost_type = Ghost::AroundNodes;

    /**
     * @brief get position within patch
     *
     * @param p grid patch
     * @param i grid index
     * @return coordinates of the grid node
     */
    static inline Point
    get_position (
        const Patch * p,
        const IntVector & i
    )
    {
        return p->getNodePosition ( i );
    }

    /**
     * @brief get position within level
     *
     * @param l grid level
     * @param i grid index
     * @return coordinates of the grid node
     */
    static inline Point
    get_position (
        const Level * l,
        const IntVector & i
    )
    {
        return l->getNodePosition ( i );
    }

    /**
     * @brief get patch first index
     *
     * @param p grid patch
     * @return lower bound for node indices
     */
    static inline IntVector
    get_low (
        const Patch * p
    )
    {
        return p->getNodeLowIndex();
    }

    /**
     * @brief get patch past the end index
     *
     * @param p grid patch
     * @return upper bound for node indices
     */
    static inline IntVector
    get_high (
        const Patch * p
    )
    {
        return p->getNodeHighIndex();
    }

}; // class dw_interface0

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_detail_dw_interface0_NC_h
