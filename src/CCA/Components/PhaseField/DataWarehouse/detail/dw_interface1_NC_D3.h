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
 * @file CCA/Components/PhaseField/DataWarehouse/detail/dw_interface1_NC_D3.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_detail_dw_interface1_NC_D3_h
#define Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_detail_dw_interface1_NC_D3_h

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Interface for data-warehouse (3D NC implementations)
 *
 * groups together various methods to get info about patches and levels which
 * depend on the different types of variable representation and problem
 * dimensions allowing to choose the relevant implementation at compile time
 *
 * @implements dw_interface1 < VAR, DIM >
 */
template<>
class dw_interface1<NC, D3>
{
protected: // CONSTRUCTOR/DESTRUCTOR

    /// prevent coonstruction
    dw_interface1() = delete;

    /// Prevent copy (and move) constructor
    dw_interface1 ( const dw_interface1 & ) = delete;

    /// Prevent copy (and move) assignment
    dw_interface1 & operator= ( const dw_interface1 & ) = delete;

public:

    /**
     * @brief get index from position
     *
     * @param p grid patch
     * @param pt point coordinates
     * @param[out] id grid index
     * @return if point has been found within patch
     */
    static inline bool find_point ( const Patch * p, const Point & pt, IntVector & id )
    {
        bool res = p->findCell ( pt, id );
        Vector pos = 2. * ( p->getNodePosition ( id ) - pt );
        if ( pos[0] < -p->getLevel()->dCell() [0] ) id[0] += 1;
        if ( pos[1] < -p->getLevel()->dCell() [1] ) id[1] += 1;
        if ( pos[2] < -p->getLevel()->dCell() [2] ) id[2] += 1;
        ASSERT ( Vector ( p->getNodePosition ( id ) - pt ).length() < 1e-10 );
        return res;
    }

}; // class dw_interface1

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_detail_dw_interface1_h
