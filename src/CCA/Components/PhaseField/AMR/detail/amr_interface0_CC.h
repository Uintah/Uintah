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
 * @file CCA/Components/PhaseField/AMR/detail/amr_interface0_CC.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_AMR_detail_amr_interface0_CC_h
#define Packages_Uintah_CCA_Components_PhaseField_AMR_detail_amr_interface0_CC_h

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Interface for amr
 * (CC implementation)
 *
 * groups together various methods to get info about amr patches and levels which
 * depend on the different types of variable representation allowing to choose
 * the relevant implementation at compile time
 *
 * @implements amr_interface0 < VAR >
 */
template<>
class amr_interface0<CC>
{
protected: // CONSTRUCTORS/DESTRUCTOR

    /// prevent coonstruction
    amr_interface0() = delete;

    /// Prevent copy (and move) constructor
    amr_interface0 ( const amr_interface0 & ) = delete;

    /// Prevent copy (and move) assignment
    /// @return deleted
    amr_interface0 & operator= ( const amr_interface0 & ) = delete;

public: // STATIC METHODS

    /**
     * @brief Get coarser grid index
     *
     * @param l fine grid level
     * @param i fine grid index
     * @return nearest coarser grid index to the given position
     */
    static inline IntVector get_coarser ( const Level * l, const IntVector & i )
    {
        return l->mapCellToCoarser ( i );
    }

    /**
     * @brief Get finer grid index
     *
     * @param l coarse grid level
     * @param i coarse grid index
     * @return nearest finer grid index to the given position
     */
    static inline IntVector get_finer ( const Level * l, const IntVector & i )
    {
        return l->mapCellToFiner ( i );
    }

}; // class amr_interface0

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_AMR_detail_amr_interface0_CC_h
