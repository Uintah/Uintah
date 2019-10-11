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
 * @file CCA/Components/PhaseField/Views/detail/view_array.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_Views_detail_view_array_h
#define Packages_Uintah_CCA_Components_PhaseField_Views_detail_view_array_h

#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Components/PhaseField/Views/detail/view_array_view.h>

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Container for collecting multiple generic views as an array
 *
 * @remark no instantiation must be performed here
 *
 * @tparam View type of Views in the array
 * @tparam Field type of Fields in the array
 * @tparam N size of the array
 */
template<typename View, typename Field, size_t N>
class view_array
    : virtual protected view_array_view<Field, N>
{
public: // VIEW ARRAY METHODS

    /**
     * @brief Retrieve value from the DataWarehouse for a given patch
     *
     * @param dw DataWarehouse from which data is retrieved
     * @param patch grid patch to retrieve data for
     * @param use_ghosts if ghosts value are to be retrieved
     */
    inline void
    set (
        DataWarehouse * dw,
        const Patch * patch,
        bool use_ghosts = View::use_ghosts_dflt
    )
    {
        for ( auto * view : this->m_view_ptr )
            dynamic_cast<View *> ( view )->set ( dw, patch, use_ghosts );
    };

    /**
     * @brief Retrieve value from the DataWarehouse for a given region
     *
     * @param dw DataWarehouse from which data is retrieved
     * @param level grid level from which retrieve data
     * @param low lower bound of the region to retrieve
     * @param high higher bound of the region to retrieve
     * @param use_ghosts if ghosts value are to be retrieved
     */
    virtual void
    set (
        DataWarehouse * dw,
        const Level * level,
        const IntVector & low,
        const IntVector & high,
        bool use_ghosts = View::use_ghosts_dflt
    )
    {
        for ( auto * view : this->m_view_ptr )
            dynamic_cast<View *> ( view )->set ( dw, level, low, high, use_ghosts );
    };

    /**
     * @brief Get reference to field component
     *
     * @param pos component index
     * @return reference to the view of the component
     */
    View &
    operator [] (
        size_t pos
    )
    {
        return dynamic_cast<View &> ( *this->m_view_ptr[pos] );
    }

    /**
     * @brief Get reference to field component
     *
     * @param pos component index
     * @return reference to the view of the component
     */
    const View &
    operator [] (
        size_t pos
    ) const
    {
        return dynamic_cast<const View &> ( *this->m_view_ptr[pos] );
    }

}; // class view_array

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_detail_view_array_h
