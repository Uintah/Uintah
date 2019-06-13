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
 * @file CCA/Components/PhaseField/Views/detail/virtual_view.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_Views_detail_virtual_view_h
#define Packages_Uintah_CCA_Components_PhaseField_Views_detail_virtual_view_h

#include <CCA/Components/PhaseField/DataTypes/SubProblemsP.h>
#include <CCA/Components/PhaseField/DataTypes/Variable.h> // must be included after handles where swapbytes override is defined
#include <CCA/Components/PhaseField/Views/detail/view.h>
#include <CCA/Components/PhaseField/DataWarehouse/DWInterface.h>

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Abstract class for accessing variables on virtual patches
 *
 * detail implementation of variable wrapping
 *
 * @remark constant view must be used
 *
 * @tparam View type of underlying view
 * @tparam Field type of field (ScalarField < T > or VectorField < T, N >)
 */
template<typename View, typename Field> class virtual_view;

/**
 * @brief Abstract class for accessing variables on virtual patches
 * (ScalarField implementation)
 *
 * detail implementation of variable wrapping
 *
 * @remark constant view must be used
 *
 * @tparam View type of underlying view
 * @tparam T type of the field value at each point
 */
template<typename View, typename T>
class virtual_view < View, ScalarField<T> > :
    public View
{
private: // TYPES

    /// Type of field
    using Field = ScalarField<T>;

    /// Non const type of the field value
    using V = typename std::remove_const<T>::type;

private: // FRIENDS

    friend View;

private: // MEMBERS

    /// Translation vector
    IntVector m_offset;

private: // COPY CONSTRUCTOR

    /**
     * @brief Constructor
     *
     * Instantiate a copy of a given view
     *
     * @remark only View can call this constructor
     *
     * @param copy source view for copying
     * @param deep if true inner grid variable is copied as well otherwise the
     * same grid variable is referenced
     * @param offset vector specifying the translation of the support
     */
    virtual_view (
        const View * copy,
        bool deep,
        const IntVector & offset
    ) : View ( copy, deep ),
        m_offset ( offset )
    {
        ASSERTMSG ( std::is_const<T>::value, "Only constant views are allowed on virtual patches" );
    }

public: // DESTRUCTOR

    virtual ~virtual_view() = default;

    /// Prevent copy (and move) constructor
    virtual_view ( const virtual_view & ) = delete;

    /// Prevent copy (and move) assignment
    virtual_view & operator= ( const virtual_view & ) = delete;

public: // VIEW METHODS

    /**
     * @brief Retrieve values from the DataWarehouse for a given patch
     *
     * @param dw DataWarehouse from which data is retrieved
     * @param patch grid patch to retrieve data for
     * @param use_ghosts if ghosts value are to be retrieved
     */
    virtual void
    set (
        DataWarehouse * dw,
        const Patch * patch,
        bool use_ghosts
    ) override
    {
        View::set ( dw, patch, use_ghosts );
        m_offset = patch->getVirtualOffset();
    };

    /**
     * @brief Retrieve values from the DataWarehouse for a given region
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
        bool use_ghosts
    ) override
    {
        View::set ( dw, level, low - m_offset, high - m_offset, use_ghosts );
    };

    /**
     * @brief Get the region for which the view has access to the DataWarehouse
     *
     * @return support of the view
     */
    virtual Support
    get_support()
    const override
    {
        Support support = View::get_support();
        for ( auto & region : support )
        {
            region.low() += m_offset;
            region.high() += m_offset;
        }
        return support;
    };

    /**
     * @brief Check if the view has access to the position with index id
     *
     * @param id position index
     * @return check result
     */
    virtual inline bool
    is_defined_at (
        const IntVector & id
    ) const override
    {
        return View::is_defined_at ( id - m_offset );
    };

    /**
     * @brief Get/Modify value at position with index id (modifications are allowed if T is non const)
     *
     * @param id position index
     * @return reference to field value at id
     */
    virtual inline T &
    operator[] (
        const IntVector & id
    ) override
    {
        return View::operator[] ( id - m_offset );
    };

    /**
     * @brief Get value at position
     *
     * @param id position index
     * @return field value at id
     */
    virtual inline V
    operator[] (
        const IntVector & id
    ) const override
    {
        return View::operator[] ( id - m_offset );
    };

}; // dw_virtual_view

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#endif //Packages_Uintah_CCA_Components_PhaseField_Views_detail_virtual_view_h
