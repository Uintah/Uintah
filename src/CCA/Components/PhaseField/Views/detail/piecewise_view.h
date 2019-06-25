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
 * @file CCA/Components/PhaseField/Views/detail/piecewise_view.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_detail_piecewise_view_h
#define Packages_Uintah_CCA_Components_PhaseField_detail_piecewise_view_h

#include <CCA/Components/PhaseField/Views/detail/view.h>
#include <CCA/Components/PhaseField/Exceptions/IntVectorOutOfBounds.h>

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Container for collecting multiple views as a unique composite view
 *
 * It is used as abase class for views that are defined across multiple sub-problems
 * (i.e. bcs views and amr coarser and finer views)
 *
 * @remark no instantiation must be performed here
 *
 * @tparam Field type of Field (should be only ScalarField)
 */
template<typename Field> class piecewise_view;

/**
 * @brief Container for collecting multiple views as a unique composite view
 * (ScalarField implementation)
 *
 * It is used as abase class for views that are defined across multiple sub-problems
 * (i.e. bcs views and amr coarser and finer views)
 *
 * @remark no instantiation must be performed here
 *
 * @tparam T type of the field value at each point
 */
template<typename T>
class piecewise_view< ScalarField<T> >
    : virtual public view< ScalarField<T> >
{
private: // TYPES

    /// Type of field
    using Field = ScalarField<T>;

    /// Non const type of the field value
    using V = typename std::remove_const<T>::type;

protected: // MEMBERS

    /// Container for each sub views
    std::list < view<Field> * > m_views;

private: // METHODS

    /**
     * @brief Get view reference
     *
     * Get the view whose support includes id
     *
     * @param id position index
     * @return required view
     * @throw IntVectorOutOfBounds if no view is found
     */
    view<Field> &
    get_view (
        const IntVector & id
    )
    {
        for ( auto * view : m_views )
            if ( view->is_defined_at ( id ) )
                return *view;
        SCI_THROW ( IntVectorOutOfBounds ( id,  __FILE__, __LINE__ ) );
    }

    /**
     * @brief Get view const reference
     *
     * Get the view whose support includes id
     *
     * @param id position index
     * @return required view
     * @throw IntVectorOutOfBounds if no view is found
     */
    const view<Field> &
    get_view (
        const IntVector & id
    ) const
    {
        for ( const auto * view : m_views )
            if ( view->is_defined_at ( id ) )
                return *view;
        SCI_THROW ( IntVectorOutOfBounds ( id,  __FILE__, __LINE__ ) );
    }

protected: // CONSTRUCTORS

    /// Default constructor
    piecewise_view () = default;

    /**
     * @brief Constructor
     *
     * Instantiate a copy of a given view
     *
     * @param copy source view for copying
     * @param deep if true inner grid variable is copied as well otherwise the
     * same grid variable is referenced
     */

    piecewise_view (
        const piecewise_view * copy,
        bool deep
    )
    {
        for ( const auto * view : copy->m_views )
            m_views.emplace_back ( view->clone ( deep ) );
    }

public: // CONSTRUCTORS/DESTRUCTOR

    /// Default destructor
    virtual ~piecewise_view () = default;

    /// Prevent copy (and move) constructor
    piecewise_view ( const piecewise_view & ) = delete;

    /// Prevent copy (and move) assignment
    /// @return deleted
    piecewise_view & operator= ( const piecewise_view & ) = delete;

public: // VIEW METHODS

    /**
     * @brief Check if the view has access to the position with index id
     *
     * @param id position index
     * @return check result
     */
    virtual bool
    is_defined_at (
        const IntVector & id
    ) const override
    {
        for ( const auto * view : m_views )
            if ( view->is_defined_at ( id ) ) return true;
        return false;
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
        return get_view ( id ) [id];
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
        return get_view ( id ) [id];
    };

}; // class view

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_detail_piecewise_view_h
