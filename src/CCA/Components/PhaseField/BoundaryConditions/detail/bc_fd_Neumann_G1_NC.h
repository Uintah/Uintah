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
 * @file CCA/Components/PhaseField/BoundaryConditions/detail/bc_fd_Neumann_G1_NC.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_detail_bc_fd_Neumann_G1_NC_h
#define Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_detail_bc_fd_Neumann_G1_NC_h

#include <CCA/Components/PhaseField/Util/Definitions.h>

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Finite-differences scheme at Neumann boundaries interface
 * (ScalarField, node-centered, 1-ghost stencils implementation)
 *
 * approximation of ghost value across Neumann boundaries
 *
 * @tparam T type of the field value at each point
 * @tparam STN finite-difference stencil
 * @tparam F patch face on which bc is to be applied
 */
template<typename T, StnType STN, Patch::FaceType F >
class bc_fd < ScalarField<T>, STN, NC, F, BC::Neumann, 1 >
    : virtual public basic_fd_view < ScalarField<T>, STN >
{
private: // STATIC MEMBERS

    /// Number of ghosts required
    static constexpr int GN = 1;

    /// Boundary face normal vector direction
    static constexpr DirType D = get_face<F>::dir;

    /// Boundary face normal vector sign (int)
    static constexpr int SGN = get_face<F>::sgn;

    /// Boundary face normal vector sign (double)
    static constexpr double DSGN = get_face<F>::dsgn;

private: // TYPES

    /// Type of field
    using Field = ScalarField<T>;

    /// Non const type of the field value
    using V = typename std::remove_const<T>::type;

private: // MEMBERS

    /// View for grid values
    const view<Field> * m_view;

    /// Imposed value at boundary
    const V m_value;

    /// Grid level
    const Level * m_level;

    /// Grid spacing
    Vector m_h;

    /// Region where the view is defined
    Support m_support;

private: // METHODS

    /**
     * @brief Get value at position
     *
     * @param id grid index
     * @return value at id
     */
    inline V
    value (
        const IntVector & id
    ) const
    {
        return ( *m_view ) [ id ];
    };

protected: // COPY CONSTRUCTOR

    /**
     * @brief Constructor
     *
     * Instantiate a copy of a given view
     *
     * @param copy source view for copying
     * @param deep if true inner grid variable is copied as well otherwise the
     * same grid variable is referenced
     */
    bc_fd (
        const bc_fd * copy,
        bool deep
    ) : m_view ( copy->m_view ),
        m_value ( copy->m_value ),
        m_level ( copy->m_level ),
        m_h ( copy->m_h ),
        m_support ( copy->m_support )
    {}

public: // CONSTRUCTORS/DESTRUCTOR

    /**
     * @brief Constructor
     *
     * Instantiate a view without gathering info from the DataWarehouse
     *
     * @param view view to access grid values
     * @param label unused since value is already in view
     * @param material unused since value is already in view
     * @param level fine grid level
     * @param value imposed value at boundary
     */
    bc_fd (
        const view<Field> * view,
        const VarLabel * _DOXYARG ( label ),
        int _DOXYARG ( material ),
        const Level * level,
        const V & value
    ) : m_view ( view ),
        m_value ( value ),
        m_level ( level ),
        m_h ( level->dCell() )
    {
    }

    /// Destructor
    virtual ~bc_fd() = default;

    /// Prevent copy (and move) constructor
    bc_fd ( const bc_fd & ) = delete;

    /// Prevent copy (and move) assignment
    bc_fd & operator= ( const bc_fd & ) = delete;

public: // VIEW METHODS

    /**
     * @brief Retrieve value from the DataWarehouse for a given patch
     *
     * @remark Fails because cannot set bc_fd over a patch
     *
     * @param dw unused
     * @param patch unused
     * @param use_ghosts unused
     */
    virtual void
    set (
        DataWarehouse * _DOXYARG ( dw ),
        const Patch * _DOXYARG ( patch ),
        bool _DOXYARG ( use_ghosts )
    ) override
    {
        ASSERTFAIL ( "cannot set bc_fd over a patch" );
    };

    /**
     * @brief Retrieve value from the DataWarehouse for a given region
     *
     * Does not retrieve data since it is not required to enforce boundary
     * conditions (inner view is just referenced here and is already set by
     * bcs_basic_fd_view which owns it. It just set the view support.
     *
     * @param dw unused
     * @param level grid level from which retrieve data
     * @param low lower bound of the region to retrieve
     * @param high higher bound of the region to retrieve
     * @param use_ghosts unused
     */
    virtual void
    set (
        DataWarehouse * _DOXYARG ( dw ),
        const Level * level,
        const IntVector & low,
        const IntVector & high,
        bool _DOXYARG ( use_ghosts )
    ) override
    {
        // on bc ghost nodes are only those from fci or neighbors not domain bc
        m_level = level;
        m_h = level->dCell();

        m_support.clear();

        IntVector l ( low ), h ( high );
        if ( SGN > 0 ) // F=D+
        {
            l[D] = high[D];
            h[D] += GN;
        }
        else // F=D-
        {
            h[D] = low[D];
            l[D] -= GN;
        }

        m_support.emplace_back ( l, h );
    };

    /**
     * @brief get bc view support
     *
     * we want to extend this to ghost cells/nodes
     *
     * @return support of the view
     */
    virtual Support
    get_support()
    const override
    {
        return m_support;
    };

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
        IntVector low { m_support.front().getLow() }, high { m_support.front().getHigh() };
        return ( low[X] <= id[X] && id[X] < high[X] ) &&
               ( low[Y] <= id[Y] && id[Y] < high[Y] ) &&
               ( low[Z] <= id[Z] && id[Z] < high[Z] );
    };

    /**
     * @brief Get/Modify value at position with index id (modifications are allowed if T is non const)
     *
     * @remark Fails because inner view is constant
     *
     * @param id position index
     * @return reference to field value at id
     */
    virtual T &
    operator[] (
        const IntVector & _DOXYARG ( id )
    ) override VIRT;

    /**
     * @brief Get value at position
     *
     * @param id position index
     * @return field value at id
     */
    virtual V
    operator[] (
        const IntVector & id
    ) const override
    {
        IntVector i ( id );
        i[D] -= SGN;
        return value ( i ) + DSGN * m_value * m_h[D];
    };

public: // BASIC FD VIEW METHODS

    /**
     * @brief Get base view (Virtual Implementation)
     *
     * @remark Fails because inner view is constant
     *
     * @return nothing
     */
    virtual inline view<Field> *
    get_view()
    override VIRT;

    /**
     * @brief Get base view
     *
     * @return const pointer to base view implementation
     */
    virtual
    const view<Field> *
    get_view()
    const override
    {
        return this;
    };

public: // BC FD MEMBERS

    /**
     * @brief First order derivative
     * (parallel direction implementation)
     *
     * First order derivative along DIR at index id at boundary interface
     *
     * @remark derivative parallel to the face should not be computed by this bc_fd
     *
     * @tparam DIR Direction along with derivative is approximated
     * @param id unused
     * @return nothing
     */
    template < DirType DIR >
    inline typename std::enable_if < D != DIR, T >::type
    d (
        const IntVector & id
    ) const VIRT;

    /**
     * @brief First order derivative
     * (normal direction implementation)
     *
     * First order derivative along DIR at index id at boundary interface
     * @remark Dirichlet condition is applied to cell edge
     *
     * @tparam DIR Direction along with derivative is approximated
     * @param id index where to evaluate the finite-difference
     * @return approximated value at id
     */
    template <DirType DIR >
    inline typename std::enable_if < D == DIR, T >::type
    d (
        const IntVector & id
    ) const
    {
        return m_value;
    }

    /**
     * @brief Second order derivative
     * (parallel direction implementation)
     *
     * Second order derivative along DIR at index id at boundary interface
     *
     * @remark derivative parallel to the face should not be computed by this bc_fd
     *
     * @tparam DIR Direction along with derivative is approximated
     * @param id unused
     * @return nothing
     */
    template < DirType DIR >
    inline typename std::enable_if < D != DIR, T >::type
    d2 (
        const IntVector & id
    ) const VIRT;

    /**
     * @brief Second order derivative
     * (normal direction implementation)
     *
     * Second order derivative along DIR at index id at boundary interface
     *
     * @tparam DIR Direction along with derivative is approximated
     * @param id index where to evaluate the finite-difference
     * @return approximated value at id
     */
    template < DirType DIR >
    inline typename std::enable_if < D == DIR, T >::type
    d2 (
        const IntVector & id
    ) const
    {
        IntVector j ( id );
        j[D] -= SGN;
        return 2. * ( value ( j ) - value ( id ) + DSGN * m_h[D] * m_value ) / ( m_h[D] * m_h[D] );
    }

}; // class bc_fd

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_detail_bc_fd_Neumann_G1_NC_h
