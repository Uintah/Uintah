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
 * FITNESS FOR stencil_entries PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/**
 * @file CCA/Components/PhaseField/BoundaryConditions/detail/bcs_basic_fd_view.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_detail_bcs_basic_fd_view_h
#define Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_detail_bcs_basic_fd_view_h

#include <CCA/Components/PhaseField/Util/Expressions.h>
#include <CCA/Components/PhaseField/DataTypes/Support.h>
#include <CCA/Components/PhaseField/Views/detail/piecewise_view.h>
#include <CCA/Components/PhaseField/BoundaryConditions/detail/bc_basic_fd_view.h>
#include <CCA/Components/PhaseField/AMR/detail/amr_interpolator.h>

#include <memory>

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Detail implementation of variables wrapper for both basic differential
 * operations over both physical and amr boundaries
 *
 * Group together multiple views (one for eatch edge the boundary belongs to,
 * and one for accessing the DataWarehouse on internal indices) and
 * expose the correct implementation of basic differential operations for each
 * direction
 *
 * @tparam Field type of Field
 * @tparam Problem type of PhaseField problem
 * @tparam Index index_sequence of Field within Problem (first element is variable index,
 * following ones, if present, are the component index within the variable)
 * @tparam P list of BC, FC, and Patch::Face packs
 */
template<typename Field, StnType STN, typename Problem, typename Index, BCF ... P > class bcs_basic_fd_view;

template<typename T, StnType STN, typename Problem, typename Index, BCF ... P >
class bcs_basic_fd_view < ScalarField<T>, STN, Problem, Index, P... >
    : virtual public basic_fd_view < ScalarField<T>, STN >
    , public piecewise_view< ScalarField<T> >
{
private: // STATIC MEMBERS

    /// Problem variable representation
    static constexpr VarType VAR = Problem::Var;

    /// Problem dimension
    static constexpr DimType DIM = Problem::Dim;

    /// Number of ghosts required by STN
    static constexpr int GN = get_stn<STN>::ghosts;

    static constexpr size_t N = sizeof ... ( P );
    static constexpr BCF Q[N] = { P ... };

private: // TYPES

    /// Type of field
    using Field = ScalarField<T>;

    /// Non const type of the field value
    using V = typename std::remove_const<T>::type;

private: // STATIC ASSERTIONS

    static_assert ( STN == STN, "stencil value must match" );

private: // MEMBERS

    /// Label for the grid variable into the DataWarehouse
    const VarLabel * m_label;

    /// Label for subprombelms in the DataWarehouse
    const VarLabel * m_subproblems_label;

    /// Material index in the DataWarehouse
    const int m_material;

    /// Inner view to grid variable
    std::unique_ptr < dw_basic_fd_view<Field, STN, VAR> > m_dw_view;

    /// Inner view pointers to grid variable (indexed by direction)
    std::vector < basic_fd_view<Field, STN> * > m_fd_view;

    /// Inner view to boundary views
    std::tuple < std::unique_ptr < bc_basic_fd_view<Field, STN, VAR, P> > ... > m_bc_view;

    /// Region where the view is defined
    Support m_support;

private: // INDEXED CONSTRUCTOR

    /**
     * @brief Indexed constructor
     *
     * Instantiate a copy of a given view
     *
     * @tparam J indices for boundary views
     * @param unused to allow template argument deduction
     * @param copy source view for copying
     * @param deep if true inner grid variable is copied as well otherwise the
     * same grid variable is referenced
     */
    template < size_t ... J >
    bcs_basic_fd_view (
        index_sequence<J...>,
        const bcs_basic_fd_view * copy,
        bool deep
    ) : piecewise_view<Field> (), // copy is made in this constructor we don't want to duplicate clones
        m_label ( copy->m_label ),
        m_subproblems_label ( copy->m_subproblems_label ),
        m_material ( copy->m_material ),
        m_dw_view ( dynamic_cast < dw_basic_fd_view < Field, STN, VAR > * > ( copy->m_dw_view->clone ( deep ) ) ), m_fd_view ( DIM, m_dw_view.get () ),
        m_bc_view { std::unique_ptr < bc_basic_fd_view < Field, STN, VAR, P > > ( dynamic_cast < bc_basic_fd_view < Field, STN, VAR, P > * > ( m_fd_view[get_face< get_bcf<P>::face >::dir] = dynamic_cast < bc_basic_fd_view<Field, STN, VAR, P> * > ( std::get<J> ( copy->m_bc_view )->clone ( deep ) ) ) ) ... }
    {
        std::array<bool, N> {{ push_back_bc<J> () ... }};

        // dw view has to be last ( when dw has ghost they may overlap physical bc_views
        this->m_views.push_back ( m_dw_view.get() );
    }

    /**
     * @brief Indexed constructor
     *
     * Instantiate a view without gathering info from the DataWarehouse
     *
     * @tparam J indices for boundary views
     * @param unused to allow template argument deduction
     * @param label label of variable in the DataWarehouse
     * @param subproblems_label label of subproblems in the DataWarehouse
     * @param material index of material in the DataWarehouse
     * @param level grid level on which data is retrieved
     * @param bcs vector with info on the boundary conditions
     */
    template < size_t ... J >
    bcs_basic_fd_view (
        index_sequence<J...>,
        const typename Field::label_type & label,
        const VarLabel * subproblems_label,
        int material,
        const Level * level,
        const std::vector < BCInfo<Field> > & bcs
    ) : piecewise_view<Field> (),
        m_label ( label ),
        m_subproblems_label ( subproblems_label ),
        m_material ( material ),
        m_dw_view ( scinew dw_basic_fd_view<Field, STN, VAR> { label, material, level } ),
              m_fd_view ( DIM, m_dw_view.get () ),
    m_bc_view { std::unique_ptr< bc_basic_fd_view< Field, STN, VAR, P > > ( dynamic_cast < bc_basic_fd_view < Field, STN, VAR, P > * > ( m_fd_view[get_face< get_bcf<P>::face >::dir] = scinew bc_basic_fd_view<Field, STN, VAR, P> ( this, label, material, level, get_value<J> ( bcs ) ) ) ) ...  }
    {
        std::array<bool, N> {{ push_back_bc<J> () ... }};

        // dw view has to be last ( when dw has ghost they may overlap physical bc_views
        this->m_views.push_back ( m_dw_view.get() );
    }

private: // SINGLE INDEX METHODS

    /**
     * @brief push a bc_view back in the piece-views vector
     * @tparam J index of the bc
     * @return unused value (to allow call in array initialization for variadic
     * template expansions)
     */
    template < size_t J >
    bool
    push_back_bc ()
    {
        if ( get_bcf< Q[J] >::bc != BC::FineCoarseInterface )
            this->m_views.push_back ( std::get<J> ( m_bc_view ).get() );
        return true;
    }

    /**
     * @brief Get BC value (non FineCoarseInterface implementation)
     * @tparam J index of the bc
     * @return the value to impose as BC
     */
    template<size_t J, BC B = get_bcf< Q[J] >::bc >
    typename std::enable_if < B != BC::FineCoarseInterface, T >::type
    get_value (
        const std::vector < BCInfo<Field> > & bc
    )
    {
        return bc[J].value;
    }

    /**
     * @brief Get BC value (FineCoarseInterface implementation)
     * @tparam J index of the bc
     * @return newly created interpolator to impose continuity across fine/coarse
     * interfaces
     */
    template < size_t J, BC B = get_bcf< Q[J] >::bc, FC C2F = get_bcf< Q[J] >::c2f >
    typename std::enable_if < B == BC::FineCoarseInterface, view<Field> * >::type
    get_value (
        const std::vector < BCInfo<Field> > & bc
    )
    {
        return scinew amr_interpolator < Field, Problem, Index, get_fc<C2F>::fci, DIM > ( m_label, m_subproblems_label, m_material ); // deleted by bc_fd_FineCoarseInterface destructor
    }

    /**
     * Modify the given region bounds to avoid to retrieved from DataWarehouse
     * values that should instead be computed as per imposed boundary conditions
     *
     * @param low [in,out] lower bound
     * @param high [in,out] requested upper bound
     * @return unused value (to allow call in array initialization for variadic
     * template expansions)
     */
    template <BCF Q>
    bool adjust_dw_region (
        IntVector & low,
        IntVector & high
    )
    {
        constexpr Patch::FaceType F = get_bcf<Q>::face;
        constexpr DirType D = get_face<F>::dir;
        constexpr int SGN = get_face<F>::sgn;
        if ( SGN > 0 ) // F=D+
            high[D] -= GN ;
        else // F=D-
            low[D] += GN;
        return true;
    }

    /**
     * @brief Retrieve value from the DataWarehouse for a given region for inner
     * dw_view
     *
     * @param dw DataWarehouse from which data is retrieved
     * @param level grid level from which retrieve data
     * @param low lower bound of the region to retrieve
     * @param high higher bound of the region to retrieve
     * @param use_ghosts if ghosts value are to be retrieved
     */
    void
    set_dw (
        DataWarehouse * dw,
        const Level * level,
        const IntVector & low,
        const IntVector & high,
        bool use_ghosts
    )
    {
        m_dw_view->set ( dw, level, low, high, use_ghosts );
        IntVector l { low };
        IntVector h { high };
        if ( use_ghosts )
        {
            l -= get_dim<DIM>::template scalar_vector<GN>();
            h += get_dim<DIM>::template scalar_vector<GN>();
            std::array<bool, N> {{ adjust_dw_region<P> ( l, h )... }};
        }
        m_support.emplace_back ( l, h );
    }

    /**
     * @brief Retrieve value from the DataWarehouse for a given region for inner
     * bc_view
     *
     * @tparam J index of the bc
     * @param dw DataWarehouse from which data is retrieved
     * @param level grid level from which retrieve data
     * @param low lower bound of the region to retrieve
     * @param high higher bound of the region to retrieve
     * @param use_ghosts if ghosts value are to be retrieved
     */
    template < size_t J >
    bool
    set_bc (
        DataWarehouse * dw,
        const Level * level,
        const IntVector & low,
        const IntVector & high,
        bool use_ghosts
    )
    {
        basic_fd_view<Field, STN> * bc_view = std::get<J> ( m_bc_view ).get();
        bc_view->set ( dw, level, low, high, use_ghosts );
        if ( get_bcf< Q[J] >::bc != BC::FineCoarseInterface )
            m_support.splice ( m_support.end(), bc_view->get_support() );
        return true;
    }

private: // INDEXED VIEW METHODS

    /**
     * @brief Retrieve value from the DataWarehouse for a given region
     *
     * Retrieve data for all internal views
     *
     * @tparam J indices for boundary views
     * @param unused to allow template argument deduction
     * @param dw DataWarehouse from which data is retrieved
     * @param level grid level from which retrieve data
     * @param low lower bound of the region to retrieve
     * @param high higher bound of the region to retrieve
     * @param use_ghosts if ghosts value are to be retrieved
     */
    template < size_t ... J >
    void
    set (
        index_sequence<J...>,
        DataWarehouse * dw,
        const Level * level,
        const IntVector & low,
        const IntVector & high,
        bool use_ghosts
    )
    {
        m_support.clear();
        set_dw ( dw, level, low, high, use_ghosts );
        std::array<bool, N> {{ set_bc<J> ( dw, level, low, high, use_ghosts ) ... }};
    }

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
    bcs_basic_fd_view (
        const bcs_basic_fd_view * copy,
        bool deep
    ) : bcs_basic_fd_view ( make_index_sequence<N> {}, copy, deep )
    {
    }

public: // CONSTRUCTOR

    /**
     * @brief Constructor
     *
     * Instantiate a view without gathering info from the DataWarehouse
     *
     * @param label unused since value is already in view
     * @param subproblems_label label of subproblems in the DataWarehouse
     * @param material unused since value is already in view
     * @param level fine grid level
     * @param bcs vector with info on the boundary conditions
     */
    bcs_basic_fd_view (
        const typename Field::label_type & label,
        const VarLabel * subproblems_label,
        int material,
        const Level * level,
        const std::vector < BCInfo<Field> > & bcs
    ) : bcs_basic_fd_view ( make_index_sequence<N> {}, label, subproblems_label, material, level, bcs )
    {
    }

    /**
     * @brief Constructor
     *
     * Instantiate a view and gather info from dw
     *
     * @param dw DataWarehouse from which data is retrieved
     * @param label label of variable in the DataWarehouse
     * @param subproblems_label label of subproblems in the DataWarehouse
     * @param material index of material in the DataWarehouse
     * @param patch grid patch
     * @param bcs vector with info on the boundary conditions
     * @param use_ghosts if ghosts value are to be retrieved
     */
    bcs_basic_fd_view (
        DataWarehouse * dw,
        const typename Field::label_type & label,
        const VarLabel * subproblems_label,
        int material,
        const Patch * patch,
        const std::vector < BCInfo<Field> > & bcs,
        bool use_ghosts
    ) : m_label ( label ),
        m_subproblems_label ( subproblems_label ),
        m_material ( material )
    {
        ASSERTFAIL ( "cannot set bc_fd over a patch" );
    }

    /// Destructor
    ~bcs_basic_fd_view () = default;

    /// Prevent copy (and move) constructor
    bcs_basic_fd_view ( const bcs_basic_fd_view & ) = delete;

    /// Prevent copy (and move) assignment
    bcs_basic_fd_view & operator= ( const bcs_basic_fd_view & ) = delete;

public: // VIEW METHODS

    /**
     * @brief Retrieve value from the DataWarehouse for a given patch
     *
     * @remark bcs_basic_fd_view should not be set to range over a patch
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
     * Retrieve data for all internal views
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
        set ( make_index_sequence<N> {}, dw, level, low, high, use_ghosts );
    };

    /**
     * @brief Get a copy of the view
     *
     * @param deep if true inner grid variable is copied as well otherwise the
     * same grid variable is referenced
     *
     * @return new view instance
     */
    virtual view<Field> *
    clone (
        bool deep
    ) const override
    {
        return scinew bcs_basic_fd_view ( this, deep );
    };

    /**
     * @brief Get a copy of the view and apply translate the support
     *
     * @remark It is meant to be used for virtual patches (i.e. periodic boundaries)
     *
     * @param deep if true inner grid variable is copied as well otherwise the
     * same grid variable is referenced
     * @param offset vector specifying the translation of the support
     * @return new view instance
     */
    virtual view<Field> *
    clone (
        bool deep,
        const IntVector & offset
    ) const override
    {
        return scinew virtual_view<bcs_basic_fd_view, Field> ( this, deep, offset );
    };

    /**
     * @brief Get the region for which the view has access to the DataWarehouse
     *
     * It is the boundary region plus ghosts
     *
     * @return support of the view
     */
    virtual
    Support
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
    virtual
    bool
    is_defined_at (
        const IntVector & id
    ) const override
    {
        for ( auto & region : m_support )
        {
            IntVector low { region.getLow() }, high { region.getHigh() };
            if ( ( low[X] <= id[X] && id[X] < high[X] ) &&
                    ( low[Y] <= id[Y] && id[Y] < high[Y] ) &&
                    ( low[Z] <= id[Z] && id[Z] < high[Z] ) )
                return true;
        }
        return false;
    };

public: // BASIC FD VIEW METHODS

    /**
     * @brief Get base view
     *
     * @return non const pointer to base view implementation
     */
    virtual view<Field> *
    get_view()
    override
    {
        return this;
    }

    /**
     * @brief Get base view
     *
     * @return const pointer to base view implementation
     */
    virtual const view<Field> *
    get_view()
    const override
    {
        return this;
    }

    /**
     * @brief Partial x derivative
     *
     * First order derivative along x at index id
     *
     * @param id index where to evaluate the finite-difference
     * @return approximated value at id
     */
    virtual inline V
    dx (
        const IntVector & id
    ) const override
    {
        return m_fd_view[X]->dx ( id );
    }

    /**
     * @brief Partial y derivative
     *
     * First order derivative along y at index id
     *
     * @param id index where to evaluate the finite-difference
     * @return approximated value at id
     */
    virtual inline V
    dy (
        const IntVector & id
    ) const override
    {
        return m_fd_view[Y]->dy ( id );
    }

    /**
     * @brief Partial z derivative
     *
     * First order derivative along z at index id
     *
     * @param id index where to evaluate the finite-difference
     * @return approximated value at id
     */
    virtual inline V
    dz (
        const IntVector & id
    ) const override
    {
        return m_fd_view[Z]->dz ( id );
    };

    /**
     * @brief Partial x second order derivative
     *
     * Second order derivative along x at index id
     *
     * @param id index where to evaluate the finite-difference
     * @return approximated value at id
     */
    virtual inline V
    dxx (
        const IntVector & id
    ) const override
    {
        return m_fd_view[X]->dxx ( id );
    }

    /**
     * @brief Partial y second order derivative
     *
     * Second order derivative along y at index id
     *
     * @param id index where to evaluate the finite-difference
     * @return approximated value at id
     */
    virtual inline V
    dyy (
        const IntVector & id
    ) const override
    {
        return m_fd_view[Y]->dyy ( id );
    }

    /**
     * @brief Partial z second order derivative
     *
     * Second order derivative along z at index id
     *
     * @param id index where to evaluate the finite-difference
     * @return approximated value at id
     */
    virtual inline V
    dzz (
        const IntVector & id
    ) const override
    {
        return m_fd_view[Z]->dzz ( id );
    }

}; // bcs_basic_fd_view

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_detail_bcs_basic_fd_view_h
