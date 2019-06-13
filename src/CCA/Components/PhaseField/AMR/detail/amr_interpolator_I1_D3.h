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
 * IMPLIED, ILUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/**
 * @file CCA/Components/PhaseField/AMR/detail/amr_interpolator_I1_D3.h
 * @author Jon Matteo Church
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_AMR_detail_amr_interpolator_I1_D3_h
#define Packages_Uintah_CCA_Components_PhaseField_AMR_detail_amr_interpolator_I1_D3_h

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Wrapper of grid variables for interpolation from coarser to finer levels
 * (3D linear implementation)
 *
 * Implements linear interpolation of a variable from coarser to finer
 * levels in 3D
 *
 * @bug this is actually broken since it could happen that interpolation is
 * requiring values from coarse level that are not extrapolated by bcs!
 * @todo implment as recursion over dimension to avoid this issue!
 *
 * @tparam T variable data type (must be constant)
 * @tparam Problem type of PhaseField problem
 * @tparam I list of indices corresponding to the variable within the subproblems
 *
 * @todo generalize implementation to arbitrary dimension
 *
 * @implements amr_interpolator< Field, Problem, Index, FCI, DIM >
 */
template<typename T, typename Problem, size_t... I>
class amr_interpolator < ScalarField<T>, Problem, index_sequence<I...>, I1, D3 >
    : virtual public view < ScalarField<T> >
{
public: // STATIC MEMBERS

    static constexpr bool use_ghosts_dflt = false;

private: // STATIC MEMBERS

    /// Problem variable representation
    static constexpr VarType VAR = Problem::Var;

    /// interpolation type: linear-interpolation (I1)
    static constexpr FCIType FCI = I1;

    /// problem dimension
    static constexpr DimType DIM = D3;

    /// number elements in the coarse region required by interpolation
    static constexpr int E = get_fci<FCI>::elems;

private: // TYPES

    /// Type of field
    using Field = ScalarField<T>;

    /// Field index
    using Index = index_sequence<I...>;

    /// Non const type of the field value
    using V = typename std::remove_const<T>::type;

private: // STATIC ASSERTIONS

    static_assert ( Problem::Dim == DIM, "non consistent dimension" );
    static_assert ( std::is_same< typename Problem::template get_field<I...>::type, ScalarField<T> >::value, "non consistent field types" );
    static_assert ( std::is_const<T>::value, "amr_interpolator with non const base" );

private: // MEMBERS

    /// inner view to variable on coarser level
    amr_coarser_view<Field, Problem, Index> * m_view_coarse;

    /// finer level
    const Level * m_level_fine;

    /// coarser level
    const Level * m_level_coarse;

    Support m_support;

private: // METHODS

    /**
     * @brief Get coarse value at position
     *
     * @param id coarse index
     * @return fine value at id
     */
    const T
    coarse_value (
        const IntVector & id
    ) const
    {
        const auto & view_coarse = *m_view_coarse;
        return view_coarse[id];
    }

    /**
     * @brief Compute required fine region
     *
     * Compute fine region corresponding to the coarse one required to
     * compute the interpolation at fine indices within the given bounds
     *
     * @param low lower bound of the region where to perform interpolation
     * @param high higher bound of the region where to perform interpolation
     * @return required region
     */
    Region
    compute_fine_region (
        const IntVector & low,
        const IntVector & high
    )
    {
        IntVector r = m_level_fine->getRefinementRatio();
        IntVector l_ghosts {0, 0, 0}, h_ghosts {0, 0, 0};
        for ( size_t d = 0; d < DIM; ++d )
        {
            int total = r[d] * ( E - 1 );
            h_ghosts[d] = total / 2;
            l_ghosts[d] = total - h_ghosts[d];
            if ( VAR == NC )
            {
                if ( ! ( low[d] % r[d] ) && l_ghosts[d] > 0 ) l_ghosts[d] -= 1;
                if ( ! ( ( high[d] - 1 ) % r[d] ) && h_ghosts[d] > 0 ) h_ghosts[d] -= 1;
            }
        }

        Region fine_region;
        fine_region.low() = low - l_ghosts;
        fine_region.high() = high + h_ghosts;

        return fine_region;
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
    amr_interpolator (
        const amr_interpolator * copy,
        bool deep
    ) : m_view_coarse ( dynamic_cast < amr_coarser_view<Field, Problem, Index> * > ( copy->m_view_coarse->clone ( deep ) ) ),
        m_level_fine ( copy->m_level_fine ),
        m_level_coarse ( copy->m_level_coarse ),
        m_support ( copy->m_support )
    {}

public: // CONSTRUCTORS/DESTRUCTOR

    /**
     * @brief construct interpolator without retrieving inner variable data from
     *        the DataWarehouse
     *
     * @param label label of variable in the DataWarehouse
     * @param subproblems_label label of subproblems in the DataWarehouse
     * @param material index of material in the DataWarehouse
     */
    amr_interpolator (
        const VarLabel * label,
        const VarLabel * subproblems_label,
        int material
    ) : m_view_coarse ( scinew amr_coarser_view<Field, Problem, Index> ( label, subproblems_label, material ) ),
        m_level_fine ( nullptr ),
        m_level_coarse ( nullptr )
    {
    }

    /**
     * @brief construct interpolator and retrieve inner variable data from the
     *        DataWarehouse whitin a given fine patch.
     *
     * the number of ghost cells/nodes and the corresponding region on the
     * coarser level is automatically computed to match the interpolation type
     *
     * @param dw DataWarehouse which data is retrieved from
     * @param label label of variable in the DataWarehouse
     * @param subproblems_label label of subproblems in the DataWarehouse
     * @param material index of material in the DataWarehouse
     * @param patch patch on which data is retrieved
     * @param use_ghosts if ghosts value are to be retrieved (must be false)
     */
    amr_interpolator (
        DataWarehouse * dw,
        const VarLabel * label,
        const VarLabel * subproblems_label,
        int material,
        const Patch * patch,
        bool use_ghosts = use_ghosts_dflt
    ) : m_view_coarse ( scinew amr_coarser_view<Field, Problem, Index> ( label, subproblems_label, material ) ),
        m_level_fine ( patch->getLevel() ),
        m_level_coarse ( m_level_fine->getCoarserLevel().get_rep() )
    {
        ASSERTMSG ( !use_ghosts, "amr_interpolator doesn't support ghosts" );
        set ( dw, patch );
    }

    /// Destructor
    virtual ~amr_interpolator()
    {
        delete m_view_coarse;
    }

    /// Prevent copy (and move) constructor
    amr_interpolator ( const amr_interpolator & ) = delete;

    /// Prevent copy (and move) assignment
    amr_interpolator & operator= ( const amr_interpolator & ) = delete;

public: // VIEW METHODS

    /**
     * @brief retrieve inner variable data from the DataWarehouse whitin a given
     *        patch.
     *
     * the number of ghost cells/nodes and the corresponding region on the
     * coarser level is automatically computed to match the interpolation type
     *
     * @param dw DataWarehouse which data is retrieved from
     * @param patch patch on which data is retrieved
     * @param use_ghosts if ghosts value are to be retrieved (must be false)
     */
    virtual void
    set (
        DataWarehouse * dw,
        const Patch * patch,
        bool use_ghosts = use_ghosts_dflt
    ) override
    {
        ASSERTMSG ( !use_ghosts, "amr_interpolator doesn't support ghosts" );
        set ( dw, patch->getLevel(), DWInterface<VAR, DIM>::get_low ( patch ), DWInterface<VAR, DIM>::get_high ( patch ) );
    }

    /**
     * @brief retrieve inner variable data from the DataWarehouse whitin a given
     *        region.
     *
     * the number of ghost cells/nodes and the corresponding region on the
     * coarser level is automatically computed to match the interpolation type
     *
     * @param dw DataWarehouse which data is retrieved from
     * @param level level of the fine region
     * @param low start index for the fine region
     * @param high past the end index for the fine region
     * @param use_ghosts if ghosts value are to be retrieved (must be false)
     */
    virtual void
    set (
        DataWarehouse * dw,
        const Level * level,
        const IntVector & low,
        const IntVector & high,
        bool use_ghosts = use_ghosts_dflt
    ) override
    {
        ASSERTMSG ( !use_ghosts, "amr_interpolator doesn't support ghosts" )
        m_support.clear();
        m_level_fine = level;
        m_level_coarse = m_level_fine->getCoarserLevel().get_rep();
        Region fine_region = compute_fine_region ( low, high );
        m_view_coarse->set ( dw, level, fine_region.low(), fine_region.high() );
        m_support.emplace_back ( low, high );
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
    )
    const override
    {
        return scinew amr_interpolator ( this, deep );
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
    )
    const override
    {
        return scinew virtual_view<amr_interpolator, Field> ( this, deep, offset );
    };

    /**
     * @brief get interpolator's fine range
     *
     * @return fine range
     */
    virtual Support
    get_support()
    const override
    {
        return m_support;
    };

    /**
     * @brief Check if the view has access to the fine position with index id
     *
     * @param id fine position index
     * @return check result
     */
    virtual bool
    is_defined_at (
        const IntVector & id
    ) const override
    {
        const IntVector low
        {
            m_support.front().getLow()
        };
        const IntVector high
        {
            m_support.front().getHigh()
        };
        return ( low[X] <= id[X] && id[X] < high[X] ) &&
               ( low[Y] <= id[Y] && id[Y] < high[Y] ) &&
               ( low[Z] <= id[Z] && id[Z] < high[Z] );
    };

    /**
     * @brief Get/Modify value at position with index id (virtual implementation)
     *
     * @remark interpolated value is computed at runtime thus doesn't exist in the
     * DataWarehouse
     *
     * @param id unused
     * @return nothing
     */
    virtual T &
    operator[] (
        const IntVector & _DOXYARG ( id )
    ) override VIRT;

    /**
     * @brief get interpolated value
     *
     * value at fine index is computed by tri-linear interpolation over the
     * nearest 8 indexes on the coarser level
     *
     * @param id_fine fine index
     * @return interpolated value at the given fine index
     */
    virtual V
    operator[] (
        const IntVector & id_fine
    ) const override
    {
        IntVector id_coarse ( m_level_fine->mapCellToCoarser ( id_fine ) );
        Point p_fine ( DWInterface<VAR, DIM>::get_position ( m_level_fine, id_fine ) );
        Point p_coarse ( DWInterface<VAR, DIM>::get_position ( m_level_coarse, id_coarse ) );
        Vector dist = ( p_fine.asVector() - p_coarse.asVector() ) / m_level_coarse->dCell();
        double w[2][2][2] = {{{ 1., 1. }, { 1., 1. }}, {{ 1., 1. }, { 1., 1. }}};
        IntVector n[2][2][2] = {{{ id_coarse, id_coarse }, { id_coarse, id_coarse }}, {{ id_coarse, id_coarse }, { id_coarse, id_coarse }}};
        const double & dx = dist[X];
        const double & dy = dist[Y];
        const double & dz = dist[Z];
        if ( dx < 0. )
        {
            n[0][0][0][X] = n[0][1][0][X] = n[0][0][1][X] = n[0][1][1][X] -= 1;
            w[0][0][0] *= -dx;
            w[0][1][0] *= -dx;
            w[0][0][1] *= -dx;
            w[0][1][1] *= -dx;
            w[1][0][0] *= 1 + dx;
            w[1][1][0] *= 1 + dx;
            w[1][0][1] *= 1 + dx;
            w[1][1][1] *= 1 + dx;
        }
        else if ( dx > 0. )
        {
            n[1][0][0][X] = n[1][1][0][X] = n[1][0][1][X] = n[1][1][1][X] += 1;
            w[0][0][0] *= 1 - dx;
            w[0][1][0] *= 1 - dx;
            w[0][0][1] *= 1 - dx;
            w[0][1][1] *= 1 - dx;
            w[1][0][1] *= dx;
            w[1][1][1] *= dx;
            w[1][0][1] *= dx;
            w[1][1][1] *= dx;
        }
        else
        {
            w[1][0][0] = 0.;
            w[1][1][0] = 0.;
            w[1][0][1] = 0.;
            w[1][1][1] = 0.;
        }

        if ( dy < 0. )
        {
            n[0][0][0][Y] = n[1][0][0][Y] = n[0][0][1][Y] = n[1][0][1][Y] -= 1;
            w[0][0][0] *= -dy;
            w[1][0][0] *= -dy;
            w[0][0][1] *= -dy;
            w[1][0][1] *= -dy;
            w[0][1][0] *= 1 + dy;
            w[1][1][0] *= 1 + dy;
            w[0][1][1] *= 1 + dy;
            w[1][1][1] *= 1 + dy;
        }
        else if ( dy > 0. )
        {
            n[0][1][0][Y] = n[1][1][0][Y] = n[0][1][1][Y] = n[1][1][1][Y] += 1;
            w[0][0][0] *= 1 - dy;
            w[1][0][0] *= 1 - dy;
            w[0][0][1] *= 1 - dy;
            w[1][0][1] *= 1 - dy;
            w[0][1][0] *= dy;
            w[1][1][0] *= dy;
            w[0][1][1] *= dy;
            w[1][1][1] *= dy;
        }
        else
        {
            w[0][1][0] = 0.;
            w[1][1][0] = 0.;
            w[0][1][1] = 0.;
            w[1][1][1] = 0.;
        }

        if ( dz < 0. )
        {
            n[0][0][0][Z] = n[0][1][0][Z] = n[1][0][0][Z] = n[1][1][0][Z] -= 1;
            w[0][0][0] *= -dz;
            w[0][1][0] *= -dz;
            w[1][0][0] *= -dz;
            w[1][1][0] *= -dz;
            w[0][0][1] *= 1 + dz;
            w[0][1][1] *= 1 + dz;
            w[1][0][1] *= 1 + dz;
            w[1][1][1] *= 1 + dz;
        }
        else if ( dz > 0. )
        {
            n[0][0][1][Z] = n[0][1][1][Z] = n[1][0][1][Z] = n[1][1][1][Z] += 1;
            w[0][0][0] *= 1 - dz;
            w[0][1][0] *= 1 - dz;
            w[1][0][0] *= 1 - dz;
            w[1][1][0] *= 1 - dz;
            w[0][0][1] *= dz;
            w[0][1][1] *= dz;
            w[1][0][1] *= dz;
            w[1][1][1] *= dz;
        }
        else
        {
            w[0][0][1] = 0.;
            w[0][1][1] = 0.;
            w[1][0][1] = 0.;
            w[1][1][1] = 0.;
        }

        return w[0][0][0] * coarse_value ( n[0][0][0] ) +
               w[0][0][1] * coarse_value ( n[0][0][1] ) +
               w[0][1][0] * coarse_value ( n[0][1][0] ) +
               w[0][1][1] * coarse_value ( n[0][1][1] ) +
               w[1][0][0] * coarse_value ( n[1][0][0] ) +
               w[1][0][1] * coarse_value ( n[1][0][1] ) +
               w[1][1][0] * coarse_value ( n[1][1][0] ) +
               w[1][1][1] * coarse_value ( n[1][1][1] );
    }

}; // class amr_interpolator <I1, D3>

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_AMR_detail_amr_interpolator_I1_D3_h
