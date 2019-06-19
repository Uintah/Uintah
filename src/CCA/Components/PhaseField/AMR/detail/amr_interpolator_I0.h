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
 * @file CCA/Components/PhaseField/AMR/detail/amr_interpolator_I0.h
 * @author Jon Matteo Church
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_AMR_detail_amr_interpolator_I0_h
#define Packages_Uintah_CCA_Components_PhaseField_AMR_detail_amr_interpolator_I0_h

#include <CCA/Components/PhaseField/AMR/detail/amr_coarser_view.h>

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Wrapper of grid variables for interpolation from coarser to finer levels
 * (piecewise implementation)
 *
 * implements piecewise constant interpolation of a variable from coarser
 * to finer levels in 1D
 *
 * @tparam T variable data type (must be constant)
 * @tparam Problem type of PhaseField problem
 * @tparam I list of indices corresponding to the variable within the subproblems
 * @tparam DIM problem dimension
 *
 * @implements amr_interpolator< Field, Problem, Index, FCI, DIM >
 */
template<typename T, typename Problem, size_t... I, DimType DIM >
class amr_interpolator < ScalarField<T>, Problem, index_sequence<I...>, I0, DIM >
    : virtual public view < ScalarField<T> >
{
public: // STATIC MEMBERS

    /// Default value for use_ghost when retrieving data
    static constexpr bool use_ghosts_dflt = false;

private: // STATIC MEMBERS

    /// Problem variable representation
    static constexpr VarType VAR = Problem::Var;

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

    /// Region where the view is defined
    Support m_support;

private: // METHODS

    /**
     * @brief Get coarse value at position
     *
     * @param id coarse index
     * @return coarse value at id
     */
    inline T
    coarse_value (
        const IntVector & id
    ) const
    {
        const auto & view_coarse = *m_view_coarse;
        return view_coarse[id];
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
     * @brief Constructor
     *
     * construct interpolator without retrieving inner variable data from
     * the DataWarehouse
     *
     * @param label label of variable in the DataWarehouse
     * @param subproblems_label label of subproblems in the DataWarehouse
     * @param material index of material in the DataWarehouse
     */
    amr_interpolator (
        const typename Field::label_type & label,
        const VarLabel * subproblems_label,
        int material
    ) : m_view_coarse ( scinew amr_coarser_view<Field, Problem, Index> ( label, subproblems_label, material ) ),
        m_level_fine ( nullptr ),
        m_level_coarse ( nullptr )
    {}

    /**
     * @brief Constructor
     *
     * construct interpolator and retrieve inner variable data from the
     * DataWarehouse within a given fine patch.
     *
     * @remark the number of ghost cells/nodes and the corresponding region on the
     * coarser level is automatically computed to match the interpolation type
     *
     * @param dw DataWarehouse which data is retrieved from
     * @param label label of variable in the DataWarehouse
     * @param subproblems_label label of subproblems in the DataWarehouse
     * @param material index of material in the DataWarehouse
     * @param patch patch on which data is retrieved
     * @param use_ghosts must be false
     */
    amr_interpolator (
        DataWarehouse * dw,
        const typename Field::label_type & label,
        const VarLabel * subproblems_label,
        int material,
        const Patch * patch,
        bool use_ghosts = use_ghosts_dflt
    ) : m_view_coarse ( scinew amr_coarser_view<Field, Problem, Index> ( dw, label, subproblems_label, material, patch ) ),
        m_level_fine ( patch->getLevel() ),
        m_level_coarse ( m_level_fine->getCoarserLevel().get_rep() )
    {
        ASSERTMSG ( !use_ghosts, "amr_interpolator doesn't support ghosts" );
        m_support.emplace_back ( DWInterface<VAR, DIM>::get_low ( patch ), DWInterface<VAR, DIM>::get_high ( patch ) );
    }

    /// Destructor
    virtual ~amr_interpolator()
    {
        delete m_view_coarse;
    }

    /// Prevent copy (and move) constructor
    amr_interpolator ( const amr_interpolator & ) = delete;

    /// Prevent copy (and move) assignment
    /// @return deleted
    amr_interpolator & operator= ( const amr_interpolator & ) = delete;

public: // VIEW METHODS

    /**
     * @brief retrieve inner variable data from the DataWarehouse within a given
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
        m_support.clear();
        m_view_coarse->set ( dw, patch );
        m_level_fine = patch->getLevel();
        m_level_coarse = m_level_fine->getCoarserLevel().get_rep();
        m_support.emplace_back ( DWInterface<VAR, DIM>::get_low ( patch ), DWInterface<VAR, DIM>::get_high ( patch ) );
    }

    /**
     * @brief retrieve inner variable data from the DataWarehouse within a given
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
    virtual void set (
        DataWarehouse * dw,
        const Level * level,
        const IntVector & low,
        const IntVector & high,
        bool use_ghosts = use_ghosts_dflt
    ) override
    {
        ASSERTMSG ( !use_ghosts, "amr_interpolator doesn't support ghosts" );
        m_support.clear();
        m_view_coarse->set ( dw, level, low, high );
        m_level_fine = level;
        m_level_coarse = m_level_fine->getCoarserLevel().get_rep();
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
    virtual inline view<Field> *
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
    virtual inline view<Field> *
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
    virtual inline Support
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
     * value at fine index is computed using the value at the nearset index
     * on the coarser level
     *
     * @param id_fine fine index
     * @return interpolated value at the given fine index
     */
    virtual inline V
    operator[] (
        const IntVector & id_fine
    ) const override
    {
        IntVector id_coarse ( AMRInterface<VAR, DIM>::get_coarser ( m_level_fine, id_fine ) );
        return coarse_value ( id_coarse );
    }

}; // class amr_interpolator<I0>

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_AMR_detail_amr_interpolator_I0_h
