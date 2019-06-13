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
 * @file CCA/Components/PhaseField/AMR/detail/amr_restrictor_I1_CC.h
 * @author Jon Matteo Church
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_AMR_detail_amr_restrictor_I0_NC_h
#define Packages_Uintah_CCA_Components_PhaseField_AMR_detail_amr_restrictor_I0_NC_h

#include <CCA/Components/PhaseField/AMR/AMRInterface.h>
#include <CCA/Components/PhaseField/DataTypes/Variable.h>
#include <CCA/Components/PhaseField/DataWarehouse/detail/dw_view.h>
#include <CCA/Components/PhaseField/AMR/detail/amr_finer_view.h>

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Wrapper of grid variables for restriction from finer to coarser levels
 * (node-centered piecewise constant implementation)
 *
 * @brief implements  piecewise constant restriction of a variable from finer to
 *        coarser levels.
 *
 * @tparam T variable data type (must be constant)
 * @tparam Problem type of PhaseField problem
 * @tparam I list of indices corresponding to the variable within the subproblems
 *
 * @implements amr_restrictor< FCI, VAR, DIM, T, Problem, I >
 */
template<typename T, typename Problem, size_t... I>
class amr_restrictor < ScalarField<T>, Problem, index_sequence<I...>, I0, NC >
    : virtual public view < ScalarField<T> >
{
public: // STATIC MEMBERS

    /// Default value for use_ghost when retrieving data
    static constexpr bool use_ghosts_dflt = false;

private: // STATIC MEMBERS

    /// restriction type: piecewise constant (I0) or linear-interpolation (I1)
    static constexpr FCIType FCI = I0;

    /// Problem dimension
    static constexpr DimType DIM = Problem::Dim;

    /// variable rapresentation type: cell-cenetered (CC) or node-centered (NC)
    static constexpr VarType VAR = NC;

    /// number of coarse ghosts which corresponds to the coarse region required by restriction
    static constexpr int E = 0;

private: // TYPES

    /// Type of field
    using Field = ScalarField<T>;

    /// Field index
    using Index = index_sequence<I...>;

    /// Non const type of the field value
    using V = typename std::remove_const<T>::type;

private: // STATIC ASSERTIONS

    static_assert ( Problem::Dim == DIM, "non consistent dimension" );
    static_assert ( Problem::Var == VAR, "non consistent variable type" );
    static_assert ( std::is_same< typename Problem::template get_field<I...>::type, ScalarField<T> >::value, "non consistent field types" );
    static_assert ( std::is_const<T>::value, "amr_restrictor with non const base" );

private: // MEMBERS

    /// inner view to variable on finer level
    amr_finer_view<Field, Problem, Index> * m_view_fine;

    /// coarser level
    const Level * m_level_coarse;

    /// finer level
    const Level * m_level_fine;

    /// Region where the view is defined
    Support m_support;

private: // METHODS

    /**
     * @brief Get fine value at position
     *
     * @param id fine index
     * @return fine value at id
     */
    inline T
    fine_value (
        const IntVector & id
    ) const
    {
        const auto & view_fine = *m_view_fine;
        return view_fine[id];
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
    amr_restrictor (
        const amr_restrictor * copy,
        bool deep
    ) : m_view_fine ( dynamic_cast < amr_finer_view<Field, Problem, Index> * > ( copy->m_view_fine->clone ( deep ) ) ),
        m_level_coarse ( copy->m_level_coarse ),
        m_level_fine ( copy->m_level_fine ),
        m_support ( copy->m_support )
    {}


public: // CONSTRUCTORS/DESTRUCTOR


    /**
     * @brief construct restrictor without retrieving inner variable data from
     *        the DataWarehouse
     *
     * @param label label of variable in the DataWarehouse
     * @param subproblems_label label of subproblems in the DataWarehouse
     * @param material index of material in the DataWarehouse
     */
    amr_restrictor (
        const VarLabel * label,
        const VarLabel * subproblems_label,
        int material
    ) : m_view_fine ( scinew amr_finer_view<Field, Problem, Index> ( label, subproblems_label, material ) ),
        m_level_coarse ( nullptr ),
        m_level_fine ( nullptr )
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
    amr_restrictor (
        DataWarehouse * dw,
        const VarLabel * label,
        const VarLabel * subproblems_label,
        int material,
        const Patch * patch,
        bool use_ghosts = use_ghosts_dflt
    ) : m_view_fine ( scinew amr_finer_view<Field, Problem, Index> ( dw, label, subproblems_label, material, patch ) ),
        m_level_coarse ( patch->getLevel() ),
        m_level_fine ( m_level_coarse->getFinerLevel().get_rep() )
    {
        ASSERTMSG ( !use_ghosts, "amr_restrictor doesn't support ghosts" );
        m_support.splice ( m_support.end(), m_view_fine->get_support() );
        for ( auto & region : m_support )
        {
            region.low() = AMRInterface<VAR, DIM>::get_coarser ( m_level_fine, region.low() );
            region.high() = AMRInterface<VAR, DIM>::get_coarser ( m_level_fine, region.high() );
        }
    }

    /// Destructor
    virtual ~amr_restrictor()
    {
        delete m_view_fine;
    };

    /// Prevent copy (and move) constructor
    amr_restrictor ( const amr_restrictor & ) = delete;

    /// Prevent copy (and move) assignment
    amr_restrictor & operator= ( const amr_restrictor & ) = delete;


public: // VIEW METHODS

    /**
     * @brief retrieve inner variable data from the DataWarehouse whitin a given
     *        patch.
     *
     * the number of ghost cells/nodes and the corresponding region on the
     * finer level is automatically computed to match the interpolation type
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
        m_view_fine->set ( dw, patch, use_ghosts );
        m_level_coarse = patch->getLevel();
        m_level_fine = m_level_coarse->getFinerLevel().get_rep();
        m_support.splice ( m_support.end(), m_view_fine->get_support() );
        for ( auto & region : m_support )
        {
            region.low() = AMRInterface<VAR, DIM>::get_coarser ( m_level_fine, region.low() );
            region.high() = AMRInterface<VAR, DIM>::get_coarser ( m_level_fine, region.high() );
        }
    };

    /**
     * @brief retrieve inner variable data from the DataWarehouse whitin a given
     *        region.
     *
     * the number of ghost cells/nodes and the corresponding region on the
     * finer level is automatically computed to match the interpolation type
     *
     * @param dw DataWarehouse which data is retrieved from
     * @param level level of the coarse region
     * @param low start index for the coarse region
     * @param high past the end index for the coarse region
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
        ASSERTMSG ( !use_ghosts, "amr_interpolator doesn't support ghosts" );
        m_support.clear();
        m_view_fine->set ( dw, level, low, high, use_ghosts );
        m_level_coarse = level;
        m_level_fine = m_level_coarse->getFinerLevel().get_rep();
        m_support.splice ( m_support.end(), m_view_fine->get_support() );
        for ( auto & region : m_support )
        {
            region.low() = AMRInterface<VAR, DIM>::get_coarser ( m_level_fine, region.low() );
            region.high() = AMRInterface<VAR, DIM>::get_coarser ( m_level_fine, region.high() );
        }
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
        return scinew amr_restrictor ( this, deep );
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
        return scinew virtual_view<amr_restrictor, Field> ( this, deep, offset );
    };

    /**
     * @brief get restrictor's coarse range
     *
     * @return coarse range
     */
    virtual Support
    get_support ()
    const override
    {
        Support support ( m_view_fine->get_support() );
        for ( auto & region : support )
        {
            region.low() = AMRInterface<VAR, DIM>::get_coarser ( m_level_fine, region.low() );
            region.high() = AMRInterface<VAR, DIM>::get_coarser ( m_level_fine, region.high() );
        }
        return support;
    };

    /**
     * @brief Check if the view has access to the coarse position with index id
     *
     * @param id_coarse coarse position index
     * @return check result
     */
    virtual
    bool
    is_defined_at (
        const IntVector & id_coarse
    ) const override
    {
        IntVector id_fine ( AMRInterface<VAR, DIM>::get_finer ( m_level_coarse, id_coarse ) );
        return m_view_fine->is_defined_at ( id_fine );
    };

    /**
      * @brief Get/Modify value at position with index id (virtual implementation)
      *
      * @remark restricted value is computed at runtime thus doesn't exist in the
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
     * @brief get restricted value
     *
     * value at fine index is computed avareging over the corresponding cells the finer level
     *
     * @param id_coarse coarse index
     * @return restricted value at the given fine index
     */
    virtual V
    operator[] (
        const IntVector & id_coarse
    ) const override
    {
        IntVector id_fine ( AMRInterface<VAR, DIM>::get_finer ( m_level_coarse, id_coarse ) );

        ASSERT ( ( m_level_coarse->getNodePosition ( id_coarse ).asVector() - m_level_coarse->getFinerLevel()->getNodePosition ( id_fine ).asVector() ).length() == 0 );

        return fine_value ( id_fine );
    }

}; // class amr_restrictor

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_AMR_detail_amr_restrictor_I0_NC_h
