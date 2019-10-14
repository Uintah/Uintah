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

#ifndef Packages_Uintah_CCA_Components_PhaseField_AMR_detail_amr_finer_view_h
#define Packages_Uintah_CCA_Components_PhaseField_AMR_detail_amr_finer_view_h
//
#include <CCA/Components/PhaseField/Views/detail/piecewise_view.h>
#include <CCA/Components/PhaseField/DataTypes/SubProblems.h>

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Composite view for accessing coarser grid variables
 *
 * @tparam Field type of Field (should be only ScalarField)
 * @tparam Problem type of PhaseField problem
 * @tparam Index index_sequence of Field within Problem (first element is variable index,
 * following ones, if present, are the component index within the variable)
 */
template<typename Field, typename Problem, typename Index> class amr_finer_view;

/**
 * @brief Composite view for accessing finer grid variables
 * (ScalarField implementation)
 *
 * @remark deletion of views instantiated in piecewise_view are performed here
 *
 * @tparam T type of the field value at each point
 * @tparam Problem type of PhaseField problem
 * @tparam I list of indices of Field within Problem (first element is variable index,
 * following ones, if present, are the component index within the variable)
 */
template<typename T, typename Problem, size_t... I>
class amr_finer_view < ScalarField<T>, Problem, index_sequence<I...> >
    : virtual public view < ScalarField<T> >
    , public piecewise_view < ScalarField<T> >
{
public: // STATIC MEMBERS

    /// Default value for use_ghost when retrieving data
    static constexpr bool use_ghosts_dflt = false;

private: // STATIC MEMBERS

    /// Problem variable representation
    static constexpr VarType VAR = Problem::Var;

    /// Problem dimension
    static constexpr DimType DIM = Problem::Dim;

private: // TYPES

    /// Type of field
    using Field = ScalarField<T>;

private: // STATIC ASSERTS

    static_assert ( std::is_same< typename Problem::template get_field<I...>::type, ScalarField<T> >::value, "non consistent field types" );
    static_assert ( std::is_const<T>::value, "amr_finer_view with non const base" );

private: // MEMBERS

    /// Label of variable in the DataWarehouse
    const VarLabel * m_label;

    /// Label for subprombelms in the DataWarehouse
    const VarLabel * m_subproblems_label;

    /// Material index in the DataWarehouse
    const int m_material;

    /// Region where the view is defined
    Support m_support;

private: // METHODS

    /**
     * @brief Create subviews
     *
     * Instantiate subviews in piecewise_view base and retrieve finer grid
     * data from dw over a given coarse region
     *
     * @param dw DataWarehouse from which data is retrieved
     * @param level_coarse grid level of the coarse region below the level from which retrieve data
     * @param low_coarse lower bound of the region to retrieve
     * @param high_coarse higher bound of the region to retrieve
     */
    void
    create_views (
        DataWarehouse * dw,
        const Level * level_coarse,
        const IntVector & low_coarse,
        const IntVector & high_coarse
    )
    {
        // Get finer region aboce given coarse region
        const Level * level_fine = level_coarse->getFinerLevel().get_rep();
        IntVector l_fine = AMRInterface<VAR, DIM>::get_finer ( level_coarse, low_coarse );
        IntVector h_fine = AMRInterface<VAR, DIM>::get_finer ( level_coarse, high_coarse -  IntVector {1, 1, 1} ) + level_fine->getRefinementRatio();

        // get list of fine patches in region
        std::vector<const Patch *> patches_fine;
        level_fine->selectPatches ( l_fine, h_fine, patches_fine );

        for ( const auto & patch_fine : patches_fine )
        {
            // virtual fine patches
            IntVector offset ( 0, 0, 0 );
            if ( patch_fine->isVirtual() )
                offset = patch_fine->getVirtualOffset();

            // need to check which BC applies on finer level
            Variable < PP, SubProblems<Problem> > subproblems_fine;

            // get fine subproblems (all logic for handling correctly all possible fine geometries is already there)
            dw->getOtherDataWarehouse ( Task::NewDW )->get ( subproblems_fine, m_subproblems_label, m_material, patch_fine );
            auto problems_fine = subproblems_fine.get().get_rep();

            for ( const auto & p : *problems_fine )
            {
                // check that the fine problem is above coarse region
                IntVector low_fine { Max ( p.get_low() + offset, l_fine ) };
                IntVector high_fine { Min ( p.get_high() + offset, h_fine ) };
                if (
                    low_fine[X] < high_fine[X] &&
                    low_fine[Y] < high_fine[Y] &&
                    low_fine[Z] < high_fine[Z]
                )
                {
                    // get problem (virtual) view
                    if ( patch_fine->isVirtual() )
                        this->m_views.push_back ( p.template get_view<I...> ().clone ( false, offset ) );
                    else
                        this->m_views.push_back ( p.template get_view<I...> ().clone ( false ) );

                    // retrieve fine data
                    this->m_views.back()->set ( dw, level_fine, low_fine, high_fine );

                    // add problem view support to view support
                    m_support.emplace_back ( low_fine, high_fine );
                }
            }
        }

        // after all subviews are fetched simplify view support
        m_support.simplify();
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
    amr_finer_view (
        const amr_finer_view * copy,
        bool deep
    ) : piecewise_view<Field> ( copy, deep ),
        m_label ( copy->m_label ),
        m_subproblems_label ( copy->m_subproblems_label ),
        m_material ( copy->m_material ),
        m_support ( copy->m_support )
    {}

public: // CONSTRUCTORS/DESTRUCTOR

    /**
     * @brief Constructor
     *
     * construct finer view without retrieving inner variable data from
     * the DataWarehouse
     *
     * @param label label of variable in the DataWarehouse
     * @param subproblems_label label of subproblems in the DataWarehouse
     * @param material index of material in the DataWarehouse
     */
    amr_finer_view (
        const VarLabel * label,
        const VarLabel * subproblems_label,
        int material
    ) : piecewise_view<Field> (),
        m_label ( label ),
        m_subproblems_label ( subproblems_label ),
        m_material ( material )
    {}

    /**
     * @brief Constructor
     *
     * construct finer view and retrieve inner variable data from the
     * DataWarehouse for the region lying under a given fine patch.
     *
     * @remark the number of ghost cells/nodes and the corresponding region on the
     * finer level is automatically computed to match the restriction type
     *
     * @param dw DataWarehouse which data is retrieved from
     * @param label label of variable in the DataWarehouse
     * @param subproblems_label label of subproblems in the DataWarehouse
     * @param material index of material in the DataWarehouse
     * @param patch patch on which data is retrieved
     * @param use_ghosts must be false
     */
    amr_finer_view (
        DataWarehouse * dw,
        const VarLabel * label,
        const VarLabel * subproblems_label,
        int material,
        const Patch * patch,
        bool use_ghosts = use_ghosts_dflt
    ) : piecewise_view<Field> (),
        m_label ( label ),
        m_subproblems_label ( subproblems_label ),
        m_material ( material )
    {
        ASSERTMSG ( !use_ghosts, "amr_finer_view doesn't support ghosts" );
        create_views ( dw, patch->getLevel(), DWInterface<VAR, DIM>::get_low ( patch ), DWInterface<VAR, DIM>::get_high ( patch ) );
    }

    /// Destructor
    virtual ~amr_finer_view()
    {
        for ( view<Field> * view : this->m_views ) delete view;
    }

    /// Prevent copy (and move) constructor
    amr_finer_view ( const amr_finer_view & ) = delete;

    /// Prevent copy (and move) assignment
    /// @return deleted
    amr_finer_view & operator= ( const amr_finer_view & ) = delete;

public: // VIEW METHODS

    /**
     * @brief Retrieve value from the DataWarehouse for accessing the finer
     * region over a given coarse patch
     *
     * @param dw DataWarehouse from which data is retrieved
     * @param patch coarse grid patch to retrieve data for
     * @param use_ghosts must be false
     */
    virtual void
    set (
        DataWarehouse * dw,
        const Patch * patch,
        bool use_ghosts = use_ghosts_dflt
    ) override
    {
        ASSERTMSG ( !use_ghosts, "amr_finer_view doesn't support ghosts" );
        m_support.clear();
        for ( view<Field> * view : this->m_views ) delete view;
        this->m_views.clear();
        create_views ( dw, patch->getLevel(), DWInterface<VAR, DIM>::get_low ( patch ), DWInterface<VAR, DIM>::get_high ( patch ) );
    }

    /**
     * @brief Retrieve value from the DataWarehouse for accessing the finer
     * region over a given coarse region
     *
     * @param dw DataWarehouse from which data is retrieved
     * @param level coarse grid level from which retrieve data
     * @param low lower bound of the coarse region to retrieve
     * @param high higher bound of the coarse region to retrieve
     * @param use_ghosts must be false
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
        ASSERTMSG ( !use_ghosts, "amr_finer_view doesn't support ghosts" );
        m_support.clear();
        for ( view<Field> * view : this->m_views ) delete view;
        this->m_views.clear();
        create_views ( dw, level, low, high );
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
        return scinew amr_finer_view ( this, deep );
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
        return scinew virtual_view<amr_finer_view, Field> ( this, deep, offset );
    };

    /**
     * @brief Get the coarse region corresponding to the finer one for which
     * the view has access to the DataWarehouse
     *
     * @return support of the view
     */
    virtual Support
    get_support()
    const override
    {
        return m_support;
    };

}; // class amr_finer_view

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_AMR_detail_amr_finer_view_h
