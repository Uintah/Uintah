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
 * @file CCA/Components/PhaseField/BoundaryConditions/detail/bcfd_view.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_detail_bcfd_view_h
#define Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_detail_bcfd_view_h

#include <CCA/Components/PhaseField/BoundaryConditions/detail/bcs_basic_fd_view.h>
#include <CCA/Components/PhaseField/DataWarehouse/detail/dw_fd_view.h>

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Wrapper of DataWarehouse variables for both basic and complex
 * differential operations over both physical and amr boundaries
 *
 * Adds to dw_view the possibility to compute finite-difference approximation of
 * of differential operations at boundary cells/points
 *
 * @tparam Field type of Field
 * @tparam Problem type of PhaseField problem
 * @tparam Index index_sequence of Field within Problem (first element is variable index,
 * following ones, if present, are the component index within the variable)
 * @tparam P list of BC, FC, and Patch::Face packs
 */
template<typename Field, StnType STN, typename Problem, typename Index, BCF ... P > class bcfd_view;

/**
 * @brief Wrapper of DataWarehouse variables for both basic and complex
 * differential operations over both physical and amr boundaries
 * (ScalarField implementation)
 *
 * Adds to dw_view the possibility to compute finite-difference approximation of
 * of differential operations at boundary cells/points
 *
 * @tparam T type of the field value at each point
 * @tparam Problem type of PhaseField problem
 * @tparam Index index_sequence of Field within Problem (first element is variable index,
 * following ones, if present, are the component index within the variable)
 * @tparam P list of BC, FC, and Patch::Face packs
 */
template<typename T, StnType STN, typename Problem, typename Index, BCF ... P >
class bcfd_view < ScalarField<T>, STN, Problem, Index, P... >
    : virtual public fd_view < ScalarField<T>, STN >
    , public bcs_basic_fd_view < ScalarField<T>, STN, Problem, Index, P... >
    , public dw_fd_view < ScalarField<T>, STN, Problem::Var >
{
private: // TYPES

    /// Type of field
    using Field = ScalarField<T>;

public: // CONSTRUCTORS/DESTRUCTOR

    /// View is constructed by bcs_basic_fd_view
    using bcs_basic_fd_view<Field, STN, Problem, Index, P...>::bcs_basic_fd_view;

    /// Default destructor
    virtual ~bcfd_view() = default;

    /// Prevent copy (and move) constructor
    bcfd_view ( const bcfd_view & ) = delete;

    /// Prevent copy (and move) assignment
    /// @return deleted
    bcfd_view & operator= ( const bcfd_view & ) = delete;

}; // class bcfd_view

/**
 * @brief Wrapper of DataWarehouse variables for both basic and complex
 * differential operations over both physical and amr boundaries
 * (VectorField implementation)
 *
 * Adds to dw_view the possibility to compute finite-difference approximation of
 * of differential operations at boundary cells/points
 *
 * @tparam T type of each component of the field value at each point
 * @tparam N dimension of the vetor field
 * @tparam Problem type of PhaseField problem
 * @tparam I variable index within Problem
 * @tparam P list of BC, FC, and Patch::Face packs
 */
template<typename T, size_t N, StnType STN, typename Problem, size_t I, BCF ... P >
class bcfd_view < VectorField<T, N>, STN, Problem, index_sequence<I>, P... >
    : virtual public view_array < fd_view < ScalarField<T>, STN >, ScalarField<T>, N >
{
private: // TYPES

    /// Type of field
    using Field = VectorField<T, N>;

    /// Type of View of each component
    using View = fd_view < ScalarField<T>, STN >;

private: // SINGLE INDEX METHODS

    /**
     * @brief Create bcfd view for a given component
     *
     * Instantiate a view without gathering info from the DataWarehouse
     *
     * @tparam J component index
     * @param label label of variable in the DataWarehouse
     * @param subproblems_label label of subproblems in the DataWarehouse
     * @param material index of material in the DataWarehouse
     * @param level level on which data is retrieved
     * @param bcs vector with info on the boundary conditions
     * @return pointer to the newly created view
     */
    template<size_t J>
    void * create_element (
        const typename Field::label_type & label,
        const VarLabel * subproblems_label,
        int material,
        const Level * level,
        const std::vector< BCInfo<Field> > & bcs
    )
    {
        std::vector< BCInfo< ScalarField<T> > > bci;
        for ( const auto & b : bcs )
            bci.push_back ( { b.value[J], b.bc, b.c2f } );
        return this->m_view_ptr[J] = new bcfd_view < ScalarField<T>, STN, Problem, index_sequence<I, J>, P... > ( label[J], subproblems_label, material, level, bci );
    }

    /**
     * @brief Create bcfd view for a given component
     *
     * Instantiate a view and gather info from dw
     *
     * @tparam J component index
     * @param dw DataWarehouse from which data is retrieved
     * @param label label of variable in the DataWarehouse
     * @param subproblems_label label of subproblems in the DataWarehouse
     * @param material index of material in the DataWarehouse
     * @param patch grid patch on which data is retrieved
     * @param bcs vector with info on the boundary conditions
     * @param use_ghosts if ghosts value are to be retrieved
     * @return pointer to the newly created view
     */
    template<size_t J>
    void * create_element (
        DataWarehouse * dw,
        const typename Field::label_type & label,
        const VarLabel * subproblems_label,
        int material,
        const Patch * patch,
        const std::vector< BCInfo<Field> > & bcs,
        bool use_ghosts
    )
    {
        std::vector< BCInfo< ScalarField<T> > > bci;
        for ( const auto & b : bcs )
            bci.push_back ( { b.value[J], b.bc, b.c2f } );
        return this->m_view_ptr[J] = new bcfd_view < ScalarField<T>, STN, Problem, index_sequence<I, J>, P... > ( dw, label[J], subproblems_label, material, patch, bci, use_ghosts );
    }

private: // INDEXED CONSTRUCTOR

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
    template<size_t... J>
    bcfd_view (
        index_sequence<J...> _DOXYARG ( unused ),
        const typename Field::label_type & label,
        const VarLabel * subproblems_label,
        int material,
        const Level * level,
        const std::vector< BCInfo<Field> > & bcs
    )
    {
        std::array<bool, N> {{ create_element<J> ( label, subproblems_label, material, level, bcs )... }};
    }

    /**
     * @brief Indexed constructor
     *
     * Instantiate a view and gather info from dw
     *
     * @tparam J indices for boundary views
     * @param unused to allow template argument deduction
     * @param dw DataWarehouse from which data is retrieved
     * @param label label of variable in the DataWarehouse
     * @param subproblems_label label of subproblems in the DataWarehouse
     * @param material index of material in the DataWarehouse
     * @param patch grid patch
     * @param bcs vector with info on the boundary conditions
     * @param use_ghosts if ghosts value are to be retrieved
     */
    template<size_t... J>
    bcfd_view (
        index_sequence<J...> _DOXYARG ( unused ),
        DataWarehouse * dw,
        const typename Field::label_type & label,
        const VarLabel * subproblems_label,
        int material,
        const Patch * patch,
        const std::vector< BCInfo<Field> > & bcs,
        bool use_ghosts
    )
    {
        std::vector< BCInfo< ScalarField<T> > > bci;
        for ( const auto & b : bcs )
            bci.push_back ( { b.value[I], b.bc, b.c2f } );
        std::array<bool, N> {{ create_element<J> ( dw, label, subproblems_label, material, patch, bcs, use_ghosts )... }};
    }

public: // CONSTRUCTORS/DESTRUCTOR

    /**
     * @brief Constructor
     *
     * Instantiate a view without gathering info from the DataWarehouse
     *
     * @param label label of variable in the DataWarehouse
     * @param subproblems_label label of subproblems in the DataWarehouse
     * @param material index of material in the DataWarehouse
     * @param level grid level on which data is retrieved
     * @param bcs vector with info on the boundary conditions
     */
    bcfd_view (
        const typename Field::label_type & label,
        const VarLabel * subproblems_label,
        int material,
        const Level * level,
        const std::vector< BCInfo<Field> > & bcs
    ) : bcfd_view ( make_index_sequence<N> {}, label, subproblems_label, material, level, bcs )
    {};

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
    bcfd_view (
        DataWarehouse * dw,
        const typename Field::label_type & label,
        const VarLabel * subproblems_label,
        int material,
        const Patch * patch,
        const std::vector< BCInfo<Field> > & bcs,
        bool use_ghosts
    ) : bcfd_view ( make_index_sequence<N> {}, dw, label, subproblems_label, material, patch, bcs, use_ghosts )
    {}

    /// Destructor
    virtual ~bcfd_view()
    {
        for ( auto view : this->m_view_ptr )
            delete view;
    }

    /// Prevent copy (and move) constructor
    bcfd_view ( const bcfd_view & ) = delete;

    /// Prevent copy (and move) assignment
    /// @return deleted
    bcfd_view & operator= ( const bcfd_view & ) = delete;

}; // bcfd_view

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_BCFDView_h
