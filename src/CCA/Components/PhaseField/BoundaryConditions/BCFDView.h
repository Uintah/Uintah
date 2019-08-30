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
 * @file CCA/Components/PhaseField/BoundaryConditions/BCFDView.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_BCFDView_h
#define Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_BCFDView_h

#include <CCA/Components/PhaseField/Factory/Implementation.h>
#include <CCA/Components/PhaseField/Views/FDView.h>
#include <CCA/Components/PhaseField/BoundaryConditions/detail/bcfd_view.h>

namespace Uintah
{
namespace PhaseField
{

/**
 * @brief Wrapper of DataWarehouse variables for both basic and complex
 * differential operations at both physical and amr boundaries
 *
 * Factory Implementation for dynamic instantiation
 *
 * @tparam Problem type of PhaseField problem
 * @tparam I index of Field within Problem
 * @tparam P list of BC, FC, and Patch::Face packs
 */
template < typename Problem, size_t I, BCF... P >
class BCFDView
    : virtual public FDView < typename Problem::template get_field<I>::type, Problem::Stn >
    , public Implementation < BCFDView<Problem, I, P...>, FDView < typename Problem::template get_field<I>::type, Problem::Stn >, const typename Problem::template get_field<I>::type::label_type &, const VarLabel *, int, const Level *, const std::vector < BCInfo < typename Problem::template get_field<I>::type > > & >
    , public detail::bcfd_view< typename Problem::template get_field<I>::type, Problem::Stn, Problem, index_sequence<I>, P... >
{
private: // TYPES

    /// Type of field
    using Field = typename Problem::template get_field<I>::type;

    /// Index of Field within Problem (first element is variable index, following
    /// ones, if present, are the component index within the variable)
    using Index = index_sequence<I>;

private: // STATIC MEMBERS

    /// Finite-difference stencil
    static constexpr StnType STN = Problem::Stn;

public: // STATIC MEMBERS

    /// Default value for use_ghost when retrieving data
    static constexpr bool use_ghosts_dflt = true;

    /// Implementation identifier within Factory
    static const std::string Name;

public: // CONSTRUCTOR

    /**
     * @brief Constructor
     *
     * Instantiate view components without gathering info from the DataWarehouse
     *
     * @param label list of variable labels for each component
     * @param subproblems_label label of subproblems in the DataWarehouse
     * @param material material index
     * @param level grid level
     * @param bcs vector with info on the boundary conditions
     */
    BCFDView (
        const typename Field::label_type & label,
        const VarLabel * subproblems_label,
        int material,
        const Level * level,
        const std::vector< BCInfo<Field> > & bcs
    ) : detail::bcfd_view<Field, STN, Problem, Index, P...> ( label, subproblems_label, material, level, bcs )
    {}

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
    BCFDView (
        DataWarehouse * dw,
        const typename Field::label_type & label,
        const VarLabel * subproblems_label,
        int material,
        const Patch * patch,
        const std::vector< BCInfo<Field> > & bcs,
        bool use_ghosts = use_ghosts_dflt
    ) : detail::bcfd_view<Field, STN, Problem, Index, P...> ( dw, label, subproblems_label, material, patch, bcs, use_ghosts )
    {}

    /// Destructor
    virtual ~BCFDView() = default;

    /// Prevent copy (and move) constructor
    BCFDView ( const BCFDView & ) = delete;

    /// Prevent copy (and move) assignment
    /// @return deleted
    BCFDView & operator= ( const BCFDView & ) = delete;

}; // class BCFDView

} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_BCFDView_h
