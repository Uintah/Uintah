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
 * @file CCA/Components/PhaseField/BoundaryConditions/BCInterface.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_BCInterface_h
#define Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_BCInterface_h

#include <CCA/Components/PhaseField/Util/Definitions.h>
#include <CCA/Components/PhaseField/BoundaryConditions/detail/get_bcs.h>
#include <CCA/Components/PhaseField/BoundaryConditions/detail/partition_range.h>
#include <CCA/Components/PhaseField/DataWarehouse/DWInterface.h>

namespace Uintah
{
namespace PhaseField
{

/**
 * @brief Interface for boundary conditions
 *
 * groups together various methods to get info about boundary conditions which
 * depend on the different types of variable representation and fine-differences
 * stencils allowing to choose the relevant implementation at compile time
 * @tparam VAR type of variable representation
 * @tparam STN finite-difference stencil
 */
template < VarType VAR, StnType STN>
struct BCInterface
{
    /**
     * @brief Partition a patch into a list of Problems
     *
     * Retrieves boundary conditions and partirion a patch accordingly
     *
     * @tparam Field list of type of fields (ScalarField < T> or VectorField < T, N >) of the Problems
     * @param labels list of lables for each variable of the problem
     * @param subproblems_label label of subproblems in the DataWarehouse
     * @param material problem material index
     * @param patch grid patch to be partitioned
     * @param c2f which fine/coarse interface conditions to use on each variable
     * @return list of partitioned Problems
     */
    template <typename... Field>
    static std::list < Problem<VAR, STN, Field...> >
    partition_patch (
        const typename Field::label_type & ...  labels,
        const VarLabel * subproblems_label,
        int material,
        const Patch * patch,
        const std::map<std::string, FC> * c2f = nullptr
    )
    {
        /// Problem Dimension
        constexpr DimType DIM = get_stn<STN>::dim;
        constexpr DirType DIR = get_dim<DIM>::highest_dir;
        constexpr int GN = get_stn<STN>::ghosts;

        // output container
        std::list < Problem<VAR, STN, Field...> > problems;

        // one flag for each patch face to store if it is a boundary face
        std::array<bool, 2 * DIM> flags;
        flags.fill ( false );

        // start partitioning from the highest_dir detail::partition_range iterates on lower directions
        detail::partition_range < DIR, GN, 0, Field... >::template exec<VAR, STN> ( labels..., subproblems_label, material, patch->getLevel(), DWInterface<VAR, DIM>::get_low ( patch ), DWInterface<VAR, DIM>::get_high ( patch ), {}, detail::get_bcs<DIM, Field>::exec ( patch, 0, 0, labels, c2f, flags )..., flags, problems );
        return problems;
    }

}; // struct BCInterface

} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_BCInterface_h
