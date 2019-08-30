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
 * @file CCA/Components/PhaseField/BoundaryConditions/detail/partition_range.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_detail_partition_range_h
#define Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_detail_partition_range_h

#include <CCA/Components/PhaseField/DataTypes/Problem.h>

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Partition range static functor
 *
 * @tparam DIR Direction to be partitioned
 * @tparam GN Number of ghosts required
 * @tparam DONE Number of directions already partitioned (for internal checks)
 * @tparam Field list of type of fields (ScalarField < T > or VectorField < T, N >)
 */
template <DirType DIR, int GN, int DONE, typename... Field>
struct partition_range
{
    /**
     * @brief Execute the functor
     *
     * Check if boundary conditions are applied to DIR plus and minus faces and
     * partition the range accordingly, then iterate on each partition on lower
     * directions
     * Problems are instantiated only on lowest direction
     *
     * @tparam VAR type of variable representation
     * @tparam STN finite-difference stencil
     * @param labels list of labels for each variable of the problem
     * @param subproblems_label label for subproblems in the DW (required by AMR boundary Problems)
     * @param material index of material in the DataWarehouse
     * @param level grid level to be partitions
     * @param low lower bound of the region to partition
     * @param high higher bound of the region to partition
     * @param faces list of faces on which boundary conditions are applied
     * @param bcs list of arrays of BC info for each one of face of the patch for each one of the problem labels
     * @param flags array of flags to check if any bc is applied to each one of faces
     * @param[out] problems list of subproblems to populate
     */
    template < VarType VAR, StnType STN >
    static void
    exec (
        const typename Field::label_type & ... labels,
        const VarLabel * subproblems_label,
        int material,
        const Level * level,
        IntVector low,
        IntVector high,
        std::list<Patch::FaceType> faces,
        const std::array < BCInfo<Field>, 2 * get_stn<STN>::dim > & ... bcs,
        std::array < bool, 2 * get_stn<STN>::dim > & flags,
        std::list < Problem<VAR, STN, Field...> > & problems
    )
    {
        static_assert ( get_dim < ( DimType ) ( get_stn<STN>::dim - DONE ) >::highest_dir == DIR, "cannot partition along DIR if higher directions have not been processed yet" );

        if ( flags [ get_dir<DIR>::plus_face ] )
        {
            IntVector l {low};
            l[DIR] = high[DIR] - GN;
            auto f = faces;
            f.emplace_front ( get_dir<DIR>::plus_face );
            partition_range < get_dir<DIR>::lower, GN, DONE + 1, Field... >::template exec<VAR, STN> ( labels..., subproblems_label, material, level, l, high, f, bcs..., flags, problems );
            high[DIR] -= GN;
        }

        if ( flags [ get_dir<DIR>::minus_face ] )
        {
            IntVector h {high};
            h[DIR] = low[DIR] + GN;
            auto f = faces;
            f.emplace_front ( get_dir<DIR>::minus_face );
            partition_range < get_dir<DIR>::lower, GN, DONE + 1, Field... >::template exec<VAR, STN> ( labels..., subproblems_label, material, level, low, h, f, bcs..., flags, problems );
            low[DIR] += GN;
        }

        partition_range < get_dir<DIR>::lower, GN, DONE + 1, Field... >::template exec<VAR, STN> ( labels..., subproblems_label, material, level, low, high, faces, bcs..., flags, problems );
    }
};

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#include <CCA/Components/PhaseField/BoundaryConditions/detail/partition_range_X.h>

#endif // Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_detail_partition_range_h
