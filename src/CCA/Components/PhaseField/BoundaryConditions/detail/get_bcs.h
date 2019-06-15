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
 * @file CCA/Components/PhaseField/BoundaryConditions/detail/get_bcs.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_detail_get_bcs_h
#define Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_detail_get_bcs_h

#include <CCA/Components/PhaseField/Util/Definitions.h>
#include <CCA/Components/PhaseField/Util/Expressions.h>
#include <CCA/Components/PhaseField/BoundaryConditions/detail/get_bc.h>

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Get boundary conditions on multiple faces static functor
 *
 * @tparam DIM problem dimension
 * @tparam Field type of field (ScalarField < T > or VectorField < T, N >)
 */
template<DimType DIM, typename Field>
class get_bcs
{
    /**
     * @brief Execute the functor (internal indexed implementation)
     *
     * For each face F call get_bc and build an array of BCInfo for the variable
     * identified by the given label on the given patch
     *
     * @tparam F list of faces to check
     * @param patch grid patch to be checked
     * @param material problem material index
     * @param label variable label to check
     * @param c2f which fine/coarse interface conditions to use on each variable
     * @param[in,out] flags array of flags to check if any bc is applied to each one of faces
     * @return array of BCInfo
     */
    template < size_t... F >
    inline static std::array < BCInfo<Field>, get_dim<DIM>::face_end >
    exec (
        index_sequence<F...>,
        const Patch * patch,
        const int & child,
        const int & material,
        const typename Field::label_type & label,
        const std::map<std::string, FC> * c2f,
        std::array<bool, get_dim<DIM>::face_end> & flags
    )
    {
        return { get_bc<Field>::exec ( patch, ( Patch::FaceType ) F, child, material, label, c2f, flags[F] )... };
    }

public:
    /**
     * @brief Execute the functor
     *
     * For each face constuct the list of faces for the patch and call the
     * internal indexed implementation
     *
     * @tparam F list of faces to check
     * @param patch grid patch to be checked
     * @param child child index of face boundary condition (as per input file)
     * @param material problem material index
     * @param label variable label to check
     * @param c2f which fine/coarse interface conditions to use on each variable
     * @param[in,out] flags array of flags to check if any bc is applied to each one of faces
     * @return array of BCInfo
     */
    inline static std::array < BCInfo<Field>, get_dim<DIM>::face_end >
    exec (
        const Patch * patch,
        const int & child,
        const int & material,
        const typename Field::label_type & label,
        const std::map<std::string, FC> * c2f,
        std::array<bool, get_dim<DIM>::face_end> & flags
    )
    {
        return exec ( make_index_sequence < get_dim<DIM>::face_end > {}, patch, child, material, label, c2f, flags );
    }
};

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_detail_get_bcs_h
