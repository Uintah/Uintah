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
 * @file CCA/Components/PhaseField/BoundaryConditions/detail/bc_fd.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_detail_bc_fd_h
#define Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_detail_bc_fd_h

#include <CCA/Components/PhaseField/Util/Definitions.h>

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Abstract finite-differences scheme for variables at grid boundary
 *
 * To implement a new boundary-condition it should be sufficient to
 * define a new BC and code the relevant implementations of this class
 * (for amr simulations also the corresponding bc_fd implementation for
 * fine/coarse interfaces must be provided)
 *
 * @implements basic_fd_view < Field, STN >
 *
 * @remark the method implemented in this class are ment to be called only by
 * bcs_basic_fd_view to compute bc dependent quantities only
 *
 * @tparam Field type of field (ScalarField < T > or VectorField < T, N >)
 * @tparam STN finite-difference stencil
 * @tparam F patch face on which bc is to be applied
 * @tparam B type of boundary conditions
 * @tparam GN Number of ghosts required
 * @tparam C2F Fine/Coarse Interface conditions
 * @tparam CNEW Wheter to use new datawarehouse for retrieving corse grid data
 */
template <typename Field, StnType STN, VarType VAR, Patch::FaceType F, BC B, int GN, FC C2F = FC::None, bool CNEW = false> class bc_fd;
_DOXYBDY (
    /// Non const type of the field value
    using V;

    /**
     * @brief First order derivative
     *
     * First order derivative along DIR at index id
     *
     * @tparam DIR Direction along with derivative is approximated
     * @param id index where to evaluate the finite-difference
     * @return approximated value at id
     */
    template <DirType DIR> inline V d ( const IntVector & id ) const = 0;

    /**
     * @brief Second order derivative
     *
     * Second order derivative along DIR at index id
     *
     * @tparam DIR Direction along with derivative is approximated
     * @param id index where to evaluate the finite-difference
     * @return approximated value at id
     */
    template <DirType DIR> inline V d2 ( const IntVector & id ) const = 0;
             );

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#include <CCA/Components/PhaseField/BoundaryConditions/detail/bc_fd_Dirichlet_G1_CC.h>
#include <CCA/Components/PhaseField/BoundaryConditions/detail/bc_fd_Dirichlet_G1_NC.h>
#include <CCA/Components/PhaseField/BoundaryConditions/detail/bc_fd_Neumann_G1_CC.h>
#include <CCA/Components/PhaseField/BoundaryConditions/detail/bc_fd_Neumann_G1_NC.h>
#include <CCA/Components/PhaseField/BoundaryConditions/detail/bc_fd_FineCoarseInterface_G1.h>
#include <CCA/Components/PhaseField/BoundaryConditions/detail/bc_fd_FineCoarseInterface_G1_FCSimple.h>
#include <CCA/Components/PhaseField/BoundaryConditions/detail/bc_fd_FineCoarseInterface_G1_FCLinear.h>
#include <CCA/Components/PhaseField/BoundaryConditions/detail/bc_fd_FineCoarseInterface_G1_FCBilinear.h>

#endif //Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_detail_bc_fd_h
