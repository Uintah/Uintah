/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#ifndef Packages_Uintah_CCA_Components_Heat_CCHeat2D_h
#define Packages_Uintah_CCA_Components_Heat_CCHeat2D_h

#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <Core/Grid/Ghost.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/BlockRange.hpp>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/Util/DebugStream.h>
#include <CCA/Components/Heat/TimeScheme.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SimulationInterface.h>
#include <CCA/Ports/SolverInterface.h>

#define DBG_MATRIX

namespace Uintah
{

class CCHeat2D :
    public UintahParallelComponent,
    public SimulationInterface
{
protected:
    using ConstVariable = constCCVariable<double>;
    using Variable = CCVariable<double>;
    using ConstMatrix = constCCVariable<Stencil7>;
    using Matrix = CCVariable<Stencil7>;
#ifdef UINTAH_ENABLE_KOKKOS
    using ConstVariableView = KokkosView3<double const>;
    using VariableView = KokkosView3<double>;
    using MatrixView = KokkosView3<Stencil7>;
#else
    using ConstVariableView = ConstVariable & ;
    using VariableView = Variable & ;
    using MatrixView = Matrix & ;
#endif

    static constexpr Patch::FaceType start_face = Patch::xminus;
    static constexpr Patch::FaceType end_face   = Patch::yplus;

    static constexpr inline ConstVariableView get_view ( ConstVariable & var );
    static constexpr inline VariableView get_view ( Variable & var );
    static constexpr inline MatrixView get_view ( Matrix & var );

    static inline IntVector get_low ( Patch const * patch );
    static inline IntVector get_high ( Patch const * patch );
    static inline BlockRange get_range ( Patch const * patch );
    static inline BlockRange get_inner_range ( Patch const * patch );
    static inline BlockRange get_face_range ( Patch const * patch, Patch::FaceType face );
    static inline void get_bc ( Patch const * patch, Patch::FaceType face, const int child, const std::string & desc, const int mat_id, BlockRange & range, std::string & bc_kind, double & bc_value );
    static inline Point get_position ( Patch const * patch, IntVector const & i );
    static inline bool is_internal ( const Patch * patch, const IntVector & i, int d, const std::string bc_kind[4] );

    static inline double dx ( int i, int j, int k, Patch const * patch, ConstVariableView psi );
    static inline double dy ( int i, int j, int k, Patch const * patch, ConstVariableView psi );
    static inline double dxx ( int i, int j, int k, Patch const * patch, ConstVariableView psi );
    static inline double dyy ( int i, int j, int k, Patch const * patch, ConstVariableView psi );
    static inline double laplacian ( int i, int j, int k, Patch const * patch, ConstVariableView psi );

protected:
    double const tol = 1.e-6;

    DebugStream dbg_out1, dbg_out2, dbg_out3, dbg_out4;

    double delt, alpha, r0, gamma;
    TimeScheme time_scheme;

    const VarLabel * u_label;
    const VarLabel * matrix_label, * rhs_label;
#ifdef DBG_MATRIX
#   define MATRIX_VIEW_A MatrixView A, VariableView Ap, VariableView Aw, VariableView Ae, VariableView As, VariableView An, VariableView Ab, VariableView At
#   define REF_A &A, &Ap, &Aw, &Ae, &As, &An, &Ab, &At
#   define GET_VIEW_A get_view ( A ), get_view ( Ap ), get_view ( Aw ), get_view ( Ae ), get_view ( As ), get_view ( An ), get_view ( Ab ), get_view ( At )
    const VarLabel * Ap_label, * Aw_label, * Ae_label, * As_label, * An_label, * Ab_label, * At_label;
#else
#   define MATRIX_VIEW_A MatrixView A
#   define REF_A &A
#   define GET_VIEW_A get_view ( A )
#endif

    SimulationStateP state;
    SolverInterface * solver;
    SolverParameters * solver_parameters;

public:
    CCHeat2D ( const ProcessorGroup * myworld, int verbosity = 0 );
    virtual ~CCHeat2D();

protected:
    CCHeat2D ( CCHeat2D const & ) = delete;
    CCHeat2D & operator= ( CCHeat2D const & ) = delete;

    virtual void problemSetup ( ProblemSpecP const & params, ProblemSpecP const & restart_prob_spec, GridP & grid, SimulationStateP & state ) override;
    virtual void scheduleInitialize ( LevelP const & level, SchedulerP & sched ) override;
    virtual void scheduleRestartInitialize ( LevelP const & /*level*/, SchedulerP & /*sched*/ ) override {} // TODO
    virtual void scheduleComputeStableTimestep ( LevelP const & level, SchedulerP & sched ) override;
    virtual void scheduleTimeAdvance ( LevelP const & level, SchedulerP & ) override;

    void scheduleTimeAdvance_forward_euler ( LevelP const & level, SchedulerP & );
    void scheduleTimeAdvance_backward_euler_assemble ( LevelP const & level, SchedulerP & );
    void scheduleTimeAdvance_backward_euler_solve ( LevelP const & level, SchedulerP & );

    void task_initialize ( ProcessorGroup const * myworld, PatchSubset const * patches, MaterialSubset const * matls, DataWarehouse * dw_old, DataWarehouse * dw_new );
    void task_compute_stable_timestep ( ProcessorGroup const * myworld, PatchSubset const * patches, MaterialSubset const * matls, DataWarehouse * dw_old, DataWarehouse * dw_new );
    void task_farward_euler_time_advance ( ProcessorGroup const * myworld, PatchSubset const * patches, MaterialSubset const * matls, DataWarehouse * dw_old, DataWarehouse * dw_new );
    void task_backward_euler_assemble ( ProcessorGroup const * myworld, PatchSubset const * patches, MaterialSubset const * matls, DataWarehouse * dw_old, DataWarehouse * dw_new );

    void initialize ( int i, int j, int k, Patch const * patch, VariableView u );
    void forward_euler_time_advance ( int i, int j, int k, Patch const * patch, ConstVariableView u_old, VariableView u_new );
    void forward_euler_time_advance ( int i, int j, int k, Patch const * patch, ConstVariableView u_old, VariableView u_new, Patch::FaceType face, const std::string bc_kind[4], const double bc_value[4] );
    void backward_euler_assemble ( int i, int j, int k, Patch const * patch, ConstVariableView u, MATRIX_VIEW_A, VariableView b );
    void backward_euler_assemble ( int i, int j, int k, Patch const * patch, ConstVariableView u, MATRIX_VIEW_A, VariableView b, Patch::FaceType face, const std::string bc_kind[4], const double bc_value[4] );
}; // class CCHeat2D

} // namespace Uintah

constexpr inline Uintah::CCHeat2D::ConstVariableView Uintah::CCHeat2D::get_view ( ConstVariable & var )
{
#ifdef UINTAH_ENABLE_KOKKOS
    return var.getKokkosView();
#else
    return var;
#endif
}

constexpr inline Uintah::CCHeat2D::VariableView Uintah::CCHeat2D::get_view ( Variable & var )
{
#ifdef UINTAH_ENABLE_KOKKOS
    return var.getKokkosView();
#else
    return var;
#endif
}

constexpr inline Uintah::CCHeat2D::MatrixView Uintah::CCHeat2D::get_view ( Matrix & var )
{
#ifdef UINTAH_ENABLE_KOKKOS
    return var.getKokkosView();
#else
    return var;
#endif
}

inline Uintah::Point Uintah::CCHeat2D::get_position ( Patch const * patch, IntVector const & i )
{
    return patch->getCellPosition ( i );
}

inline Uintah::IntVector Uintah::CCHeat2D::get_low ( Patch const * patch )
{
    return patch->getCellLowIndex();
}

inline Uintah::IntVector Uintah::CCHeat2D::get_high ( Patch const * patch )
{
    return patch->getCellHighIndex();
}

inline Uintah::BlockRange Uintah::CCHeat2D::get_range ( Patch const * patch )
{
    return { get_low ( patch ), get_high ( patch ) };
}

inline Uintah::BlockRange Uintah::CCHeat2D::get_inner_range ( Patch const * patch )
{
    return { get_low ( patch ) + IntVector (
                 patch->getBCType ( Patch::xminus ) == Patch::Neighbor ? 0 : 1,
                 patch->getBCType ( Patch::yminus ) == Patch::Neighbor ? 0 : 1,
                 0
             ),
             get_high ( patch ) - IntVector (
                 patch->getBCType ( Patch::xplus ) == Patch::Neighbor ? 0 : 1,
                 patch->getBCType ( Patch::yplus ) == Patch::Neighbor ? 0 : 1,
                 0
             )
           };
}

inline Uintah::BlockRange Uintah::CCHeat2D::get_face_range ( Patch const * patch, Patch::FaceType face )
{
    IntVector l, h;
    patch->getFaceCells ( face, 0, l, h );
    return { l, h };
}

inline void Uintah::CCHeat2D::get_bc ( const Patch * patch, Patch::FaceType face, const int child, const std::string & desc, const int mat_id, BlockRange & range, std::string & bc_kind, double & bc_value )
{
    range = get_face_range ( patch, face );
    bc_kind = "NotSet";
    bc_value = 0.;

    if ( desc == "zeroNeumann" )
    {
        bc_kind = "zeroNeumann";
        return;
    }

    if ( patch->getBCType ( face ) == Patch::Coarse )
    {
        bc_kind = "FineCoarseInterface";
        return;
    }

    if ( patch->getBCType ( face ) == Patch::Neighbor )
    {
        bc_kind = "Neighbor";
        return;
    }

    const BCDataArray * bcd = patch->getBCDataArray ( face );
    if ( !bcd ) return;

    const BoundCondBase * bc;
    if ( ( bc = bcd->getBoundCondData ( mat_id, desc, child ) ) )
    {
        const BoundCond<double> * bcs = dynamic_cast<const BoundCond<double> *> ( bc );;
        if ( bcs )
        {
            bc_kind  = bcs->getBCType();
            bc_value = bcs->getValue();
        }
    }
    else if ( ( bc = bcd->getBoundCondData ( mat_id, "Symmetric", child ) ) )
    {
        if ( bc->getBCType() == "symmetry" ) bc_kind  = "symmetry";
    }
    if ( bc ) delete bc;
    return;
}

bool Uintah::CCHeat2D::is_internal ( const Patch * patch, const IntVector & i, int d, const std::string bc_kind[4] )
{
    IntVector l = get_low ( patch );
    IntVector h = get_high ( patch ) - IntVector ( 1, 1, 1 );

    for ( int k = 0; k < d; ++k )
    {
        if ( bc_kind[2 * k] == "Dirichlet" && i[k] <= l[k] ) return false;
        if ( bc_kind[2 * k + 1] == "Dirichlet" && i[k] >= h[k] ) return false;
    }
    return true;
}

inline double Uintah::CCHeat2D::dx ( int i, int j, int k, Patch const * patch, ConstVariableView psi )
{
    Vector const d ( patch->getLevel()->dCell() );
    return ( psi ( i + 1, j, k ) - psi ( i - 1, j, k ) ) / ( 2. * d.x() );
}

inline double Uintah::CCHeat2D::dy ( int i, int j, int k, Patch const * patch, ConstVariableView psi )
{
    Vector const d ( patch->getLevel()->dCell() );
    return ( psi ( i, j + 1, k ) - psi ( i, j - 1, k ) ) / ( 2. * d.y() );
}

inline double Uintah::CCHeat2D::dxx ( int i, int j, int k, Patch const * patch, ConstVariableView psi )
{
    Vector const d ( patch->getLevel()->dCell() );
    return ( psi ( i + 1, j, k ) + psi ( i - 1, j, k ) - 2. * psi ( i, j, k ) ) / ( d.x() * d.x() );
}

inline double Uintah::CCHeat2D::dyy ( int i, int j, int k, Patch const * patch, ConstVariableView psi )
{
    Vector const d ( patch->getLevel()->dCell() );
    return ( psi ( i, j + 1, k ) + psi ( i, j - 1, k ) - 2. * psi ( i, j, k ) ) / ( d.y() * d.y() );
}

inline double Uintah::CCHeat2D::laplacian ( int i, int j, int k, Patch const * patch, ConstVariableView psi )
{
    return dxx ( i, j, k, patch, psi ) + dyy ( i, j, k, patch, psi );
}

#endif // Packages_Uintah_CCA_Components_Heat_CCHeat2D_h
