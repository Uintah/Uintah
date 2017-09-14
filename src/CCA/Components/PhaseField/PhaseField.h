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

#ifndef Packages_Uintah_CCA_Components_PhaseField_PhaseField_h
#define Packages_Uintah_CCA_Components_PhaseField_PhaseField_h

#include <Core/Grid/Ghost.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/BlockRange.hpp>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/Util/DebugStream.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SimulationInterface.h>

namespace Uintah
{

class TypeDescription;

namespace PF
{
enum VariableType { CellCentered = Ghost::AroundCells, NodeCentered = Ghost::AroundNodes };

template<VariableType VariableType>
using ConstVariable = typename std::conditional< VariableType == PF::CellCentered, constCCVariable<double>, constNCVariable<double> >::type;

template<VariableType VariableType>
using Variable = typename std::conditional < VariableType == PF::CellCentered, CCVariable<double>, NCVariable<double> >::type;

#ifdef UINTAH_ENABLE_KOKKOS

template<VariableType VariableType>
using ConstView = KokkosView3<double const>;

template<VariableType VariableType>
using View = KokkosView3<double>;

#else

template<VariableType VariableType>
using ConstView = ConstVariable<VariableType> & ;

template<VariableType VariableType>
using View = Variable<VariableType> & ;

#endif

template <int n>
struct static_false
{
    enum { value = n != n };
};

template<VariableType VariableType>
Point get_position ( Patch const * patch, IntVector const & i )
{
    static_assert ( static_false<VariableType>::value, "get_position<VariableType> not implemented" );
    return { NAN, NAN, NAN };
}

template<>
Point get_position<CellCentered> ( Patch const * patch, IntVector const & i )
{
    return patch->getCellPosition ( i );
}

template<>
Point get_position<NodeCentered> ( Patch const * patch, IntVector const & i )
{
    return patch->getNodePosition ( i );
}

template<VariableType VariableType>
IntVector get_low ( Patch const * patch )
{
    static_assert ( static_false<VariableType>::value, "get_low<VariableType> not implemented" );
    return { 0, 0, 0 };
}

template<>
IntVector get_low<CellCentered> ( Patch const * patch )
{
    return patch->getCellLowIndex();
}

template<>
IntVector get_low<NodeCentered> ( Patch const * patch )
{
    return patch->getNodeLowIndex();
}

template<VariableType VariableType>
IntVector get_high ( Patch const * patch )
{
    static_assert ( static_false<VariableType>::value, "get_high<VariableType> not implemented" );
    return { 0, 0, 0 };
}

template<>
IntVector get_high<CellCentered> ( Patch const * patch )
{
    return patch->getCellHighIndex();
}

template<>
IntVector get_high<NodeCentered> ( Patch const * patch )
{
    return patch->getNodeHighIndex();
}

template<VariableType VariableType>
BlockRange get_range ( Patch const * patch )
{
    return { get_low<VariableType> ( patch ), get_high<VariableType> ( patch ) };
}

template<VariableType VariableType, int NumGhosts, int Dimension>
typename std::enable_if < NumGhosts != 1 || ( Dimension != 2 && Dimension != 3 ), BlockRange >::type get_inner_range ( Patch const * patch )
{
    static_assert ( static_false<VariableType>::value, "get_inner_range<VariableType, NumGhosts, Dimension> not implemented" );
    return { IntVector ( 0, 0, 0 ), IntVector ( 0, 0, 0 ) };
}

template<VariableType VariableType, int NumGhosts, int Dimension>
typename std::enable_if < NumGhosts == 1 && Dimension == 2, BlockRange >::type get_inner_range ( Patch const * patch )
{
    return { get_low<VariableType> ( patch ) + IntVector (
                 patch->getBCType ( Patch::xminus ) == Patch::Neighbor ? 0 : 1,
                 patch->getBCType ( Patch::yminus ) == Patch::Neighbor ? 0 : 1,
                 0
             ),
             get_high<VariableType> ( patch ) - IntVector (
                 patch->getBCType ( Patch::xplus ) == Patch::Neighbor ? 0 : 1,
                 patch->getBCType ( Patch::yplus ) == Patch::Neighbor ? 0 : 1,
                 0
             )
           };
}

template<VariableType VariableType, int NumGhosts, int Dimension>
typename std::enable_if < NumGhosts == 1 && Dimension == 3, BlockRange >::type get_inner_range ( Patch const * patch )
{
    return { get_low<VariableType> ( patch ) + IntVector (
                 patch->getBCType ( Patch::xminus ) == Patch::Neighbor ? 0 : 1,
                 patch->getBCType ( Patch::yminus ) == Patch::Neighbor ? 0 : 1,
                 patch->getBCType ( Patch::zminus ) == Patch::Neighbor ? 0 : 1
             ),
             get_high<VariableType> ( patch ) - IntVector (
                 patch->getBCType ( Patch::xplus ) == Patch::Neighbor ? 0 : 1,
                 patch->getBCType ( Patch::yplus ) == Patch::Neighbor ? 0 : 1,
                 patch->getBCType ( Patch::zplus ) == Patch::Neighbor ? 0 : 1
             )
           };
}

template<VariableType VariableType>
BlockRange get_face_range ( Patch const * patch, Patch::FaceType face )
{
    static_assert ( static_false<VariableType>::value, "get_face_range<VariableType> not implemented" );
    return { IntVector ( 0, 0, 0 ), IntVector ( 0, 0, 0 ) };
}

template<>
BlockRange get_face_range<CellCentered> ( Patch const * patch, Patch::FaceType face )
{
    IntVector l, h;
    patch->getFaceCells ( face, 0, l, h );
    return { l, h };
}

template<>
BlockRange get_face_range<NodeCentered> ( Patch const * patch, Patch::FaceType face )
{
    IntVector l, h;
    patch->getFaceNodes ( face, 0, l, h );
    return { l, h };
}

template<VariableType VariableType, int NumGhosts>
typename std::enable_if < NumGhosts != 1, double >::type dx ( int i, int j, int k, Patch const * patch, ConstView<VariableType> psi )
{
    static_assert ( static_false<VariableType>::value, "dx<VariableType, NumGhosts> not implemented" );
    return NAN;
}

template<VariableType VariableType, int NumGhosts>
typename std::enable_if<NumGhosts == 1, double>::type dx ( int i, int j, int k, Patch const * patch, ConstView<VariableType> psi )
{
    Vector const d ( patch->getLevel()->dCell() );
    return ( psi ( i + 1, j, k ) - psi ( i - 1, j, k ) ) / ( 2. * d.x() );
}

template<VariableType VariableType, int NumGhosts>
typename std::enable_if < NumGhosts != 1, double >::type dy ( int i, int j, int k, Patch const * patch, ConstView<VariableType> psi )
{
    static_assert ( static_false<VariableType>::value, "dy<VariableType, NumGhosts> not implemented" );
    return NAN;
}

template<VariableType VariableType, int NumGhosts>
typename std::enable_if<NumGhosts == 1, double>::type dy ( int i, int j, int k, Patch const * patch, ConstView<VariableType> psi )
{
    Vector const d ( patch->getLevel()->dCell() );
    return ( psi ( i, j + 1, k ) - psi ( i, j - 1, k ) ) / ( 2. * d.y() );
}

template<VariableType VariableType, int NumGhosts>
typename std::enable_if < NumGhosts != 1, double >::type dz ( int i, int j, int k, Patch const * patch, ConstView<VariableType> psi )
{
    static_assert ( static_false<VariableType>::value, "dz<VariableType, NumGhosts> not implemented" );
    return NAN;
}

template<VariableType VariableType, int NumGhosts>
typename std::enable_if<NumGhosts == 1, double>::type dz ( int i, int j, int k, Patch const * patch, ConstView<VariableType> psi )
{
    Vector const d ( patch->getLevel()->dCell() );
    return ( psi ( i, j, k + 1 ) - psi ( i, j, k - 1 ) ) / ( 2. * d.z() );
}

template<VariableType VariableType, int NumGhosts>
typename std::enable_if < NumGhosts != 1, double >::type dxx ( int i, int j, int k, Patch const * patch, ConstView<VariableType> psi )
{
    static_assert ( static_false<VariableType>::value, "dxx<VariableType, NumGhosts> not implemented" );
    return NAN;
}

template<VariableType VariableType, int NumGhosts>
typename std::enable_if<NumGhosts == 1, double>::type dxx ( int i, int j, int k, Patch const * patch, ConstView<VariableType> psi )
{
    Vector const d ( patch->getLevel()->dCell() );
    return ( psi ( i + 1, j, k ) + psi ( i - 1, j, k ) - 2. * psi ( i, j, k ) ) / ( d.x() * d.x() );
}

template<VariableType VariableType, int NumGhosts>
typename std::enable_if < NumGhosts != 1, double >::type dyy ( int i, int j, int k, Patch const * patch, ConstView<VariableType> psi )
{
    static_assert ( static_false<VariableType>::value, "dyy<VariableType, NumGhosts> not implemented" );
    return NAN;
}

template<VariableType VariableType, int NumGhosts>
typename std::enable_if<NumGhosts == 1, double>::type dyy ( int i, int j, int k, Patch const * patch, ConstView<VariableType> psi )
{
    Vector const d ( patch->getLevel()->dCell() );
    return ( psi ( i, j + 1, k ) + psi ( i, j - 1, k ) - 2. * psi ( i, j, k ) ) / ( d.y() * d.y() );
}

template<VariableType VariableType, int NumGhosts>
typename std::enable_if < NumGhosts != 1, double >::type dzz ( int i, int j, int k, Patch const * patch, ConstView<VariableType> psi )
{
    static_assert ( static_false<VariableType>::value, "dzz<VariableType, NumGhosts> not implemented" );
    return NAN;
}

template<VariableType VariableType, int NumGhosts>
typename std::enable_if<NumGhosts == 1, double>::type dzz ( int i, int j, int k, Patch const * patch, ConstView<VariableType> psi )
{
    Vector const d ( patch->getLevel()->dCell() );
    return ( psi ( i, j, k + 1 ) + psi ( i, j, k - 1 ) - 2. * psi ( i, j, k ) ) / ( d.z() * d.z() );
}

template<VariableType VariableType, int NumGhosts, int Dimension>
typename std::enable_if < NumGhosts == 1 && Dimension == 2, double >::type laplacian ( int i, int j, int k, Patch const * patch, ConstView<VariableType> psi )
{
    return dxx<VariableType, NumGhosts> ( i, j, k, patch, psi ) + dyy<VariableType, NumGhosts> ( i, j, k, patch, psi );
}

template<VariableType VariableType, int NumGhosts, int Dimension>
typename std::enable_if < NumGhosts == 1 && Dimension == 3, double >::type laplacian ( int i, int j, int k, Patch const * patch, ConstView<VariableType> psi )
{
    return dxx<VariableType, NumGhosts> ( i, j, k, patch, psi ) + dyy<VariableType, NumGhosts> ( i, j, k, patch, psi ) + dzz<VariableType, NumGhosts> ( i, j, k, patch, psi );
}

template<VariableType VariableType, int NumGhosts, int Dimension>
typename std::enable_if < NumGhosts != 1 || ( Dimension != 2 && Dimension != 3 ), double >::type laplacian ( int i, int j, int k, Patch const * patch, ConstView<VariableType> psi )
{
    static_assert ( static_false<VariableType>::value, "laplacian<VariableType, NumGhosts, Dimension> not implemented" );
    return NAN;
}

}

template<int N> struct factorial
{
    enum { value = N * factorial < N - 1 >::value };
};

template<> struct factorial<0>
{
    enum { value = 1 };
};

template<int N, int R> struct combinations
{
    enum { value = factorial<N>::value / ( factorial<R>::value * factorial < N - R >::value ) };
};

std::ostream& operator << ( std::ostream& stream, const BlockRange& range )
{
    return stream << "(" << range.begin ( 0 ) << "," <<  range.begin ( 1 ) <<"," <<  range.begin ( 2 ) <<") - (" << range.end ( 0 ) <<"," << range.end ( 1 ) <<"," << range.end ( 2 ) << ")";
}


template<PF::VariableType VariableType, int NumGhosts, int Dimension>
class PhaseField :
    public UintahParallelComponent,
    public SimulationInterface
{
protected:
    constexpr static Patch::FaceType start_face = static_cast<Patch::FaceType> ( 0 );
    constexpr static Patch::FaceType end_face = static_cast<Patch::FaceType> ( Dimension * 2 );
    constexpr static int b_size = combinations<Dimension, 2>::value;
    constexpr static int xy = 0;
    constexpr static int xz = 1;
    constexpr static int yz = 2;

    using ConstVariable = PF::ConstVariable<VariableType>;
    using Variable = PF::Variable<VariableType>;
    using ConstView = PF::ConstView<VariableType>;
    using View = PF::View<VariableType>;
    template < int N > using ConstVariables = std::array < ConstVariable, N >;
    template < int N > using Variables = std::array < Variable, N >;

#ifdef UINTAH_ENABLE_KOKKOS
    template < int N > using ConstViews = std::array < ConstView, N >;
    template < int N > using Views = std::array < View, N >;

    static constexpr ConstView get_view ( ConstVariable & var )
    {
        return var.getKokkosView();
    }

    static constexpr View get_view ( Variable & var )
    {
        return var.getKokkosView();
    }

    template < int N >6
    static constexpr ConstViews<N> get_views ( ConstVariables<N> & vars )
    {
        ConstViews<N> views;
        for ( int d = 0; d < N; ++d )
            views[d] = get_view ( vars[d] );
        return views;
    }

    template < int N >
    static constexpr Views<N> get_views ( Variables<N> & vars )
    {
        Views<N> views;
        for ( int d = 0; d < N; ++d )
            views[d] = get_view ( vars[d] );
        return views;
    }
#else
    template < int N > using ConstViews = ConstVariables<N> & ;
    template < int N > using Views = Variables<N> & ;

    static constexpr ConstView get_view ( ConstVariable & var )
    {
        return var;
    }
    static constexpr View get_view ( Variable & var )
    {
        return var;
    }

    template < int N >
    static constexpr ConstViews<N> get_views ( ConstVariables<N> & vars )
    {
        return vars;
    }

    template < int N >
    static constexpr Views<N> get_views ( Variables<N> & vars )
    {
        return vars;
    }
#endif


    template <typename... Args>
    static Point get_position ( Args && ... args )
    {
        return PF::get_position<VariableType> ( std::forward<Args> ( args )... );
    }

    template <typename... Args>
    static IntVector get_low ( Args && ... args )
    {
        return PF::get_low<VariableType> ( std::forward<Args> ( args )... );
    }

    template <typename... Args>
    static IntVector get_high ( Args && ... args )
    {
        return PF::get_high<VariableType> ( std::forward<Args> ( args )... );
    }

    template <typename... Args>
    static BlockRange get_range ( Args && ... args )
    {
        return PF::get_range<VariableType> ( std::forward<Args> ( args )... );
    }

    template <typename... Args>
    static BlockRange get_inner_range ( Args && ... args )
    {
        return PF::get_inner_range<VariableType, NumGhosts, Dimension> ( std::forward<Args> ( args )... );
    }

    template <typename... Args>
    static BlockRange get_face_range ( Args && ... args )
    {
        return PF::get_face_range<VariableType> ( std::forward<Args> ( args )... );
    }

    template <typename... Args>
    static double dx ( Args && ... args )
    {
        return PF::dx<VariableType, NumGhosts> ( std::forward<Args> ( args )... );
    }

    template <typename... Args>
    static double dy ( Args && ... args )
    {
        return PF::dy<VariableType, NumGhosts> ( std::forward<Args> ( args )... );
    }

    template <typename... Args>
    static double dz ( Args && ... args )
    {
        return PF::dz<VariableType, NumGhosts> ( std::forward<Args> ( args )... );
    }

    template <typename... Args>
    static double dxx ( Args && ... args )
    {
        return PF::dxx<VariableType, NumGhosts> ( std::forward<Args> ( args )... );
    }

    template <typename... Args>
    static double dyy ( Args && ... args )
    {
        return PF::dyy<VariableType, NumGhosts> ( std::forward<Args> ( args )... );
    }

    template <typename... Args>
    static double dzz ( Args && ... args )
    {
        return PF::dzz<VariableType, NumGhosts> ( std::forward<Args> ( args )... );
    }

    template <typename... Args>
    static double laplacian ( Args && ... args )
    {
        return PF::laplacian<VariableType, NumGhosts, Dimension> ( std::forward<Args> ( args )... );
    }

    static const Ghost::GhostType GhostType = static_cast<Ghost::GhostType> ( VariableType );

    double const tol = 1.e-6;

    DebugStream dbg_out1, dbg_out2, dbg_out3, dbg_out4;

    double delt, alpha, r0, delta, epsilon, phi0, lambda, gamma_psi, gamma_u;
    VarLabel const * psi_label, * u_label;
    VarLabel const * grad_psi_norm2_label, * grad_psi_label[Dimension];
    VarLabel const * a_label, * a2_label, * b_label[b_size];

    SimulationStateP state;


public:
    PhaseField ( const ProcessorGroup * myworld, int verbosity = 0 );
    virtual ~PhaseField();

protected:
    PhaseField ( PhaseField const & ) = delete;
    PhaseField & operator= ( PhaseField const & ) = delete;

    virtual void problemSetup ( ProblemSpecP const & params, ProblemSpecP const & restart_prob_spec, GridP & grid, SimulationStateP & state ) override;
    virtual void scheduleInitialize ( LevelP const & level, SchedulerP & sched ) override;
    virtual void scheduleRestartInitialize ( LevelP const & level, SchedulerP & sched )
    {
        /*TODO*/
    };
    virtual void scheduleComputeStableTimestep ( LevelP const & level, SchedulerP & sched ) override;
    virtual void scheduleTimeAdvance ( LevelP const & level, SchedulerP & ) override;

protected:
    virtual void task_time_advance_grad_psi_requires ( LevelP const & level, Task * task );
    virtual void task_time_advance_grad_psi_computes ( Task * task );
    virtual void task_time_advance_anisotropy_terms_requires ( LevelP const & level, Task * task );
    virtual void task_time_advance_anisotropy_terms_computes ( Task * task );
    virtual void task_time_advance_current_solution_requires ( LevelP const & level, Task * task );
    virtual void task_time_advance_current_solution_computes ( Task * task );

    void task_initialize ( ProcessorGroup const * myworld, PatchSubset const * patches, MaterialSubset const * matls, DataWarehouse * dw_old, DataWarehouse * dw_new );
    void task_compute_stable_timestep ( ProcessorGroup const * myworld, PatchSubset const * patches, MaterialSubset const * matls, DataWarehouse * dw_old, DataWarehouse * dw_new );
    void task_time_advance_grad_psi ( ProcessorGroup const * myworld, PatchSubset const * patches, MaterialSubset const * matls, DataWarehouse * dw_old, DataWarehouse * dw_new );
    void task_time_advance_anisotropy_terms ( ProcessorGroup const * myworld, PatchSubset const * patches, MaterialSubset const * matls, DataWarehouse * dw_old, DataWarehouse * dw_new );
    void task_time_advance_current_solution ( ProcessorGroup const * myworld, PatchSubset const * patches, MaterialSubset const * matls, DataWarehouse * dw_old, DataWarehouse * dw_new );

    void initialize_current_solution ( int i, int j, int k, Patch const * patch, View psi, View u );
    void time_advance_grad_psi ( int i, int j, int k, Patch const * patch, ConstView psi, Views<Dimension> grad_psi, View grad_psi_norm2 );
    void time_advance_grad_psi ( int i, int j, int k, Patch::FaceType face, Patch const * patch, ConstView psi, Views<Dimension> grad_psi, View grad_psi_norm2 );
    void time_advance_anisotropy_terms ( int i, int j, int k, Patch const * patch, ConstViews<Dimension> grad_psi, ConstView grad_psi_norm2, View a, View a2, Views<b_size> b );
    void time_advance_solution ( int i, int j, int k, Patch const * patch, ConstView psi_old, ConstView u_old, ConstViews<Dimension> grad_psi, ConstView grad_psi_norm2, ConstView a, ConstView a2, ConstViews<b_size> b, View psi_new, View u_new );
    void time_advance_solution ( int i, int j, int k, Patch::FaceType face, Patch const * patch, ConstView psi_old, ConstView u_old, ConstViews<Dimension> grad_psi, ConstView grad_psi_norm2, ConstView a, ConstView a2, ConstViews<b_size> b6, View psi_new, View u_new );
};

extern template class PhaseField <PF::CellCentered, 1, 2>;
extern template class PhaseField <PF::NodeCentered, 1, 2>;
extern template class PhaseField <PF::CellCentered, 1, 3>;
extern template class PhaseField <PF::NodeCentered, 1, 3>;

using CCPhaseField2D = PhaseField <PF::CellCentered, 1, 2>;
using NCPhaseField2D = PhaseField <PF::NodeCentered, 1, 2>;
using CCPhaseField3D = PhaseField <PF::CellCentered, 1, 3>;
using NCPhaseField3D = PhaseField <PF::NodeCentered, 1, 3>;

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
PhaseField<VariableType, NumGhosts, Dimension>::PhaseField ( ProcessorGroup const * myworld, int verbosity )
    : UintahParallelComponent ( myworld )
    , dbg_out1 ( "PhaseField", verbosity > 0 )
    , dbg_out2 ( "PhaseField", verbosity > 1 )
    , dbg_out3 ( "PhaseField", verbosity > 2 )
    , dbg_out4 ( "PhaseField", verbosity > 3 )
{
    psi_label = VarLabel::create ( "psi", Variable::getTypeDescription() );
    u_label = VarLabel::create ( "u", Variable::getTypeDescription() );
    grad_psi_norm2_label = VarLabel::create ( "grad_psi_norm2", Variable::getTypeDescription() );
    grad_psi_label[0] = VarLabel::create ( "psi_x", Variable::getTypeDescription() );
    grad_psi_label[1] = VarLabel::create ( "psi_y", Variable::getTypeDescription() );
    a_label = VarLabel::create ( "A", Variable::getTypeDescription() );
    a2_label = VarLabel::create ( "A2", Variable::getTypeDescription() );
    b_label[xy] = VarLabel::create ( "Bxy", Variable::getTypeDescription() );
    if ( Dimension > 2 )
    {
        grad_psi_label[2] = VarLabel::create ( "psi_z", Variable::getTypeDescription() );
        b_label[xz] = VarLabel::create ( "Bxz", Variable::getTypeDescription() );
        b_label[yz] = VarLabel::create ( "Byz", Variable::getTypeDescription() );
    }
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
PhaseField<VariableType, NumGhosts, Dimension>::~PhaseField()
{
    VarLabel::destroy ( psi_label );
    VarLabel::destroy ( u_label );
    VarLabel::destroy ( grad_psi_norm2_label );
    VarLabel::destroy ( a_label );
    VarLabel::destroy ( a2_label );
    for ( int d = 0; d < Dimension; ++d )
        VarLabel::destroy ( grad_psi_label[d] );
    for ( int d = 0; d < b_size; ++d )
        VarLabel::destroy ( b_label[d] );
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void PhaseField<VariableType, NumGhosts, Dimension>::problemSetup ( ProblemSpecP const & params, ProblemSpecP const & /*restart_prob_spec*/, GridP & /*grid*/, SimulationStateP & simulation_state )
{
    state = simulation_state;
    state->setIsLockstepAMR ( true );
    state->registerSimpleMaterial ( scinew SimpleMaterial() );

    ProblemSpecP phase_field = params->findBlock ( "PhaseField" );
    phase_field->require ( "delt", delt );
    phase_field->require ( "alpha", alpha );
    phase_field->require ( "R0", r0 );
    phase_field->require ( "Delta", delta );
    phase_field->require ( "epsilon", epsilon );
    phase_field->getWithDefault ( "gamma_psi", gamma_psi, 1. );
    phase_field->getWithDefault ( "gamma_u", gamma_u, 1. );

    lambda = alpha / 0.6267;
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void PhaseField<VariableType, NumGhosts, Dimension>::scheduleInitialize ( LevelP const & level, SchedulerP & sched )
{
    Task * task = scinew Task ( "PhaseField::task_initialize", this, &PhaseField::task_initialize );
    task->computes ( psi_label );
    task->computes ( u_label );
    task->computes ( grad_psi_norm2_label );
    task->computes ( a_label );
    task->computes ( a2_label );
    for ( int d = 0; d < Dimension; ++d )
        task->computes ( grad_psi_label[d] );
    for ( int d = 0; d < b_size; ++d )
        task->computes ( b_label[d] );
    sched->addTask ( task, level->eachPatch(), state->allMaterials() );
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void PhaseField<VariableType, NumGhosts, Dimension>::scheduleComputeStableTimestep ( LevelP const & level, SchedulerP & sched )
{
    Task * task = scinew Task ( "PhaseField::task_compute_stable_timestep", this, &PhaseField::task_compute_stable_timestep );
    task->computes ( state->get_delt_label(), level.get_rep() );
    sched->addTask ( task, level->eachPatch(), state->allMaterials() );
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void PhaseField<VariableType, NumGhosts, Dimension>::scheduleTimeAdvance ( LevelP const & level, SchedulerP & sched )
{
    dbg_out1 << "*** PhaseField::scheduleTimeAdvance " << * ( level.get_rep() ) << " " << sched << std::endl;

    Task * task_psi_grad = scinew Task ( "PhaseField::task_time_advance_grad_psi", this, &PhaseField::task_time_advance_grad_psi );
    Task * task_anisotropy_terms = scinew Task ( "PhaseField::task_time_advance_anisotropy_terms", this, &PhaseField::task_time_advance_anisotropy_terms );
    Task * task_current_solution = scinew Task ( "PhaseField::task_time_advance_current_solution", this, &PhaseField::task_time_advance_current_solution );

    task_time_advance_grad_psi_requires ( level, task_psi_grad );
    task_time_advance_grad_psi_computes ( task_psi_grad );

    task_time_advance_anisotropy_terms_requires ( level, task_anisotropy_terms );
    task_time_advance_anisotropy_terms_computes ( task_anisotropy_terms );

    task_time_advance_current_solution_requires ( level, task_current_solution );
    task_time_advance_current_solution_computes ( task_current_solution );

    sched->addTask ( task_psi_grad, level->eachPatch(), state->allMaterials() );
    sched->addTask ( task_anisotropy_terms, level->eachPatch(), state->allMaterials() );
    sched->addTask ( task_current_solution, level->eachPatch(), state->allMaterials() );

    dbg_out2 << std::endl;
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void PhaseField<VariableType, NumGhosts, Dimension>::task_time_advance_grad_psi_requires ( LevelP const & level, Task * task )
{
    task->requires ( Task::OldDW, psi_label, GhostType, NumGhosts );
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void PhaseField<VariableType, NumGhosts, Dimension>::task_time_advance_grad_psi_computes ( Task * task )
{
    task->computes ( grad_psi_norm2_label );
    for ( int d = 0; d < Dimension; ++d )
        task->computes ( grad_psi_label[d] );
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void PhaseField<VariableType, NumGhosts, Dimension>::task_time_advance_anisotropy_terms_requires ( LevelP const & level, Task * task )
{
    task->requires ( Task::NewDW, grad_psi_norm2_label, Ghost::None, 0 );
    for ( int d = 0; d < Dimension; ++d )
        task->requires ( Task::NewDW, grad_psi_label[d], Ghost::None, 0 );
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void PhaseField<VariableType, NumGhosts, Dimension>::task_time_advance_anisotropy_terms_computes ( Task * task )
{
    task->computes ( a_label );
    task->computes ( a2_label );
    for ( int d = 0; d < b_size; ++d )
        task->computes ( b_label[d] );
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void PhaseField<VariableType, NumGhosts, Dimension>::task_time_advance_current_solution_requires ( LevelP const & level, Task * task )
{
    task->requires ( Task::OldDW, psi_label, GhostType, NumGhosts );
    task->requires ( Task::OldDW, u_label, GhostType, NumGhosts );
    task->requires ( Task::NewDW, grad_psi_norm2_label, GhostType, NumGhosts );
    task->requires ( Task::NewDW, a_label, Ghost::Ghost::None, 0 );
    task->requires ( Task::NewDW, a2_label, GhostType, NumGhosts );
    for ( int d = 0; d < Dimension; ++d )
        task->requires ( Task::NewDW, grad_psi_label[d], GhostType, NumGhosts );
    for ( int d = 0; d < b_size; ++d )
        task->requires ( Task::NewDW, b_label[d], GhostType, NumGhosts );
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void PhaseField<VariableType, NumGhosts, Dimension>::task_time_advance_current_solution_computes ( Task * task )
{
    task->computes ( psi_label );
    task->computes ( u_label );
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void PhaseField<VariableType, NumGhosts, Dimension>::task_initialize ( ProcessorGroup const * /*myworld*/, PatchSubset const * patches, MaterialSubset const * /*matls*/, DataWarehouse * /*dw_old*/, DataWarehouse * dw_new )
{
    dbg_out1 << "==== PhaseField::task_initialize ====" << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << "== Patch: " << *patch << std::endl;

        Variable psi, u, grad_psi_norm2, a, a2;
        Variables<Dimension> grad_psi;
        Variables<b_size> b;
        dw_new->allocateAndPut ( psi, psi_label, 0, patch );
        dw_new->allocateAndPut ( u, u_label, 0, patch );
        dw_new->allocateAndPut ( grad_psi_norm2, grad_psi_norm2_label, 0, patch );
        dw_new->allocateAndPut ( a, a_label, 0, patch );
        dw_new->allocateAndPut ( a2, a2_label, 0, patch );
        dbg_out4 << "psi \t range " << psi.getLowIndex() << psi.getHighIndex() << std::endl;
        dbg_out4 << "u \t range " << u.getLowIndex() << u.getHighIndex() << std::endl;
        dbg_out4 << "grad_psi_norm2 \t range " << grad_psi_norm2.getLowIndex() << grad_psi_norm2.getHighIndex() << std::endl;
        dbg_out4 << "a \t range " << a.getLowIndex() << a.getHighIndex() << std::endl;
        dbg_out4 << "a2 \t range " << a2.getLowIndex() << a2.getHighIndex() << std::endl;

        for ( int d = 0; d < Dimension; ++d )
        {
            dw_new->allocateAndPut ( grad_psi[d], grad_psi_label[d], 0, patch );
            dbg_out4 << grad_psi_label[d]->getName() << "\t range " << grad_psi[d].getLowIndex() << grad_psi[d].getHighIndex() << std::endl;
        }

        for ( int d = 0; d < b_size; ++d )
        {
            dw_new->allocateAndPut ( b[d], b_label[d], 0, patch );
            dbg_out4 << b_label[d]->getName() << "\t range " << b[d].getLowIndex() <<  b[d].getHighIndex() << std::endl;
        }

        BlockRange range ( get_range ( patch ) );

        dbg_out3 << "= Iterating over range " << range << std::endl;
        parallel_for ( range, [patch, &psi, &u, this] ( int i, int j, int k )->void { initialize_current_solution ( i, j, k, patch, get_view ( psi ), get_view ( u ) ); } );

        a.initialize ( 0. );
        a2.initialize ( 0. );
        grad_psi_norm2.initialize ( 0. );
        for ( int d = 0; d < Dimension; ++d )
            grad_psi[d].initialize ( 0. );
        for ( int d = 0; d < b_size; ++d )
            b[d].initialize ( 0. );
    }

    dbg_out2 << std::endl;
}


template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void PhaseField<VariableType, NumGhosts, Dimension>::task_compute_stable_timestep ( ProcessorGroup const * /*myworld*/, PatchSubset const * patches, MaterialSubset const * /*matls*/, DataWarehouse * /*dw_old*/, DataWarehouse * dw_new )
{
    dbg_out1 << "==== PhaseField::task_compute_stable_timestep ====" << std::endl;
    dw_new->put ( delt_vartype ( delt ), state->get_delt_label(), getLevel ( patches ) );
    dbg_out2 << std::endl;
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void PhaseField<VariableType, NumGhosts, Dimension>::task_time_advance_grad_psi ( ProcessorGroup const * /*myworld*/, PatchSubset const * patches, MaterialSubset const * /*matls*/, DataWarehouse * dw_old, DataWarehouse * dw_new )
{
    dbg_out1 << "==== PhaseField::task_time_advance_grad_psi ====" << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << "== Patch: " << *patch << ")" << std::endl;

        ConstVariable psi;
        dw_old->get ( psi, psi_label, 0, patch, GhostType, NumGhosts );
        dbg_out4 << "psi \t range " << psi.getLowIndex() << psi.getHighIndex() << std::endl;

        Variable grad_psi_norm2;
        dw_new->allocateAndPut ( grad_psi_norm2, grad_psi_norm2_label, 0, patch );
        dbg_out4 << "grad_psi_norm2 \t range " << grad_psi_norm2.getLowIndex() << grad_psi_norm2.getHighIndex() << std::endl;

        Variables<Dimension> grad_psi;
        for ( int d = 0; d < Dimension; ++d )
        {
            dw_new->allocateAndPut ( grad_psi[d], grad_psi_label[d], 0, patch );
            dbg_out4 << grad_psi_label[d]->getName() << "\t range " << grad_psi[d].getLowIndex() << grad_psi[d].getHighIndex() << std::endl;
        }

        BlockRange range ( get_inner_range ( patch ) );

        dbg_out3 << "= Iterating over inner range" << range << std::endl;
        parallel_for ( range, [patch, &psi, &grad_psi, &grad_psi_norm2, this] ( int i, int j, int k )->void { time_advance_grad_psi ( i, j, k, patch, get_view ( psi ), get_views<Dimension> ( grad_psi ), get_view ( grad_psi_norm2 ) ); } );

        for ( auto face = start_face; face < end_face; face = Patch::nextFace ( face ) )
            if ( patch->getBCType ( face ) != Patch::Neighbor )
            {
                BlockRange range ( get_face_range ( patch, face ) );

                dbg_out3 << "= Iterating over " << face << " face range " << range << std::endl;
                parallel_for ( range, [face, patch, &psi, &grad_psi, &grad_psi_norm2, this] ( int i, int j, int k )->void { time_advance_grad_psi ( i, j, k, face, patch, get_view ( psi ), get_views<Dimension> ( grad_psi ), get_view ( grad_psi_norm2 ) ); } );
            }
    }

    dbg_out2 << std::endl;
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void PhaseField<VariableType, NumGhosts, Dimension>::task_time_advance_anisotropy_terms ( ProcessorGroup const * /*myworld*/, PatchSubset const * patches, MaterialSubset const * /*matls*/, DataWarehouse * /*dw_old*/, DataWarehouse * dw_new )
{
    dbg_out1 << "==== PhaseField::task_time_advance_anisotropy_terms ====" << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << "== Patch: " << *patch << std::endl;

        ConstVariable grad_psi_norm2;
        dw_new->get ( grad_psi_norm2, grad_psi_norm2_label, 0, patch, Ghost::None, 0 );
        dbg_out4 << "grad_psi_norm2 \t range " << grad_psi_norm2.getLowIndex() << grad_psi_norm2.getHighIndex() << std::endl;

        ConstVariables<Dimension> grad_psi;
        for ( int d = 0; d < Dimension; ++d )
        {
            dw_new->get ( grad_psi[d], grad_psi_label[d], 0, patch, Ghost::None, 0 );
            dbg_out4 << grad_psi_label[d]->getName() << "\t range " << grad_psi[d].getLowIndex() << grad_psi[d].getHighIndex() << std::endl;
        }

        Variable a, a2;
        dw_new->allocateAndPut ( a, a_label, 0, patch );
        dw_new->allocateAndPut ( a2, a2_label, 0, patch );
        dbg_out4 << "a \t range " << a.getLowIndex() << a.getHighIndex() << std::endl;
        dbg_out4 << "a2 \t range " << a2.getLowIndex() << a2.getHighIndex() << std::endl;

        Variables<b_size> b;
        for ( int d = 0; d < b_size; ++d )
        {
            dw_new->allocateAndPut ( b[d], b_label[d], 0, patch );
            dbg_out4 << b_label[d]->getName() << "\t range " << b[d].getLowIndex() << b[d].getHighIndex() << std::endl;
        }

        BlockRange range ( get_range ( patch ) );

        dbg_out3 << "= Iterating over inner range " << range << std::endl;
        parallel_for ( range, [patch, &grad_psi, &grad_psi_norm2, &a, &a2, &b, this] ( int i, int j, int k )->void { time_advance_anisotropy_terms ( i, j, k, patch, get_views<Dimension> ( grad_psi ), get_view ( grad_psi_norm2 ), get_view ( a ), get_view ( a2 ), get_views<b_size> ( b ) ); } );
    }

    dbg_out2 << std::endl;
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void PhaseField<VariableType, NumGhosts, Dimension>::task_time_advance_current_solution ( ProcessorGroup const * /*myworld*/, PatchSubset const * patches, MaterialSubset const * /*matls*/, DataWarehouse * dw_old, DataWarehouse * dw_new )
{
    dbg_out1 << "==== PhaseField::task_time_advance_current_solution ====" << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << "== Patch: " << *patch << std::endl;

        ConstVariable psi_old, u_old, grad_psi_norm2, a, a2;
        dw_old->get ( psi_old, psi_label, 0, patch, GhostType, NumGhosts );
        dw_old->get ( u_old, u_label, 0, patch, GhostType, NumGhosts );
        dw_new->get ( grad_psi_norm2, grad_psi_norm2_label, 0, patch, GhostType, NumGhosts );
        dw_new->get ( a, a_label, 0, patch, Ghost::None, 0 );
        dw_new->get ( a2, a2_label, 0, patch, GhostType, NumGhosts );
        dbg_out4 << "psi_old \t range " << psi_old.getLowIndex() << psi_old.getHighIndex() << std::endl;
        dbg_out4 << "u_old \t range " << u_old.getLowIndex() << u_old.getHighIndex() << std::endl;
        dbg_out4 << "grad_psi_norm2 \t range " << grad_psi_norm2.getLowIndex() << grad_psi_norm2.getHighIndex() << std::endl;
        dbg_out4 << "a \t range " << a.getLowIndex() << a.getHighIndex() << std::endl;
        dbg_out4 << "a2 \t range " << a2.getLowIndex() << a2.getHighIndex() << std::endl;

        ConstVariables<Dimension> grad_psi;
        ConstVariables<b_size> b;
        for ( int d = 0; d < Dimension; ++d )
        {
            dw_new->get ( grad_psi[d], grad_psi_label[d], 0, patch, GhostType, NumGhosts );
            dbg_out4 << grad_psi_label[d]->getName() << "\t range " << grad_psi[d].getLowIndex() << grad_psi[d].getHighIndex() << std::endl;
        }
        for ( int d = 0; d < b_size; ++d )
        {
            dw_new->get ( b[d], b_label[d], 0, patch, GhostType, NumGhosts );
            dbg_out4 << b_label[d]->getName() << "\t range " << b[d].getLowIndex() << b[d].getHighIndex() << std::endl;
        }

        Variable psi_new, u_new;
        dw_new->allocateAndPut ( psi_new, psi_label, 0, patch );
        dw_new->allocateAndPut ( u_new, u_label, 0, patch );
        dbg_out4 << "psi_new \t range " << psi_new.getLowIndex() << psi_new.getHighIndex() << std::endl;
        dbg_out4 << "u_new \t range " << u_new.getLowIndex() << u_new.getHighIndex() << std::endl;

        psi_new.copyPatch ( psi_old, psi_new.getLowIndex(), psi_new.getHighIndex() );
        u_new.copyPatch ( u_old, u_new.getLowIndex(), u_new.getHighIndex() );

        BlockRange range ( get_inner_range ( patch ) );

        dbg_out3 << "= Iterating over inner range " << range << std::endl;
        parallel_for ( range, [patch, &psi_old, &u_old, &grad_psi, &grad_psi_norm2, &a, &a2, &b, &psi_new, &u_new, this] ( int i, int j, int k )->void { time_advance_solution ( i, j, k, patch, psi_old, get_view ( u_old ), get_views<Dimension> ( grad_psi ), get_view ( grad_psi_norm2 ), get_view ( a ), get_view ( a2 ), get_views<b_size> ( b ), get_view ( psi_new ), get_view ( u_new ) ); } );

        for ( auto face = start_face; face < end_face; face = Patch::nextFace ( face ) )
            if ( patch->getBCType ( face ) != Patch::Neighbor )
            {
                BlockRange range ( get_face_range ( patch, face ) );

                dbg_out3 << "= Iterating over " << face << " face range " << range << std::endl;
                parallel_for ( range, [face, patch, &psi_old, &u_old, &grad_psi, &grad_psi_norm2, &a, &a2, &b, &psi_new, &u_new, this] ( int i, int j, int k )->void { time_advance_solution ( i, j, k, face, patch, ( psi_old ), get_view ( u_old ), get_views<Dimension> ( grad_psi ), get_view ( grad_psi_norm2 ), get_view ( a ), get_view ( a2 ), get_views<b_size> ( b ), get_view ( psi_new ), get_view ( u_new ) ); } );
            }
    }

    dbg_out2 << std::endl;
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void PhaseField<VariableType, NumGhosts, Dimension>::initialize_current_solution ( int i, int j, int k, Patch const * patch, View psi, View u )
{
    IntVector n ( i, j, k );
    Point p = get_position ( patch, n );
    double r2 = p.x() * p.x() + p.y() * p.y();
    if ( Dimension > 2 )
        r2 += p.z() * p.z();
    double tmp = r2 - r0 * r0;
    psi[n] = - tanh ( gamma_psi * tmp );
    u[n] = -delta * ( 1. + tanh ( gamma_u * tmp ) ) / 2.;
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void PhaseField<VariableType, NumGhosts, Dimension>::time_advance_grad_psi ( int i, int j, int k, const Patch * patch, ConstView psi, Views<Dimension> grad_psi, View grad_psi_norm2 )
{
    static_assert ( Dimension == 2 || Dimension == 3, "time_advance_grad_psi not implemented" );

    double psi_x = dx ( i, j, k, patch, psi );
    double psi_y = dy ( i, j, k, patch, psi );
    grad_psi[0] ( i, j, k ) = psi_x;
    grad_psi[1] ( i, j, k ) = psi_y;

    if ( Dimension == 2 )
    {
        grad_psi_norm2 ( i, j, k ) = psi_x * psi_x + psi_y * psi_y;
    }
    if ( Dimension == 3 )
    {
        double psi_z = dz ( i, j, k, patch, psi );
        grad_psi[2] ( i, j, k ) = psi_z;
        grad_psi_norm2 ( i, j, k ) = psi_x * psi_x + psi_y * psi_y + psi_z * psi_z;
    }
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void PhaseField<VariableType, NumGhosts, Dimension>::time_advance_grad_psi ( int i, int j, int k, Patch::FaceType face, Patch const * patch, ConstView psi, Views<Dimension> grad_psi, View grad_psi_norm2 )
{
    static_assert ( Dimension == 2 || Dimension == 3, "time_advance_grad_psi not implemented" );

    IntVector l ( psi.getLowIndex() ), h ( psi.getHighIndex() );
    Vector d ( patch->dCell() );

    int im = ( i > l.x() ) ? i - 1 : i, ip = ( i < h.x() - 1 ) ? i + 1 : i;
    int jm = ( j > l.y() ) ? j - 1 : j, jp = ( j < h.y() - 1 ) ? j + 1 : j;
    double psi_x = ( psi ( ip, j, k ) - psi ( im, j, k ) ) / ( d.x() * ( ip - im ) );
    double psi_y = ( psi ( i, jp, k ) - psi ( i, jm, k ) ) / ( d.y() * ( jp - jm ) );
    grad_psi[0] ( i, j, k ) = psi_x;
    grad_psi[1] ( i, j, k ) = psi_y;

    if ( Dimension == 2 )
    {
        grad_psi_norm2 ( i, j, k ) = psi_x * psi_x + psi_y * psi_y;
    }
    if ( Dimension == 3 )
    {
        int km = ( k > l.z() ) ? k - 1 : k, kp = ( k < h.z() - 1 ) ? k + 1 : k;
        double psi_z = ( psi ( i, j, kp ) - psi ( i, j, km ) ) / ( d.z() * ( kp - km ) );
        grad_psi[2] ( i, j, k ) = psi_z;
        grad_psi_norm2 ( i, j, k ) = psi_x * psi_x + psi_y * psi_y + psi_z * psi_z;
    }
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void PhaseField<VariableType, NumGhosts, Dimension>::time_advance_anisotropy_terms ( int i, int j, int k, Patch const * patch, ConstViews<Dimension> grad_psi, ConstView grad_psi_norm2, View a, View a2, Views<b_size> b )
{
    double tmp = 1. + epsilon;
    double n2 = grad_psi_norm2 ( i, j, k );
    if ( n2 < tol )
    {
        a ( i, j, k ) = tmp;
        a2 ( i, j, k ) = tmp * tmp;
        for ( int d = 0; d < b_size; ++d )
            b[d] ( i, j, k ) = 0.;
    }
    else
    {
        double psi_x = grad_psi[0] ( i, j, k );
        double psi_y = grad_psi[1] ( i, j, k );
        double n4 = n2 * n2;
        double psi_x2= psi_x * psi_x;
        double psi_y2= psi_y * psi_y;

        double psi_z, psi_z2, tmp4;
        if ( Dimension == 2 ) // Compile time if
            tmp4 = 4. * ( psi_x * psi_x * psi_x * psi_x + psi_y * psi_y * psi_y * psi_y ) / n4;
        if ( Dimension == 3 )   // Compile time if
        {
            psi_z = grad_psi[2] ( i, j, k );
            psi_z2 = psi_z * psi_z;
            tmp4 = 4. * ( psi_x2 * psi_x2 + psi_y2 * psi_y2 + psi_z2 * psi_z2 ) / n4;
        }
        double tmp = 1. + epsilon * ( tmp4 - 3. );
        a ( i, j, k ) = tmp;
        a2 ( i, j, k ) = tmp * tmp;
        b[xy] ( i, j, k ) = 16. * epsilon * tmp * ( psi_x * psi_y )  * ( psi_x2 - psi_y2 ) / n4;
        if ( Dimension == 3 )   // Compile time if
        {
            b[xz] ( i, j, k ) = 16. * epsilon * tmp * ( psi_x * psi_z ) * ( psi_x2 - psi_z2 ) / n4;
            b[yz] ( i, j, k ) = 16. * epsilon * tmp * ( psi_y * psi_z ) * ( psi_y2 - psi_z2 ) / n4;
        }
    }
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void PhaseField<VariableType, NumGhosts, Dimension>::time_advance_solution ( int i, int j, int k, Patch const * patch, ConstView psi_old, ConstView u_old, ConstViews<Dimension> grad_psi, ConstView grad_psi_norm2, ConstView a, ConstView a2, ConstViews<b_size> b, View psi_new, View u_new )
{
    double source = 1. - psi_old ( i, j, k ) * psi_old ( i, j, k );
    source *= ( psi_old ( i, j, k ) - lambda * u_old ( i, j, k ) * source );

    double delta_psi = 0;
    if ( Dimension == 2 )
        delta_psi = delt * ( laplacian ( i, j, k, patch, psi_old ) * a2 ( i, j, k )
                             + ( dx ( i, j, k, patch, a2 ) - dy ( i,j,k, patch, b[xy] ) ) * grad_psi[0] ( i, j, k )
                             + ( dx ( i,j,k, patch, b[xy] ) + dy ( i, j, k, patch, a2 ) ) * grad_psi[1] ( i, j, k )
                             + source ) / a ( i, j, k );
    if ( Dimension == 3 )
        delta_psi = delt * ( laplacian ( i, j, k, patch, psi_old ) * a2 ( i, j, k )
                             + ( dx ( i, j, k, patch, a2 ) - dy ( i,j,k, patch, b[xy] ) - dz ( i,j,k, patch, b[xz] ) ) * grad_psi[0] ( i, j, k )
                             + ( dx ( i,j,k, patch, b[xy] ) + dy ( i, j, k, patch, a2 ) - dz ( i,j,k, patch, b[yz] ) ) * grad_psi[1] ( i, j, k )
                             + ( dx ( i,j,k, patch, b[xz] ) + dy ( i,j,k, patch, b[yz] ) + dz ( i, j, k, patch, a2 ) ) * grad_psi[1] ( i, j, k )
                             + source ) / a ( i, j, k );

    double delta_u = delt * laplacian ( i, j, k, patch, u_old ) * alpha + delta_psi / 2.;

    psi_new ( i, j, k ) = psi_old ( i, j, k ) + delta_psi;
    u_new ( i, j, k ) = u_old ( i, j, k ) + delta_u;
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void PhaseField<VariableType, NumGhosts, Dimension>::time_advance_solution ( int i, int j, int k, Patch::FaceType face, Patch const * patch, ConstView psi_old, ConstView u_old, ConstViews<Dimension> grad_psi, ConstView grad_psi_norm2, ConstView a, ConstView a2, ConstViews<b_size> b, View psi_new, View u_new )
{
    psi_new ( i, j, k ) = psi_old ( i, j, k );
    u_new ( i, j, k ) = u_old ( i, j, k );
}

}

#endif





