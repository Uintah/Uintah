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

#include <CCA/Components/Heat/CCHeat2D.h>

#include <CCA/Components/Heat/blockrange_io.h>

using namespace Uintah;

CCHeat2D::CCHeat2D ( ProcessorGroup const * myworld, int verbosity )
    : UintahParallelComponent ( myworld )
    , dbg_out1 ( "CCHeat2D", verbosity > 0 )
    , dbg_out2 ( "CCHeat2D", verbosity > 1 )
    , dbg_out3 ( "CCHeat2D", verbosity > 2 )
    , dbg_out4 ( "CCHeat2D", verbosity > 3 )
    , solver ( nullptr )
    , solver_parameters ( nullptr )
{
    u_label = VarLabel::create ( "u", Variable::getTypeDescription() );
    matrix_label = VarLabel::create ( "A", Matrix::getTypeDescription() );
    rhs_label = VarLabel::create ( "b", Variable::getTypeDescription() );
#ifdef DBG_MATRIX
    Ap_label = VarLabel::create ( "Ap", Variable::getTypeDescription() );
    Aw_label = VarLabel::create ( "Aw", Variable::getTypeDescription() );
    Ae_label = VarLabel::create ( "Ae", Variable::getTypeDescription() );
    An_label = VarLabel::create ( "An", Variable::getTypeDescription() );
    As_label = VarLabel::create ( "As", Variable::getTypeDescription() );
    At_label = VarLabel::create ( "At", Variable::getTypeDescription() );
    Ab_label = VarLabel::create ( "Ab", Variable::getTypeDescription() );
#endif
}

CCHeat2D::~CCHeat2D()
{
    VarLabel::destroy ( u_label );
    VarLabel::destroy ( matrix_label );
    VarLabel::destroy ( rhs_label );
#ifdef DBG_MATRIX
    VarLabel::destroy ( Ap_label );
    VarLabel::destroy ( Aw_label );
    VarLabel::destroy ( Ae_label );
    VarLabel::destroy ( An_label );
    VarLabel::destroy ( As_label );
    VarLabel::destroy ( At_label );
    VarLabel::destroy ( Ab_label );
#endif
}

void CCHeat2D::problemSetup ( ProblemSpecP const & params, ProblemSpecP const & /*restart_prob_spec*/, GridP & /*grid*/, SimulationStateP & simulation_state )
{
    state = simulation_state;
    state->setIsLockstepAMR ( true );
    state->registerSimpleMaterial ( scinew SimpleMaterial() );

    ProblemSpecP heat = params->findBlock ( "FDHeat" );
    std::string scheme;
    heat->require ( "delt", delt );
    heat->require ( "alpha", alpha );
    heat->require ( "R0", r0 );
    heat->getWithDefault ( "gamma", gamma, 1. );
    heat->getWithDefault ( "scheme", scheme, "forward_euler" );
    time_scheme = from_str ( scheme );
    if ( time_scheme & TimeScheme::Implicit )
    {
        ProblemSpecP solv = params->findBlock ( "Solver" );
        solver = dynamic_cast<SolverInterface *> ( getPort ( "solver" ) );
        if ( !solver )
        {
            throw InternalError ( "CCHeat2D:couldn't get solver port", __FILE__, __LINE__ );
        }
        solver_parameters = solver->readParameters ( solv, "u" , state );
        solver_parameters->setSolveOnExtraCells ( false );
    }
}

void CCHeat2D::scheduleInitialize ( LevelP const & level, SchedulerP & sched )
{
    Task * task = scinew Task ( "CCHeat2D::task_initialize", this, &CCHeat2D::task_initialize );
    task->computes ( u_label );
    sched->addTask ( task, level->eachPatch(), state->allMaterials() );
}

void CCHeat2D::scheduleComputeStableTimestep ( LevelP const & level, SchedulerP & sched )
{
    Task * task = scinew Task ( "CCHeat2D::task_compute_stable_timestep", this, &CCHeat2D::task_compute_stable_timestep );
    task->computes ( state->get_delt_label(), level.get_rep() );
    sched->addTask ( task, level->eachPatch(), state->allMaterials() );
}

void CCHeat2D::scheduleTimeAdvance ( LevelP const & level, SchedulerP & sched )
{
    switch ( time_scheme )
    {
    case TimeScheme::ForwardEuler:
        scheduleTimeAdvance_forward_euler ( level, sched );
        break;
    case TimeScheme::BackwardEuler:
        scheduleTimeAdvance_backward_euler_assemble ( level, sched );
        scheduleTimeAdvance_backward_euler_solve ( level, sched );
        break;
    default:
        break;
    }
}

void CCHeat2D::scheduleTimeAdvance_forward_euler ( LevelP const & level, SchedulerP & sched )
{
    Task * task = scinew Task ( "CCHeat2D::task_farward_euler_time_advance", this, &CCHeat2D::task_farward_euler_time_advance );
    task->requires ( Task::OldDW, u_label, Ghost::AroundCells, 1 );
    task->computes ( u_label );
    sched->addTask ( task, level->eachPatch(), state->allMaterials() );
}

void CCHeat2D::scheduleTimeAdvance_backward_euler_assemble ( LevelP const & level, SchedulerP & sched )
{
    Task * task = scinew Task ( "CCHeat2D::task_backward_euler_assemble", this, &CCHeat2D::task_backward_euler_assemble );
    task->requires ( Task::OldDW, u_label, Ghost::AroundCells, 1 );
    task->computes ( matrix_label );
    task->computes ( rhs_label );
#ifdef DBG_MATRIX
    task->computes ( Ap_label );
    task->computes ( Aw_label );
    task->computes ( Ae_label );
    task->computes ( An_label );
    task->computes ( As_label );
    task->computes ( At_label );
    task->computes ( Ab_label );
#endif
    sched->addTask ( task, level->allPatches(), state->allMaterials() );
}

void CCHeat2D::scheduleTimeAdvance_backward_euler_solve ( LevelP const & level, SchedulerP & sched )
{
    solver->scheduleSolve ( level, sched, state->allMaterials(),
                            matrix_label, Task::NewDW, // A
                            u_label, false,            // x
                            rhs_label, Task::NewDW,    // b
                            u_label, Task::OldDW,      // guess
                            solver_parameters );
}

void CCHeat2D::task_initialize ( ProcessorGroup const * /*myworld*/, PatchSubset const * patches, MaterialSubset const * /*matls*/, DataWarehouse * /*dw_old*/, DataWarehouse * dw_new )
{
    dbg_out1 << "==== CCHeat2D::task_initialize ====" << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << "== Patch: " << *patch << std::endl;

        Variable u;
        dw_new->allocateAndPut ( u, u_label, 0, patch );
        dbg_out4 << "u \t range " << u.getLowIndex() << u.getHighIndex() << std::endl;

        BlockRange range ( get_range ( patch ) );
        dbg_out3 << "= Iterating over range " << range << std::endl;
        parallel_for ( range, [patch, &u, this] ( int i, int j, int k )->void { initialize ( i, j, k, patch, get_view ( u ) ); } );
    }

    dbg_out2 << std::endl;
}

void CCHeat2D::task_compute_stable_timestep ( ProcessorGroup const * /*myworld*/, PatchSubset const * patches, MaterialSubset const * /*matls*/, DataWarehouse * /*dw_old*/, DataWarehouse * dw_new )
{
    dbg_out1 << "==== CCHeat2D::task_compute_stable_timestep ====" << std::endl;
    dw_new->put ( delt_vartype ( delt ), state->get_delt_label(), getLevel ( patches ) );
    dbg_out2 << std::endl;
}

void CCHeat2D::task_farward_euler_time_advance ( ProcessorGroup const * /*myworld*/, PatchSubset const * patches, MaterialSubset const * /*matls*/, DataWarehouse * dw_old, DataWarehouse * dw_new )
{
    dbg_out1 << "==== CCHeat2D::task_farward_euler_time_advance ====" << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << "== Patch: " << *patch << std::endl;

        ConstVariable u_old;
        dw_old->get ( u_old, u_label, 0, patch, Ghost::AroundCells, 1 );
        dbg_out4 << "u_old \t range " << u_old.getLowIndex() << u_old.getHighIndex() << std::endl;

        Variable u_new;
        dw_new->allocateAndPut ( u_new, u_label, 0, patch );
        dbg_out4 << "u_new \t range " << u_new.getLowIndex() << u_new.getHighIndex() << std::endl;

        u_new.copyPatch ( u_old, u_new.getLowIndex(), u_new.getHighIndex() );
        BlockRange range ( get_inner_range ( patch ) );

        dbg_out3 << "= Iterating over inner range " << range << std::endl;
        Uintah::parallel_for ( range, [patch, &u_old, &u_new, this] ( int i, int j, int k )->void { forward_euler_time_advance ( i, j, k, patch, get_view ( u_old ), get_view ( u_new ) ); } );

        std::string bc_kind[4];
        double bc_value[4];
        for ( auto face = start_face; face <= end_face; face = Patch::nextFace ( face ) )
            if ( patch->getBCType ( face ) != Patch::Neighbor )
            {
                get_bc ( patch, face, 0, "u", 0, range, bc_kind[face], bc_value[face] ); // I don't need to know bc_kind for faces > face
                dbg_out3 << "= Iterating over " << face << " face range " << range << " BC " << bc_kind[face] << " " << bc_value[face] << std::endl;
                parallel_for ( range, [face, bc_value, bc_kind, patch, &u_old, &u_new, this] ( int i, int j, int k )->void { forward_euler_time_advance ( i, j, k, patch, get_view ( u_old ), get_view ( u_new ), face, bc_kind, bc_value ); } );
            }
    }

    dbg_out2 << std::endl;
}

void Uintah::CCHeat2D::task_backward_euler_assemble ( const Uintah::ProcessorGroup * myworld, const PatchSubset * patches, const MaterialSubset * matls, Uintah::DataWarehouse * dw_old, Uintah::DataWarehouse * dw_new )
{
    dbg_out1 << "==== CCHeat2D::task_backward_euler_assemble ====" << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << "== Patch: " << *patch << std::endl;

        ConstVariable u;
        dw_old->get ( u, u_label, 0, patch, Ghost::AroundCells, 1 );
        dbg_out4 << "u\t range " << u.getLowIndex() << u.getHighIndex() << std::endl;

        Matrix A;
        dw_new->allocateAndPut ( A, matrix_label, 0, patch );
        dbg_out4 << "A \t range " << A.getLowIndex() << A.getHighIndex() << std::endl;

#ifdef DBG_MATRIX
        Variable Ap;
        dw_new->allocateAndPut ( Ap, Ap_label, 0, patch );
        dbg_out4 << "Ap \t range " << Ap.getLowIndex() << Ap.getHighIndex() << std::endl;

        Variable Aw;
        dw_new->allocateAndPut ( Aw, Aw_label, 0, patch );
        dbg_out4 << "Aw \t range " << Aw.getLowIndex() << Aw.getHighIndex() << std::endl;

        Variable Ae;
        dw_new->allocateAndPut ( Ae, Ae_label, 0, patch );
        dbg_out4 << "Ae \t range " << Ae.getLowIndex() << Ae.getHighIndex() << std::endl;

        Variable An;
        dw_new->allocateAndPut ( An, An_label, 0, patch );
        dbg_out4 << "An \t range " << An.getLowIndex() << An.getHighIndex() << std::endl;

        Variable As;
        dw_new->allocateAndPut ( As, As_label, 0, patch );
        dbg_out4 << "As \t range " << As.getLowIndex() << As.getHighIndex() << std::endl;

        Variable At;
        dw_new->allocateAndPut ( At, At_label, 0, patch );
        dbg_out4 << "At \t range " << At.getLowIndex() << At.getHighIndex() << std::endl;

        Variable Ab;
        dw_new->allocateAndPut ( Ab, Ab_label, 0, patch );
        dbg_out4 << "Ab \t range " << Ab.getLowIndex() << Ab.getHighIndex() << std::endl;
#endif

        Variable b;
        dw_new->allocateAndPut ( b, rhs_label, 0, patch );
        dbg_out4 << "b\t range " << b.getLowIndex() << b.getHighIndex() << std::endl;

        BlockRange range ( get_range ( patch ) );
        dbg_out3 << "= Iterating over range " << range << std::endl;
        Uintah::parallel_for ( range, [patch, &u, REF_A, &b, this] ( int i, int j, int k )->void { backward_euler_assemble ( i, j, k, patch, get_view ( u ), GET_VIEW_A, get_view ( b ) ); } );

        std::string bc_kind[4];
        double bc_value[4];
        for ( auto face = start_face; face <= end_face; face = Patch::nextFace ( face ) )
            if ( patch->getBCType ( face ) != Patch::Neighbor )
            {
                get_bc ( patch, face, 0, "u", 0, range, bc_kind[face], bc_value[face] ); // I don't need to know bc_kind for faces > face
                dbg_out3 << "= Iterating over " << face << " face range " << range << " - BC " << bc_kind[face] << " " << bc_value[face] << std::endl;
                parallel_for ( range, [face, bc_value, bc_kind, patch, &u, REF_A, &b, this] ( int i, int j, int k )->void { backward_euler_assemble ( i, j, k, patch, get_view ( u ), GET_VIEW_A, get_view ( b ), face, bc_kind, bc_value ); } );
            }
    }

    dbg_out2 << std::endl;
}

void CCHeat2D::initialize ( int i, int j, int k, Patch const * patch, VariableView u )
{
    IntVector n ( i, j, k );
    Point p = get_position ( patch, n );
    double r2 = p.x() * p.x() + p.y() * p.y();
    double tmp = r2 - r0 * r0;
    u[n] = - tanh ( gamma * tmp );
}

void CCHeat2D::forward_euler_time_advance ( int i, int j, int k, Patch const * patch, ConstVariableView u_old, VariableView u_new )
{
    double delta_u = delt * alpha * laplacian ( i, j, k, patch, u_old );
    u_new ( i, j, k ) = u_old ( i, j, k ) + delta_u;
}

void CCHeat2D::forward_euler_time_advance ( int i, int j, int k, Patch const * patch, ConstVariableView u_old, VariableView u_new, Patch::FaceType face, const std::string bc_kind[4], const double bc_value[4] )
{
    const double d ( face / 2 );
    const double h ( patch->getLevel()->dCell() [d] );

    const double dx ( patch->getLevel()->dCell().x() );
//  const double dy ( patch->getLevel()->dCell().y() );

    const IntVector cl = get_low ( patch );
    const IntVector ch = get_high ( patch ) - IntVector ( 1, 1, 1 );

    const IntVector c0 ( i, j, k );
    const IntVector cm = c0 - patch->faceDirection ( face );
//  const IntVector cp = c0 + patch->faceDirection ( face );

    const IntVector cw ( i - 1, j, k );
    const IntVector ce ( i + 1, j, k );
//  const IntVector cs ( i, j - 1, k );
//  const IntVector cn ( i, j + 1, k );

    if ( bc_kind[face] == "Dirichlet" )
    {
        u_new[c0] = bc_value[face];
        if ( is_internal ( patch, cm, d, bc_kind ) ) // check if a dirichlet bc has already been imposed on cm
        {
            u_new[cm] += ( delt * alpha * ( bc_value[face] - u_old[c0] ) ) / ( h * h );
        }
        return;
    }
    if ( bc_kind[face] == "Neumann" )
    {
        double delta_u = delt * alpha;
        const double sgn ( patch->faceDirection ( face ) [d] );
        double uxx = 0.;
        double uyy = 0.;

        switch ( face )
        {
        case Patch::xminus:
        case Patch::xplus:
            if ( j == cl.y() && patch->getBCType ( Patch::yminus ) != Patch::Neighbor )
            {
                return;    // handled by yminus
            }
            if ( j == ch.y() && patch->getBCType ( Patch::yplus ) != Patch::Neighbor )
            {
                return;    // handled by yplus
            }
            uxx = sgn * ( u_old[c0] - u_old[cm] + bc_value[face] * h ) / ( h * h );
            uyy = dyy ( i, j, k, patch, u_old );
            delta_u *= uxx + uyy;
            break;
        case Patch::yminus:
        case Patch::yplus:
            if ( i == cl.x() && patch->getBCType ( Patch::xminus ) != Patch::Neighbor )
            {
                if ( bc_kind[Patch::xminus] == "Dirichlet" )
                {
                    return;
                }
                if ( bc_kind[Patch::xminus] == "Neumann" )
                {
                    uxx = ( u_old[ce] - u_old[c0] - bc_value[Patch::xminus] * dx ) / ( dx * dx );
                }
            }
            else if ( i == ch.x() && patch->getBCType ( Patch::xplus ) != Patch::Neighbor )
            {
                if ( bc_kind[Patch::xplus] == "Dirichlet" )
                {
                    return;
                }
                if ( bc_kind[Patch::xplus] == "Neumann" )
                {
                    uxx = ( u_old[c0] - u_old [cw] + bc_value[Patch::xplus] * dx ) / ( dx * dx );
                }
            }
            else
            {
                uxx = dxx ( i, j, k, patch, u_old );
            }
            uyy = sgn * ( u_old[c0] - u_old[cm] + bc_value[face] * h ) / ( h * h );
            delta_u *=  uxx + uyy;
            break;
        default:
            delta_u = 0.;
        }
        u_new[c0] = u_old[c0] + delta_u;
        return;
    }
}

void CCHeat2D::backward_euler_assemble ( int i, int j, int k, Patch const * patch, ConstVariableView u, MATRIX_VIEW_A, VariableView b )
{
    Vector const d ( patch->getLevel()->dCell() );
    IntVector n ( i, j, k );
    double a = - alpha * delt;
    double ax = a / ( d.x() * d.x() );
    double ay = a / ( d.y() * d.y() );
    A[n].p = 1. - 2. * ( ax + ay );
    A[n].e = A[n].w = ax;
    A[n].n = A[n].s = ay;
    A[n].t = A[n].b = 0.;
    b[n] = u[n];
#ifdef DBG_MATRIX
    Ap[n] = A[n].p;
    Aw[n] = A[n].w;
    Ae[n] = A[n].e;
    As[n] = A[n].s;
    An[n] = A[n].n;
    At[n] = A[n].t;
    Ab[n] = A[n].b;
#endif
}

void CCHeat2D::backward_euler_assemble ( int i, int j, int k, Patch const * patch, ConstVariableView u, MATRIX_VIEW_A, VariableView b, Patch::FaceType face, const std::string bc_kind[4], const double bc_value[4] )
{
    switch ( patch->getBCType ( face ) )
    {
    case Patch::None:
    {
        const double d ( face / 2 );
        const double h ( patch->getLevel()->dCell() [d] );
        IntVector n0 ( i, j, k );
        IntVector n1 = n0 - patch->faceDirection ( face );
        double a = ( alpha * delt ) / ( h * h );
        {
            if ( bc_kind[face] == "Dirichlet" )
            {
                b[n0] = bc_value[face];
                if ( is_internal ( patch, n1, d, bc_kind ) ) // check if a dirichlet bc has already been imposed on n1
                {
                    b[n1] += a * bc_value[face];
                }
                A[n0].p = 1;
                A[n0].e = A[n0].w = 0.;
                A[n0].n = A[n0].s = 0.;
                A[n0].t = A[n0].b = 0.;
                A[n1][face] = 0.;
#ifdef DBG_MATRIX
                Ap[n0] = A[n0].p;
                Aw[n0] = A[n0].w;
                Ae[n0] = A[n0].e;
                As[n0] = A[n0].s;
                An[n0] = A[n0].n;
                At[n0] = A[n0].t;
                Ab[n0] = A[n0].b;

                Ap[n1] = A[n1].p;
                Aw[n1] = A[n1].w;
                Ae[n1] = A[n1].e;
                As[n1] = A[n1].s;
                An[n1] = A[n1].n;
                At[n1] = A[n1].t;
                Ab[n1] = A[n1].b;
#endif
                return;
            }
            if ( bc_kind[face] == "Neumann" )
            {
                if ( !is_internal ( patch, n0, d, bc_kind ) ) // check if a dirichlet bc has already been imposed on n0
                {
                    return;
                }
                const double sgn ( patch->faceDirection ( face ) [d] );
                A[n0].p -= a;
                A[n0][face] = 0.;
#ifdef DBG_MATRIX
                Ap[n0] = A[n0].p;
                Aw[n0] = A[n0].w;
                Ae[n0] = A[n0].e;
                As[n0] = A[n0].s;
                An[n0] = A[n0].n;
                At[n0] = A[n0].t;
                Ab[n0] = A[n0].b;
#endif
                b[n0] -= a * sgn * h * bc_value[face];
                return;
            }
        }
    }
    default:
        return;
    }
}
