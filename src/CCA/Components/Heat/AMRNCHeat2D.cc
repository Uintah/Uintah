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

#include <CCA/Components/Heat/AMRNCHeat2D.h>

#include <Core/Exceptions/InvalidValue.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <CCA/Components/Heat/blockrange_io.h>
#include <CCA/Components/Regridder/PerPatchVars.h>

#ifdef CUSTOM_OUT
#include <iomanip>
#include <sstream>
#include <Core/OS/Dir.h>
#include <CCA/Components/Heat/vtkfile.hpp>
#include <CCA/Components/Heat/pvtkfile.hpp>
#endif

using namespace Uintah;

Uintah::AMRNCHeat2D::AMRNCHeat2D ( const Uintah::ProcessorGroup * myworld, int verbosity )
    : NCHeat2D ( myworld, verbosity )
{}

AMRNCHeat2D::~AMRNCHeat2D ()
{}

void AMRNCHeat2D::problemSetup ( ProblemSpecP const & params, ProblemSpecP const & restart_prob_spec, GridP & grid, SimulationStateP & state )
{
    NCHeat2D::problemSetup ( params, restart_prob_spec, grid, state );

    ProblemSpecP diffusion = params->findBlock ( "FDHeat" );
    diffusion->require ( "refine_threshold", refine_threshold );
}

void AMRNCHeat2D::scheduleTimeAdvance ( LevelP const & level, SchedulerP & sched )
{
    GridP grid = level->getGrid();

    switch ( time_scheme )
    {
    case TimeScheme::ForwardEuler:
        if ( level->hasCoarserLevel() )
        {
            scheduleTimeAdvance_forward_euler_refinement ( level, sched );
        }
        else
        {
            scheduleTimeAdvance_forward_euler ( level, sched );
        }
        break;
    case TimeScheme::BackwardEuler:
//      if ( solver->getName() == "hypre" )
//      {
//          scheduleTimeAdvance_backward_euler_assemble ( level, sched );
//          scheduleTimeAdvance_backward_euler_solve ( level, sched );
//      }
//      if ( solver->getName() == "hypreamr" )
//      {
//          if ( level->getIndex() != 0 )
//          {
//              return;    // only schedule on the coarsest level.
//          }
//
//          // all assemble task must be sent to the scheduler before the solve task
//          for ( int l = 0; l < grid->numLevels(); ++l )
//          {
//              scheduleTimeAdvance_backward_euler_assemble ( grid->getLevel ( l ), sched );
//          }
//
//          scheduleTimeAdvance_backward_euler_solve ( level, sched );
//      }
//      break;
        throw InternalError ( "\n ERROR: BackwardEuler time scheme not implemented for node centered variables\n", __FILE__, __LINE__ );
    default:
        throw InternalError ( "\n ERROR: Unknown time scheme\n", __FILE__, __LINE__ );
    }
}

void AMRNCHeat2D::scheduleRefine ( PatchSet const * patches, SchedulerP & sched )
{
    if ( getLevel ( patches )->hasCoarserLevel() )
    {
        Task * task = scinew Task ( "AMRNCHeat2D::task_refine", this, &AMRNCHeat2D::task_refine );
        task->requires ( Task::NewDW, u_label, 0, Task::CoarseLevel, 0, Task::NormalDomain, Ghost::AroundNodes, 1 );
        task->computes ( u_label );
        sched->addTask ( task, patches, state->allMaterials() );
    }
}

void AMRNCHeat2D::scheduleCoarsen ( LevelP const & level_coarse, SchedulerP & sched )
{
    Task * task = scinew Task ( "AMRNCHeat2D::task_coarsen", this, &AMRNCHeat2D::task_coarsen );
    task->requires ( Task::NewDW, u_label, 0, Task::FineLevel, 0, Task::NormalDomain, Ghost::None, 0 );
    task->modifies ( u_label );
    sched->addTask ( task, level_coarse->eachPatch(), state->allMaterials() );
}

void AMRNCHeat2D::scheduleErrorEstimate ( LevelP const & level_coarse, SchedulerP & sched )
{
    Task * task = scinew Task ( "AMRNCHeat2D::task_error_estimate", this, &AMRNCHeat2D::task_error_estimate );
    task->requires ( Task::NewDW, u_label, Ghost::AroundNodes, 1 ); // this is actually the old value of this
    task->modifies ( state->get_refineFlag_label(), state->refineFlagMaterials() );
    task->modifies ( state->get_refinePatchFlag_label(), state->refineFlagMaterials() );
    sched->addTask ( task, level_coarse->eachPatch(), state->allMaterials() );
}

void AMRNCHeat2D::scheduleInitialErrorEstimate ( LevelP const & level_coarse, SchedulerP & sched )
{
    scheduleErrorEstimate ( level_coarse, sched );
}

void AMRNCHeat2D::scheduleTimeAdvance_forward_euler_refinement ( LevelP const & level, SchedulerP & sched )
{
    Task * task = scinew Task ( "AMRNCHeat2D::task_forward_euler_time_advance_refinement", this, &AMRNCHeat2D::task_forward_euler_time_advance_refinement );
    task->requires ( Task::OldDW, u_label, Ghost::AroundNodes, 1 );
    task->requires ( Task::OldDW, u_label, 0, Task::CoarseLevel, 0, Task::NormalDomain, Ghost::AroundNodes, 1 );
    task->computes ( u_label );
    sched->addTask ( task, level->eachPatch(), state->allMaterials() );
}

void AMRNCHeat2D::task_forward_euler_time_advance_refinement ( ProcessorGroup const * myworld, PatchSubset const * patches, MaterialSubset const * matls, DataWarehouse * dw_old, DataWarehouse * dw_new )
{
    dbg_out1 << "==== AMRNCHeat2D::task_farward_euler_time_advance_refinement ====" << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << "== Patch: " << *patch << std::endl;

        ConstVariable u_old;
        dw_old->get ( u_old, u_label, 0, patch, Ghost::AroundNodes, 1 );
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
        ConstVariable u_coarse_old;
        bool get_coarser = true;
        for ( auto face = start_face; face <= end_face; face = Patch::nextFace ( face ) )
            if ( patch->getBCType ( face ) != Patch::Neighbor )
            {
                get_bc ( patch, face, 0, "u", 0, range, bc_kind[face], bc_value[face] ); // I don't need to know bc_kind for faces > face
                if ( patch->getBCType ( face ) == Patch::Coarse && get_coarser )
                {
                    const Level * level = patch->getLevel();
                    const Level * level_coarse = level->getCoarserLevel().get_rep();
                    IntVector l = get_low ( patch );
                    IntVector h = get_high ( patch );
                    IntVector l_coarse = map_to_coarser ( level, l ) - IntVector ( 1, 1, 0 );
                    IntVector h_coarse = map_to_coarser ( level, h ) + IntVector ( 1, 1, 0 );

                    dw_old->getRegion ( u_coarse_old, u_label, 0, level_coarse, l_coarse, h_coarse );
                    get_coarser = false;
                }
                dbg_out3 << "= Iterating over " << face << " face range " << range << " - BC " << bc_kind[face] << " " << bc_value[face] << " face direction" << patch->faceDirection ( face ) << std::endl;
                parallel_for ( range, [face, bc_value, bc_kind, patch, &u_old, &u_coarse_old, &u_new, this] ( int i, int j, int k )->void { forward_euler_time_advance_refinement ( i, j, k, patch, get_view ( u_old ), get_view ( u_new ), face, bc_kind, bc_value, u_coarse_old ); } );
            }
    }

    dbg_out2 << std::endl;
}

void AMRNCHeat2D::task_refine ( ProcessorGroup const * /*myworld*/, PatchSubset const * patches_fine, MaterialSubset const * /*matls*/, DataWarehouse * /*dw_old*/, DataWarehouse * dw_new )
{
    dbg_out1 << "==== AMRNCHeat2D::task_refine ====" << std::endl;

    const Level * level_fine = getLevel ( patches_fine );
    const Level * level_coarse = level_fine->getCoarserLevel().get_rep();

    for ( int p = 0; p < patches_fine->size(); ++p )
    {
        const Patch * patch_fine = patches_fine->get ( p );
        dbg_out2 << "== Fine Patch: " << *patch_fine << std::endl;

        Variable u_fine;
        dw_new->allocateAndPut ( u_fine, u_label, 0, patch_fine );
        dbg_out4 << "u_fine \t window " << u_fine.getLowIndex() << u_fine.getHighIndex() << std::endl;

        IntVector l_fine = get_low ( patch_fine );
        IntVector h_fine = get_high ( patch_fine );

        IntVector l_coarse = map_to_coarser ( level_fine, l_fine );
        IntVector h_coarse = map_to_coarser ( level_fine, h_fine );
        // Extening in order to select nodes on right edges
        h_coarse += IntVector ( 1, 1, 0 );

        dbg_out4 << "fine range" << BlockRange ( l_fine, h_fine ) << std::endl;
        dbg_out4 << "coarse range" << BlockRange ( l_coarse, h_coarse ) << std::endl;

        ConstVariable u_coarse;
        dw_new->getRegion ( u_coarse, u_label, 0, level_coarse, l_coarse, h_coarse );
        dbg_out4 << "u_coarse \t window " << u_coarse.getLowIndex() << u_coarse.getHighIndex() << std::endl;

        BlockRange range_fine ( l_fine, h_fine );
        dbg_out3 << "= Iterating over fine range" << range_fine << std::endl;
        parallel_for ( range_fine, [level_fine, level_coarse, &u_coarse, &u_fine, this] ( int i, int j, int k )->void { refine ( i, j, k, level_fine, level_coarse, get_view ( u_coarse ), get_view ( u_fine ) ); } );
    }

    dbg_out2 << std::endl;
}

void AMRNCHeat2D::task_coarsen ( ProcessorGroup const * /*myworld*/, const PatchSubset * patches_coarse, const MaterialSubset * /*matls*/, DataWarehouse * dw_old, DataWarehouse * dw_new )
{
    dbg_out1 << "==== AMRNCHeat2D::task_coarsen " << std::endl;

    const Level * level_coarse = getLevel ( patches_coarse );
    const Level * level_fine = level_coarse->getFinerLevel().get_rep();

    for ( int p = 0; p < patches_coarse->size(); ++p )
    {
        const Patch * patch_coarse = patches_coarse->get ( p );
        dbg_out2 << "== Coarse Patch: " << *patch_coarse << std::endl;

        Variable u_coarse;
        dw_new->getModifiable ( u_coarse, u_label, 0, patch_coarse );
        dbg_out4 << "u_coarse \t window " << u_coarse.getLowIndex() << u_coarse.getHighIndex() << std::endl;

        IntVector l_coarse = get_low ( patch_coarse );
        IntVector h_coarse = get_high ( patch_coarse );

        IntVector l_fine = map_to_finer ( level_coarse, l_coarse );
        IntVector h_fine = map_to_finer ( level_coarse, h_coarse );

        Level::selectType patches_fine;
        level_fine->selectPatches ( l_fine, h_fine, patches_fine );

        for ( int i = 0; i < patches_fine.size(); ++i )
        {
            const Patch * patch_fine = patches_fine[i];
            dbg_out3 << "= Fine Patch " << *patch_fine << std::endl;

            ConstVariable u_fine;
            dw_new->get ( u_fine, u_label, 0, patch_fine, Ghost::None, 0 );
            dbg_out4 << "u_fine \t window " << u_fine.getLowIndex() << u_fine.getHighIndex() << std::endl;

            BlockRange range_coarse (
                Max ( l_coarse, map_to_coarser ( level_fine, get_low ( patch_fine ) ) ),
                Min ( h_coarse, map_to_coarser ( level_fine, get_high ( patch_fine ) ) )
            );

            dbg_out3 << "= Iterating over coarse cells window " << range_coarse << std::endl;
            parallel_for ( range_coarse, [level_coarse, level_fine, &u_fine, &u_coarse, this] ( int i, int j, int k )->void { coarsen ( i, j, k, level_coarse, level_fine, get_view ( u_fine ), get_view ( u_coarse ) ); } );
        }
    }

    dbg_out2 << std::endl;
}

void AMRNCHeat2D::task_error_estimate ( ProcessorGroup const * /*myworld*/, PatchSubset const * patches, MaterialSubset const * /*matls*/, DataWarehouse * /*dw_old*/, DataWarehouse * dw_new )
{
    dbg_out1 << "==== AMRNCHeat2D::task_error_estimate " << std::endl;

#ifdef CUSTOM_OUT
    PVtkFile * out_pvtk = nullptr;
    const Level * level = getLevel ( patches );
    const int & levelID = level->getIndex();
    const int & timestep = state->getCurrentTopLevelTimeStep();
    std::string out_path, time_path, level_path, patch_path;

    if ( timestep + 1 >= dataArchiver->getNextOutputTimestep() )
    {
        dbg_out1 << "==== AMRNCHeat2D::task_save" << std::endl;
        std::stringstream time_ss, level_ss;
        out_path = dataArchiver->getOutputLocation();
        time_ss << "t" << std::setw ( 5 ) << std::setfill ( '0' ) << timestep;
        level_ss << "l" << levelID;
        time_path = time_ss.str();
        level_path = level_ss.str();
        MKDIR ( std::string ( out_path + "/" + time_path ).c_str(), 0777 );
        MKDIR ( std::string ( out_path + "/" + time_path + "/rf" + level_path ).c_str(), 0777 );

        // WARNING only for one process/thread

        out_pvtk = new PVtkFile ( out_path +  "/" + time_path + "/rf" + level_path, time_path, timestep * delt );
    }
#endif

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << "== Patch: " << *patch << std::endl;

        FlagVariable flag_refine;
        PerPatch<PatchFlagP> flag_refine_patch;
        dw_new->getModifiable ( flag_refine, state->get_refineFlag_label(), 0, patch );
        dw_new->get ( flag_refine_patch, state->get_refinePatchFlag_label(), 0, patch );
        dbg_out4 << "flag_refine \t window " << flag_refine.getLowIndex() << flag_refine.getHighIndex() << std::endl;

        PatchFlag * patch_flag_refine = flag_refine_patch.get().get_rep();
        ConstVariable u;
        dw_new->get ( u, u_label, 0, patch, Ghost::AroundNodes, 1 );
        dbg_out4 << "u \t window " << u.getLowIndex() << u.getHighIndex() << std::endl;

        bool refine_patch = false;
        BlockRange range = get_flag_range ( patch );
        dbg_out3 << "= Iterating over cells window " << range << std::endl;
        parallel_reduce ( range, [patch, &u, &flag_refine, this] ( int i, int j, int k, bool & refine_patch )->void { error_estimate ( i, j, k, patch, get_view ( u ), get_view ( flag_refine ), refine_patch, refine_threshold ); }, refine_patch );

        if ( refine_patch )
        {
            patch_flag_refine->set();
        }

#ifdef CUSTOM_OUT
        if ( out_pvtk )
        {
            IntVector l = get_low ( patch );
            IntVector h = get_high ( patch ) + IntVector (
                              patch->getBCType ( Patch::xplus ) == Patch::Neighbor ? 1 : 0,
                              patch->getBCType ( Patch::yplus ) == Patch::Neighbor ? 1 : 0,
                              0
                          );
            Point p0 = get_position ( patch, l );

            VtkFile * out_vtk = new VtkFile ( out_path + "/" + time_path + "/rf" + level_path, patch->getID() );
            out_vtk->set_grid ( patch->getLevel()->dCell().x(), patch->getLevel()->dCell().y(), patch->getLevel()->dCell().z(), h.x() - l.x(), h.y() - l.y(), h.z() - l.z(), p0.x(), p0.y(), p0.z() );
            out_vtk->add_cell_data ( state->get_refineFlag_label()->getName() + "/" + std::to_string ( levelID ), flag_refine, flag_refine.getLowIndex(), flag_refine.getHighIndex() );
            out_vtk->save();
            out_pvtk->add ( out_vtk->file_name() );
            delete out_vtk;
        }
#endif
    }

#ifdef CUSTOM_OUT
    if ( out_pvtk )
    {
        out_pvtk->save();

        if ( rf_out_visit.find ( levelID ) == rf_out_visit.end() )
        {
            rf_out_visit.emplace ( levelID, new VisitFile ( dataArchiver->getOutputLocation(), "rf_level" + std::to_string ( levelID ), false ) );
        }
        rf_out_visit[levelID]->add ( time_path + "/rf" + level_path + "/" + out_pvtk->file_name() );
        delete out_pvtk;
    }
#endif

    dbg_out2 << std::endl;
}

void AMRNCHeat2D::forward_euler_time_advance_refinement ( int i, int j, int k, Patch const * patch, ConstVariableView u_old, VariableView u_new, Patch::FaceType face, const std::string bc_kind[4], const double bc_value[4], ConstVariable u_coarse_old )
{
    const Level * level = patch->getLevel();

    const double d ( face / 2 );
    const double h ( level->dCell() [d] );

    const double dx ( level->dCell().x() );
//  const double dy ( level->dCell().y() );

    const IntVector nl = get_low ( patch );
    const IntVector nh = get_high ( patch ) - IntVector ( 1, 1, 1 );

    const IntVector n0 ( i, j, k );
    const IntVector nm = n0 - patch->faceDirection ( face );
    const IntVector np = n0 + patch->faceDirection ( face );

    const IntVector nw ( i - 1, j, k );
    const IntVector ne ( i + 1, j, k );
//  const IntVector ns ( i, j - 1,  k );
//  const IntVector nn ( i, j + 1,  k );

    if ( bc_kind[face] == "Dirichlet" )
    {
        u_new[n0] = bc_value[face];
        if ( is_internal ( patch, nm, d, bc_kind ) ) // check if a dirichlet bc has already been imposed on nm
        {
            u_new[nm] += ( delt * alpha * ( bc_value[face] - u_old[n0] ) ) / ( h * h );
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
            if ( j == nl.y() && patch->getBCType ( Patch::yminus ) != Patch::Neighbor )
            {
                return;    // handled by yminus
            }
            if ( j == nh.y() && patch->getBCType ( Patch::yplus ) != Patch::Neighbor )
            {
                return;    // handled by yplus
            }
            uxx = sgn * 2. * ( u_old[n0] - u_old[nm] + bc_value[face] * h ) / ( h * h );
            uyy = dyy ( i, j, k, patch, u_old );
            delta_u *= uxx + uyy;
            break;
        case Patch::yminus:
        case Patch::yplus:
            if ( i == nl.x() && patch->getBCType ( Patch::xminus ) != Patch::Neighbor )
            {
                if ( bc_kind[Patch::xminus] == "Dirichlet" )
                {
                    return;
                }
                else if ( bc_kind[Patch::xminus] == "Neumann" )
                {
                    uxx = 2. * ( u_old[ne] - u_old[n0] - bc_value[Patch::xminus] * dx ) / ( dx * dx );
                }
                else if ( bc_kind[Patch::xminus] == "FineCoarseInterface" )
                {
                    double u_old_nw = interpolate_coarser ( nw, level, level->getCoarserLevel().get_rep(), u_coarse_old );
                    uxx = ( u_old[ne] + u_old_nw - 2. * u_old[n0] ) / ( dx * dx );
                }
            }
            else if ( i == nh.x() && patch->getBCType ( Patch::xplus ) != Patch::Neighbor )
            {
                if ( bc_kind[Patch::xplus] == "Dirichlet" )
                {
                    return;
                }
                else if ( bc_kind[Patch::xplus] == "Neumann" )
                {
                    uxx = 2. * ( u_old[n0] - u_old[nw] + bc_value[Patch::xplus] * dx ) / ( dx * dx );
                }
                else if ( bc_kind[Patch::xplus] == "FineCoarseInterface" )
                {
                    double u_old_ne = interpolate_coarser ( ne, level, level->getCoarserLevel().get_rep(), u_coarse_old );
                    uxx = ( u_old_ne + u_old[nw] - 2. * u_old[n0] ) / ( dx * dx );
                }
            }
            else
            {
                uxx = dxx ( i, j, k, patch, u_old );
            }
            uyy = sgn * 2. * ( u_old[n0] - u_old[nm] + bc_value[face] * h ) / ( h * h );
            delta_u *=  uxx + uyy;
            break;
        default:
            delta_u = 0.;
        }
        u_new[n0] = u_old[n0] + delta_u;
        return;
    }

    if ( bc_kind[face] == "FineCoarseInterface" )
    {
        double delta_u = delt * alpha;
        double uxx = 0.;
        double uyy = 0.;

        double u_old_np;

        switch ( face )
        {
        case Patch::xminus:
        case Patch::xplus:
            if ( j == nl.y() && patch->getBCType ( Patch::yminus ) != Patch::Neighbor )
            {
                return;    // handled by yminus
            }
            if ( j == nh.y() && patch->getBCType ( Patch::yplus ) != Patch::Neighbor )
            {
                return;    // handled by yplus
            }
            u_old_np = interpolate_coarser ( np, level, level->getCoarserLevel().get_rep(), u_coarse_old );
            uxx = ( u_old[nm] + u_old_np - 2. * u_old[n0] ) / ( h * h );
            uyy = dyy ( i, j, k, patch, u_old );
            delta_u *= uxx + uyy;
            break;
        case Patch::yminus:
        case Patch::yplus:
            if ( i == nl.x() && patch->getBCType ( Patch::xminus ) != Patch::Neighbor )
            {
                if ( bc_kind[Patch::xminus] == "Dirichlet" )
                {
                    return;
                }
                else if ( bc_kind[Patch::xminus] == "Neumann" )
                {
                    uxx = 2. * ( u_old[ne] - u_old[n0] - bc_value[Patch::xminus] * dx ) / ( dx * dx );
                }
                else if ( bc_kind[Patch::xminus] == "FineCoarseInterface" )
                {
                    double u_old_nw = interpolate_coarser ( nw, level, level->getCoarserLevel().get_rep(), u_coarse_old );
                    uxx = ( u_old[ne] + u_old_nw - 2. * u_old[n0] ) / ( dx * dx );
                }
            }
            else if ( i == nh.x() && patch->getBCType ( Patch::xplus ) != Patch::Neighbor )
            {
                if ( bc_kind[Patch::xplus] == "Dirichlet" )
                {
                    return;
                }
                else if ( bc_kind[Patch::xplus] == "Neumann" )
                {
                    uxx = 2. * ( u_old[n0] - u_old[nw] + bc_value[Patch::xplus] * dx ) / ( dx * dx );
                }
                else if ( bc_kind[Patch::xplus] == "FineCoarseInterface" )
                {
                    double u_old_ne = interpolate_coarser ( ne, level, level->getCoarserLevel().get_rep(), u_coarse_old );
                    uxx = ( u_old_ne + u_old[nw] - 2. * u_old[n0] ) / ( dx * dx );
                }
            }
            else
            {
                uxx = dxx ( i, j, k, patch, u_old );
            }
            u_old_np = interpolate_coarser ( np, level, level->getCoarserLevel().get_rep(), u_coarse_old );
            uyy = ( u_old[nm] + u_old_np - 2. * u_old[n0] ) / ( h * h );
            delta_u *= uxx + uyy;
            break;
        default:
            delta_u = 0.;
        }
        u_new[n0] = u_old[n0] + delta_u;
        return;
    }

    std::ostringstream msg;
    msg << "\n ERROR: Unknown BC condition (" << bc_kind[face] << ") on patch " << *patch << " face " << face << "\n";
    throw InvalidValue ( msg.str(), __FILE__, __LINE__ );
}

void Uintah::AMRNCHeat2D::refine ( int i_fine, int j_fine, int k_fine, const Uintah::Level * level_fine, const Uintah::Level * level_coarse, ConstVariableView u_coarse, VariableView u_fine )
{
    IntVector node_fine ( i_fine, j_fine, k_fine );
    u_fine[node_fine] = interpolate_coarser ( node_fine, level_fine, level_coarse, u_coarse );
}

void Uintah::AMRNCHeat2D::coarsen ( int i_coarse, int j_coarse, int k_coarse, const Uintah::Level * level_coarse, const Uintah::Level * level_fine, ConstVariableView u_fine, VariableView u_coarse )
{
    IntVector node_coarse ( i_coarse, j_coarse, k_coarse );
    u_coarse[node_coarse] = restrict_finer ( node_coarse, level_coarse, level_fine, u_fine );
}

void Uintah::AMRNCHeat2D::error_estimate ( int i, int j, int k, const Uintah::Patch * patch, ConstVariableView u, FlagView flag_refine, bool & refine_patch, const double & refine_threshold )
{
    Vector d ( patch->dCell() );

    double u_x_0 = ( u ( i + 1, j,     k ) - u ( i,     j,     k ) ) / d.x();
    double u_x_1 = ( u ( i + 1, j + 1, k ) - u ( i,     j + 1, k ) ) / d.x();
    double u_y_0 = ( u ( i,     j + 1, k ) - u ( i,     j,     k ) ) / d.y();
    double u_y_1 = ( u ( i + 1, j + 1, k ) - u ( i + 1, j,     k ) ) / d.y();
    bool tmp = u_x_0 * u_x_0 > refine_threshold ||
               u_x_1 * u_x_1 > refine_threshold ||
               u_y_0 * u_y_0 > refine_threshold ||
               u_y_1 * u_y_1 > refine_threshold;
    flag_refine ( i, j, k ) = tmp;
    refine_patch |= tmp;
}

double AMRNCHeat2D::interpolate_coarser ( const IntVector & node_fine, const Level * level_fine, const Level * level_coarse, ConstVariableView u_coarse )
{
    IntVector node_coarse ( level_fine->mapNodeToCoarser ( node_fine ) );
    Point point_fine ( level_fine->getNodePosition ( node_fine ) );
    Point point_coarse ( level_coarse->getNodePosition ( node_coarse ) );
    Vector dist = ( point_fine.asVector() - point_coarse.asVector() ) / level_coarse->dCell();
    double w00 ( 1 ), w01 ( 1 ), w10 ( 1 ), w11 ( 1 );
    IntVector n00 ( node_coarse ), n01 ( node_coarse ), n10 ( node_coarse ), n11 ( node_coarse );
    if ( dist.x() < 0. )
    {
        n00[0] = n01[0] -= 1;
        w00 *= -dist.x();
        w01 *= -dist.x();
        w10 *= ( 1 + dist.x() );
        w11 *= ( 1 + dist.x() );
    }
    else if ( dist.x() > 0. )
    {
        n10[0] = n11[0] += 1;
        w00 *= ( 1 - dist.x() );
        w01 *= ( 1 - dist.x() );
        w10 *= dist.x();
        w11 *= dist.x();
    }
    else
    {
        w10 = 0.;
        w11 = 0.;
    }

    if ( dist.y() < 0. )
    {
        n00[1] = n10[1] -= 1;
        w00 *= -dist.y();
        w10 *= -dist.y();
        w01 *= ( 1 + dist.y() );
        w11 *= ( 1 + dist.y() );
    }
    else if ( dist.y() > 0. )
    {
        n01[1] = n11[1] += 1;
        w00 *= ( 1 - dist.y() );
        w10 *= ( 1 - dist.y() );
        w01 *= dist.y();
        w11 *= dist.y();
    }
    else
    {
        w01 = 0.;
        w11 = 0.;
    }

    return w00 * u_coarse[n00] +
           w01 * u_coarse[n01] +
           w10 * u_coarse[n10] +
           w11 * u_coarse[n11];
}

double AMRNCHeat2D::restrict_finer ( const IntVector & node_coarse, const Level * level_coarse, const Level * level_fine, ConstVariableView u_fine )
{
    IntVector node_fine ( level_coarse->mapNodeToFiner ( node_coarse ) );

    Point point_coarse ( level_coarse->getNodePosition ( node_coarse ) );
    Point point_fine ( level_fine->getNodePosition ( node_fine ) );

    assert ( ( point_fine.asVector() - point_coarse.asVector() ).length() == 0 );

    return u_fine[node_fine];
}
