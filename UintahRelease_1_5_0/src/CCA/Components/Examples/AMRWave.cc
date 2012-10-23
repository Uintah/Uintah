/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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


#include <CCA/Components/Examples/AMRWave.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/SimulationState.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Components/Regridder/PerPatchVars.h>
#include <Core/Util/DebugStream.h>

// TODO:
// Implement flux registers
// Linear interpolation might not be enough
// Possible bad interpolation

using namespace Uintah;

DebugStream amrwave("AMRWave", false);

AMRWave::AMRWave(const ProcessorGroup* myworld)
  : Wave(myworld)
{
}

AMRWave::~AMRWave()
{
}

//______________________________________________________________________
//
void AMRWave::problemSetup(const ProblemSpecP& params, 
                           const ProblemSpecP& restart_prob_spec, 
                           GridP& grid, SimulationStateP& sharedState)
{
  Wave::problemSetup(params, restart_prob_spec,grid, sharedState);
  ProblemSpecP wave = params->findBlock("Wave");
  wave->require("refine_threshold", refine_threshold);

  do_refineFaces = false;
  do_refine = false;
  do_coarsen = false;

  if (wave->findBlock("do_refineFaces"))
    do_refineFaces = true;
  if (wave->findBlock("do_refine"))
    do_refine = true;
  if ( wave->findBlock("do_coarsen"))
    do_coarsen = true;
}

void AMRWave::scheduleRefineInterface(const LevelP& fineLevel,
                                      SchedulerP& scheduler,
                                      bool needCoarseOld, bool needCoarseNew)
{
}
//______________________________________________________________________
//
void AMRWave::scheduleCoarsen(const LevelP& coarseLevel, SchedulerP& sched)
{
  if (!do_coarsen)
    return;
  Task* task = scinew Task("coarsen", this, &AMRWave::coarsen);
  task->requires(Task::NewDW, phi_label, 0, Task::FineLevel, 0, Task::NormalDomain, Ghost::None, 0);
  task->modifies(phi_label);
  task->requires(Task::NewDW, pi_label, 0, Task::FineLevel, 0, Task::NormalDomain, Ghost::None, 0);
  task->modifies(pi_label);
  sched->addTask(task, coarseLevel->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void AMRWave::scheduleRefine (const PatchSet* patches, SchedulerP& sched)
{
  if (!do_refine)
    return;
  Task* task = scinew Task("refine", this, &AMRWave::refine);
  task->requires(Task::NewDW, phi_label, 0, Task::CoarseLevel, 0, Task::NormalDomain, Ghost::AroundCells, 1);
  task->requires(Task::NewDW, pi_label, 0, Task::CoarseLevel, 0, Task::NormalDomain, Ghost::AroundCells, 1);

  // if this is a new level, then we need to schedule compute, otherwise, the copydata will yell at us.
  if (patches == getLevel(patches->getSubset(0))->eachPatch()) {
    task->computes(phi_label);
    task->computes(pi_label);
  }
  sched->addTask(task, patches, sharedState_->allMaterials());
}
//______________________________________________________________________
//
void AMRWave::scheduleErrorEstimate(const LevelP& coarseLevel,
				       SchedulerP& sched)
{
  Task* task = scinew Task("errorEstimate", this, &AMRWave::errorEstimate);
  task->requires(Task::NewDW, phi_label, Ghost::AroundCells, 1);
  task->modifies(sharedState_->get_refineFlag_label(), sharedState_->refineFlagMaterials());
  task->modifies(sharedState_->get_refinePatchFlag_label(), sharedState_->refineFlagMaterials());
  sched->addTask(task, coarseLevel->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void AMRWave::scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                              SchedulerP& sched)
{
  scheduleErrorEstimate(coarseLevel, sched);
}
//______________________________________________________________________
//
void AMRWave::scheduleTimeAdvance( const LevelP& level, 
                                   SchedulerP& sched)
{
  Wave::scheduleTimeAdvance(level, sched);
}
//______________________________________________________________________
//
void AMRWave::errorEstimate(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse*, DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    //if (patch->getLevel()->getIndex() > 0) cout << "  Doing errorEstimate on patch " << patch->getID() 
    //                                           << " low " << patch->getCellLowIndex() << " hi " << patch->getCellHighIndex() 
    //                                           << endl;
    CCVariable<int> refineFlag;
    PerPatch<PatchFlagP> refinePatchFlag;
    
    new_dw->getModifiable(refineFlag, sharedState_->get_refineFlag_label(),
                          0, patch);
    new_dw->get(refinePatchFlag, sharedState_->get_refinePatchFlag_label(),
                0, patch);

    PatchFlag* refinePatch = refinePatchFlag.get().get_rep();

    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      constCCVariable<double> phi;
      new_dw->get(phi, phi_label, matl, patch, Ghost::AroundCells, 1);

      // No boundary conditions - only works with periodic grids...

      Vector dx = patch->dCell();
      double thresh = refine_threshold/(dx.x()*dx.y()*dx.z());
      double sumdx2 = -2 / (dx.x()*dx.x()) -2/(dx.y()*dx.y()) - 2/(dx.z()*dx.z());
      Vector inv_dx2(1./(dx.x()*dx.x()), 1./(dx.y()*dx.y()), 1./(dx.z()*dx.z()));
      int numFlag = 0;
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        const IntVector& c = *iter;

        IntVector low = patch->getCellLowIndex();
        IntVector high = patch->getCellHighIndex();

        // Compute curl
        double curlPhi = curl(phi, c, sumdx2, inv_dx2);

        if(curlPhi > thresh){
          numFlag++;
          refineFlag[c] = true;
        }
      }
      // cerr << "numFlag=" << numFlag << '\n';
      if(numFlag != 0)
        refinePatch->set();
    }
  }
}
//______________________________________________________________________
//
void AMRWave::refineCell(CCVariable<double>& finevar, constCCVariable<double>& coarsevar, IntVector fineIndex,
                    const Level* fineLevel, const Level* coarseLevel)
{
  Point cell_pos = fineLevel->getCellPosition(fineIndex);
  int i = 1, j = 1, k = 1; // where to go for the next cell

  // starting coarse point - will interp 8 cells starting here.  
  IntVector coarseStart(fineLevel->mapCellToCoarser(fineIndex));
  Point coarse_pos = coarseLevel->getCellPosition(coarseStart);

  // adjust the coarse starting point.  If a coarse coordinate is greater than 
  // a fine one, then subtract 1, so our 8 cells will surround the fine point.
  // (TODO - compensate for non-2 refinement ratio?
  Vector dist = coarse_pos.asVector() - cell_pos.asVector();
  if (dist.x() > 0)
    i = -1;
  if (dist.y() > 0)
    j = -1;
  if (dist.z() > 0)
    k = -1;

  // dist should now be the the fraction where the fine cell is between two coarse cells
  dist *= Vector(1.0)/coarseLevel->dCell();
  dist = Abs(dist);

  amrwave << "RefineCell on level " << fineLevel->getIndex() << " Fine Index " << fineIndex << " " << fineLevel->getCellPosition(fineIndex) << " coarse range = " 
          << coarseStart << " " << IntVector(coarseStart.x()+i, coarseStart.y()+j, coarseStart.z()+k) << endl;

  // weights of each cell
  double w0 = (1. - dist.x()) * (1. - dist.y()) * (1. - dist.z()); 
  double w1 = dist.x() * (1. - dist.y()) * (1. - dist.z());
  double w2 = dist.y() * (1. - dist.x()) * (1. - dist.z());
  double w3 = dist.x() * dist.y() * (1. - dist.z());
  double w4 = (1. - dist.x()) * (1. - dist.y()) * dist.z();
  double w5 = dist.x() * (1. - dist.y()) * dist.z();
  double w6 = dist.y() * (1. - dist.x()) * dist.z();
  double w7 = dist.x() * dist.y() * dist.z();

  amrwave << "  CVs: " << coarsevar[coarseStart] << " " << coarsevar[coarseStart + IntVector(i,0,0)] << " "
          << coarsevar[coarseStart + IntVector(0,j,0)] << " " << coarsevar[coarseStart + IntVector(i,j,0)] << " " 
          << coarsevar[coarseStart + IntVector(0,0,k)] << " " << coarsevar[coarseStart + IntVector(i,0,k)] << " "
          << coarsevar[coarseStart + IntVector(0,j,k)] << " " << coarsevar[coarseStart + IntVector(i,j,k)] << endl;

  // add up the weighted values
  finevar[fineIndex] = 
    w0*coarsevar[coarseStart] +
    w1*coarsevar[coarseStart + IntVector(i,0,0)] +
    w2*coarsevar[coarseStart + IntVector(0,j,0)] +
    w3*coarsevar[coarseStart + IntVector(i,j,0)] +
    w4*coarsevar[coarseStart + IntVector(0,0,k)] +
    w5*coarsevar[coarseStart + IntVector(i,0,k)] +
    w6*coarsevar[coarseStart + IntVector(0,j,k)] +
    w7*coarsevar[coarseStart + IntVector(i,j,k)];

  //amrwave << ": " << finevar[fineIndex] << endl;

}
//______________________________________________________________________
//
void AMRWave::coarsenCell(CCVariable<double>& coarsevar, constCCVariable<double>& finevar, IntVector coarseIndex,
                    const Level* fineLevel, const Level* coarseLevel)
{
  double tmp=0; 

  // starting coarse point - will interp 8 cells starting here.  On the +x, +y, +z faces, subtract 1
  IntVector fineStart(coarseLevel->mapCellToFiner(coarseIndex));
  IntVector crr = fineLevel->getRefinementRatio();

  // amrwave << "coarsenCell on level " << coarseLevel->getIndex() << " coarse Index " << coarseIndex << " " << coarseLevel->getCellPosition(coarseIndex) << " fine range = " 
  //        << fineStart << " " << fineStart+crr << endl << "  FVs: ";

  // find all the fine cells, average the values; no need to weight, as they are equidistant
  for(CellIterator inside(IntVector(0,0,0), crr); !inside.done(); inside++){
    // amrwave << finevar[fineStart + *inside] << ' ';
    tmp +=finevar[fineStart + *inside];
  }

  coarsevar[coarseIndex]=tmp/(crr.x()*crr.y()*crr.z());
  // amrwave << ": " << coarsevar[coarseIndex] << endl;
}
//______________________________________________________________________
//
void AMRWave::refine(const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse*, DataWarehouse* new_dw)
{
  const Level* fineLevel = getLevel(patches);
  const Level* coarseLevel = fineLevel->getCoarserLevel().get_rep();

  for (int p = 0; p < patches->size(); p++) {  
    const Patch* finePatch = patches->get(p);
    // amrwave << "    DOING AMRWave::Refine on patch " << finePatch->getID() << ": " << finePatch->getLowIndex() << " " << finePatch->getHighIndex() <<  endl;

    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      CCVariable<double> pi;
      CCVariable<double> phi;
      constCCVariable<double> coarse_pi;
      constCCVariable<double> coarse_phi;
      
      new_dw->allocateAndPut(phi, phi_label, matl, finePatch);
      new_dw->allocateAndPut(pi, pi_label, matl, finePatch);
      pi.initialize(0);
      phi.initialize(0);

      IntVector coarsePhiLow  = fineLevel->mapCellToCoarser(phi.getLowIndex()) - IntVector(1,1,1);
      IntVector coarsePhiHigh = fineLevel->mapCellToCoarser(phi.getHighIndex()+fineLevel->getRefinementRatio() - IntVector(1,1,1)) + IntVector(1,1,1);
      IntVector coarsePiLow  = fineLevel->mapCellToCoarser(pi.getLowIndex()) - IntVector(1,1,1);
      IntVector coarsePiHigh = fineLevel->mapCellToCoarser(pi.getHighIndex()+fineLevel->getRefinementRatio() - IntVector(1,1,1)) + IntVector(1,1,1);

      // amrwave << "   Calling getRegion for Phi: " << coarsePhiLow << " " << coarsePhiHigh << endl;
      new_dw->getRegion(coarse_phi, phi_label, matl, coarseLevel, coarsePhiLow, coarsePhiHigh);
      // amrwave << "   Calling getRegion for Pi: " << coarsePiLow << " " << coarsePiHigh << endl;
      new_dw->getRegion(coarse_pi, pi_label, matl, coarseLevel, coarsePiLow, coarsePiHigh);

      // simple linear interpolation (maybe)
      for(CellIterator iter(phi.getLowIndex(), phi.getHighIndex()); !iter.done(); iter++){
        // refine phi
        refineCell(phi, coarse_phi, *iter, fineLevel, coarseLevel);

        // phi extends (potentially) one more cell than pi.
        IntVector pilow, pihigh;
        pilow = pi.getLow();
        pihigh = pi.getHigh();
        
        if ((*iter).x() >= pilow.x() && (*iter).y() >= pilow.y() && (*iter).z() >= pilow.z() &&
          (*iter).x() < pihigh.x() && (*iter).y() < pihigh.y() && (*iter).z() < pihigh.z()) {
          refineCell(pi, coarse_pi, *iter, fineLevel, coarseLevel);
        }
      }
    } // matls
  } // finePatches
}
//______________________________________________________________________
//
void AMRWave::coarsen(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse*, DataWarehouse* new_dw)
{
  const Level* coarseLevel = getLevel(patches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();

  for (int p = 0; p < patches->size(); p++) {  
    const Patch* coarsePatch = patches->get(p);
    //amrwave << "    DOING AMRWave::coarsen on patch " << coarsePatch->getID() << ": " << coarsePatch->getLowIndex() << " " << coarsePatch->getHighIndex() << endl ;

    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      CCVariable<double> pi;
      CCVariable<double> phi;
      constCCVariable<double> fine_pi;
      constCCVariable<double> fine_phi;
      
      new_dw->getModifiable(phi, phi_label, matl, coarsePatch);
      new_dw->getModifiable(pi, pi_label, matl, coarsePatch);
      
      IntVector fineLow  = coarseLevel->mapCellToFiner(pi.getLowIndex());
      IntVector fineHigh = coarseLevel->mapCellToFiner(pi.getHighIndex());

      Level::selectType finePatches;
      fineLevel->selectPatches(fineLow, fineHigh, finePatches);
      
      for (int i = 0; i < finePatches.size(); i++) {
        const Patch* finePatch = finePatches[i];
        new_dw->get(fine_phi, phi_label, matl, finePatch, Ghost::None, 0);
        new_dw->get(fine_pi, pi_label, matl, finePatch, Ghost::None, 0);

        // simple linear interpolation (maybe) - don't coarsen to phi's boundary
        for(CellIterator iter(Max(pi.getLowIndex(), fineLevel->mapCellToCoarser(fine_pi.getLowIndex())), 
                              Min(pi.getHighIndex(), fineLevel->mapCellToCoarser(fine_pi.getHighIndex()))); 
            !iter.done(); iter++){
          coarsenCell(phi, fine_phi, *iter, fineLevel, coarseLevel);
          coarsenCell(pi, fine_phi, *iter, fineLevel, coarseLevel);
        }
      } // finePatches
    } // matls
  } // patches
}
//______________________________________________________________________
//
void AMRWave::addRefineDependencies(Task* task, const VarLabel* var,
                                    bool needCoarseOld, bool needCoarseNew)
{
  if (!do_refineFaces)
    return;
  Ghost::GhostType gc = Ghost::AroundCells;

  if(needCoarseOld)
    task->requires(Task::CoarseOldDW, var,
		   0, Task::CoarseLevel, 0, Task::NormalDomain, gc, 1);
  if(needCoarseNew)
    task->requires(Task::CoarseNewDW, var,
		   0, Task::CoarseLevel, 0, Task::NormalDomain, gc, 1);
}
//______________________________________________________________________
//
void AMRWave::refineFaces(const Patch* finePatch, 
                 const Level* fineLevel,
		 const Level* coarseLevel, 
		 CCVariable<double>& finevar, 
                 const VarLabel* label, int matl, 
                 DataWarehouse* coarse_old_dw,
		 DataWarehouse* coarse_new_dw)
{
  DataWarehouse* fine_new_dw = coarse_old_dw->getOtherDataWarehouse(Task::NewDW);
  double subCycleProgress = getSubCycleProgress(fine_new_dw);
  if (!do_refineFaces)
    return;
  for (int f = 0; f < 6; f++) {
    Patch::FaceType face = (Patch::FaceType) f;
    if (finePatch->getBCType(face) == Patch::Coarse) {

      IntVector low, high;
      IntVector fineLow, fineHigh;
      finePatch->getFace(face, IntVector(2,2,2), IntVector(2,2,2), low, high);
      finePatch->getFace(face, IntVector(0,0,0), IntVector(1,1,1), fineLow, fineHigh);

      // grab one more cell...
      IntVector coarseLow = fineLevel->mapCellToCoarser(low) - IntVector(1,1,1);
      IntVector coarseHigh = fineLevel->mapCellToCoarser(high) + IntVector(1,1,1);

      amrwave << "    DOING AMRWave::RefineFaces on patch " << finePatch->getID() << ": " << finePatch->getCellLowIndex() << " " << finePatch->getCellHighIndex() << " face: " << fineLow << " " << fineHigh << endl;

      constCCVariable<double> coarse_old_var;
      constCCVariable<double> coarse_new_var;

      if (subCycleProgress < 1.0-1e-10)
        coarse_old_dw->getRegion(coarse_old_var, label, matl, coarseLevel, coarseLow, coarseHigh);

      if (subCycleProgress > 0.0)
        coarse_new_dw->getRegion(coarse_new_var, label, matl, coarseLevel, coarseLow, coarseHigh);

      // loop across the face, and depending on where we are in time, use the values 
      // from the old dw, new dw, or both (based on a weight)
      for (CellIterator iter(fineLow, fineHigh); !iter.done(); iter++) {
        double x0 = 0, x1 = 0;
        double w1 = subCycleProgress;
        double w0 = 1-w1;

        if (subCycleProgress < 1.0-1e-10) {
          refineCell(finevar, coarse_old_var, *iter, fineLevel, coarseLevel);
          x0 = finevar[*iter];
        }
        if (subCycleProgress > 0.0) {
          refineCell(finevar, coarse_new_var, *iter, fineLevel, coarseLevel);
          x1 = finevar[*iter];
        }
        finevar[*iter] = x0*w0 + x1*w1;
        amrwave << "Index " << *iter << " SubCycle: " << subCycleProgress << " X0 " << x0 << " X1 " << x1 << " final " << finevar[*iter] << endl;
      }
    } // if (finePatch->getBCType(face) == Patch::Coarse) {
  } // for (Patch::FaceType face = 0; face < Patch::numFaces; face++) {
}
