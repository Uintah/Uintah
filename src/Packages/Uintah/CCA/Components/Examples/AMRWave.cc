
#include <Packages/Uintah/CCA/Components/Examples/AMRWave.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Components/Regridder/PerPatchVars.h>

// TODO:
// Implement flux registers
// refine patches
// Refine faces for boundaries
// periodic boundaries broken...

using namespace Uintah;

AMRWave::AMRWave(const ProcessorGroup* myworld)
  : Wave(myworld)
{
}

AMRWave::~AMRWave()
{
}


void AMRWave::problemSetup(const ProblemSpecP& params, GridP& grid,
		      SimulationStateP& sharedState)
{
  Wave::problemSetup(params, grid, sharedState);
  ProblemSpecP wave = params->findBlock("Wave");
  wave->require("refine_threshold", refine_threshold);
}

void AMRWave::scheduleRefineInterface(const LevelP& fineLevel,
					 SchedulerP& scheduler,
					 int step, int nsteps)
{
}

void AMRWave::scheduleCoarsen(const LevelP& coarseLevel, SchedulerP& sched)
{
  Task* task = scinew Task("coarsen", this, &AMRWave::coarsen);
  task->requires(Task::NewDW, phi_label, 0, Task::FineLevel, 0, Task::NormalDomain, Ghost::None, 0);
  task->modifies(phi_label);
  task->requires(Task::NewDW, pi_label, 0, Task::FineLevel, 0, Task::NormalDomain, Ghost::None, 0);
  task->modifies(pi_label);
  sched->addTask(task, coarseLevel->eachPatch(), sharedState_->allMaterials());
}

void AMRWave::scheduleRefine (const PatchSet* patches, SchedulerP& sched)
{
  Task* task = scinew Task("refine", this, &AMRWave::refine);
  task->requires(Task::NewDW, phi_label, 0, Task::CoarseLevel, 0, Task::NormalDomain, Ghost::AroundCells, 1);
  //task->computes(phi_label);
  task->requires(Task::NewDW, pi_label, 0, Task::CoarseLevel, 0, Task::NormalDomain, Ghost::AroundCells, 1);
  //task->computes(pi_label);
  sched->addTask(task, patches, sharedState_->allMaterials());
}

void AMRWave::scheduleErrorEstimate(const LevelP& coarseLevel,
				       SchedulerP& sched)
{
  Task* task = scinew Task("errorEstimate", this, &AMRWave::errorEstimate);
  task->requires(Task::NewDW, phi_label, Ghost::AroundCells, 1);
  task->modifies(sharedState_->get_refineFlag_label(), sharedState_->refineFlagMaterials());
  task->modifies(sharedState_->get_refinePatchFlag_label(), sharedState_->refineFlagMaterials());
  sched->addTask(task, coarseLevel->eachPatch(), sharedState_->allMaterials());
}

void AMRWave::scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                              SchedulerP& sched)
{
  scheduleErrorEstimate(coarseLevel, sched);
}

void AMRWave::scheduleTimeAdvance( const LevelP& level, 
                                   SchedulerP& sched, int step, int nsteps )
{
  Wave::scheduleTimeAdvance(level, sched, step, nsteps);
}

void AMRWave::errorEstimate(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse*, DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    //if (patch->getLevel()->getIndex() > 0) cout << "  Doing errorEstimate on patch " << patch->getID() 
    //                                           << " low " << patch->getLowIndex() << " hi " << patch->getHighIndex() 
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

        IntVector low = patch->getLowIndex();
        IntVector high = patch->getHighIndex();

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

void AMRWave::refineCell(CCVariable<double>& finevar, constCCVariable<double>& coarsevar, IntVector fineIndex,
                    const Level* fineLevel, const Level* coarseLevel)
{
  Point cell_pos = fineLevel->getCellPosition(fineIndex);
  double tmp=0;
  double total_weight = 0;

  // starting coarse point - will interp 8 cells starting here.  On the +x, +y, +z faces, subtract 1
  IntVector coarseStart(fineLevel->mapCellToCoarser(fineIndex));
  IntVector coarseHigh(coarsevar.getHighIndex());
  if (coarseStart.x() == coarseHigh.x()-1)
    coarseStart[0]--;
  if (coarseStart.y() == coarseHigh.y()-1)
    coarseStart[1]--;
  if (coarseStart.z() == coarseHigh.z()-1)
    coarseStart[2]--;
  
  // find eight cells, add up values multiplied by the weight of their distance
  for(CellIterator inside(IntVector(0,0,0), IntVector(2,2,2)); !inside.done(); inside++){
    IntVector coarse_idx = coarseStart+*inside;
    Point coarse_pos = coarseLevel->getCellPosition(coarse_idx);
    double distance = (coarse_pos - cell_pos).length();
    
    tmp +=coarsevar[coarse_idx]*distance;
    total_weight += distance;
  }

  // average value
  finevar[fineIndex]=tmp/total_weight;

}

void AMRWave::coarsenCell(CCVariable<double>& coarsevar, constCCVariable<double>& finevar, IntVector coarseIndex,
                    const Level* fineLevel, const Level* coarseLevel)
{
  Point cell_pos = coarseLevel->getCellPosition(coarseIndex);
  double tmp=0; 
  double total_weight = 0;

  // starting coarse point - will interp 8 cells starting here.  On the +x, +y, +z faces, subtract 1
  IntVector fineStart(coarseLevel->mapCellToFiner(coarseIndex));
  
  // find eight cells, add up values multiplied by the weight of their distance
  for(CellIterator inside(IntVector(0,0,0), IntVector(2,2,2)); !inside.done(); inside++){
    IntVector fine_idx = fineStart+*inside;
    Point fine_pos = fineLevel->getCellPosition(fine_idx);
    double distance = (fine_pos - cell_pos).length();
    
    tmp +=finevar[fine_idx]*distance;
    total_weight += distance;
  }

  // average value
  coarsevar[coarseIndex]=tmp/total_weight;

}

void AMRWave::refine(const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse*, DataWarehouse* new_dw)
{
  const Level* fineLevel = getLevel(patches);
  const Level* coarseLevel = fineLevel->getCoarserLevel().get_rep();

  for (int p = 0; p < patches->size(); p++) {  
    const Patch* finePatch = patches->get(p);

    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      CCVariable<double> pi;
      CCVariable<double> phi;
      constCCVariable<double> coarse_pi;
      constCCVariable<double> coarse_phi;
      
      new_dw->allocateAndPut(phi, phi_label, matl, finePatch);
      new_dw->allocateAndPut(pi, pi_label, matl, finePatch);
      
      IntVector coarsePhiLow  = fineLevel->mapCellToCoarser(phi.getLowIndex());
      IntVector coarsePhiHigh = fineLevel->mapCellToCoarser(phi.getHighIndex()+fineLevel->getRefinementRatio() - IntVector(1,1,1));
      IntVector coarsePiLow  = fineLevel->mapCellToCoarser(pi.getLowIndex());
      IntVector coarsePiHigh = fineLevel->mapCellToCoarser(pi.getHighIndex()+fineLevel->getRefinementRatio() - IntVector(1,1,1));

      cout << "  Doing refine on patch " << finePatch->getID() << " matl " << matl << endl;
      cout << "  Getting phi from " << coarsePhiLow << " " << coarsePhiHigh << " and pi from " << coarsePiLow << " " << coarsePiHigh << endl;

      new_dw->getRegion(coarse_phi, phi_label, matl, coarseLevel, coarsePhiLow, coarsePhiHigh);
      new_dw->getRegion(coarse_pi, pi_label, matl, coarseLevel, coarsePiLow, coarsePiHigh);

      // simple linear interpolation (maybe)
      for(CellIterator iter(phi.getLowIndex(), phi.getHighIndex()); !iter.done(); iter++){
        // refine phi
        refineCell(phi, coarse_phi, *iter, fineLevel, coarseLevel);

        // phi extends (potentially) one more cell than pi.
        IntVector pilow, pihigh;
        pilow = pi.getLow();
        pihigh = pi.getHigh();
        
        if ((*iter).x() < pilow.x() || (*iter).y() < pilow.y() || (*iter).z() < pilow.z() || 
          (*iter).x() > pihigh.x() || (*iter).y() < pihigh.y() || (*iter).z() < pihigh.z()) {
          refineCell(pi, coarse_pi, *iter, fineLevel, coarseLevel);
        }
      }
    } // matls
  } // finePatches
}

void AMRWave::coarsen(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse*, DataWarehouse* new_dw)
{
  const Level* coarseLevel = getLevel(patches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();

  for (int p = 0; p < patches->size(); p++) {  
    const Patch* coarsePatch = patches->get(p);

    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      CCVariable<double> pi;
      CCVariable<double> phi;
      constCCVariable<double> fine_pi;
      constCCVariable<double> fine_phi;
      
      new_dw->getModifiable(phi, phi_label, matl, coarsePatch);
      new_dw->getModifiable(pi, pi_label, matl, coarsePatch);
      
      IntVector fineLow  = coarseLevel->mapCellToFiner(phi.getLowIndex());
      IntVector fineHigh = coarseLevel->mapCellToFiner(phi.getHighIndex());

      cout << "  Doing coarsen on patch " << coarsePatch->getID() << " matl " << matl << endl;
      cout << "  Getting phi from " << fineLow << " " << fineHigh << endl;

      new_dw->getRegion(fine_phi, phi_label, matl, fineLevel, fineLow, fineHigh);
      new_dw->getRegion(fine_pi, pi_label, matl, fineLevel, fineLow, fineHigh);

      // simple linear interpolation (maybe)
      for(CellIterator iter(phi.getLowIndex(), phi.getHighIndex()); !iter.done(); iter++){
        // refine phi
        coarsenCell(phi, fine_phi, *iter, fineLevel, coarseLevel);
        coarsenCell(pi, fine_phi, *iter, fineLevel, coarseLevel);
      }
    } // matls
  } // finePatches
}

void AMRWave::addRefineDependencies(Task* task, const VarLabel* var,
                                    int step, int nsteps)
{
  ASSERTRANGE(step, 0, nsteps+1);
  Ghost::GhostType gc = Ghost::AroundCells;

  if(step != nsteps)
    task->requires(Task::CoarseOldDW, var,
		   0, Task::CoarseLevel, 0, Task::NormalDomain, gc, 1);
  if(step != 0)
    task->requires(Task::CoarseNewDW, var,
		   0, Task::CoarseLevel, 0, Task::NormalDomain, gc, 1);
}

void AMRWave::refineFaces(const Patch* finePatch, 
                 const Level* fineLevel,
		 const Level* coarseLevel, 
		 CCVariable<double>& finevar, 
                 const VarLabel* label,
		 int step, int nsteps, int matl, 
                 DataWarehouse* coarse_old_dw,
		 DataWarehouse* coarse_new_dw)
{
  for (int f = 0; f < 6; f++) {
    Patch::FaceType face = (Patch::FaceType) f;
    if (finePatch->getBCType(face) == Patch::Coarse) {
      IntVector low;
      IntVector high;
      finePatch->getFace(face, IntVector(0,0,0), IntVector(1,1,1), low, high);

      constCCVariable<double> coarse_old_var;
      constCCVariable<double> coarse_new_var;

      if (step != nsteps)
        coarse_old_dw->getRegion(coarse_old_var, label, matl, coarseLevel, low, high);

      if (step != 0)
        coarse_new_dw->getRegion(coarse_new_var, label, matl, coarseLevel, low, high);

      // loop across the face, and depending on where we are in time, use the values 
      // from the old dw, new dw, or both (based on a weight)
      for (CellIterator iter(low, high); !iter.done(); iter++) {
        double x0 = 0, x1 = 0;
        double w1 = ((double) step) / (double) nsteps;
        double w0 = 1-w1;

        if (step != nsteps) {
          refineCell(finevar, coarse_old_var, *iter, fineLevel, coarseLevel);
          x0 = finevar[*iter];
        }
        if (step != 0) {
          refineCell(finevar, coarse_new_var, *iter, fineLevel, coarseLevel);
          x1 = finevar[*iter];
        }
        finevar[*iter] = x0*w0 + x1*w1;
      }

    } // if (finePatch->getBCType(face) == Patch::Coarse) {
  } // for (Patch::FaceType face = 0; face < Patch::numFaces; face++) {
}
