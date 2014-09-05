
#include <Packages/Uintah/CCA/Components/Examples/AMRWave.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
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

void AMRWave::scheduleRefine (const LevelP& fineLevel, SchedulerP& sched)
{
  Task* task = scinew Task("refine", this, &AMRWave::refine);
  task->requires(Task::NewDW, phi_label, 0, Task::CoarseLevel, 0, Task::NormalDomain, Ghost::None, 0);
  task->computes(phi_label);
  task->requires(Task::NewDW, pi_label, 0, Task::CoarseLevel, 0, Task::NormalDomain, Ghost::None, 0);
  task->computes(pi_label);
  sched->addTask(task, fineLevel->eachPatch(), sharedState_->allMaterials());
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

        // Compute curl
        double curlPhi = sumdx2 * phi[c]
          + (phi[c+IntVector(1,0,0)] + phi[c-IntVector(1,0,0)]) * inv_dx2.x()
          + (phi[c+IntVector(0,1,0)] + phi[c-IntVector(0,1,0)]) * inv_dx2.y()
          + (phi[c+IntVector(0,0,1)] + phi[c-IntVector(0,0,1)]) * inv_dx2.z();

        if(curlPhi > thresh){
          numFlag++;
          refineFlag[c] = true;
        }
      }
      cerr << "numFlag=" << numFlag << '\n';
      if(numFlag != 0)
        refinePatch->set();
    }
  }
}


void AMRWave::refine(const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse*, DataWarehouse* new_dw)
{
  cerr << "AMRWave::refine not finished\n";
}

void AMRWave::coarsen(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse*, DataWarehouse* new_dw)
{
  cerr << "AMRWave::coarsen not finished\n";
}
