#include <CCA/Components/Examples/AMRHeat.hpp>
#include <Core/Grid/Variables/PerPatchVars.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/Regridder.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/PerPatch.h>

using namespace Uintah;

AMRHeat::AMRHeat(const ProcessorGroup* myworld,
		 const MaterialManagerP materialManager) :
  Heat(myworld, materialManager)
{

}

AMRHeat::~AMRHeat()
{

}

void AMRHeat::problemSetup(const ProblemSpecP&     ps,
                           const ProblemSpecP&     restart_ps,
                                 GridP&            grid)
{
  Heat::problemSetup(ps, restart_ps, grid);
  ProblemSpecP heat_ps   = ps->findBlock("Heat");

  heat_ps->require("refine_threshold", d_refine_threshold);
}

void AMRHeat::scheduleInitialErrorEstimate(const LevelP&     coarseLevel,
                                                 SchedulerP& sched)
{
  scheduleErrorEstimate(coarseLevel, sched);
}

void AMRHeat::scheduleErrorEstimate(const LevelP&     coarseLevel,
                                          SchedulerP& sched)
{
  Task* task = scinew Task("AMRHeat::errorEstimate", this, &AMRHeat::errorEstimate);
  task->requires(Task::NewDW, d_lb->temperature_nc, Ghost::AroundNodes, 1);
  task->modifies(m_regridder->getRefineFlagLabel(), m_regridder->refineFlagMaterials());
  task->modifies(m_regridder->getRefinePatchFlagLabel(), m_regridder->refineFlagMaterials());
  sched->addTask(task, coarseLevel->eachPatch(), m_materialManager->allMaterials());
}

void AMRHeat::errorEstimate(const ProcessorGroup* pg,
                            const PatchSubset*    patches,
                            const MaterialSubset* matls,
                                  DataWarehouse*  old_dw,
                                  DataWarehouse*  new_dw)
{
  double error;

  for(int p = 0; p < patches->size(); p++){
    const Patch *patch = patches->get(p);

    constNCVariable<double> temp;

    CCVariable<int> refine_flag;
    PerPatch<PatchFlagP> refine_patch_flag;

    new_dw->getModifiable(refine_flag, m_regridder->getRefineFlagLabel(), 0, patch);
    new_dw->get(refine_patch_flag, m_regridder->getRefinePatchFlagLabel(), 0, patch);
    new_dw->get(temp, d_lb->temperature_nc, 0, patch, Ghost::AroundNodes, 1);

    PatchFlag* refine_patch = refine_patch_flag.get().get_rep();

    int num_flags = 0;
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      const IntVector& c = *iter;
      error = computeError(c, patch, temp);
      if(error > d_refine_threshold){
        num_flags++;
        refine_flag[c] = true;
      }
    }

    if(num_flags > 0)
      refine_patch->set();

  }
}

double AMRHeat::computeError(const IntVector& c,
                             const Patch* patch,
                                   constNCVariable<double>& temp)
{
  IntVector xoffset(1,0,0);
  IntVector yoffset(0,1,0);
  Vector d = patch->dCell();

  double u_x_0 = (temp[c+xoffset] - temp[c])/d.x();
  double u_x_1 = (temp[c+xoffset+yoffset] - temp[c+yoffset])/d.x();
  double u_y_0 = (temp[c+yoffset] - temp[c])/d.y();
  double u_y_1 = (temp[c+xoffset+yoffset] - temp[c+xoffset])/d.y();

  u_x_0 *= u_x_0;
  u_x_1 *= u_x_1;
  u_y_0 *= u_y_0;
  u_y_1 *= u_y_1;

  if(u_x_0 > d_refine_threshold){
    return u_x_0;
  }else if(u_x_1 > d_refine_threshold){
    return u_x_1;
  }else if(u_y_0 > d_refine_threshold){
    return u_y_0;
  }else if(u_y_1 > d_refine_threshold){
    return u_y_1;
  }else{
    return d_refine_threshold;
  }

}

void AMRHeat::scheduleRefineInterface(const LevelP&     fineLevel,
                                            SchedulerP& scheduler,
                                            bool        needCoarseOld,
                                            bool        needCoarseNew)
{

}

void AMRHeat::scheduleRefine (const PatchSet*   patches,
                                    SchedulerP& sched)
{
  if(getLevel(patches)->hasCoarserLevel()){
    Task* task = scinew Task("AMRHeat::refine", this, &AMRHeat::refine);
    task->requires(Task::NewDW, d_lb->temperature_nc, 0, Task::CoarseLevel,
                   0, Task::NormalDomain, Ghost::AroundNodes, 1);
    task->computes(d_lb->temperature_nc);
    sched->addTask(task, patches, m_materialManager->allMaterials());
  }
}

void AMRHeat::refine(const ProcessorGroup* pg,
                     const PatchSubset*    patches,
                     const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw)
{
  const Level* fine_level = getLevel(patches);
  const Level* coarse_level = fine_level->getCoarserLevel().get_rep();

  std::cout << "Level: " << fine_level->getIndex() << ", Num Patches: " << patches->size() << std::endl;
  for(int p = 0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);

    NCVariable<double> temp;
    constNCVariable<double> coarse_temp;

    new_dw->allocateAndPut(temp, d_lb->temperature_nc, 0, patch);

    IntVector fine_low  = patch->getNodeLowIndex();
    IntVector fine_high = patch->getNodeHighIndex();
    IntVector coarse_low  = fine_level->mapNodeToCoarser(fine_low);
    IntVector coarse_high = fine_level->mapNodeToCoarser(fine_high);
    coarse_high += IntVector(1,1,0);

    new_dw->getRegion(coarse_temp, d_lb->temperature_nc, 0, coarse_level,
                      coarse_low, coarse_high);

    for(NodeIterator iter(fine_low, fine_high); !iter.done(); iter++){
      IntVector n = *iter;
      refineNode(temp, coarse_temp, n, fine_level, coarse_level);
    }
  } // End Patches Loop
}

/**
 * Node Arrangement
 *  n01 ----- n11
 *   |         |
 *   |         |
 *  n00 ----- n10
*/
void AMRHeat::refineNode(NCVariable<double>& temp, constNCVariable<double>& coarse_temp,
                         IntVector fine_index,
                         const Level* fine_level, const Level* coarse_level)
{
  IntVector x_offset(1,0,0);
  IntVector y_offset(0,1,0);
  IntVector crs_index = fine_level->mapNodeToCoarser(fine_index);
  IntVector refine_ratio = coarse_level->getRefinementRatio();
  Point node_pos = fine_level->getNodePosition(fine_index);
  Point coarse_node_pos = coarse_level->getNodePosition(crs_index);
  Vector dcell = coarse_level->dCell();


  Vector dist = (node_pos - coarse_node_pos)/dcell;
  double w00(1), w01(1), w11(1), w10(1);
  IntVector n00(crs_index), n01(crs_index), n11(crs_index), n10(crs_index);

  if((fine_index.x() + refine_ratio.x() - 1)%refine_ratio.x() > 0){
    w00 *= (1-dist.x());
    w01 *= (1-dist.x());
    w10 *= dist.x();
    w11 *= dist.x();
    n10 += x_offset;
    n11 += x_offset;
  }else{
    w10 = 0.0;
    w11 = 0.0;
  }

  if((fine_index.y() + refine_ratio.y() - 1)%refine_ratio.y() > 0){
    w00 *= (1-dist.y());
    w01 *= dist.y();
    w10 *= (1-dist.y());
    w11 *= dist.y();
    n01 += y_offset;
    n11 += y_offset;
  }else{
    w01 = 0.0;
    w11 = 0.0;
  }

  temp[fine_index] = w00*coarse_temp[n00] + w01*coarse_temp[n01]
                   + w10*coarse_temp[n10] + w11*coarse_temp[n11];

}

void AMRHeat::scheduleCoarsen(const LevelP&     coarseLevel,
                                    SchedulerP& sched)
{

}
