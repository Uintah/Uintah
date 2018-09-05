#include <CCA/Components/Examples/Heat.hpp>
#include <CCA/Ports/Scheduler.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BCUtils.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/ReductionVariable.h>
#include <Core/Geometry/IntVector.h>

#include <iostream>
#include <vector>

using namespace Uintah;

Heat::Heat(const ProcessorGroup* myworld,
	   const MaterialManagerP materialManager) :
  ApplicationCommon(myworld, materialManager)
{
  d_lb    = scinew ExamplesLabel();
  d_mat   = 0;
  d_delt  = 0.0;
  d_alpha = 0.0;
  d_r0    = 0.0;
  d_gamma = 1.0;
}

Heat::~Heat()
{
  delete d_lb;
}

void Heat::problemSetup(const ProblemSpecP&     params,
                        const ProblemSpecP&     restart_prob_spec,
                              GridP&            grid)
{
  d_mat = scinew SimpleMaterial();

  ProblemSpecP heat = params->findBlock("Heat");
  heat->require("delt", d_delt);
  heat->require("alpha", d_alpha);
  heat->require("R0", d_r0);
  heat->getWithDefault("gamma", d_gamma, 1.0);

  m_materialManager->registerSimpleMaterial(d_mat);
}

void Heat::scheduleInitialize(const LevelP&     level,
                                    SchedulerP& sched)
{
  Task *task = scinew Task("Heat::initialize", this, &Heat::initialize);
  task->computes(d_lb->temperature_nc);
  sched->addTask(task, level->eachPatch(), m_materialManager->allMaterials());
}

void Heat::initialize(const ProcessorGroup* pg,
                      const PatchSubset*    patches,
                      const MaterialSubset* matls,
                            DataWarehouse*  old_dw,
                            DataWarehouse*  new_dw)
{
  for(int p = 0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    NCVariable<double> temp;
    new_dw->allocateAndPut(temp, d_lb->temperature_nc, 0, patch);
    temp.initialize(0.0);
    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();

    // Just iterate over internal nodes
    /**
    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0:1,
                   patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0:1,
                   0);
                   //patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0:1);

    h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor ? 0:1,
                   patch->getBCType(Patch::yplus) == Patch::Neighbor ? 0:1,
                   0);
                   //patch->getBCType(Patch::zplus) == Patch::Neighbor ? 0:1);
    */

    for(NodeIterator iter(l,h); !iter.done(); iter++){
      IntVector n = *iter;
      Point pnt = patch->getNodePosition(n);
      double r2 = pnt.x()*pnt.x() + pnt.y()*pnt.y();
      double tmp = r2 - d_r0*d_r0;
      temp[n] = tanh(d_gamma*tmp);
    }
  }

}

void Heat::scheduleRestartInitialize(const LevelP&     level,
                                           SchedulerP& sched)
{

}

void Heat::scheduleComputeStableTimeStep(const LevelP&     level,
                                               SchedulerP& sched)
{
  Task *task = scinew Task("Heat::computeStableTimeStep", this,
                           &Heat::computeStableTimeStep);
  task->computes(getDelTLabel(), level.get_rep());
  sched->addTask(task, level->eachPatch(), m_materialManager->allMaterials());
}

void Heat::computeStableTimeStep(const ProcessorGroup* pg,
                                 const PatchSubset*    patches,
                                 const MaterialSubset* matls,
                                       DataWarehouse*  old_dw,
                                       DataWarehouse*  new_dw)
{
  new_dw->put(delt_vartype(d_delt), getDelTLabel(), getLevel(patches));
}

void Heat::scheduleTimeAdvance( const LevelP&     level,
                                      SchedulerP& sched)
{
  Task *task = scinew Task("Heat::timeAdvance", this, &Heat::timeAdvance);
  task->requires(Task::OldDW, d_lb->temperature_nc,
                 Ghost::AroundNodes, Ghost::AroundNodes, 1);
  task->computes(d_lb->temperature_nc);
  sched->addTask(task, level->eachPatch(), m_materialManager->allMaterials());

}

void Heat::timeAdvance(const ProcessorGroup* pg,
                       const PatchSubset*    patches,
                       const MaterialSubset* matls,
                             DataWarehouse*  old_dw,
                             DataWarehouse*  new_dw)
{
  IntVector xoffset(1,0,0);
  IntVector yoffset(0,1,0);

  for(int p = 0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    constNCVariable<double> temp_old;
    NCVariable<double> temp_new;

    Vector d = patch->getLevel()->dCell();
    double dx2 = d.x()*d.x();
    double dy2 = d.y()*d.y();
    //double dz2 = d.z()*d.z();

    old_dw->get(temp_old, d_lb->temperature_nc, 0 , patch, Ghost::AroundNodes, 1);
    new_dw->allocateAndPut(temp_new, d_lb->temperature_nc, 0, patch);
    temp_new.initialize(1.0);

    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();
    IntVector l_in = l;
    IntVector h_in = h;

    // Just iterate over internal nodes
    l_in += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0:1,
                      patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0:1,
                      0);
                      //patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0:1);

    h_in -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor ? 0:1,
                      patch->getBCType(Patch::yplus) == Patch::Neighbor ? 0:1,
                      0);
                      //patch->getBCType(Patch::yplus) == Patch::Neighbor ? 0:1);

    for(NodeIterator iter(l_in,h_in); !iter.done(); iter++){
      IntVector n = *iter;
      double lap = (temp_old[n-xoffset] + temp_old[n+xoffset] - 2*temp_old[n])/dx2;
      lap += (temp_old[n-yoffset] + temp_old[n+yoffset] - 2*temp_old[n])/dy2;
      temp_new[n] = temp_old[n] + d_delt*d_alpha*lap;
    }

    // Start Boundary Condition Section
    std::vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);
    for(std::vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){
      Patch::FaceType face = *itr;
      IntVector oneCell = patch->faceDirection(face);
      std::string bc_kind  = "NotSet";
      std::string face_label = "NotSet";
      int nCells = 0;
      int numChildren = patch->getBCDataArray(face)->getNumberChildren(0);
      Iterator bound_ptr;

      for (int child = 0;  child < numChildren; child++) {
        double bc_value = -9;

        getBCValue<double>(patch, face, child, "Temp", 0, bc_value);
        getBCKind(patch, face, child, "Temp", 0, bc_kind, face_label);

        if(face == Patch::zminus || face == Patch::zplus){
          // This is currently a 2d node centered problem in the xy plane and thus
          // by applying boundary conditions to the z faces computed solutions will
          // be overwritten.
        }else{
          patch->getBCDataArray(face)->getNodeFaceIterator(0, bound_ptr, child);
          if(bc_kind == "Dirichlet"){
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              temp_new[*bound_ptr] = bc_value;
            }
            nCells += bound_ptr.size();
          }
        }
      } // end child loop
    } // end face loop
    // End Boundary Condition Section
  } // end patch loop
}



