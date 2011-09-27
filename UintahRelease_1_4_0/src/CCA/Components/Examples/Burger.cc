/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#include <CCA/Components/Examples/Burger.h>
#include <CCA/Components/Examples/ExamplesLabel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>

using namespace Uintah;
//______________________________________________________________________
//  Preliminary
Burger::Burger(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
  u_label = VarLabel::create("u", NCVariable<double>::getTypeDescription());
}

Burger::~Burger()
{
  VarLabel::destroy(u_label);
}
//______________________________________________________________________
//
void Burger::problemSetup(const ProblemSpecP& params, 
                          const ProblemSpecP& restart_prob_spec, 
                          GridP& /*grid*/,  
                          SimulationStateP& sharedState)
{
  sharedState_ = sharedState;
  ProblemSpecP burger = params->findBlock("Burger");
  burger->require("delt", delt_);
  mymat_ = scinew SimpleMaterial();
  sharedState->registerSimpleMaterial(mymat_);
}
 
//______________________________________________________________________
// 
void Burger::scheduleInitialize(const LevelP& level,
                                   SchedulerP& sched)
{
  Task* task = scinew Task("Burger::initialize",
                     this, &Burger::initialize);
  task->computes(u_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
// 
void Burger::scheduleComputeStableTimestep(const LevelP& level,
                                          SchedulerP& sched)
{
  Task* task = scinew Task("Burger::computeStableTimestep",
                     this, &Burger::computeStableTimestep);
                     
  task->computes(sharedState_->get_delt_label(),level.get_rep());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void  Burger::scheduleTimeAdvance( const LevelP& level, 
                                   SchedulerP& sched)
{
  Task* task = scinew Task("Burger::timeAdvance",
                     this, &Burger::timeAdvance);
                     
  task->requires(Task::OldDW, u_label, Ghost::AroundNodes, 1);
  task->requires(Task::OldDW, sharedState_->get_delt_label());
  
  task->computes(u_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}

//______________________________________________________________________
//
void Burger::computeStableTimestep(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset*,
                                   DataWarehouse*, 
                                   DataWarehouse* new_dw)
{
  new_dw->put(delt_vartype(delt_), sharedState_->get_delt_label(),getLevel(patches));
}

//______________________________________________________________________
//
void Burger::initialize(const ProcessorGroup*,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse*, 
                        DataWarehouse* new_dw)
{
  int matl = 0;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    NCVariable<double> u;
    new_dw->allocateAndPut(u, u_label, matl, patch);
    
    //Initialize
    // u = sin( pi*x ) + sin( pi*2*y ) + sin(pi*3z )
    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();
    
    for( NodeIterator iter=patch->getNodeIterator(); !iter.done(); iter++ ){
      IntVector n = *iter;
      Point p = patch->nodePosition(n);
      u[n] = sin( p.x() * 3.14159265358 ) + sin( p.y() * 2*3.14159265358)  +  sin( p.z() * 3*3.14159265358);
    }
  }
}

//______________________________________________________________________
//
void Burger::timeAdvance(const ProcessorGroup*,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw, 
                         DataWarehouse* new_dw)
{
  int matl = 0;
  //Loop for all patches on this processor
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    //  Get data from the data warehouse including 1 layer of
    // "ghost" nodes from surrounding patches
    constNCVariable<double> u;
    old_dw->get(u, u_label, matl, patch, Ghost::AroundNodes, 1);

    // dt, dx
    Vector dx = patch->getLevel()->dCell();
    delt_vartype dt;
    old_dw->get(dt, sharedState_->get_delt_label());
    
    // allocate memory
    NCVariable<double> new_u;
    new_dw->allocateAndPut(new_u, u_label, matl, patch);
    
    // define iterator range
    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();

    //offset to prevent accessing memory out-of-bounds
    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::yminus) == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::zminus) == Patch::Neighbor?0:1 );
                   
    h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::yplus) == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::zplus) == Patch::Neighbor?0:1 );
                   
    //Iterate through all the nodes
    for(NodeIterator iter(l, h);!iter.done(); iter++){    
      IntVector n = *iter;
      double dudx = (u[n+IntVector(1,0,0)] - u[n-IntVector(1,0,0)]) /(2.0 * dx.x());
      double dudy = (u[n+IntVector(0,1,0)] - u[n-IntVector(0,1,0)]) /(2.0 * dx.y());
      double dudz = (u[n+IntVector(0,0,1)] - u[n-IntVector(0,0,1)]) /(2.0 * dx.z());
      double du = - u[n] * dt * (dudx + dudy + dudz);
      new_u[n]= u[n] + du;
    }

    //__________________________________
    // Boundary conditions: Neumann
    // Iterate over the faces encompassing the domain
    vector<Patch::FaceType>::const_iterator iter;
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);
    for (iter  = bf.begin(); iter != bf.end(); ++iter){
      Patch::FaceType face = *iter;

      IntVector axes = patch->getFaceAxes(face);
      int P_dir = axes[0]; // find the principal dir of that face

      IntVector offset(0,0,0);
      if (face == Patch::xminus || face == Patch::yminus || face == Patch::zminus){
        offset[P_dir] += 1; 
      }
      if (face == Patch::xplus || face == Patch::yplus || face == Patch::zplus){
        offset[P_dir] -= 1;
      }

      Patch::FaceIteratorType FN = Patch::FaceNodes;
      for (CellIterator iter = patch->getFaceIterator(face,FN);!iter.done(); iter++){
        IntVector n = *iter;
        new_u[n] = new_u[n + offset];
      }
    }
  }
}
