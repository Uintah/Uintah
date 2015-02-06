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



#include <CCA/Components/Examples/Poisson4.h>
#include <CCA/Components/Examples/ExamplesLabel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/GridSurfaceIterator.h>
#include <Core/Grid/Variables/GridIterator.h>
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

using namespace std;
using namespace Uintah;

Poisson4::Poisson4(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{

  phi_label = VarLabel::create("phi", 
                               NCVariable<double>::getTypeDescription());
  residual_label = VarLabel::create("residual", 
                                    sum_vartype::getTypeDescription());

}

Poisson4::~Poisson4()
{
  VarLabel::destroy(phi_label);
  VarLabel::destroy(residual_label);
}
//______________________________________________________________________
//
void Poisson4::problemSetup(const ProblemSpecP& params, 
                            const ProblemSpecP& restart_prob_spec, 
                            GridP& /*grid*/, 
                            SimulationStateP& sharedState)
{
  sharedState_ = sharedState;
  ProblemSpecP poisson = params->findBlock("Poisson");
  
  poisson->require("delt", delt_);
  
  mymat_ = scinew SimpleMaterial();
  
  sharedState->registerSimpleMaterial(mymat_);
}
//______________________________________________________________________
//
void Poisson4::scheduleInitialize(const LevelP& level,
                                  SchedulerP& sched)
{
  Task* task = scinew Task("Poisson4::initialize",
                     this, &Poisson4::initialize);
                     
  task->computes(phi_label);
  task->computes(residual_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}

//______________________________________________________________________
//
void Poisson4::scheduleComputeStableTimestep(const LevelP& level,
                                             SchedulerP& sched)
{
  Task* task = scinew Task("Poisson4::computeStableTimestep",
                     this, &Poisson4::computeStableTimestep);
                     
  task->requires(Task::NewDW, residual_label);
  task->computes(sharedState_->get_delt_label(),level.get_rep());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void Poisson4::scheduleTimeAdvance0(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls)
{
  Task* task = scinew Task("Poisson4::timeAdvance",
                     this, &Poisson4::timeAdvance);
                     
  task->requires(Task::OldDW, phi_label, Ghost::AroundNodes, 1);
  task->computes(phi_label);
  task->computes(residual_label);
  sched->addTask(task, patches, matls);
}
//______________________________________________________________________
//

void Poisson4::scheduleTimeAdvance1(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls)
{
  Task* task = scinew Task("Poisson4::timeAdvance",
                     this, &Poisson4::timeAdvance1);
                     
  //  task->requires(Task::NewDW, phi_label, Ghost::AroundNodes, 1);
  task->modifies(phi_label);
  task->modifies(residual_label);
  sched->addTask(task, patches, matls);
}
//______________________________________________________________________
//
void
Poisson4::scheduleTimeAdvance( const LevelP& level, 
                               SchedulerP& sched)
{

  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = sharedState_->allMaterials();

  scheduleTimeAdvance0(sched,patches,matls);

  scheduleTimeAdvance1(sched,patches,matls);


#if 0
  Task* task = scinew Task("Poisson4::timeAdvance",
                     this, &Poisson4::timeAdvance);
                     
  task->requires(Task::OldDW, phi_label, Ghost::AroundNodes, 1);
  task->computes(phi_label);
  task->computes(residual_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
#endif
}
//______________________________________________________________________
//
void Poisson4::computeStableTimestep(const ProcessorGroup* pg,
                                     const PatchSubset* patches,
                                     const MaterialSubset* /*matls*/,
                                     DataWarehouse*,
                                     DataWarehouse* new_dw)
{
  if(pg->myrank() == 0){
    sum_vartype residual;
    new_dw->get(residual, residual_label);
    cerr << "Residual=" << residual << '\n';
  }
  new_dw->put(delt_vartype(delt_), sharedState_->get_delt_label(),getLevel(patches));
}

//______________________________________________________________________
//
void Poisson4::initialize(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* /*old_dw*/, DataWarehouse* new_dw)
{
  int matl = 0;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    NCVariable<double> phi;
    new_dw->allocateAndPut(phi, phi_label, matl, patch);
    phi.initialize(0.);

    for (Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
         face=Patch::nextFace(face)) {

      IntVector l = patch->getNodeLowIndex();
      IntVector h = patch->getNodeHighIndex();
#if 0
      if (patch->getBCType(face) == Patch::None) {
        int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl);
        for (int child = 0; child < numChildren; child++) {
          Iterator nbound_ptr, nu;
          
          const BoundCondBase* bcb = patch->getArrayBCValues(face,matl,"Phi",nu,
                                                             nbound_ptr,child);
          
          const BoundCond<double>* bc = 
            dynamic_cast<const BoundCond<double>*>(bcb); 
          double value = bc->getValue();
          for (nbound_ptr.reset(); !nbound_ptr.done();nbound_ptr++) {

            phi[*nbound_ptr]=value;
            // cout << "phi" << *nbound_ptr  << "=" << phi[*nbound_ptr] << endl;

          }
          delete bcb;
        }
      }
      for(NodeIterator iter(l, h);!iter.done(); iter++) {
        cout << "phi" << *iter  << "=" << phi[*iter] << endl;
      }
    }
#endif            
#if 1
    if(patch->getBCType(Patch::xminus) != Patch::Neighbor){
       IntVector l,h;
       patch->getFaceNodes(Patch::xminus, 1, l, h);
 
      for(NodeIterator iter(l,h); !iter.done(); iter++){
        if (phi[*iter] != 1.0) {
          cout << "phi_old[" << *iter << "]=" << phi[*iter] << endl;
        }
         phi[*iter]=1;
      }
    }
    
    IntVector l_e = patch->getExtraNodeLowIndex();
    IntVector h_e = patch->getExtraNodeHighIndex();
    for (NodeIterator iter(l_e,h_e); !iter.done(); iter++) {
      cout << "phi[" << *iter << "]=" << phi[*iter] << endl;
    }
  }
#endif

    new_dw->put(sum_vartype(-1), residual_label);
  }
}
//______________________________________________________________________
//
void Poisson4::timeAdvance(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw, 
                           DataWarehouse* new_dw)
{
  int matl = 0;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    constNCVariable<double> phi;
 
    int ngn=1;
    old_dw->get(phi, phi_label, matl, patch, Ghost::AroundNodes, ngn);
    NCVariable<double> newphi;
 
    new_dw->allocateAndPut(newphi, phi_label, matl, patch);
    newphi.copyPatch(phi, newphi.getLowIndex(), newphi.getHighIndex());
 
    double residual=0;
    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();
    IntVector l_ngn = patch->getNodeLowIndex(-ngn);
    IntVector h_ngn = patch->getNodeHighIndex(-ngn);
    IntVector lextra = patch->getExtraNodeLowIndex();
    IntVector hextra = patch->getExtraNodeHighIndex();
 
    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::yminus) == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::zminus) == Patch::Neighbor?0:1);
    h -= IntVector(patch->getBCType(Patch::xplus)  == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::yplus)  == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::zplus)  == Patch::Neighbor?0:1);
    
    //__________________________________
    //  Stencil 

#define SURFACE_ITERATOR
    //#undef SURFACE_ITERATOR
#ifdef SURFACE_ITERATOR
    // Iterate over interior nodes
      GridIterator iter_interior(l,h);

    // Iterate over exterior nodes
      GridSurfaceIterator iter_exterior(lextra,hextra,l,h);

#else
    for(NodeIterator iter(l, h);!iter.done(); iter++) {
#endif

#ifdef SURFACE_ITERATOR
      cout << "Doing interior iteration" << endl;
      while (!iter_interior.done()) {
        IntVector n = *iter_interior;
#else
        IntVector n = *iter;
#endif
        
        newphi[n]=(1./6)*(
                          phi[n+IntVector(1,0,0)] + phi[n+IntVector(-1,0,0)] +
                          phi[n+IntVector(0,1,0)] + phi[n+IntVector(0,-1,0)] +
                          phi[n+IntVector(0,0,1)] + phi[n+IntVector(0,0,-1)]);
        
        double diff = newphi[n] - phi[n];
        residual += diff * diff;
#ifdef SURFACE_ITERATOR
        iter_interior++;
#else
        iter++;
#endif
      }
#ifdef SURFACE_ITERATOR
      cout << "Doing exterior iteration" << endl;
      while (!iter_exterior.done()) {
        IntVector n = *iter_exterior;
        cout << "iterator = " << n << endl;
        newphi[n]=(1./6)*(
                          phi[n+IntVector(1,0,0)] + phi[n+IntVector(-1,0,0)] +
                          phi[n+IntVector(0,1,0)] + phi[n+IntVector(0,-1,0)] +
                          phi[n+IntVector(0,0,1)] + phi[n+IntVector(0,0,-1)]);
        
        double diff = newphi[n] - phi[n];
        residual += diff * diff;
        iter_exterior++;
      }
#endif
    new_dw->put(sum_vartype(residual), residual_label);
  }
}


void Poisson4::timeAdvance1(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw, 
                            DataWarehouse* new_dw)
{
  int matl = 0;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    NCVariable<double> phi;
 
    int ngn=1;
    new_dw->getModifiable(phi, phi_label, matl, patch);
    NCVariable<double> newphi;
 
    new_dw->allocateAndPut(newphi, phi_label, matl, patch);
    newphi.copyPatch(phi, newphi.getLowIndex(), newphi.getHighIndex());
 
    double residual=0;
    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();
    IntVector l_ngn = patch->getNodeLowIndex(-ngn);
    IntVector h_ngn = patch->getNodeHighIndex(-ngn);
    IntVector lextra = patch->getExtraNodeLowIndex();
    IntVector hextra = patch->getExtraNodeHighIndex();
 
    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::yminus) == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::zminus) == Patch::Neighbor?0:1);
    h -= IntVector(patch->getBCType(Patch::xplus)  == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::yplus)  == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::zplus)  == Patch::Neighbor?0:1);
    
    //__________________________________
    //  Stencil 

#define SURFACE_ITERATOR
    //#undef SURFACE_ITERATOR
#ifdef SURFACE_ITERATOR
    // Iterate over interior nodes
      GridIterator iter_interior(l,h);

    // Iterate over exterior nodes
      GridSurfaceIterator iter_exterior(lextra,hextra,l,h);

#else
    for(NodeIterator iter(l, h);!iter.done(); iter++) {
#endif

#ifdef SURFACE_ITERATOR
      cout << "Doing interior iteration" << endl;
      while (!iter_interior.done()) {
        IntVector n = *iter_interior;
#else
        IntVector n = *iter;
#endif
        
        newphi[n]=(1./6)*(
                          phi[n+IntVector(1,0,0)] + phi[n+IntVector(-1,0,0)] +
                          phi[n+IntVector(0,1,0)] + phi[n+IntVector(0,-1,0)] +
                          phi[n+IntVector(0,0,1)] + phi[n+IntVector(0,0,-1)]);
        
        double diff = newphi[n] - phi[n];
        residual += diff * diff;
#ifdef SURFACE_ITERATOR
        iter_interior++;
#else
        iter++;
#endif
      }
#ifdef SURFACE_ITERATOR
      cout << "Doing exterior iteration" << endl;
      while (!iter_exterior.done()) {
        IntVector n = *iter_exterior;
        cout << "iterator = " << n << endl;
        newphi[n]=(1./6)*(
                          phi[n+IntVector(1,0,0)] + phi[n+IntVector(-1,0,0)] +
                          phi[n+IntVector(0,1,0)] + phi[n+IntVector(0,-1,0)] +
                          phi[n+IntVector(0,0,1)] + phi[n+IntVector(0,0,-1)]);
        
        double diff = newphi[n] - phi[n];
        residual += diff * diff;
        iter_exterior++;
      }
#endif
    new_dw->put(sum_vartype(residual), residual_label);
  }
}
