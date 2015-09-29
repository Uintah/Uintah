/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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


#include <CCA/Components/Examples/Poisson1.h>
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

#include <Kokkos_Core.hpp>

#include <sys/time.h>


namespace {
/** \brief  Time since construction */

class MyTimer {
private:
  struct timeval m_old ;
  MyTimer( const MyTimer & );
  MyTimer & operator = ( const MyTimer & );
public:

  inline
    void reset() {
      gettimeofday( & m_old , ((struct timezone *) NULL ) );
    }

  inline
    ~MyTimer() {}

  inline
    MyTimer() { reset(); }

  inline
    double seconds() const
    {
      struct timeval m_new ;

      ::gettimeofday( & m_new , ((struct timezone *) NULL ) );

      return ( (double) ( m_new.tv_sec  - m_old.tv_sec ) ) +
        ( (double) ( m_new.tv_usec - m_old.tv_usec ) * 1.0e-6 );
    }
};

} // namespace


using namespace std;
using namespace Uintah;

Poisson1::Poisson1(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{

  phi_label = VarLabel::create("phi", 
                               NCVariable<double>::getTypeDescription());
  residual_label = VarLabel::create("residual", 
                                    sum_vartype::getTypeDescription());
}

Poisson1::~Poisson1()
{
  VarLabel::destroy(phi_label);
  VarLabel::destroy(residual_label);
}
//______________________________________________________________________
//
void Poisson1::problemSetup(const ProblemSpecP& params, 
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
void Poisson1::scheduleInitialize(const LevelP& level,
                                  SchedulerP& sched)
{
  Task* task = scinew Task("Poisson1::initialize",
                     this, &Poisson1::initialize);
                     
  task->computes(phi_label);
  task->computes(residual_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void Poisson1::scheduleRestartInitialize(const LevelP& level,
                                         SchedulerP& sched)
{
}
//______________________________________________________________________
//
void Poisson1::scheduleComputeStableTimestep(const LevelP& level,
                                             SchedulerP& sched)
{
  Task* task = scinew Task("Poisson1::computeStableTimestep",
                     this, &Poisson1::computeStableTimestep);
                     
  task->requires(Task::NewDW, residual_label);
  task->computes(sharedState_->get_delt_label(),level.get_rep());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void
Poisson1::scheduleTimeAdvance( const LevelP& level, 
                               SchedulerP& sched)
{
  Task* task = scinew Task("Poisson1::timeAdvance",
                     this, &Poisson1::timeAdvance);
                     
  task->requires(Task::OldDW, phi_label, Ghost::AroundNodes, 1);
  task->computes(phi_label);
  task->computes(residual_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void Poisson1::computeStableTimestep(const ProcessorGroup* pg,
                                     const PatchSubset* patches,
                                     const MaterialSubset* /*matls*/,
                                     DataWarehouse*,
                                     DataWarehouse* new_dw)
{
  if(pg->myrank() == 0){
    sum_vartype residual;
    new_dw->get(residual, residual_label);
    //cerr << "Residual=" << residual << '\n';
  }
  new_dw->put(delt_vartype(delt_), sharedState_->get_delt_label(),getLevel(patches));
}

//______________________________________________________________________
//
void Poisson1::initialize(const ProcessorGroup*,
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

          }
          delete bcb;
        }
      }
    }            
#if 0
    if(patch->getBCType(Patch::xminus) != Patch::Neighbor){
       IntVector l,h;
       patch->getFaceNodes(Patch::xminus, 0, l, h);
 
      for(NodeIterator iter(l,h); !iter.done(); iter++){
        if (phi[*iter] != 1.0) {
          cout << "phi_old[" << *iter << "]=" << phi[*iter] << endl;
        }
         phi[*iter]=1;
      }
    }
#endif

    new_dw->put(sum_vartype(-1), residual_label);
  }
}
//______________________________________________________________________
//

namespace {

struct TimeAdvanceFunctor
{
  constNCVariable<double> & m_phi;
  NCVariable<double> & m_newphi;

  typedef double value_type;

  TimeAdvanceFunctor( constNCVariable<double> & phi, NCVariable<double> & newphi )
    : m_phi( phi )
    , m_newphi( newphi )
  {}

  void operator()(int i, int j, int k, double & residual) const
  {
    const IntVector n(i,j,k);
    m_newphi[n]=(1./6)*(
        m_phi[IntVector(i+1,j,k)] + m_phi[IntVector(i-1,j,k)] +
        m_phi[IntVector(i,j+1,k)] + m_phi[IntVector(i,j-1,k)] +
        m_phi[IntVector(i,j,k+1)] + m_phi[IntVector(i,j,k-1)]);

    double diff = m_newphi[n] - m_phi[n];
    residual += diff * diff;
  }
};

} // namespace

void Poisson1::timeAdvance(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw, 
                           DataWarehouse* new_dw)
{
  int matl = 0;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    constNCVariable<double> phi;
 
    old_dw->get(phi, phi_label, matl, patch, Ghost::AroundNodes, 1);
    NCVariable<double> newphi;
 
    new_dw->allocateAndPut(newphi, phi_label, matl, patch);
    newphi.copyPatch(phi, newphi.getLowIndex(), newphi.getHighIndex());
 
    double residual=0;

    TimeAdvanceFunctor func( phi, newphi );

    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();
 
    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::yminus) == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::zminus) == Patch::Neighbor?0:1);
    h -= IntVector(patch->getBCType(Patch::xplus)  == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::yplus)  == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::zplus)  == Patch::Neighbor?0:1);

    {
     // MyTimer timer;
      Kokkos::parallel_reduce( Kokkos::Range3Policy<int>(l[0],l[1],l[2], h[0],h[1],h[2])
          , func
          , residual
          );
     // std::cerr << timer.seconds() << std::endl;
    }
#if 0
    //__________________________________
    //  Stencil 
    for(NodeIterator iter(l, h);!iter.done(); iter++){
      IntVector n = *iter;
 
      newphi[n]=(1./6)*(
        phi[n+IntVector(1,0,0)] + phi[n+IntVector(-1,0,0)] +
        phi[n+IntVector(0,1,0)] + phi[n+IntVector(0,-1,0)] +
        phi[n+IntVector(0,0,1)] + phi[n+IntVector(0,0,-1)]);
 
      double diff = newphi[n] - phi[n];
      residual += diff * diff;
    }
#endif
    new_dw->put(sum_vartype(residual), residual_label);
  }
}
