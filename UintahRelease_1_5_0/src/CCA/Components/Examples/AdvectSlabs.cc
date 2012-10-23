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


#include <CCA/Components/Examples/AdvectSlabs.h>
#include <CCA/Components/Examples/ExamplesLabel.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>
#include <Core/Util/FancyAssert.h>

using namespace Uintah;

AdvectSlabs::AdvectSlabs(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
  mass_label = VarLabel::create("mass", 
                               CCVariable<double>::getTypeDescription());
  massAdvected_label = VarLabel::create("massAdvected", 
                                  CCVariable<double>::getTypeDescription());
    
    //__________________________________
    //  outflux/influx slabs
    OF_slab[RIGHT] = RIGHT;         IF_slab[RIGHT]  = LEFT;
    OF_slab[LEFT]  = LEFT;          IF_slab[LEFT]   = RIGHT;
    OF_slab[TOP]   = TOP;           IF_slab[TOP]    = BOTTOM;
    OF_slab[BOTTOM]= BOTTOM;        IF_slab[BOTTOM] = TOP;  
    OF_slab[FRONT] = FRONT;         IF_slab[FRONT]  = BACK;
    OF_slab[BACK]  = BACK;          IF_slab[BACK]   = FRONT;   
    
    // Slab adjacent cell
    S_ac[RIGHT]  =  IntVector( 1, 0, 0);   
    S_ac[LEFT]   =  IntVector(-1, 0, 0);   
    S_ac[TOP]    =  IntVector( 0, 1, 0);   
    S_ac[BOTTOM] =  IntVector( 0,-1, 0);   
    S_ac[FRONT]  =  IntVector( 0, 0, 1);   
    S_ac[BACK]   =  IntVector( 0, 0,-1);
}

AdvectSlabs::~AdvectSlabs()
{
    
}

void AdvectSlabs::problemSetup(const ProblemSpecP& params,
                            const ProblemSpecP& restart_prob_spec,
                            GridP&, SimulationStateP& sharedState)
{
  sharedState_ = sharedState;
  ProblemSpecP ps = params->findBlock("AdvectSlabs");
  ps->require("delt", delt_);
  mymat_ = scinew SimpleMaterial();
  sharedState->registerSimpleMaterial(mymat_);
}
 
void AdvectSlabs::scheduleInitialize(const LevelP& level,
			       SchedulerP& sched)
{
  Task* task = scinew Task("initialize",
			   this, &AdvectSlabs::initialize);
  task->computes(mass_label);
  task->computes(massAdvected_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
 
void AdvectSlabs::scheduleComputeStableTimestep(const LevelP& level,
					  SchedulerP& sched)
{
  Task* task = scinew Task("computeStableTimestep",
			   this, &AdvectSlabs::computeStableTimestep);
  task->computes(sharedState_->get_delt_label(),level.get_rep());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}

void
AdvectSlabs::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched)
{
  Task* task = scinew Task("timeAdvance",
			   this, &AdvectSlabs::timeAdvance);

  task->requires(Task::OldDW, mass_label, Ghost::AroundCells, 2);
  task->computes(mass_label);
  task->computes(massAdvected_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

}

void AdvectSlabs::computeStableTimestep(const ProcessorGroup*,
				  const PatchSubset* patches,
				  const MaterialSubset*,
				  DataWarehouse*, DataWarehouse* new_dw)
{
  new_dw->put(delt_vartype(delt_), sharedState_->get_delt_label(),getLevel(patches));
}

void AdvectSlabs::initialize(const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset* matls,
		       DataWarehouse*old_dw, DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
      
    new_dw->allocateTemporary(d_OFS, patch, Ghost::AroundCells,1);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      CCVariable<double> mass, massAd;
      new_dw->allocateAndPut(mass,   mass_label,         matl, patch, Ghost::AroundCells, 2);
      new_dw->allocateAndPut(massAd, massAdvected_label, matl, patch, Ghost::AroundCells, 2);
      mass.initialize(0.0);
      massAd.initialize(0.0);
        
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++)
      {
        // set initial value for fluxes
        for(int face = TOP; face <= BACK; face++ )  {
          d_OFS[*iter].d_fflux[face]= 1;
        }
        // set up the initial mass
        mass[*iter]=1;
      }
    }
  }
}

void AdvectSlabs::timeAdvance(const ProcessorGroup* pg,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw, DataWarehouse* new_dw)
{
    struct fflux ff; 
    for(int p=0;p<patches->size();p++){
        const Patch* patch = patches->get(p);
        Vector dx = patch->dCell();            
        double invvol = 1.0/(dx.x() * dx.y() * dx.z());                     


        d_OFS.initialize(ff);
        for(int m = 0;m<matls->size();m++){
          int matl = matls->get(m);



          // variable to get
          constCCVariable<double> mass;
          CCVariable<double>      mass2, massAd;
        
          old_dw->get(mass, mass_label, matl, patch, Ghost::AroundCells, 2);
          new_dw->allocateAndPut(mass2, mass_label, matl, patch, Ghost::AroundCells, 2 );
          new_dw->allocateAndPut(massAd, massAdvected_label, matl, patch, Ghost::AroundCells, 2 );
          for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
              const IntVector& c = *iter;  
        
              double q_face_flux[6];
              double faceVol[6];
          
              double sum_q_face_flux(0.0);   
              for(int f = TOP; f <= BACK; f++ )  {    
                //__________________________________
                //   S L A B S
                // q_CC: vol_frac, mass, momentum, int_eng....
                //      for consistent units you need to divide by cell volume
                // 
                IntVector ac = c + S_ac[f];     // slab adjacent cell

                double outfluxVol = d_OFS[c ].d_fflux[OF_slab[f]];
                double influxVol  = d_OFS[ac].d_fflux[IF_slab[f]];
                
                double q_faceFlux_tmp  =   mass[ac] * influxVol - mass[c] * outfluxVol;
            
                faceVol[f]       =  outfluxVol +  influxVol;
                q_face_flux[f]   = q_faceFlux_tmp; 
                sum_q_face_flux += q_faceFlux_tmp;
              }  
              massAd[c] = sum_q_face_flux*invvol;
              mass2[c] = mass[c] - massAd[c];
           }
        }
    }
}



