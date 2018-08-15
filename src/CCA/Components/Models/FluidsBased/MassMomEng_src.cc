/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <CCA/Components/Models/FluidsBased/MassMomEng_src.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/ICE/Core/ICELabel.h>

using namespace Uintah;
using namespace std;

MassMomEng_src::MassMomEng_src(const ProcessorGroup* myworld,
                               const MaterialManagerP& materialManager,
                               const ProblemSpecP& params)
  : FluidsBasedModel(myworld, materialManager), d_params(params)
{
  mymatls = 0;
  Ilb = scinew ICELabel();
  totalMass_srcLabel = 0;
  totalEng_srcLabel = 0;
  d_src = scinew src();
}

MassMomEng_src::~MassMomEng_src()
{
  delete Ilb;
  delete d_src;
  if(mymatls && mymatls->removeReference()){
    delete mymatls;
  }

  if(0!=totalMass_srcLabel){
    VarLabel::destroy(totalMass_srcLabel);
  }
  if(0!=totalMom_srcLabel){
    VarLabel::destroy(totalMom_srcLabel);
  }
  if(0!=totalEng_srcLabel){
    VarLabel::destroy(totalEng_srcLabel);
  }
}

//______________________________________________________________________
void MassMomEng_src::problemSetup(GridP&, const bool isRestart )
{
  ProblemSpecP src_ps = d_params->findBlock("MassMomEng_src");
  d_matl = m_materialManager->parseAndLookupMaterial(src_ps, "material");

  src_ps->require("momentum_src", d_src->mom_src_rate);
  src_ps->require("mass_src",     d_src->mass_src_rate);
  src_ps->require("energy_src",   d_src->eng_src_rate);
  src_ps->getWithDefault("mme_src_t_start",d_src->d_mme_src_t_start,0.0);
  src_ps->getWithDefault("mme_src_t_final",d_src->d_mme_src_t_final,9.e99);


  vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  mymatls = scinew MaterialSet();
  mymatls->addAll(m);
  mymatls->addReference();   
  
  totalMass_srcLabel  = VarLabel::create( "TotalMass_src",
                                        sum_vartype::getTypeDescription() );
  totalMom_srcLabel  = VarLabel::create("TotalMom_src",
                                        sumvec_vartype::getTypeDescription() );
  totalEng_srcLabel  = VarLabel::create("TotalEng_src",
                                        sum_vartype::getTypeDescription() );
}

//______________________________________________________________________
void MassMomEng_src::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute("type","mass_momentum_energy_src");
  ProblemSpecP src_ps = model_ps->appendChild("MassMomEng_src");
  src_ps->appendElement("material",     d_matl->getName());
  src_ps->appendElement("momentum_src", d_src->mom_src_rate);
  src_ps->appendElement("mass_src",     d_src->mass_src_rate);
  src_ps->appendElement("energy_src",   d_src->eng_src_rate);
}
 
//______________________________________________________________________
void MassMomEng_src::scheduleInitialize(SchedulerP&,
                                   const LevelP& level)
{
  // None necessary...
}

//______________________________________________________________________     
void MassMomEng_src::scheduleComputeStableTimeStep(SchedulerP&,
                                              const LevelP&)
{
  // None necessary...
}

//__________________________________      
void MassMomEng_src::scheduleComputeModelSources(SchedulerP& sched,
                                                const LevelP& level)
{ 
  Task* t = scinew Task("MassMomEng_src::computeModelSources",this, 
                        &MassMomEng_src::computeModelSources);
  t->modifies(Ilb->modelMass_srcLabel);
  t->modifies(Ilb->modelMom_srcLabel);
  t->modifies(Ilb->modelEng_srcLabel);
  t->modifies(Ilb->modelVol_srcLabel);
  
  t->requires(Task::OldDW, Ilb->simulationTimeLabel);
  t->requires(Task::OldDW, Ilb->delTLabel,        level.get_rep());
  t->requires(Task::NewDW, Ilb->sp_vol_CCLabel,   Ghost::None,0);
  t->requires(Task::NewDW, Ilb->vol_frac_CCLabel, Ghost::None,0);
  
  t->computes(MassMomEng_src::totalMass_srcLabel);
  t->computes(MassMomEng_src::totalMom_srcLabel);
  t->computes(MassMomEng_src::totalEng_srcLabel);
  
   
  sched->addTask(t, level->eachPatch(), mymatls);
}

//__________________________________
void MassMomEng_src::computeModelSources(const ProcessorGroup*, 
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)
{
  simTime_vartype simTimeVar;
  old_dw->get(simTimeVar, Ilb->simulationTimeLabel);
  double simTime = simTimeVar;

  delt_vartype delT;
  old_dw->get(delT, Ilb->delTLabel,getLevel(patches));
  double dt = delT;

  int indx = d_matl->getDWIndex();
  double totalMass_src = 0.0;
  double totalEng_src = 0.0;
  Vector totalMom_src(0,0,0);

  // double simTime = m_materialManager->getElapsedSimTime();
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);  
    
    if(simTime>d_src->d_mme_src_t_start && simTime <=d_src->d_mme_src_t_final){
      Vector dx = patch->dCell();
      double vol = dx.x()*dx.y()*dx.z();

      CCVariable<double> mass_src;
      CCVariable<Vector> mom_src;
      CCVariable<double> eng_src;
      CCVariable<double> vol_src;
      constCCVariable<double> sp_vol_CC;
      constCCVariable<double> vol_frac;
    
      new_dw->getModifiable(mass_src, Ilb->modelMass_srcLabel, indx, patch);
      new_dw->getModifiable(mom_src,  Ilb->modelMom_srcLabel,  indx, patch);
      new_dw->getModifiable(eng_src,  Ilb->modelEng_srcLabel,  indx, patch);
      new_dw->getModifiable(vol_src,  Ilb->modelVol_srcLabel,  indx, patch);
      new_dw->get(sp_vol_CC,          Ilb->sp_vol_CCLabel,    indx, patch, Ghost::None,0);
      new_dw->get(vol_frac,           Ilb->vol_frac_CCLabel,  indx, patch, Ghost::None,0);
    
      //__________________________________
      //  Do some work
      // To maintain constant temperature, eng_src must be:
      // mass_src * specific_heat * initial_temperature
      double usr_eng_src  = d_src->eng_src_rate  * dt * vol;
      double usr_mass_src = d_src->mass_src_rate * dt * vol;
      Vector usr_mom_src  = d_src->mom_src_rate  * dt * vol;
    
      vector<Region> regions;
      patch->getFinestRegionsOnPatch(regions);

      for(vector<Region>::iterator region=regions.begin();region!=regions.end();region++){
    
        for (CellIterator iter(region->getLow(), region->getHigh()); !iter.done(); iter++){
          IntVector c = *iter;
        
          if ( vol_frac[c] > 0.001) {
            eng_src[c]  += usr_eng_src*vol_frac[c];
            mass_src[c] += usr_mass_src*vol_frac[c];
            mom_src[c]  += usr_mom_src*vol_frac[c];
//          vol_src[c]  += usr_mass_src * sp_vol_CC[c]*vol_frac[c];// volume src
            totalMass_src += usr_mass_src*vol_frac[c];
            totalMom_src  += usr_mom_src*vol_frac[c];
            totalEng_src  += usr_eng_src*vol_frac[c];
          }
        }
      } // region
      new_dw->put(sum_vartype(totalMass_src),    MassMomEng_src::totalMass_srcLabel);
      new_dw->put(sumvec_vartype(totalMom_src),  MassMomEng_src::totalMom_srcLabel);
      new_dw->put(sum_vartype(totalEng_src),     MassMomEng_src::totalEng_srcLabel);
    }
  }
}
//______________________________________________________________________  
   
void MassMomEng_src::scheduleModifyThermoTransportProperties(SchedulerP&,
                                                        const LevelP&,
                                                        const MaterialSet*)
{
  // do nothing      
}
void MassMomEng_src::computeSpecificHeat(CCVariable<double>&,
                                    const Patch*,   
                                    DataWarehouse*, 
                                    const int)      
{
  //do nothing
}
//______________________________________________________________________
//
void MassMomEng_src::scheduleErrorEstimate(const LevelP&,
                                      SchedulerP&)
{
  // Not implemented yet
}
//__________________________________
void MassMomEng_src::scheduleTestConservation(SchedulerP&,
                                         const PatchSet*)
{
  // Not implemented yet
}
