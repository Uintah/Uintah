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



#include <CCA/Components/Models/HEChem/ZeroOrder.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Material.h>
#include <CCA/Components/ICE/ICEMaterial.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/ICELabel.h>
#include <CCA/Components/ICE/ICEMaterial.h>
#include <CCA/Components/ICE/BoundaryCond.h>
#include <iostream>
#include <Core/Util/DebugStream.h>

using namespace Uintah;
using namespace std;
//__________________________________
//  setenv SCI_DEBUG "MODELS_DOING_COUT:+"
//  MODELS_DOING_COUT:   dumps when tasks are scheduled and performed
static DebugStream cout_doing("MODELS_DOING_COUT", false);

ZeroOrder::ZeroOrder(const ProcessorGroup* myworld, 
             ProblemSpecP& params,
             const ProblemSpecP& prob_spec)
  : ModelInterface(myworld), d_params(params), d_prob_spec(prob_spec)
{
  mymatls = 0;
  Ilb  = scinew ICELabel();
  d_saveConservedVars = scinew saveConservedVars();
  
  //__________________________________
  //  diagnostic labels
  reactedFractionLabel   = VarLabel::create("F",
                                       CCVariable<double>::getTypeDescription());
  delFLabel              = VarLabel::create("delF",
                                       CCVariable<double>::getTypeDescription());
                     
  totalMassBurnedLabel  = VarLabel::create( "totalMassBurned",
                                            sum_vartype::getTypeDescription() );
  totalHeatReleasedLabel= VarLabel::create( "totalHeatReleased",
                                            sum_vartype::getTypeDescription() );
}

ZeroOrder::~ZeroOrder()
{
  delete Ilb;
  delete d_saveConservedVars;

  VarLabel::destroy(reactedFractionLabel);
  VarLabel::destroy(delFLabel);
  VarLabel::destroy(totalMassBurnedLabel);
  VarLabel::destroy(totalHeatReleasedLabel);
    
  if(mymatls && mymatls->removeReference())
    delete mymatls;
}
//______________________________________________________________________
//
void ZeroOrder::problemSetup(GridP&, SimulationStateP& sharedState, ModelSetup*)
{
  d_sharedState = sharedState;
  bool defaultActive=true;
  d_params->getWithDefault("Active",          d_active, defaultActive);
  d_params->getWithDefault("ThresholdVolFrac",d_threshold_volFrac, 0.01);
  
  d_params->require("ThresholdPressure", d_threshold_pressure);
  d_params->require("fromMaterial",fromMaterial);
  d_params->require("toMaterial",  toMaterial);
  d_params->require("G",    d_G);
  d_params->require("b",    d_b);
  d_params->require("E0",   d_E0);
  d_params->require("rho0", d_rho0);

  //__________________________________
  //  Are we saving the total burned mass and total burned energy
  ProblemSpecP DA_ps = d_prob_spec->findBlock("DataArchiver");
  for (ProblemSpecP child = DA_ps->findBlock("save");
       child != 0;
       child = child->findNextBlock("save") ){
    map<string,string> var_attr;
    child->getAttributes(var_attr);
    
    if (var_attr["label"] == "totalMassBurned"){
      d_saveConservedVars->mass  = true;
    }
    if (var_attr["label"] == "totalHeatReleased"){
      d_saveConservedVars->energy = true;
    }
  }
  
  if(d_active){
    matl0 = sharedState->parseAndLookupMaterial(d_params, "fromMaterial");
    matl1 = sharedState->parseAndLookupMaterial(d_params, "toMaterial");

    //__________________________________
    //  define the materialSet
    mymatls = scinew MaterialSet();

    vector<int> m;
    m.push_back(0);                       // needed for the pressure and NC_CCWeight
    m.push_back(matl0->getDWIndex());
    m.push_back(matl1->getDWIndex());

    mymatls->addAll_unique(m);            // elimiate duplicate entries
    mymatls->addReference(); 
  }
}
//______________________________________________________________________
//
void ZeroOrder::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute("type","ZeroOrder");

  model_ps->appendElement("Active",d_active);
  model_ps->appendElement("ThresholdPressure",d_threshold_pressure);
  model_ps->appendElement("ThresholdVolFrac", d_threshold_volFrac);
  model_ps->appendElement("fromMaterial",fromMaterial);
  model_ps->appendElement("toMaterial",toMaterial);
  model_ps->appendElement("G",    d_G);
  model_ps->appendElement("b",    d_b);
  model_ps->appendElement("E0",   d_E0);
  model_ps->appendElement("rho0", d_rho0);
  
}

//______________________________________________________________________
//
void ZeroOrder::activateModel(GridP&, SimulationStateP& sharedState, ModelSetup*)
{
  d_active=true;

  matl0 = sharedState->parseAndLookupMaterial(d_params, "fromMaterial");
  matl1 = sharedState->parseAndLookupMaterial(d_params, "toMaterial");
 
  //__________________________________
  //  define the materialSet
  mymatls = scinew MaterialSet();

  vector<int> m;
  m.push_back(0);                       // needed for the pressure and NC_CCWeight
  m.push_back(matl0->getDWIndex());
  m.push_back(matl1->getDWIndex());

  mymatls->addAll_unique(m);            // elimiate duplicate entries
  mymatls->addReference(); 
}

//______________________________________________________________________
//     
void ZeroOrder::scheduleInitialize(SchedulerP&,
                                   const LevelP&,
                                   const ModelInfo*)
{
  // None necessary...
}
//______________________________________________________________________
//      
void ZeroOrder::scheduleComputeStableTimestep(SchedulerP&,
                                              const LevelP&,
                                              const ModelInfo*)
{
  // None necessary...
}

//______________________________________________________________________
//     
void ZeroOrder::scheduleComputeModelSources(SchedulerP& sched,
                                            const LevelP& level,
                                            const ModelInfo* mi)
{
  if(d_active){
    Task* t = scinew Task("ZeroOrder::computeModelSources", this, 
                          &ZeroOrder::computeModelSources, mi);
    cout_doing << "ZeroOrder::scheduleComputeModelSources "<<  endl;  
   
    Ghost::GhostType  gn  = Ghost::None;
    const MaterialSubset* react_matl = matl0->thisMaterial();
    const MaterialSubset* prod_matl  = matl1->thisMaterial();
    MaterialSubset* one_matl     = scinew MaterialSubset();
    one_matl->add(0);
    one_matl->addReference();
    MaterialSubset* press_matl   = one_matl;
  
    t->requires(Task::OldDW, mi->delT_Label,         level.get_rep());
    //__________________________________
    // Products
    t->requires(Task::NewDW,  Ilb->rho_CCLabel,      prod_matl, gn);
  
    //__________________________________
    // Reactants
    t->requires(Task::NewDW, Ilb->sp_vol_CCLabel,    react_matl, gn);
    t->requires(Task::OldDW, Ilb->vel_CCLabel,       react_matl, gn);
    t->requires(Task::OldDW, Ilb->temp_CCLabel,      react_matl, gn);
    t->requires(Task::NewDW, Ilb->rho_CCLabel,       react_matl, gn);
    t->requires(Task::NewDW, Ilb->vol_frac_CCLabel,  react_matl, gn);

    t->requires(Task::NewDW, Ilb->press_equil_CCLabel, press_matl,gn);
    t->computes(reactedFractionLabel, react_matl);
    t->computes(delFLabel,            react_matl);

    t->modifies(mi->modelMass_srcLabel);
    t->modifies(mi->modelMom_srcLabel);
    t->modifies(mi->modelEng_srcLabel);
    t->modifies(mi->modelVol_srcLabel); 
    
    if(d_saveConservedVars->mass ){
      t->computes(ZeroOrder::totalMassBurnedLabel);
    }
    if(d_saveConservedVars->energy){
      t->computes(ZeroOrder::totalHeatReleasedLabel);
    } 
    sched->addTask(t, level->eachPatch(), mymatls);

    if (one_matl->removeReference())
      delete one_matl;
  }
}
//______________________________________________________________________
//
void ZeroOrder::scheduleCheckNeedAddMaterial(SchedulerP& sched,
                                             const LevelP& level,
                                             const ModelInfo* mi)
{
    Task* t = scinew Task("ZeroOrder::checkNeedAddMaterial", this, 
                          &ZeroOrder::checkNeedAddMaterial, mi);
    cout_doing << "ZeroOrder::scheduleCheckNeedAddMaterial "<<  endl;  

    Ghost::GhostType  gn  = Ghost::None;

    MaterialSet* one_matl     = scinew MaterialSet();
    one_matl->add(0);
    one_matl->addReference();

    t->requires(Task::NewDW, Ilb->press_equil_CCLabel, one_matl->getUnion(),gn);
    t->computes(Ilb->NeedAddIceMaterialLabel);

    sched->addTask(t, level->eachPatch(), one_matl);

    if (one_matl->removeReference())
      delete one_matl;
}

//______________________________________________________________________
//
void
ZeroOrder::checkNeedAddMaterial(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse* /*old_dw*/,
                                DataWarehouse* new_dw,
                                const ModelInfo* /*mi*/)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);  
    
    cout_doing << "Doing checkNeedAddMaterial on patch "<< patch->getID()
               <<"\t\t\t\t  ZeroOrder" << endl;

    Ghost::GhostType  gn  = Ghost::None;

    constCCVariable<double> press_CC;
    new_dw->get(press_CC,   Ilb->press_equil_CCLabel,0,  patch,gn, 0);

    double need_add=0.;

    if(!d_active){
      bool add = false;
      for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        if (press_CC[c] > .9*d_threshold_pressure){
          add = true;
        }
      }

      if(add){
        need_add=1.;
      }
      else{
        need_add=0.;
      }
    }  //only add a new material once
    else{
      need_add=0.;
    }
    new_dw->put(sum_vartype(need_add),     Ilb->NeedAddIceMaterialLabel);
  }
}
//______________________________________________________________________
//
void ZeroOrder::computeModelSources(const ProcessorGroup*, 
                                    const PatchSubset* patches,
                                    const MaterialSubset*,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw,
                                    const ModelInfo* mi)
{
  delt_vartype delT;
  const Level* level = getLevel(patches);
  old_dw->get(delT, mi->delT_Label, level);

  int m0 = matl0->getDWIndex(); /* reactant material */
  int m1 = matl1->getDWIndex(); /* product material */
  double totalBurnedMass = 0;
  double totalHeatReleased = 0;
 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);  
    
    cout_doing << "Doing computeModelSources on patch "<< patch->getID()
               <<"\t\t\t\t  ZeroOrder" << endl;
    CCVariable<double> mass_src_0, mass_src_1, mass_0;
    CCVariable<Vector> momentum_src_0, momentum_src_1;
    CCVariable<double> energy_src_0, energy_src_1;
    CCVariable<double> sp_vol_src_0, sp_vol_src_1;

    new_dw->getModifiable(mass_src_0,    mi->modelMass_srcLabel,  m0,patch);
    new_dw->getModifiable(momentum_src_0,mi->modelMom_srcLabel,   m0,patch);
    new_dw->getModifiable(energy_src_0,  mi->modelEng_srcLabel,   m0,patch);
    new_dw->getModifiable(sp_vol_src_0,  mi->modelVol_srcLabel,   m0,patch);

    new_dw->getModifiable(mass_src_1,    mi->modelMass_srcLabel,  m1,patch);
    new_dw->getModifiable(momentum_src_1,mi->modelMom_srcLabel,   m1,patch);
    new_dw->getModifiable(energy_src_1,  mi->modelEng_srcLabel,   m1,patch);
    new_dw->getModifiable(sp_vol_src_1,  mi->modelVol_srcLabel,   m1,patch);

    constCCVariable<double> press_CC, cv_reactant,rctVolFrac;
    constCCVariable<double> rctTemp,rctRho,rctSpvol,prodRho;
    constCCVariable<Vector> rctvel_CC;
    CCVariable<double> Fr;
    CCVariable<double> delF;
            
    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();
    Ghost::GhostType  gn  = Ghost::None;    
   
    //__________________________________
    // Reactant data
    old_dw->get(rctTemp,       Ilb->temp_CCLabel,      m0,patch,gn, 0); 
    old_dw->get(rctvel_CC,     Ilb->vel_CCLabel,       m0,patch,gn, 0); 
    new_dw->get(rctRho,        Ilb->rho_CCLabel,       m0,patch,gn, 0); 
    new_dw->get(rctSpvol,      Ilb->sp_vol_CCLabel,    m0,patch,gn, 0); 
    new_dw->get(rctVolFrac,    Ilb->vol_frac_CCLabel,  m0,patch,gn, 0);
    new_dw->allocateAndPut(Fr,   reactedFractionLabel, m0,patch);
    new_dw->allocateAndPut(delF, delFLabel,            m0,patch);
    Fr.initialize(0.);
    delF.initialize(0.);

    //__________________________________
    // Product Data, 
    new_dw->get(prodRho,       Ilb->rho_CCLabel,   m1,patch,gn, 0);

    //__________________________________
    //   Misc.
    new_dw->get(press_CC,         Ilb->press_equil_CCLabel,0,  patch,gn, 0);

    // Get the specific heat, this is the value from the input file
    double cv_rct = -1.0; 
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial *>(d_sharedState->getMaterial(m0));
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial *>(d_sharedState->getMaterial(m0));
    if(mpm_matl) {
      cv_rct = mpm_matl->getSpecificHeat();
    } else if(ice_matl){
      cv_rct = ice_matl->getSpecificHeat();  
    }
    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      if (press_CC[c] > d_threshold_pressure && rctVolFrac[c] > d_threshold_volFrac){          
        //__________________________________
        // Insert Burn Model Here
        double burnedMass;
        double F = prodRho[c]/(rctRho[c]+prodRho[c]);
        if(F >= 0.0 && F < 1.0){
          delF[c] = d_G*pow(press_CC[c],d_b);
        }
        delF[c]*=delT;
        Fr[c] = F;
        double rctMass = rctRho[c]*cell_vol;
        double prdMass = prodRho[c]*cell_vol;
        burnedMass = min(delF[c]*(prdMass+rctMass), rctMass);

        // Prevent burning a whole cell in a timestep (this is to maintain stability)
        burnedMass = min(burnedMass, 0.2*d_rho0*cell_vol);

        //__________________________________
        // conservation of mass, momentum and energy                           
        mass_src_0[c]   -= burnedMass;
        mass_src_1[c]   += burnedMass;
        totalBurnedMass += burnedMass;
           
        Vector momX        = rctvel_CC[c] * burnedMass;
        momentum_src_0[c] -= momX;
        momentum_src_1[c] += momX;

        double energyX   = cv_rct*rctTemp[c]*burnedMass; 
        double releasedHeat = burnedMass * d_E0;
        energy_src_0[c]   -= energyX;
        energy_src_1[c]   += energyX + releasedHeat;
        totalHeatReleased += releasedHeat;

        double createdVolx  = burnedMass * rctSpvol[c];
        sp_vol_src_0[c] -= createdVolx;
        sp_vol_src_1[c] += createdVolx;
      }  // if (pressure)
    }  // cell iterator  

    //__________________________________
    //  set symetric BC
    setBC(mass_src_0, "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
    setBC(mass_src_1, "set_if_sym_BC",patch, d_sharedState, m1, new_dw);
    setBC(delF,       "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
    setBC(Fr,         "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
  }
  //__________________________________
  //save total quantities
  if(d_saveConservedVars->mass ){
    new_dw->put(sum_vartype(totalBurnedMass),   ZeroOrder::totalMassBurnedLabel);
  }
  if(d_saveConservedVars->energy){
    new_dw->put(sum_vartype(totalHeatReleased), ZeroOrder::totalHeatReleasedLabel);
  }
}
//______________________________________________________________________
//
void ZeroOrder::scheduleModifyThermoTransportProperties(SchedulerP&,
                                                        const LevelP&,
                                                        const MaterialSet*)
{
  // do nothing      
}
void ZeroOrder::computeSpecificHeat(CCVariable<double>&,
                                    const Patch*,   
                                    DataWarehouse*, 
                                    const int)      
{
  //do nothing
}
//______________________________________________________________________
//
void ZeroOrder::scheduleErrorEstimate(const LevelP&,
                                      SchedulerP&)
{
  // Not implemented yet
}
//__________________________________
void ZeroOrder::scheduleTestConservation(SchedulerP&,
                                         const PatchSet*,                      
                                         const ModelInfo*)                     
{
  // Not implemented yet
}
