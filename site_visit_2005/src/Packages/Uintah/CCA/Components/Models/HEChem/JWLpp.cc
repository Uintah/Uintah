
#include <Packages/Uintah/CCA/Components/Models/HEChem/JWLpp.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Material.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Labels/ICELabel.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <iostream>
#include <Core/Util/DebugStream.h>

using namespace Uintah;
using namespace std;
//__________________________________
//  setenv SCI_DEBUG "MODELS_DOING_COUT:+"
//  MODELS_DOING_COUT:   dumps when tasks are scheduled and performed
static DebugStream cout_doing("MODELS_DOING_COUT", false);

JWLpp::JWLpp(const ProcessorGroup* myworld, ProblemSpecP& params)
  : ModelInterface(myworld), params(params)
{
  mymatls = 0;
  Ilb  = scinew ICELabel();
  //__________________________________
  //  diagnostic labels
  reactedFractionLabel   = VarLabel::create("F",
                     CCVariable<double>::getTypeDescription());
                     
  delFLabel   = VarLabel::create("delF",
                     CCVariable<double>::getTypeDescription());
}

JWLpp::~JWLpp()
{
  delete Ilb;

  VarLabel::destroy(reactedFractionLabel);
  VarLabel::destroy(delFLabel);
  
  if(mymatls && mymatls->removeReference())
    delete mymatls;
}

void JWLpp::problemSetup(GridP&, SimulationStateP& sharedState,
			     ModelSetup*)
{
  cout << "I'm in problem setup" << endl;
  d_sharedState = sharedState;
  bool defaultActive=true;
  params->getWithDefault("Active", d_active, defaultActive);
  params->require("ThresholdPressure",   d_threshold_pressure);
  if(d_active){
    matl0 = sharedState->parseAndLookupMaterial(params, "fromMaterial");
    matl1 = sharedState->parseAndLookupMaterial(params, "toMaterial");
    params->require("G",    d_G);
    params->require("b",    d_b);
    params->require("E0",   d_E0);
    params->require("rho0", d_rho0);

    //__________________________________
    //  define the materialSet
    vector<int> m_tmp(2);
    m_tmp[0] = matl0->getDWIndex();
    m_tmp[1] = matl1->getDWIndex();
    mymatls = new MaterialSet();            
 
    if( m_tmp[0] != 0 && m_tmp[1] != 0){
      vector<int> m(3);
      m[0] = 0;    // needed for the pressure and NC_CCWeight 
      m[1] = m_tmp[0];
      m[2] = m_tmp[1];
      mymatls->addAll(m);
    }else{
      vector<int> m(2);
      m[0] = m_tmp[0];
      m[1] = m_tmp[1];
      mymatls->addAll(m);
    }
    mymatls->addReference();
  }
}

void JWLpp::activateModel(GridP&, SimulationStateP& sharedState, ModelSetup*)
{
  cout << "I'm in activateModel" << endl;
  d_active=true;
  matl0 = sharedState->parseAndLookupMaterial(params, "fromMaterial");
  matl1 = sharedState->parseAndLookupMaterial(params, "toMaterial");
  params->require("G",    d_G);
  params->require("b",    d_b);
  params->require("E0",   d_E0);
  params->require("rho0", d_rho0);

  //__________________________________
  //  define the materialSet
  vector<int> m_tmp(2);
  m_tmp[0] = matl0->getDWIndex();
  m_tmp[1] = matl1->getDWIndex();
  mymatls = new MaterialSet();
                                                                                
  if( m_tmp[0] != 0 && m_tmp[1] != 0){
    vector<int> m(3);
    m[0] = 0;    // needed for the pressure and NC_CCWeight
    m[1] = m_tmp[0];
    m[2] = m_tmp[1];
    mymatls->addAll(m);
  }else{
    vector<int> m(2);
    m[0] = m_tmp[0];
    m[1] = m_tmp[1];
    mymatls->addAll(m);
  }
  mymatls->addReference();
}

//______________________________________________________________________
//     
void JWLpp::scheduleInitialize(SchedulerP&,
                               const LevelP&,
                               const ModelInfo*)
{
  // None necessary...
}
//______________________________________________________________________
//      
void JWLpp::scheduleComputeStableTimestep(SchedulerP&,
                                          const LevelP&,
                                          const ModelInfo*)
{
  // None necessary...
}

//______________________________________________________________________
//     
void JWLpp::scheduleComputeModelSources(SchedulerP& sched,
                                       const LevelP& level,
                                       const ModelInfo* mi)
{
  if(d_active){
    Task* t = scinew Task("JWLpp::computeModelSources", this, 
                          &JWLpp::computeModelSources, mi);
    cout_doing << "JWLpp::scheduleComputeModelSources "<<  endl;  
    t->requires( Task::OldDW, mi->delT_Label);
    Ghost::GhostType  gn  = Ghost::None;
    const MaterialSubset* react_matl = matl0->thisMaterial();
    const MaterialSubset* prod_matl  = matl1->thisMaterial();
    MaterialSubset* one_matl     = scinew MaterialSubset();
    one_matl->add(0);
    one_matl->addReference();
    MaterialSubset* press_matl   = one_matl;
  
    //__________________________________
    // Products
    t->requires(Task::NewDW,  Ilb->rho_CCLabel,      prod_matl, gn);
  
    //__________________________________
    // Reactants
    t->requires(Task::NewDW, Ilb->sp_vol_CCLabel,    react_matl, gn);
    t->requires(Task::OldDW, Ilb->vel_CCLabel,       react_matl, gn);
    t->requires(Task::OldDW, Ilb->temp_CCLabel,      react_matl, gn);
    t->requires(Task::NewDW, Ilb->rho_CCLabel,       react_matl, gn);
    t->requires(Task::NewDW, Ilb->specific_heatLabel,react_matl, gn);

    t->requires(Task::NewDW, Ilb->press_equil_CCLabel, press_matl,gn);
    t->computes(reactedFractionLabel, react_matl);
    t->computes(delFLabel,            react_matl);

    t->modifies(mi->mass_source_CCLabel);
    t->modifies(mi->momentum_source_CCLabel);
    t->modifies(mi->energy_source_CCLabel);
    t->modifies(mi->sp_vol_source_CCLabel); 
    sched->addTask(t, level->eachPatch(), mymatls);

    if (one_matl->removeReference())
      delete one_matl;
  }
}

void JWLpp::scheduleCheckNeedAddMaterial(SchedulerP& sched,
                                         const LevelP& level,
                                         const ModelInfo* mi)
{
    Task* t = scinew Task("JWLpp::checkNeedAddMaterial", this, 
                          &JWLpp::checkNeedAddMaterial, mi);
    cout_doing << "JWLpp::scheduleCheckNeedAddMaterial "<<  endl;  

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

void
JWLpp::checkNeedAddMaterial(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset*,
                            DataWarehouse* /*old_dw*/,
                            DataWarehouse* new_dw,
                            const ModelInfo* /*mi*/)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);  
    
    cout_doing << "Doing checkNeedAddMaterial on patch "<< patch->getID()
               <<"\t\t\t\t  JWLpp" << endl;

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
void JWLpp::computeModelSources(const ProcessorGroup*, 
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw,
                                const ModelInfo* mi)
{
  delt_vartype delT;
  old_dw->get(delT, mi->delT_Label);

  int m0 = matl0->getDWIndex();
  int m1 = matl1->getDWIndex();
 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);  
    
    cout_doing << "Doing computeModelSources on patch "<< patch->getID()
               <<"\t\t\t\t  JWLpp" << endl;
    CCVariable<double> mass_src_0, mass_src_1, mass_0;
    CCVariable<Vector> momentum_src_0, momentum_src_1;
    CCVariable<double> energy_src_0, energy_src_1;
    CCVariable<double> sp_vol_src_0, sp_vol_src_1;

    new_dw->getModifiable(mass_src_0,    mi->mass_source_CCLabel,     m0,patch);
    new_dw->getModifiable(momentum_src_0,mi->momentum_source_CCLabel, m0,patch);
    new_dw->getModifiable(energy_src_0,  mi->energy_source_CCLabel,   m0,patch);
    new_dw->getModifiable(sp_vol_src_0,  mi->sp_vol_source_CCLabel,   m0,patch);

    new_dw->getModifiable(mass_src_1,    mi->mass_source_CCLabel,     m1,patch);
    new_dw->getModifiable(momentum_src_1,mi->momentum_source_CCLabel, m1,patch);
    new_dw->getModifiable(energy_src_1,  mi->energy_source_CCLabel,   m1,patch);
    new_dw->getModifiable(sp_vol_src_1,  mi->sp_vol_source_CCLabel,   m1,patch);

    constCCVariable<double> press_CC, cv_reactant;
    constCCVariable<double> rctTemp,rctRho,rctSpvol,prodRho;
    constCCVariable<Vector> rctvel_CC;
    CCVariable<double> Fr;
    CCVariable<double> delF;
	    
    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();
    Ghost::GhostType  gn  = Ghost::None;    
   
    //__________________________________
    // Reactant data
    old_dw->get(rctTemp,       Ilb->temp_CCLabel,  m0,patch,gn, 0);
    old_dw->get(rctvel_CC,     Ilb->vel_CCLabel,   m0,patch,gn, 0);
    new_dw->get(rctRho,        Ilb->rho_CCLabel,   m0,patch,gn, 0);
    new_dw->get(rctSpvol,      Ilb->sp_vol_CCLabel,m0,patch,gn, 0);
    new_dw->get(cv_reactant,   Ilb->specific_heatLabel,m0,patch,gn, 0);
    new_dw->allocateAndPut(Fr,   reactedFractionLabel,m0,patch);
    new_dw->allocateAndPut(delF, delFLabel,           m0,patch);
    Fr.initialize(0.);
    delF.initialize(0.);

    //__________________________________
    // Product Data, 
    new_dw->get(prodRho,       Ilb->rho_CCLabel,   m1,patch,gn, 0);

    //__________________________________
    //   Misc.
    new_dw->get(press_CC,         Ilb->press_equil_CCLabel,0,  patch,gn, 0);

    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      if (press_CC[c] > d_threshold_pressure){
        //__________________________________
        // Insert Burn Model Here
        double burnedMass;
        double F = prodRho[c]/(rctRho[c]+prodRho[c]);
        if(F >= 0. && F < .99){
          delF[c] = d_G*pow(press_CC[c],d_b)*(1.-F);
        }
        delF[c]*=delT;
        Fr[c] = F;
        double rctMass = rctRho[c]*cell_vol;
        double prdMass = prodRho[c]*cell_vol;
        burnedMass = min(delF[c]*(prdMass+rctMass), rctMass);
        burnedMass = min(burnedMass, .2*d_rho0*cell_vol);

        //__________________________________
        // conservation of mass, momentum and energy                           
        mass_src_0[c] -= burnedMass;
        mass_src_1[c] += burnedMass;
           
        Vector momX        = rctvel_CC[c] * burnedMass;
        momentum_src_0[c] -= momX;
        momentum_src_1[c] += momX;

        double energyX   = cv_reactant[c]*rctTemp[c]*burnedMass; 
        double releasedHeat = burnedMass * d_E0;
        energy_src_0[c] -= energyX;
        energy_src_1[c] += energyX + releasedHeat;

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
}
//______________________________________________________________________
//
void JWLpp::scheduleModifyThermoTransportProperties(SchedulerP&,
                                                    const LevelP&,
                                                    const MaterialSet*)
{
  // do nothing      
}
void JWLpp::computeSpecificHeat(CCVariable<double>&,
                                const Patch*,   
                                DataWarehouse*, 
                                const int)      
{
  //do nothing
}
//______________________________________________________________________
//
void JWLpp::scheduleErrorEstimate(const LevelP&,
                                  SchedulerP&)
{
  // Not implemented yet
}
//__________________________________
void JWLpp::scheduleTestConservation(SchedulerP&,
                                     const PatchSet*,                      
                                     const ModelInfo*)                     
{
  // Not implemented yet
}
