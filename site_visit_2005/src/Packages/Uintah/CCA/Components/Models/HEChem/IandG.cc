#include <Packages/Uintah/CCA/Components/Models/HEChem/IandG.h>
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

IandG::IandG(const ProcessorGroup* myworld, ProblemSpecP& params)
  : ModelInterface(myworld), params(params)
{
  mymatls = 0;
  Ilb  = scinew ICELabel();
  //__________________________________
  //  diagnostic labels
  reactedFractionLabel   = VarLabel::create("F",
                     CCVariable<double>::getTypeDescription());
                     
  IandGterm1Label   = VarLabel::create("IandGterm1",
                     CCVariable<double>::getTypeDescription());
  IandGterm2Label   = VarLabel::create("IandGterm2",
                     CCVariable<double>::getTypeDescription());
  IandGterm3Label   = VarLabel::create("IandGterm3",
                     CCVariable<double>::getTypeDescription());
                     
}

IandG::~IandG()
{
  delete Ilb;

  VarLabel::destroy(reactedFractionLabel);
  VarLabel::destroy(IandGterm1Label);
  VarLabel::destroy(IandGterm2Label);
  VarLabel::destroy(IandGterm3Label);
  
  if(mymatls && mymatls->removeReference())
    delete mymatls;
}

void IandG::problemSetup(GridP&, SimulationStateP& sharedState,
			     ModelSetup*)
{
  cout << "I'm in problem setup" << endl;
  d_sharedState = sharedState;
  matl0 = sharedState->parseAndLookupMaterial(params, "fromMaterial");
  matl1 = sharedState->parseAndLookupMaterial(params, "toMaterial");
  params->require("I",  d_I);
  params->require("G1", d_G1);
  params->require("G2", d_G2);
  params->require("a",  d_a);
  params->require("b",  d_b);
  params->require("c",  d_c);
  params->require("d",  d_d);
  params->require("e",  d_e);
  params->require("g",  d_g);
  params->require("x",  d_x);
  params->require("y",  d_y);
  params->require("z",  d_z);
  params->require("Figmax", d_Figmax);
  params->require("FG1max", d_FG1max);
  params->require("FG2min", d_FG2min);
  params->require("rho0",   d_rho0);
  params->require("E0",     d_E0);
  params->require("ThresholdPressure",   d_threshold_pressure);

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
void IandG::scheduleInitialize(SchedulerP&,
                               const LevelP&,
                               const ModelInfo*)
{
  // None necessary...
}
//______________________________________________________________________
//      
void IandG::scheduleComputeStableTimestep(SchedulerP&,
                                          const LevelP&,
                                          const ModelInfo*)
{
  // None necessary...
}

//______________________________________________________________________
//     
void IandG::scheduleComputeModelSources(SchedulerP& sched,
                                        const LevelP& level,
                                        const ModelInfo* mi)
{
  Task* t = scinew Task("IandG::computeModelSources", this, 
                        &IandG::computeModelSources, mi);
  cout_doing << "IandG::scheduleComputeModelSources "<<  endl;  
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
  t->computes(IandGterm1Label, react_matl);
  t->computes(IandGterm2Label, react_matl);
  t->computes(IandGterm3Label, react_matl);

  t->modifies(mi->mass_source_CCLabel);
  t->modifies(mi->momentum_source_CCLabel);
  t->modifies(mi->energy_source_CCLabel);
  t->modifies(mi->sp_vol_source_CCLabel); 
  sched->addTask(t, level->eachPatch(), mymatls);

  if (one_matl->removeReference())
    delete one_matl;
}

//______________________________________________________________________
//
void IandG::computeModelSources(const ProcessorGroup*, 
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
               <<"\t\t\t\t  IandG" << endl;
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
    CCVariable<double> term1,term2,term3;
	    
    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();
    Ghost::GhostType  gn  = Ghost::None;    
   
    //__________________________________
    // Reactant data
    old_dw->get(rctTemp,       Ilb->temp_CCLabel,      m0,patch,gn, 0);
    old_dw->get(rctvel_CC,     Ilb->vel_CCLabel,       m0,patch,gn, 0);
    new_dw->get(rctRho,        Ilb->rho_CCLabel,       m0,patch,gn, 0);
    new_dw->get(rctSpvol,      Ilb->sp_vol_CCLabel,    m0,patch,gn, 0);
    new_dw->get(cv_reactant,   Ilb->specific_heatLabel,m0,patch,gn, 0);
    new_dw->allocateAndPut(Fr,reactedFractionLabel,m0,patch);
    new_dw->allocateAndPut(term1, IandGterm1Label, m0,patch);
    new_dw->allocateAndPut(term2, IandGterm2Label, m0,patch);
    new_dw->allocateAndPut(term3, IandGterm3Label, m0,patch);
    Fr.initialize(0.);
    term1.initialize(0.);
    term2.initialize(0.);
    term3.initialize(0.);

    //__________________________________
    // Product Data, 
    new_dw->get(prodRho,       Ilb->rho_CCLabel,        m1, patch,gn, 0);

    //__________________________________
    //   Misc.
    new_dw->get(press_CC,      Ilb->press_equil_CCLabel,0,  patch,gn, 0);
  

    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      if (press_CC[c] > d_threshold_pressure){
        //__________________________________
        // Insert Burn Model Here
        double burnedMass = 0.;
        double delF = 0;
        double F = prodRho[c]/(rctRho[c]+prodRho[c]);
        if(F >= 0. && F < d_Figmax){
         term1[c]=d_I*pow((1.-F),d_b)*pow((1./(rctSpvol[c]*d_rho0)-1.-d_a),d_x);
         delF += term1[c];
        }
        if(F >= 0. && F < d_FG1max){
          term2[c] = d_G1*pow((1.-F),d_c)*pow(F,d_d)*pow(press_CC[c],d_y);
          delF += term2[c];
        }
        if(F >= d_FG2min && F < 0.99){
          term3[c] = d_G2*pow((1.-F),d_e)*pow(F,d_g)*pow(press_CC[c],d_z);
          delF += term3[c];
        }
        delF*=delT;
        Fr[c] = F;
        double rctMass = rctRho[c]*cell_vol;
        double prdMass = prodRho[c]*cell_vol;
        burnedMass = min(delF*(prdMass+rctMass), rctMass);
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
    setBC(term1, "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
    setBC(term2, "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
    setBC(term3, "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
    setBC(Fr,    "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
  }
}
//______________________________________________________________________
//
void IandG::scheduleModifyThermoTransportProperties(SchedulerP&,
                                                    const LevelP&,         
                                                    const MaterialSet*)    
{
  // do nothing      
}
void IandG::computeSpecificHeat(CCVariable<double>&,
                                const Patch*,   
                                DataWarehouse*, 
                                const int)      
{
  //do nothing
}
//______________________________________________________________________
//
void IandG::scheduleErrorEstimate(const LevelP&,
                                  SchedulerP&)
{
  // Not implemented yet
}
//__________________________________
void IandG::scheduleTestConservation(SchedulerP&,
                                     const PatchSet*,                
                                     const ModelInfo*)               
{
  // Not implemented yet
}

