
#include <Packages/Uintah/CCA/Components/Models/HEChem/LightTime.h>
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
#include <Core/Math/MiscMath.h>

using namespace Uintah;
using namespace std;
//__________________________________
//  setenv SCI_DEBUG "MODELS_NORMAL_COUT:+,MODELS_DOING_COUT:+"
//  MODELS_DOING_COUT:   dumps when tasks are scheduled and performed
static DebugStream cout_doing("MODELS_DOING_COUT", false);

LightTime::LightTime(const ProcessorGroup* myworld, ProblemSpecP& params)
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

LightTime::~LightTime()
{
  delete Ilb;

  VarLabel::destroy(reactedFractionLabel);
  VarLabel::destroy(delFLabel);
  
  if(mymatls && mymatls->removeReference())
    delete mymatls;
}

void LightTime::problemSetup(GridP&, SimulationStateP& sharedState,
			     ModelSetup*)
{
  cout << "I'm in problem setup" << endl;
  d_sharedState = sharedState;
  matl0 = sharedState->parseAndLookupMaterial(params, "fromMaterial");
  matl1 = sharedState->parseAndLookupMaterial(params, "toMaterial");
  params->require("starting_location",    d_start_place);
  params->require("direction_if_plane",   d_direction);
  params->require("D",    d_D);
  params->require("E0",   d_E0);
  params->require("rho0", d_rho0);
  params->getWithDefault("react_mixed_cells", d_react_mixed_cells,true);

  // if point  ignition is desired, direction_if_plane = (0.,0.,0)
  // if planar ignition is desired, direction_if_plane is normal in
  // the direction of burning

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
void LightTime::scheduleInitialize(SchedulerP& sched,
                               const LevelP& level,
                               const ModelInfo*)
{
  cout_doing << "LightTime::scheduleInitialize " << endl;
  Task* t = scinew Task("LightTime::initialize", this, &LightTime::initialize);

  const MaterialSubset* react_matl = matl0->thisMaterial();

  t->computes(reactedFractionLabel,react_matl);

  sched->addTask(t, level->eachPatch(), mymatls);
}

//______________________________________________________________________
//      
void LightTime::initialize(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset*,
                           DataWarehouse*,
                           DataWarehouse* new_dw)
{
  cout_doing << "Doing Initialize \t\t\t\t\tLightTime" << endl;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int m0 = matl0->getDWIndex();
                                                                                
    CCVariable<double>  Fr;
    new_dw->allocateAndPut(Fr, reactedFractionLabel, m0, patch);
    Fr.initialize(0.);
  }
}
//______________________________________________________________________
//      
void LightTime::scheduleComputeStableTimestep(SchedulerP&,
                                          const LevelP&,
                                          const ModelInfo*)
{
  // None necessary...
}

//______________________________________________________________________
//     
void LightTime::scheduleComputeModelSources(SchedulerP& sched,
                                            const LevelP& level,
                                            const ModelInfo* mi)
{
  Task* t = scinew Task("LightTime::computeModelSources", this, 
                        &LightTime::computeModelSources, mi);
  cout_doing << "LightTime::scheduleComputeModelSources "<<  endl;  
  t->requires( Task::OldDW, mi->delT_Label);
  Ghost::GhostType  gn  = Ghost::None;
  const MaterialSubset* react_matl = matl0->thisMaterial();
  const MaterialSubset* prod_matl  = matl1->thisMaterial();
  //__________________________________
  // Products
  t->requires(Task::NewDW,  Ilb->rho_CCLabel,      prod_matl, gn);
  t->requires(Task::NewDW,  Ilb->vol_frac_CCLabel, prod_matl, gn);
  
  //__________________________________
  // Reactants
  t->requires(Task::NewDW, Ilb->vol_frac_CCLabel,  react_matl, gn);
  t->requires(Task::NewDW, Ilb->sp_vol_CCLabel,    react_matl, gn);
  t->requires(Task::OldDW, Ilb->vel_CCLabel,       react_matl, gn);
  t->requires(Task::OldDW, Ilb->int_eng_CCLabel,      react_matl, gn);
  t->requires(Task::OldDW, reactedFractionLabel,   react_matl, gn);
  t->requires(Task::NewDW, Ilb->rho_CCLabel,       react_matl, gn);

  t->computes(reactedFractionLabel, react_matl);
  t->computes(delFLabel,            react_matl);

  t->modifies(mi->mass_source_CCLabel);
  t->modifies(mi->momentum_source_CCLabel);
  t->modifies(mi->energy_source_CCLabel);
  t->modifies(mi->sp_vol_source_CCLabel); 
  sched->addTask(t, level->eachPatch(), mymatls);
}

//______________________________________________________________________
//
void LightTime::computeModelSources(const ProcessorGroup*, 
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
               <<"\t\t\t\t  LightTime" << endl;
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

    constCCVariable<double> vol_frac_rct, vol_frac_prd;
    constCCVariable<double> Fr_old;
    constCCVariable<double> rctIntEng,rctRho,rctSpvol,prodRho;
    constCCVariable<Vector> rctvel_CC;
    CCVariable<double> Fr;
    CCVariable<double> delF;
	    
    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();
    Ghost::GhostType  gn  = Ghost::None;    
   
    //__________________________________
    // Reactant data
    old_dw->get(rctIntEng,     Ilb->int_eng_CCLabel,      m0,patch,gn, 0);
    old_dw->get(rctvel_CC,     Ilb->vel_CCLabel,       m0,patch,gn, 0);
    old_dw->get(Fr_old,        reactedFractionLabel,   m0,patch,gn, 0);
    new_dw->get(rctRho,        Ilb->rho_CCLabel,       m0,patch,gn, 0);
    new_dw->get(rctSpvol,      Ilb->sp_vol_CCLabel,    m0,patch,gn, 0);
    new_dw->get(vol_frac_rct,  Ilb->vol_frac_CCLabel,  m0,patch,gn, 0);
    new_dw->get(vol_frac_prd,  Ilb->vol_frac_CCLabel,  m1,patch,gn, 0);
    new_dw->allocateAndPut(Fr, reactedFractionLabel,   m0,patch);
    new_dw->allocateAndPut(delF, delFLabel,            m0,patch);
    Fr.initialize(0.);
    delF.initialize(0.);

    //__________________________________
    // Product Data, 
    new_dw->get(prodRho,       Ilb->rho_CCLabel,   m1,patch,gn, 0);

    const Level* lvl = patch->getLevel();
    double time = d_sharedState->getElapsedTime();
    double delta_L = 1.5*pow(cell_vol,1./3.)/d_D;
//    double delta_L = 1.5*dx.x()/d_D;
    double A=d_direction.x();
    double B=d_direction.y();
    double C=d_direction.z();
    double x0=d_start_place.x();
    double y0=d_start_place.y();
    double z0=d_start_place.z();
    double D = -A*x0 - B*y0 - C*z0;
    double denom = 1.0;
    double plane = 0.;
    if(d_direction.length() > 0.0){
      plane = 1.0;
      denom = sqrt(A*A + B*B + C*C);
    }

    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;

      Point pos = lvl->getCellPosition(c);
      double dist_plane = Abs(A*pos.x() + B*pos.y() + C*pos.z() + D)/denom;
      double dist_straight = (pos - d_start_place).length();
      double dist = dist_plane*plane + dist_straight*(1.-plane);
      double t_b = dist/d_D; 
      double VF_SUM = 0.;
      if(!d_react_mixed_cells){
        VF_SUM = .99;
      }
      if((vol_frac_rct[c] + vol_frac_prd[c]) > VF_SUM){
        if (time >= t_b && rctRho[c] > d_TINY_RHO){
          Fr[c] = (time - t_b)/delta_L;
          if(Fr[c] > .96) Fr[c] = 1.0;
          delF[c] = Fr[c] - Fr_old[c];

          //__________________________________
          // Insert Burn Model Here
          double burnedMass;
          double rctMass = rctRho[c]*cell_vol;
          double prdMass = prodRho[c]*cell_vol;
          burnedMass = min(delF[c]*(prdMass+rctMass), rctMass);

          //__________________________________
          // conservation of mass, momentum and energy                           
          mass_src_0[c] -= burnedMass;
          mass_src_1[c] += burnedMass;

          Vector momX        = rctvel_CC[c] * burnedMass;
          momentum_src_0[c] -= momX;
          momentum_src_1[c] += momX;

          double energyX   = rctIntEng[c]*burnedMass; 
          double releasedHeat = burnedMass * d_E0;
          energy_src_0[c] -= energyX;
          energy_src_1[c] += energyX + releasedHeat;

          double createdVolx  = burnedMass * rctSpvol[c];
          sp_vol_src_0[c] -= createdVolx;
          sp_vol_src_1[c] += createdVolx;
        }  // if (time to light it)
      }  // if cell only contains rct and prod
      if (rctRho[c] <= d_TINY_RHO){
        Fr[c] = 1.0;
        delF[c] = 0.0;
      }  // reactant mass is already consumed
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
void LightTime::scheduleModifyThermoTransportProperties(SchedulerP&,
                                                    const LevelP&,
                                                    const MaterialSet*)
{
  // do nothing      
}
void LightTime::computeSpecificHeat(CCVariable<double>&,
                                const Patch*,   
                                DataWarehouse*, 
                                const int)      
{
  //do nothing
}
//______________________________________________________________________
//
void LightTime::scheduleErrorEstimate(const LevelP&,
                                      SchedulerP&)
{
  // Not implemented yet
}
//__________________________________
void LightTime::scheduleTestConservation(SchedulerP&,
                                         const PatchSet*,                      
                                         const ModelInfo*)                     
{
  // Not implemented yet
}
