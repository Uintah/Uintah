
#include <Packages/Uintah/CCA/Components/Models/test/TestModel.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Material.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <iostream>

using namespace Uintah;
using namespace std;

TestModel::TestModel(const ProcessorGroup* myworld, ProblemSpecP& params)
  : ModelInterface(myworld), params(params)
{
  mymatls = 0;
  MIlb  = scinew MPMICELabel();
}

TestModel::~TestModel()
{
  delete MIlb;
  if(mymatls && mymatls->removeReference())
    delete mymatls;
}
//______________________________________________________________________
void TestModel::problemSetup(GridP&, SimulationStateP& sharedState,
			     ModelSetup* )
{
  matl0 = sharedState->parseAndLookupMaterial(params, "fromMaterial");
  matl1 = sharedState->parseAndLookupMaterial(params, "toMaterial");
  params->require("rate", rate);

  vector<int> m(2);
  m[0] = matl0->getDWIndex();
  m[1] = matl1->getDWIndex();
  mymatls = new MaterialSet();
  mymatls->addAll(m);
  mymatls->addReference();
 
  // What flavor of matl it is.
  Material* matl = sharedState->getMaterial( m[0] );
  ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
  MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
  if (mpm_matl){
    d_is_mpm_matl = true;
    d_matl = mpm_matl;
  }
  if (ice_matl){
    d_is_mpm_matl = false;
    d_matl = ice_matl;
  }   
}
      
void TestModel::scheduleInitialize(SchedulerP&,
				   const LevelP& level,
				   const ModelInfo*)
{
  // None necessary...
}
      
void TestModel::scheduleComputeStableTimestep(SchedulerP&,
					      const LevelP&,
					      const ModelInfo*)
{
  // None necessary...
}

//__________________________________      
void TestModel::scheduleMassExchange(SchedulerP& sched,
				     const LevelP& level,
				     const ModelInfo* mi)
{
  Task* t = scinew Task("TestModel::massExchange",
			this, &TestModel::massExchange, mi);
  t->modifies(mi->mass_source_CCLabel);
  t->modifies(mi->momentum_source_CCLabel);
  t->modifies(mi->energy_source_CCLabel);
  t->modifies(mi->sp_vol_source_CCLabel);
  Ghost::GhostType  gn  = Ghost::None;
  
  Task::WhichDW DW;
  Task::WhichDW NDW =Task::NewDW;   
  if(d_is_mpm_matl){              // MPM (pull data from newDW)
    DW = Task::NewDW;
    t->requires( DW, MIlb->cMassLabel,     matl0->thisMaterial(), gn);
  } else { 
    DW = Task::OldDW;             // ICE (pull data from old DW)
    t->requires( DW, mi->density_CCLabel,    matl0->thisMaterial(), gn);
    t->requires( NDW,mi->specific_heatLabel, matl0->thisMaterial(), gn);
  } 
                                  // All matls
  t->requires( DW,  mi->velocity_CCLabel,   matl0->thisMaterial(), gn);
  t->requires( DW,  mi->temperature_CCLabel,matl0->thisMaterial(), gn); 
  t->requires( NDW, mi->sp_vol_CCLabel,     matl0->thisMaterial(), gn);
  
  t->requires( Task::OldDW, mi->delT_Label);
  sched->addTask(t, level->eachPatch(), mymatls);
}

//__________________________________
void TestModel::massExchange(const ProcessorGroup*, 
			     const PatchSubset* patches,
			     const MaterialSubset* matls,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw,
			     const ModelInfo* mi)
{
  delt_vartype delT;
  old_dw->get(delT, mi->delT_Label);
  double dt = delT;

  ASSERT(matls->size() == 2);
  int m0 = matl0->getDWIndex();
  int m1 = matl1->getDWIndex();
 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);  
    CCVariable<double> mass_src_0, mass_src_1, mass_0, cv;
    CCVariable<Vector> mom_src_0, mom_src_1;
    CCVariable<double> eng_src_0, eng_src_1;
    CCVariable<double> sp_vol_src_0, sp_vol_src_1;
    
    new_dw->allocateTemporary(cv, patch);
    new_dw->getModifiable(mass_src_0,   mi->mass_source_CCLabel,    m0, patch);
    new_dw->getModifiable(mom_src_0,    mi->momentum_source_CCLabel,m0, patch);
    new_dw->getModifiable(eng_src_0,    mi->energy_source_CCLabel,  m0, patch);
    new_dw->getModifiable(sp_vol_src_0, mi->sp_vol_source_CCLabel,  m0, patch);

    new_dw->getModifiable(mass_src_1,   mi->mass_source_CCLabel,    m1, patch);
    new_dw->getModifiable(mom_src_1,    mi->momentum_source_CCLabel,m1, patch);
    new_dw->getModifiable(eng_src_1,    mi->energy_source_CCLabel,  m1, patch);
    new_dw->getModifiable(sp_vol_src_1, mi->sp_vol_source_CCLabel,  m1, patch);
                       
    //__________________________________
    //  Compute the mass and specific heat of matl 0
    new_dw->allocateTemporary(mass_0, patch);
    Vector dx = patch->dCell();
    double volume = dx.x()*dx.y()*dx.z();                    
    DataWarehouse* dw;
    Ghost::GhostType  gn = Ghost::None;
   
    if(d_is_mpm_matl){
      dw = new_dw;            // MPM  (Just grab it)
      constCCVariable<double> cmass;  
      dw->get(cmass,   MIlb->cMassLabel,    m0, patch, gn, 0); 
      mass_0.copyData(cmass);
   
      cv.initialize(d_matl->getSpecificHeat());
    } else {
      dw = old_dw;            // ICE   (compute it from the density)
      constCCVariable<double> rho_tmp, cv_ice;
      old_dw->get(rho_tmp, mi->density_CCLabel,    m0, patch, gn, 0);
      new_dw->get(cv_ice,  mi->specific_heatLabel, m0, patch, gn, 0);
      
      cv.copyData(cv_ice);    
      
      for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
                                                                iter++){
        mass_0[*iter] = rho_tmp[*iter] * volume;
      }
    }

    constCCVariable<Vector> vel_0;    // MPM  pull from new_dw
    constCCVariable<double> temp_0;   // ICE  pull from old_dw
    constCCVariable<double> sp_vol_0;
    dw  ->  get(vel_0,    mi->velocity_CCLabel,    m0, patch, gn, 0);    
    dw  ->  get(temp_0,   mi->temperature_CCLabel, m0, patch, gn, 0);    
    new_dw->get(sp_vol_0, mi->sp_vol_CCLabel,      m0, patch, gn, 0);
        
    double tm = 0;
    double trate = rate*dt;
    if(trate > 1){
      trate=1;
    }
    //__________________________________
    //  Do some work
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      double massx = mass_0[c]*trate;
      mass_src_0[c] -= massx;
      mass_src_1[c] += massx;

      Vector momx = vel_0[c]*massx;
      mom_src_0[c] -= momx;
      mom_src_1[c] += momx;
      
      double energyx = temp_0[c] * massx * cv[c];
      eng_src_0[c] -= energyx;
      eng_src_1[c] += energyx;
    
      double vol_sourcex  = massx * sp_vol_0[c];
      sp_vol_src_0[c] -= vol_sourcex;
      sp_vol_src_1[c] += vol_sourcex;
            
      tm += massx;
    }
    cerr << "Total mass transferred: " << tm << '\n';
  }
}
//______________________________________________________________________  
void TestModel::scheduleMomentumAndEnergyExchange(SchedulerP&,
						  const LevelP&,
						  const ModelInfo*)
{
  // None
}
   
void TestModel::scheduleModifyThermoTransportProperties(SchedulerP&,
                                                        const LevelP&,
                                                        const MaterialSet*)
{
  // do nothing      
}
void TestModel::computeSpecificHeat(CCVariable<double>&,
                                    const Patch*,   
                                    DataWarehouse*, 
                                    const int)      
{
  //do nothing
}
