
#include <Packages/Uintah/CCA/Components/Models/HEChem/Simple_Burn.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Material.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <iostream>
#include <Core/Util/DebugStream.h>

using namespace Uintah;
using namespace std;
//__________________________________
//  setenv SCI_DEBUG "MPMICE_NORMAL_COUT:+,MPMICE_DOING_COUT:+"
//  MPMICE_DOING_COUT:   dumps when tasks are scheduled and performed
static DebugStream cout_doing("MPMICE_DOING_COUT", false);

Simple_Burn::Simple_Burn(const ProcessorGroup* myworld, ProblemSpecP& params)
  : ModelInterface(myworld), params(params)
{
  mymatls = 0;
  Mlb  = scinew MPMLabel();
  Ilb  = scinew ICELabel();
  MIlb  = scinew MPMICELabel();
  //__________________________________
  //  diagnostic labels
  onSurfaceLabel   = VarLabel::create("Simple_Burn::onSurface",
                     CCVariable<double>::getTypeDescription());
                     
  surfaceTempLabel = VarLabel::create("Simple_Burn::surfaceTemp",
                     CCVariable<double>::getTypeDescription());
                
}

Simple_Burn::~Simple_Burn()
{
  delete Ilb;
  delete Mlb;
  delete MIlb;
  
  VarLabel::destroy(surfaceTempLabel);
  VarLabel::destroy(onSurfaceLabel);
  
  if(mymatls && mymatls->removeReference())
    delete mymatls;
}

void Simple_Burn::problemSetup(GridP& grid, SimulationStateP& sharedState,
			     ModelSetup& setup)
{
  cout << "I'm in problem setup" << endl;
  d_sharedState = sharedState;
  matl0 = sharedState->parseAndLookupMaterial(params, "fromMaterial");
  matl1 = sharedState->parseAndLookupMaterial(params, "toMaterial");
  params->require("ThresholdTemp",    d_thresholdTemp);
  params->require("ThresholdPressure",d_thresholdPress);
  params->require("Enthalpy",         d_Enthalpy);
  params->require("BurnCoeff",        d_BurnCoeff);
  params->require("refPressure",      d_refPress);

  vector<int> m(2);
  m[0] = matl0->getDWIndex();
  m[1] = matl1->getDWIndex();
  mymatls = new MaterialSet();
  mymatls->addAll(m);
  mymatls->addReference();
  
}
//______________________________________________________________________
//     
void Simple_Burn::scheduleInitialize(const LevelP&,
				   SchedulerP&)
{
  // None necessary...
}
//______________________________________________________________________
//      
void Simple_Burn::scheduleComputeStableTimestep(SchedulerP&,
					      const LevelP& level,
					      const ModelInfo*)
{
  // None necessary...
}

//______________________________________________________________________
//     
void Simple_Burn::scheduleMassExchange(SchedulerP& sched,
				     const LevelP& level,
				     const ModelInfo* mi)
{
  Task* t = scinew Task("Simple_Burn::massExchange",
			this, &Simple_Burn::massExchange, mi);
  
  t->requires( Task::OldDW, mi->delT_Label);
  Ghost::GhostType  gac = Ghost::AroundCells;  
  Ghost::GhostType  gn  = Ghost::None;
  const MaterialSubset* react_matl = matl0->thisMaterial();
  const MaterialSubset* prod_matl  = matl1->thisMaterial();
  MaterialSubset* one_matl     = scinew MaterialSubset();
  one_matl->add(0);
  one_matl->addReference();
  MaterialSubset* press_matl   = one_matl;
  
  //__________________________________
  // Products
  t->requires(Task::OldDW,  Ilb->temp_CCLabel,    prod_matl, gn);       
  t->requires(Task::NewDW,  Ilb->vol_frac_CCLabel,prod_matl, gn);       
  t->requires(Task::NewDW,  Ilb->TempX_FCLabel,   prod_matl, gac,2);    
  t->requires(Task::NewDW,  Ilb->TempY_FCLabel,   prod_matl, gac,2);    
  t->requires(Task::NewDW,  Ilb->TempZ_FCLabel,   prod_matl, gac,2);
    
  t->requires(Task::NewDW,  Ilb->press_equil_CCLabel, press_matl,gn);
  t->requires(Task::OldDW,  MIlb->NC_CCweightLabel,   one_matl,  gac, 1);
  
  //__________________________________
  // Reactants
  t->requires(Task::NewDW, Ilb->sp_vol_CCLabel,   react_matl, gn);
  t->requires(Task::NewDW, Ilb->TempX_FCLabel,    react_matl, gac,2);
  t->requires(Task::NewDW, Ilb->TempY_FCLabel,    react_matl, gac,2);
  t->requires(Task::NewDW, Ilb->TempZ_FCLabel,    react_matl, gac,2);
  t->requires(Task::NewDW, MIlb->vel_CCLabel,     react_matl, gn);
  t->requires(Task::NewDW, MIlb->temp_CCLabel,    react_matl, gn);
  t->requires(Task::NewDW, MIlb->cMassLabel,      react_matl, gn);
  t->requires(Task::NewDW, Mlb->gMassLabel,       react_matl, gac,1);  

  t->computes(Simple_Burn::onSurfaceLabel,     one_matl);
  t->computes(Simple_Burn::surfaceTempLabel,   one_matl);
  
  t->modifies(mi->mass_source_CCLabel);
  t->modifies(mi->momentum_source_CCLabel);
  t->modifies(mi->energy_source_CCLabel);
  t->modifies(mi->sp_vol_source_CCLabel); 
  sched->addTask(t, level->eachPatch(), mymatls);
}

//______________________________________________________________________
//
void Simple_Burn::massExchange(const ProcessorGroup*, 
			     const PatchSubset* patches,
			     const MaterialSubset* matls,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw,
			     const ModelInfo* mi)
{

  delt_vartype delT;
  old_dw->get(delT, mi->delT_Label);

  ASSERT(matls->size() == 2);
  int m0 = matl0->getDWIndex();
  int m1 = matl1->getDWIndex();
 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);  
    
    cout_doing << "Doing massExchange on patch "<< patch->getID()
               <<"\t\t\t\t  Simple_Burn" << endl;
    CCVariable<double> mass_src_0, mass_src_1, mass_0;
    CCVariable<Vector> momentum_src_0, momentum_src_1;
    CCVariable<double> energy_src_0, energy_src_1;
    CCVariable<double> sp_vol_src_0, sp_vol_src_1;
    CCVariable<double> onSurface, surfaceTemp;
    
    new_dw->getModifiable(mass_src_0,    mi->mass_source_CCLabel,     m0, patch);    
    new_dw->getModifiable(momentum_src_0,mi->momentum_source_CCLabel, m0, patch); 
    new_dw->getModifiable(energy_src_0,  mi->energy_source_CCLabel,   m0, patch);   
    new_dw->getModifiable(sp_vol_src_0,  mi->sp_vol_source_CCLabel,   m0, patch);   

    new_dw->getModifiable(mass_src_1,    mi->mass_source_CCLabel,    m1,patch);   
    new_dw->getModifiable(momentum_src_1,mi->momentum_source_CCLabel,m1,patch);   
    new_dw->getModifiable(energy_src_1,  mi->energy_source_CCLabel,  m1,patch);   
    new_dw->getModifiable(sp_vol_src_1,  mi->sp_vol_source_CCLabel,  m1,patch);   
 
    
    constCCVariable<double> press_CC,gasTemp,gasVol_frac;
    constCCVariable<double> solidTemp,solidMass,solidSp_vol;

    constNCVariable<double> NC_CCweight,NCsolidMass;
    constSFCXVariable<double> gasTempX_FC,solidTempX_FC;
    constSFCYVariable<double> gasTempY_FC,solidTempY_FC;
    constSFCZVariable<double> gasTempZ_FC,solidTempZ_FC;
    constCCVariable<Vector> vel_CC;
    
    Vector dx = patch->dCell();
    double delX = dx.x();
    double delY = dx.y();
    double delZ = dx.z();
    Ghost::GhostType  gn  = Ghost::None;    
    Ghost::GhostType  gac = Ghost::AroundCells;   
   
    //__________________________________
    // Reactant data
    new_dw->get(solidTemp,       MIlb->temp_CCLabel, m0,patch,gn, 0);
    new_dw->get(solidMass,       MIlb->cMassLabel,   m0,patch,gn, 0);
    new_dw->get(solidSp_vol,     Ilb->sp_vol_CCLabel,m0,patch,gn,0);
    new_dw->get(solidTempX_FC,   Ilb->TempX_FCLabel, m0,patch,gac,2);
    new_dw->get(solidTempY_FC,   Ilb->TempY_FCLabel, m0,patch,gac,2);
    new_dw->get(solidTempZ_FC,   Ilb->TempZ_FCLabel, m0,patch,gac,2);
    new_dw->get(vel_CC,          MIlb->vel_CCLabel,  m0,patch,gn, 0);
    new_dw->get(NCsolidMass,     Mlb->gMassLabel,    m0,patch,gac,1);

    //__________________________________
    // Product Data, 
    new_dw->get(gasTempX_FC,      Ilb->TempX_FCLabel,m1,patch,gac,2);
    new_dw->get(gasTempY_FC,      Ilb->TempY_FCLabel,m1,patch,gac,2);
    new_dw->get(gasTempZ_FC,      Ilb->TempZ_FCLabel,m1,patch,gac,2);
    old_dw->get(gasTemp,          Ilb->temp_CCLabel, m1,patch,gn, 0);
    new_dw->get(gasVol_frac,      Ilb->vol_frac_CCLabel,  m1, patch,gn, 0);
    //__________________________________
    //   Misc.
    new_dw->get(press_CC,         Ilb->press_equil_CCLabel,0,  patch,gn, 0);
    old_dw->get(NC_CCweight,     MIlb->NC_CCweightLabel,  0,   patch,gac,1);   
  
    new_dw->allocateAndPut(onSurface,  Simple_Burn::onSurfaceLabel,   0, patch);
    new_dw->allocateAndPut(surfaceTemp,Simple_Burn::surfaceTempLabel, 0, patch);
 
    IntVector nodeIdx[8];
    
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m0);
    double cv_solid = mpm_matl->getSpecificHeat();
    
    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;

     //__________________________________
     // Find if the cell contains surface:
      patch->findNodesFromCell(*iter,nodeIdx);
      double MaxMass = d_SMALL_NUM;
      double MinMass = 1.0/d_SMALL_NUM;
      for (int nN=0; nN<8; nN++) {
        MaxMass = std::max(MaxMass,NC_CCweight[nodeIdx[nN]]*
                                   NCsolidMass[nodeIdx[nN]]);
        MinMass = std::min(MinMass,NC_CCweight[nodeIdx[nN]]*
                                   NCsolidMass[nodeIdx[nN]]); 
      }               

      if ( (MaxMass-MinMass)/MaxMass > 0.4            //--------------KNOB 1
        && (MaxMass-MinMass)/MaxMass < 1.0
        &&  MaxMass > d_SMALL_NUM){

        //__________________________________
        //  Determine the temperature
        //  to use in burn model
        double Temp = 0;

        if (gasVol_frac[c] < 0.2){             //--------------KNOB 2
          Temp =std::max(Temp, solidTempX_FC[c] );    //L
          Temp =std::max(Temp, solidTempY_FC[c] );    //Bot
          Temp =std::max(Temp, solidTempZ_FC[c] );    //BK
          Temp =std::max(Temp, solidTempX_FC[c + IntVector(1,0,0)] );
          Temp =std::max(Temp, solidTempY_FC[c + IntVector(0,1,0)] );
          Temp =std::max(Temp, solidTempZ_FC[c + IntVector(0,0,1)] );
        }else {
          Temp =std::max(Temp, gasTempX_FC[c] );    //L
          Temp =std::max(Temp, gasTempY_FC[c] );    //Bot
          Temp =std::max(Temp, gasTempZ_FC[c] );    //BK
          Temp =std::max(Temp, gasTempX_FC[c + IntVector(1,0,0)] );
          Temp =std::max(Temp, gasTempY_FC[c + IntVector(0,1,0)] );          
          Temp =std::max(Temp, gasTempZ_FC[c + IntVector(0,0,1)] );
        }
        surfaceTemp[c] = Temp;

        double gradRhoX = 0.25 *
                          ((NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
                            NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+
                            NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+
                            NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]])
                          -
                          ( NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
                            NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]]+
                            NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]]+
                            NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]])
                          ) / delX;
        double gradRhoY = 0.25 *
                          ((NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
                            NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+
                            NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
                            NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]])
                          -
                          ( NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+
                            NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]]+
                            NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]]+
                            NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]])
                          ) / delY;
        double gradRhoZ = 0.25 *
                          ((NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+
                            NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]]+
                            NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]]+
                            NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]])
                          -
                          ( NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
                            NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+
                            NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
                            NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]])
                          ) / delZ;

        double absGradRho = sqrt(gradRhoX*gradRhoX +
                                 gradRhoY*gradRhoY +
                                 gradRhoZ*gradRhoZ );

        double normalX = gradRhoX/absGradRho;
        double normalY = gradRhoY/absGradRho;
        double normalZ = gradRhoZ/absGradRho;

        double TmpX = fabs(normalX*delX);
        double TmpY = fabs(normalY*delY);
        double TmpZ = fabs(normalZ*delZ);

        double surfArea = delX*delY*delZ / (TmpX+TmpY+TmpZ); 
//             double surfArea = delX*delY; 
        onSurface[c] = surfArea; // debugging var

        //__________________________________
        //  Simple Burn Model
        double burnedMass = 0.0;
        if ((Temp > d_thresholdTemp) && (press_CC[c] > d_thresholdPress)) {
          burnedMass = delT *surfArea * d_BurnCoeff 
                       * pow((press_CC[c]/d_refPress),0.778);
        }
        if(burnedMass > solidMass[c]){
          burnedMass = solidMass[c];
        }
  
        //__________________________________
        // conservation of mass, momentum and energy                           
        mass_src_0[c] -= burnedMass;
        mass_src_1[c] += burnedMass;
           
        Vector momX        = vel_CC[c] * burnedMass;
        momentum_src_0[c] -= momX;
        momentum_src_1[c] += momX;

        double energyX   = cv_solid*solidTemp[c]*burnedMass; 
        double releasedHeat = burnedMass * d_Enthalpy;
        energy_src_0[c] -= energyX;
        energy_src_1[c] += energyX + releasedHeat;

        double createdVolx  = burnedMass * solidSp_vol[c];
        sp_vol_src_0[c] -= createdVolx;
        sp_vol_src_1[c] += createdVolx;
      }  // if (maxMass-MinMass....)
    }  // cell iterator  

    //__________________________________
    //  set symetric BC
    setBC(mass_src_0, "set_if_sym_BC",patch, d_sharedState, m0);
    setBC(mass_src_1, "set_if_sym_BC",patch, d_sharedState, m1);
   
  }
}
//______________________________________________________________________
//
void Simple_Burn::scheduleMomentumAndEnergyExchange(SchedulerP&,
				       const LevelP& level,
				       const ModelInfo*)
{
  // None
}
