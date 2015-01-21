#include <CCA/Components/Models/HEChem/Steady_Burn.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Labels/ICELabel.h>
#include <Core/Labels/MPMICELabel.h>
#include <CCA/Components/ICE/ICEMaterial.h>
#include <CCA/Components/ICE/BoundaryCond.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <SCIRun/Core/Util/DebugStream.h>
#include <iomanip>
#include <iostream>

#undef DEBUG_SCALAR
#define DEBUG_SCALAR

using namespace Uintah;
using namespace SCIRun;
using namespace std;
//__________________________________   
//  MODELS_DOING_COUT:   dumps when tasks are scheduled and performed
static DebugStream cout_doing("MODELS_DOING_COUT", false);

const double Steady_Burn::EPSILON   = 1e-6;   /* stop epsilon for Bisection-Newton method */

Steady_Burn::Steady_Burn(const ProcessorGroup* myworld, 
                         ProblemSpecP& params,
                         const ProblemSpecP& prob_spec)
  : ModelInterface(myworld), d_params(params), d_prob_spec(prob_spec) { 
  mymatls = 0;
  Mlb  = scinew MPMLabel();
  Ilb  = scinew ICELabel();
  MIlb = scinew MPMICELabel();
  d_saveConservedVars = scinew saveConservedVars();
  //__________________________________
  //  diagnostic labels
  BurningCellLabel = VarLabel::create("SteadyBurn.BurningCell", CCVariable<double>::getTypeDescription());
  TsLabel          = VarLabel::create("SteadyBurn.SurfTemp",  CCVariable<double>::getTypeDescription());
  
  totalMassBurnedLabel  = VarLabel::create( "totalMassBurned",
                                            sum_vartype::getTypeDescription() );
  
  totalHeatReleasedLabel= VarLabel::create( "totalHeatReleased",
                                            sum_vartype::getTypeDescription() );
}


Steady_Burn::~Steady_Burn(){
  delete Ilb;
  delete Mlb; 
  delete MIlb;
  delete d_saveConservedVars;
  
  VarLabel::destroy(BurningCellLabel);
  VarLabel::destroy(TsLabel);
  VarLabel::destroy(totalMassBurnedLabel);
  VarLabel::destroy(totalHeatReleasedLabel);
  
  if(mymatls && mymatls->removeReference())
    delete mymatls;
}

//______________________________________________________________________
void Steady_Burn::problemSetup(GridP&, SimulationStateP& sharedState, ModelSetup*){
  d_sharedState = sharedState;
  matl0 = sharedState->parseAndLookupMaterial(d_params, "fromMaterial");
  matl1 = sharedState->parseAndLookupMaterial(d_params, "toMaterial");  
  
  d_params->require("IdealGasConst",     R );
  d_params->require("PreExpCondPh",      Ac);
  d_params->require("ActEnergyCondPh",   Ec);
  d_params->require("PreExpGasPh",       Bg);
  d_params->require("CondPhaseHeat",     Qc);
  d_params->require("GasPhaseHeat",      Qg);
  d_params->require("HeatConductGasPh",  Kg);
  d_params->require("HeatConductCondPh", Kc);
  d_params->require("SpecificHeatBoth",  Cp);
  d_params->require("MoleWeightGasPh",   MW);
  d_params->require("BoundaryParticles", BP);
  d_params->require("ThresholdPressure", ThresholdPressure);
  d_params->require("IgnitionTemp",      ignitionTemp);
  
  /* initialize constants */
  CC1 = Ac*R*Kc/Ec/Cp;        
  CC2 = Qc/Cp/2;              
  CC3 = 4*Kg*Bg*MW*MW/Cp/R/R;  
  CC4 = Qc/Cp;                
  CC5 = Qg/Cp;                
  
  //__________________________________
  //  Are we saving the total burned mass and total burned energy
  ProblemSpecP DA_ps = d_prob_spec->findBlock("DataArchiver");
  for (ProblemSpecP child = DA_ps->findBlock("save"); child != 0;
       child = child->findNextBlock("save")) {
    map<string,string> var_attr;
    child->getAttributes(var_attr);
    if (var_attr["label"] == "totalMassBurned"){
      d_saveConservedVars->mass  = true;
    }
    if (var_attr["label"] == "totalHeatReleased"){
      d_saveConservedVars->energy = true;
    }
  }
  
  /*  define the materialSet */
  vector<int> m_tmp(2);
  m_tmp[0] = matl0->getDWIndex();
  m_tmp[1] = matl1->getDWIndex();
  mymatls = scinew MaterialSet();            
  
  if( m_tmp[0] != 0 && m_tmp[1] != 0){
    vector<int> m(3);
    m[0] = 0; /* needed for the pressure and NC_CCWeight */ 
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
void Steady_Burn::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute("type","Steady_Burn");
  
  model_ps->appendElement("fromMaterial",matl0->getName());
  model_ps->appendElement("toMaterial",  matl1->getName());
  
  model_ps->appendElement("IdealGasConst",     R );
  model_ps->appendElement("PreExpCondPh",      Ac);
  model_ps->appendElement("ActEnergyCondPh",   Ec);
  model_ps->appendElement("PreExpGasPh",       Bg);
  model_ps->appendElement("CondPhaseHeat",     Qc);
  model_ps->appendElement("GasPhaseHeat",      Qg);
  model_ps->appendElement("HeatConductGasPh",  Kg);
  model_ps->appendElement("HeatConductCondPh", Kc);
  model_ps->appendElement("SpecificHeatBoth",  Cp);
  model_ps->appendElement("MoleWeightGasPh",   MW);
  model_ps->appendElement("BoundaryParticles", BP);
  model_ps->appendElement("ThresholdPressure", ThresholdPressure);
  model_ps->appendElement("IgnitionTemp",      ignitionTemp);
}
//______________________________________________________________________
void Steady_Burn::scheduleInitialize(SchedulerP& sched, const LevelP& level, const ModelInfo*){
  printSchedule(level,"Steady_Burn::scheduleInitialize\t\t\t");
  Task* t = scinew Task("Steady_Burn::initialize", this, &Steady_Burn::initialize);                        
  const MaterialSubset* react_matl = matl0->thisMaterial();
  t->computes(TsLabel, react_matl);
  sched->addTask(t, level->eachPatch(), mymatls);
}

//______________________________________________________________________
void Steady_Burn::initialize(const ProcessorGroup*, 
                             const PatchSubset* patches, 
                             const MaterialSubset* /*matls*/, 
                             DataWarehouse*, 
                             DataWarehouse* new_dw){
  int m0 = matl0->getDWIndex();
  for(int p=0;p<patches->size();p++) {
    const Patch* patch = patches->get(p);
    cout_doing << "Doing Initialize on patch " << patch->getID()<< "\t\t\t STEADY_BURN" << endl; 
    
    CCVariable<double> Ts;
    new_dw->allocateAndPut(Ts, TsLabel, m0, patch);
    Ts.initialize(0.0);
    // What does this task do??
  }        
}

//______________________________________________________________________
void Steady_Burn::scheduleComputeStableTimestep(SchedulerP&, const LevelP&, const ModelInfo*){
  // None necessary...
}

//______________________________________________________________________
// only perform this task on the finest level
void Steady_Burn::scheduleComputeModelSources(SchedulerP& sched, 
                                              const LevelP& level, 
                                              const ModelInfo* mi){
  
  if (level->hasFinerLevel())
    return;  
  
  Task* t = scinew Task("Steady_Burn::computeModelSources", this, 
                        &Steady_Burn::computeModelSources, mi);
  
  printSchedule(level,"Steady_Burn::scheduleComputeModelSources\t\t\t");  
  t->requires( Task::OldDW, mi->delT_Label);
  
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gan = Ghost::AroundNodes;  
  Ghost::GhostType  gn  = Ghost::None;
  
  // define material subsets  
  const MaterialSet* all_matls = d_sharedState->allMaterials();
  const MaterialSubset* all_matls_sub = all_matls->getUnion();
  
  const MaterialSubset* react_matl = matl0->thisMaterial();  
  MaterialSubset* one_matl     = scinew MaterialSubset();
  one_matl->add(0);
  one_matl->addReference();
  
  Task::DomainSpec oms = Task::OutOfDomain;  //outside of mymatl set.

  t->requires(Task::OldDW, Ilb->temp_CCLabel,      all_matls_sub, oms, gac,1);
  t->requires(Task::NewDW, Ilb->vol_frac_CCLabel,  all_matls_sub, oms, gac,1);
  /*     Products     */
  /*     Reactants    */
  t->requires(Task::NewDW, Ilb->sp_vol_CCLabel,   react_matl, gn);
  t->requires(Task::NewDW, MIlb->vel_CCLabel,     react_matl, gn);
  t->requires(Task::NewDW, MIlb->cMassLabel,      react_matl, gn);
  t->requires(Task::NewDW, MIlb->gMassLabel,      react_matl, gac,1);
  t->requires(Task::OldDW, Mlb->pXLabel,          react_matl, gan,1);
  /*     Misc      */
  t->requires(Task::NewDW,  Ilb->press_equil_CCLabel, one_matl, gac, 1);
  t->requires(Task::OldDW,  MIlb->NC_CCweightLabel,   one_matl, gac, 1);  
  
  t->modifies(mi->mass_source_CCLabel);
  t->modifies(mi->momentum_source_CCLabel);
  t->modifies(mi->energy_source_CCLabel);
  t->modifies(mi->sp_vol_source_CCLabel); 
  
  t->computes(BurningCellLabel, react_matl);
  t->computes(TsLabel,          react_matl);
     
  if(d_saveConservedVars->mass ){
    t->computes(Steady_Burn::totalMassBurnedLabel);
  }
  if(d_saveConservedVars->energy){
    t->computes(Steady_Burn::totalHeatReleasedLabel);
  } 
  
  sched->addTask(t, level->eachPatch(), mymatls);
  if(one_matl->removeReference())
    delete one_matl;
}


void Steady_Burn::scheduleModifyThermoTransportProperties(SchedulerP&, 
                                                          const LevelP&, 
                                                          const MaterialSet*){
  // do nothing      
}


void Steady_Burn::computeSpecificHeat(CCVariable<double>&, const Patch*, DataWarehouse*, const int){
  //do nothing
}



/*
 ***************** Private Member Functions:******************************
 */
void Steady_Burn::computeModelSources(const ProcessorGroup*, 
                                      const PatchSubset* patches,
                                      const MaterialSubset* /*matls*/,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw,
                                      const ModelInfo* mi){
  
  delt_vartype delT;
  old_dw->get(delT, mi->delT_Label,getLevel(patches));
  
  //ASSERT(matls->size() == 2);
  int m0 = matl0->getDWIndex(); /* reactant material */
  int m1 = matl1->getDWIndex(); /* product material */
  double totalBurnedMass = 0;
  double totalHeatReleased = 0;
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);  
    
    printTask(patches,patch,"Doing computeModelSources\t\t\t\t");
    CCVariable<double> mass_src_0, mass_src_1, mass_0;
    CCVariable<Vector> momentum_src_0, momentum_src_1;
    CCVariable<double> energy_src_0, energy_src_1;
    CCVariable<double> sp_vol_src_0, sp_vol_src_1;

    /* reactant */
    new_dw->getModifiable(mass_src_0,     mi->mass_source_CCLabel,     m0, patch);    
    new_dw->getModifiable(momentum_src_0, mi->momentum_source_CCLabel, m0, patch); 
    new_dw->getModifiable(energy_src_0,   mi->energy_source_CCLabel,   m0, patch);   
    new_dw->getModifiable(sp_vol_src_0,   mi->sp_vol_source_CCLabel,   m0, patch);   
    /* product */
    new_dw->getModifiable(mass_src_1,     mi->mass_source_CCLabel,     m1, patch);   
    new_dw->getModifiable(momentum_src_1, mi->momentum_source_CCLabel, m1, patch);   
    new_dw->getModifiable(energy_src_1,   mi->energy_source_CCLabel,   m1, patch);   
    new_dw->getModifiable(sp_vol_src_1,   mi->sp_vol_source_CCLabel,   m1, patch);
    
    constCCVariable<double>   press_CC, solidTemp, solidMass, solidSp_vol;
    constNCVariable<double>   NC_CCweight, NCsolidMass;
    constCCVariable<Vector>   vel_CC;

    Ghost::GhostType  gn  = Ghost::None;    
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gan = Ghost::AroundNodes;
    /* Reactant data */
    old_dw->get(solidTemp,       MIlb->temp_CCLabel,    m0, patch, gac, 1);
    new_dw->get(solidMass,       MIlb->cMassLabel,      m0, patch, gn,  0);
    new_dw->get(solidSp_vol,     Ilb->sp_vol_CCLabel,   m0, patch, gn,  0);   
    new_dw->get(vel_CC,          MIlb->vel_CCLabel,     m0, patch, gn,  0);
    new_dw->get(NCsolidMass,     MIlb->gMassLabel,      m0, patch, gac, 1);


    constParticleVariable<Point>  px;
    ParticleSubset* pset = old_dw->getParticleSubset(m0, patch, gan,1, Mlb->pXLabel);
    old_dw->get(px, Mlb->pXLabel, pset);    
    /* Product Data */
       
    /* Misc */
    new_dw->get(press_CC,       Ilb->press_equil_CCLabel,      0, patch, gac, 1);
    old_dw->get(NC_CCweight,    MIlb->NC_CCweightLabel,        0, patch, gac, 1);

    CCVariable<double> BurningCell, surfTemp;
    new_dw->allocateAndPut(BurningCell, BurningCellLabel, m0, patch, gn, 0);
    new_dw->allocateAndPut(surfTemp,    TsLabel,          m0, patch, gn, 0);
    BurningCell.initialize(0.0);
    surfTemp.initialize(0.0);

    /* Indicating cells containing how many particles */
    CCVariable<double> pFlag;
    new_dw->allocateTemporary(pFlag, patch, gac, 1);
    pFlag.initialize(0.0);

    /* All Material Data */
    int numAllMatls = d_sharedState->getNumMatls();
    StaticArray<constCCVariable<double> >  vol_frac_CC(numAllMatls);
    StaticArray<constCCVariable<double> >  temp_CC(numAllMatls);
    for (int m = 0; m < numAllMatls; m++) {
      Material* matl = d_sharedState->getMaterial(m);
      int indx = matl->getDWIndex();
      old_dw->get(temp_CC[m],       MIlb->temp_CCLabel,      indx, patch, gac, 1);
      new_dw->get(vol_frac_CC[m],   Ilb->vol_frac_CCLabel,   indx, patch, gac, 1);
    }
    
    
    //===============================================
    //If you change the burning surface criteria logic you must also modify
    //CCA/Components/SwitchCriteria/SteadyBurn.cc
    //===============================================
    /* count how many reactant particles in each cell */
    for(ParticleSubset::iterator iter=pset->begin(), iter_end=pset->end(); iter != iter_end; iter++){
      particleIndex idx = *iter;
      IntVector c;
      patch->findCell(px[idx],c);
      pFlag[c] += 1.0;
    }    
    setBC(pFlag, "zeroNeumann", patch, d_sharedState, m0, new_dw);
 
    Vector dx = patch->dCell();
    MIN_MASS_IN_A_CELL = dx.x()*dx.y()*dx.z()*d_TINY_RHO;
    

    /* Cell Iteration */
    IntVector nodeIdx[8];
    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      patch->findNodesFromCell(*iter,nodeIdx);

      double MaxMass = d_SMALL_NUM;
      double MinMass = 1.0/d_SMALL_NUM; 
      for (int nN=0; nN<8; nN++){
        MaxMass = std::max(MaxMass,NC_CCweight[nodeIdx[nN]]*NCsolidMass[nodeIdx[nN]]);
        MinMass = std::min(MinMass,NC_CCweight[nodeIdx[nN]]*NCsolidMass[nodeIdx[nN]]); 
      }

      bool   burning = 0;
      double maxProductVolFrac  = -1.0;
      double maxReactantVolFrac = -1.0;
      double productPress = 0.0;
      double Tzero = 0.0;
      double temp_vf = 0.0;
      
      /*if( (MaxMass-MinMass)/MaxMass>0.4 && (MaxMass-MinMass)/MaxMass<1.0 && pFlag[c]>0 ){ */
      if( MinMass/MaxMass<0.7 && pFlag[c]>0 ){ 
        /* near interface and containing particles */
        for(int i = -1; i<=1; i++){
          for(int j = -1; j<=1; j++){
            for(int k = -1; k<=1; k++){
              IntVector cell = c + IntVector(i,j,k);

              /* Search for Tzero from max_vol_frac reactant cell */
              temp_vf = vol_frac_CC[m0][cell]; 
              if( temp_vf > maxReactantVolFrac ){
                maxReactantVolFrac = temp_vf;
                Tzero = solidTemp[cell];
              }//endif

              /* Search for pressure from max_vol_frac product cell */
              temp_vf = vol_frac_CC[m1][cell]; 
              if( temp_vf > maxProductVolFrac ){
                maxProductVolFrac = temp_vf;
                productPress = press_CC[cell];
              }//endif
              
              if(burning == 0 && pFlag[cell] <= BP){
                for (int m = 0; m < numAllMatls; m++){
                  if(vol_frac_CC[m][cell] > 0.2 && temp_CC[m][cell] > ignitionTemp){
                    burning = 1;
                    break;
                  }
                }
              }//endif

            }//end 3rd for
          }//end 2nd for
        }//end 1st for
      }//endif
      if(burning == 1 && productPress >= ThresholdPressure){
        BurningCell[c]=1.0;
        
        Vector rhoGradVector = computeDensityGradientVector(nodeIdx, NCsolidMass, NC_CCweight,dx);
       
        double surfArea = computeSurfaceArea(rhoGradVector, dx); 
        double Tsurf = 800.0;  // initial guess for the surface temperature.
        
        double burnedMass = computeBurnedMass(Tzero, Tsurf,  productPress, solidSp_vol[c], 
                                              surfArea, delT, solidMass[c]);
                                              
        surfTemp[c] = Tsurf;
        
        /* conservation of mass, momentum and energy   */
         mass_src_0[c]   -= burnedMass;
         mass_src_1[c]   += burnedMass;
         totalBurnedMass += burnedMass;
       
        Vector momX = vel_CC[c] * burnedMass;
        momentum_src_0[c]  -= momX;
        momentum_src_1[c]    += momX;
        
        double energyX   = Cp*solidTemp[c]*burnedMass; 
        double releasedHeat = burnedMass * (Qc + Qg);
        energy_src_0[c]   -= energyX;
        energy_src_1[c]   += energyX + releasedHeat;
        totalHeatReleased += releasedHeat;
        
        double createdVolx = burnedMass * solidSp_vol[c];
        sp_vol_src_0[c]  -= createdVolx;
        sp_vol_src_1[c]    += createdVolx;
      }  // if (cell is ignited)
    }  // cell iterator


    /*  set symetric BC  */
    setBC(mass_src_0, "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
    setBC(mass_src_1, "set_if_sym_BC",patch, d_sharedState, m1, new_dw); 
  }
  //__________________________________
  //save total quantities
  if(d_saveConservedVars->mass ){
    new_dw->put(sum_vartype(totalBurnedMass),  Steady_Burn::totalMassBurnedLabel);
  }
  if(d_saveConservedVars->energy){
    new_dw->put(sum_vartype(totalHeatReleased),Steady_Burn::totalHeatReleasedLabel);
  }
  
}

//______________________________________________________________________
double Steady_Burn::computeSurfaceArea(Vector &rhoGradVector, Vector &dx){
  double delX = dx.x();
  double delY = dx.y();
  double delZ = dx.z();
  double rgvX = fabs(rhoGradVector.x());
  double rgvY = fabs(rhoGradVector.y());
  double rgvZ = fabs(rhoGradVector.z());
  
  double max = rgvX;
  if(rgvY > max)   max = rgvY;
  if(rgvZ > max)   max = rgvZ;
  
  double coeff = pow(1.0/max, 1.0/3.0);

  double TmpX = delX*rgvX;
  double TmpY = delY*rgvY;
  double TmpZ = delZ*rgvZ;
    
  return delX*delY*delZ / (TmpX+TmpY+TmpZ) * coeff; 
}

//______________________________________________________________________
//
Vector Steady_Burn::computeDensityGradientVector(IntVector *nodeIdx, 
                                                 constNCVariable<double> &NCsolidMass, 
                                                 constNCVariable<double> &NC_CCweight, 
                                                 Vector &dx){
  double gradRhoX = 0.25 * (
                            (NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
                             NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+
                             NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+
                             NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]])
                            -
                            (NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
                             NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]]+
                             NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]]+
                             NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]])
                            )/dx.x();

  double gradRhoY = 0.25 * (
                            (NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
                             NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+
                             NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
                             NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]])
                            -
                            (NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+
                             NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]]+
                             NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]]+
                             NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]])
                            )/dx.y();

  double gradRhoZ = 0.25 * (
                            (NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+
                             NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]]+
                             NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]]+
                             NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]])
                            -
                            (NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
                             NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+
                             NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
                             NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]])
                            )/dx.z();

  double absGradRho = sqrt(gradRhoX*gradRhoX + gradRhoY*gradRhoY + gradRhoZ*gradRhoZ );

  return Vector(gradRhoX/absGradRho, gradRhoY/absGradRho, gradRhoZ/absGradRho);
}

void Steady_Burn::scheduleErrorEstimate(const LevelP&, SchedulerP&){
  // Not implemented yet
}

void Steady_Burn::scheduleTestConservation(SchedulerP&, const PatchSet*, const ModelInfo*){
  // Not implemented yet
}

/****************************************************************************/
/******************* Bisection Newton Solver ********************************/
/****************************************************************************/
double Steady_Burn::computeBurnedMass(double To, double& Ts, double P, double Vc, double surfArea, 
                                      double delT, double solidMass){  
  UpdateConstants(To, P, Vc);
  Ts = BisectionNewton(Ts);
  double m =  m_Ts(Ts);
  double burnedMass = delT * surfArea * m;
  if (burnedMass + MIN_MASS_IN_A_CELL > solidMass) 
    burnedMass = solidMass - MIN_MASS_IN_A_CELL;  
  return burnedMass;
}

//______________________________________________________________________
void Steady_Burn::UpdateConstants(double To, double P, double Vc){
  /* CC1 = Ac*R*Kc/Ec/Cp        */
  /* CC2 = Qc/Cp/2              */
  /* CC3 = 4*Kg*Bg*W*W/Cp/R/R;  */
  /* CC4 = Qc/Cp                */
  /* CC5 = Qg/Cp                */
  /* Vc = Condensed Phase Specific Volume */
  
  C1 = CC1 / Vc; 
  C2 = To + CC2; 
  C3 = CC3 * P*P;
  C4 = To + CC4; 
  C5 = CC5 * C3; 
  
  Tmin = C4;
  double Tsmax = Ts_max();
  if (Tmin < Tsmax)
    Tmax =  F_Ts(Tsmax);
  else
    Tmax = F_Ts(Tmin);
  
  IL = Tmin;
  IR = Tmax;
}


/***   
 ***   Ts = F_Ts(Ts) = Ts_m(m_Ts(Ts))                                              
 ***   f_Ts(Ts) = C4 + C5/(sqrt(m^2+C3) + m)^2 
 ***
 ***   Solve for diff(f_Ts(Ts))=0 
 ***   Ts_max = C2 - Ec/2R + sqrt(4*R^2*C2^2+Ec^2)/2R
 ***   f_Ts_max = f_Ts(Ts_max)
 ***/
double Steady_Burn::F_Ts(double Ts){
  return Ts_m(m_Ts(Ts));
}

double Steady_Burn::m_Ts(double Ts){
  return sqrt( C1*Ts*Ts/(Ts-C2)*exp(-Ec/R/Ts) );
}

double Steady_Burn::Ts_m(double m){
  double deno = sqrt(m*m+C3)+m;
  return C4 + C5/(deno*deno);
}

/* the function value for the zero finding problem */
double Steady_Burn::Func(double Ts){
  return Ts - F_Ts(Ts);
}

/* dFunc/dTs */
double Steady_Burn::Deri(double Ts){
  double m = m_Ts(Ts);
  double K1 = Ts-C2;
  double K2 = sqrt(m*m+C3);
  double K3 = (R*Ts*(K1-C2)+Ec*K1)*m*C5;
  double K4 = (K2+m)*(K2+m)*K1*K2*R*Ts*Ts;
  return 1.0 + K3/K4;
}

/* F_Ts(Ts_max) is the max of F_Ts function */
double Steady_Burn::Ts_max(){
  return 0.5*(2.0*R*C2 - Ec + sqrt(4.0*R*R*C2*C2+Ec*Ec))/R;
} 

void Steady_Burn::SetInterval(double f, double Ts){  
  /* IL <= 0,  IR >= 0 */
  if(f < 0)  
    IL = Ts;
  else if(f > 0)
    IR = Ts;
  else if(f ==0){
    IL = Ts;
    IR = Ts; 
  }
}

/* Bisection - Newton Method */
double Steady_Burn::BisectionNewton(double Ts){  
  double y = 0;
  double df_dTs = 0;
  double delta_old = 0;
  double delta_new = 0;
  
  int iter = 0;
  if(Ts>Tmax || Ts<Tmin)
    Ts = (Tmin+Tmax)/2;
  
  while(1){
    iter++;
    y = Func(Ts);
    SetInterval(y, Ts);
    
    if(fabs(y)<EPSILON)
      return Ts;
    
    delta_new = 1e100;
    while(1){
      if(iter>100){
        cout<<"Not converging after 100 iterations in Steady_Burn.cc."<<endl;
        exit(1);
      }

      df_dTs = Deri(Ts);
      if(df_dTs==0) 
        break;

      delta_old = delta_new;
      delta_new = -y/df_dTs; //Newton Step
      Ts += delta_new;
      y = Func(Ts);

      if(fabs(y)<EPSILON)
        return Ts;
      
      if(Ts<IL || Ts>IR || fabs(delta_new)>fabs(delta_old*0.7))
        break;

      iter++; 
      SetInterval(y, Ts);  
    }
    
    Ts = (IL+IR)/2.0; //Bisection Step
  }
}

//______________________________________________________________________
void Steady_Burn::printSchedule(const LevelP& level,
                                const string& where){
  if (cout_doing.active()){
    cout_doing << d_myworld->myrank() << " " 
               << where << "L-"
               << level->getIndex()<< endl;
  }  
}
//______________________________________________________________________
void Steady_Burn::printTask(const PatchSubset* patches,
                            const Patch* patch,
                            const string& where){
  if (cout_doing.active()){
    cout_doing << d_myworld->myrank() << " " 
               << where << " STEADY_BURN L-"
               << getLevel(patches)->getIndex()
               << " patch " << patch->getGridIndex()<< endl;
  }  
}
//______________________________________________________________________
