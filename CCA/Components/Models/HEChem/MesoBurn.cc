/*

The MIT License

Copyright (c) 1997-2012 The University of Utah

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

*/


#include <CCA/Components/Models/HEChem/MesoBurn.h>
#include <CCA/Components/Models/HEChem/Common.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Labels/ICELabel.h>
#include <Core/Labels/MPMICELabel.h>
#include <Core/Grid/DbgOutput.h>
#include <CCA/Components/ICE/ICEMaterial.h>
#include <CCA/Components/ICE/BoundaryCond.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/ConstitutiveModel/ElasticPlasticHP.h>
#include <Core/Util/DebugStream.h>
#include <iomanip>
#include <iostream>

#undef DEBUG_SCALAR
#define DEBUG_SCALAR

using namespace Uintah;
using namespace std;
//__________________________________   
//  MODELS_DOING_COUT:   dumps when tasks are scheduled and performed
static DebugStream cout_doing("MODELS_DOING_COUT", false);

const double MesoBurn::EPSILON   = 1e-6;   /* stop epsilon for Bisection-Newton method */
const double ONE_YEAR_MICROSECONDS = 3.1536e13;

MesoBurn::MesoBurn(const ProcessorGroup* myworld, 
                   ProblemSpecP& params,
                   const ProblemSpecP& prob_spec)
  : ModelInterface(myworld), d_params(params), d_prob_spec(prob_spec) { 
  mymatls = 0;
  Mlb  = scinew MPMLabel();
  Ilb  = scinew ICELabel();
  MIlb = scinew MPMICELabel();
  d_saveConservedVars = scinew saveConservedVars();

  //  Diagnostic labels
  BurningCellLabel = VarLabel::create("MesoBurn.BurningCell",
                                      CCVariable<double>::getTypeDescription());
  TsLabel          = VarLabel::create("MesoBurn.SurfTemp",
                                      CCVariable<double>::getTypeDescription());
  numPPCLabel      = VarLabel::create("MesoBurn.numPPC",
                                      CCVariable<double>::getTypeDescription());
  inductionTimeLabel = VarLabel::create("MesoBurn.inductionTime",
                                        CCVariable<double>::getTypeDescription());
  inducedLabel     = VarLabel::create("MesoBurn.induced",
                                      CCVariable<double>::getTypeDescription());
  inducedMassLabel = VarLabel::create("MesoBurn.inducedMass",
                                      CCVariable<double>::getTypeDescription());
  inductionTimePartLabel = VarLabel::create("MesoBurn.inductionTimePart",
                                            ParticleVariable<double>::getTypeDescription());
  timeInducedLabel = VarLabel::create("MesoBurn.timeInduced",
                                      ParticleVariable<double>::getTypeDescription());
      
  // Cummulative varibles
  totalMassBurnedLabel  = VarLabel::create("totalMassBurned",
                                           sum_vartype::getTypeDescription());
  totalHeatReleasedLabel= VarLabel::create("totalHeatReleased",
                                           sum_vartype::getTypeDescription());
  totalSurfaceAreaLabel = VarLabel::create("totalSurfaceArea",
                                           sum_vartype::getTypeDescription());
}


MesoBurn::~MesoBurn(){
  delete Ilb;
  delete Mlb; 
  delete MIlb;
  delete d_saveConservedVars;
  
  // Diagnostic Labels
  VarLabel::destroy(BurningCellLabel);
  VarLabel::destroy(TsLabel);
  VarLabel::destroy(numPPCLabel);
  VarLabel::destroy(inductionTimeLabel);
  VarLabel::destroy(inductionTimePartLabel);
  VarLabel::destroy(inducedLabel);
  VarLabel::destroy(inducedMassLabel);
  VarLabel::destroy(timeInducedLabel);
    
  // Cummulative Labels
  VarLabel::destroy(totalMassBurnedLabel);
  VarLabel::destroy(totalHeatReleasedLabel);
  VarLabel::destroy(totalSurfaceAreaLabel);
  
  if(mymatls && mymatls->removeReference())
    delete mymatls;
}

//______________________________________________________________________
void MesoBurn::problemSetup(GridP&, 
                            SimulationStateP& sharedState, 
                            ModelSetup*){
  d_sharedState = sharedState;
  matl0 = sharedState->parseAndLookupMaterial(d_params, "fromMaterial");
  matl1 = sharedState->parseAndLookupMaterial(d_params, "toMaterial");  
  
  // Burn parameters
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
   
  // Induction parameters
  d_params->require("Ta", Ta);
  d_params->require("k",  k);
  d_params->require("Cv", Cv);
  d_params->getWithDefault("AfterMelting", afterMelting, false);
  d_params->getWithDefault("resolution", resolution, 8);
    
  // Just in case... though the problem spec should pick these up
  if( Ta < 0.0 )
    throw new ProblemSetupException(std::string("Activation temperature (Ta) cannot be negative."), __FILE__, __LINE__);
  if( k < 0.0 )
    throw new ProblemSetupException(std::string("Rate constant (k) cannot be negative."), __FILE__, __LINE__);  
  if( Cv < 0.0 )
    throw new ProblemSetupException(std::string("Specific heat (Cv) cannot be negative."), __FILE__, __LINE__);
    
  /* initialize constants */
  CC1 = Ac*R*Kc/Ec/Cp;        
  CC2 = Qc/Cp/2;              
  CC3 = 4*Kg*Bg*MW*MW/Cp/R/R;  
  CC4 = Qc/Cp;                
  CC5 = Qg/Cp;                
  
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
    if (var_attr["label"] == "totalSurfaceArea"){
      d_saveConservedVars->surfaceArea = true;
    }
  }
  
  
  //__________________________________
  //  define the materialSet
  mymatls = scinew MaterialSet();

  vector<int> m;
  m.push_back(0);                                 // needed for the pressure and NC_CCWeight
  m.push_back(matl0->getDWIndex());
  m.push_back(matl1->getDWIndex());

  mymatls->addAll_unique(m);                    // elimiate duplicate entries
  mymatls->addReference(); 
    
  // Check that the Constitutive Equation if of type ElasticPlasticHP 
  //  so that it has a melting temperature model
  if(afterMelting) {
    const MPMMaterial *solid = dynamic_cast<const MPMMaterial *>(matl0);
    if(solid == NULL)
      throw new ProblemSetupException(std::string("The 'afterMelting' parameter requires a MPM based reactant."), __FILE__, __LINE__);
    if(dynamic_cast<ElasticPlasticHP *>(solid->getConstitutiveModel()) == NULL)
      throw new ProblemSetupException(std::string("The 'afterMelting' parameter requires an ElasticPlasticHP based reactant for its melting temperature."), __FILE__, __LINE__);
  }
}
//______________________________________________________________________
void MesoBurn::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute("type","MesoBurn");
  
  model_ps->appendElement("fromMaterial",matl0->getName());
  model_ps->appendElement("toMaterial",  matl1->getName());
  
  // Burn parameters
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
    
  // Induction parameters
  model_ps->appendElement("Cv", Cv);
  model_ps->appendElement("k",  k);
  model_ps->appendElement("Ta", Ta);
  model_ps->appendElement("AfterMelting", afterMelting);
}
//______________________________________________________________________
void MesoBurn::scheduleInitialize(SchedulerP& sched, 
                                  const LevelP& level, 
                                  const ModelInfo*){
  printSchedule(level, cout_doing,"MesoBurn::scheduleInitialize");
  
  Task* t = scinew Task("MesoBurn::initialize", this, &MesoBurn::initialize);                        
  const MaterialSubset* react_matl = matl0->thisMaterial();
  t->computes(TsLabel, react_matl);
  t->computes(inductionTimeLabel, react_matl);
  t->computes(inducedLabel, react_matl);
  t->computes(inductionTimePartLabel, react_matl);
  t->computes(timeInducedLabel, react_matl);
  t->computes(inducedMassLabel, react_matl);
  sched->addTask(t, level->eachPatch(), mymatls);
}

//______________________________________________________________________
void MesoBurn::initialize(const ProcessorGroup*, 
                          const PatchSubset* patches, 
                          const MaterialSubset* /*matls*/, 
                          DataWarehouse* old_dw, 
                          DataWarehouse* new_dw){
  int m0 = matl0->getDWIndex();
  for(int p=0;p<patches->size();p++) {
    const Patch* patch = patches->get(p);
    ParticleSubset* pset = old_dw->getParticleSubset(m0, patch);
    cout_doing << "Doing Initialize on patch " << patch->getID()<< "\t\t\t MesoBurn" << endl; 
    
    CCVariable<double> Ts;
    new_dw->allocateAndPut(Ts, TsLabel, m0, patch);
    Ts.initialize(0.0);
      
    CCVariable<double> inductionTime;
    CCVariable<double> induced;
    CCVariable<double> inducedMass;
    ParticleVariable<double> inductionTimePart;
    ParticleVariable<double> timeInduced;
    new_dw->allocateAndPut(inductionTime, inductionTimeLabel, m0, patch);
    new_dw->allocateAndPut(induced, inducedLabel, m0, patch);
    new_dw->allocateAndPut(inducedMass, inducedMassLabel, m0, patch);
    new_dw->allocateAndPut(inductionTimePart, inductionTimePartLabel, pset);
    new_dw->allocateAndPut(timeInduced, timeInducedLabel, pset);
    inductionTime.initialize(ONE_YEAR_MICROSECONDS);
    induced.initialize(0.0);
    inducedMass.initialize(0.0);
    for(ParticleSubset::iterator iter=pset->begin(), iter_end=pset->end();
                                                     iter != iter_end; iter++){
      particleIndex idx = *iter;
      inductionTimePart[idx] = ONE_YEAR_MICROSECONDS;
      timeInduced[idx] = 0.0;
    }
  }        
}

//______________________________________________________________________
void MesoBurn::scheduleComputeStableTimestep(SchedulerP&, 
                                             const LevelP&, 
                                             const ModelInfo*){
  // None necessary...
}

//______________________________________________________________________
// only perform this task on the finest level
void MesoBurn::scheduleComputeModelSources(SchedulerP& sched,
                                           const LevelP& level, 
                                           const ModelInfo* mi){
  
  if (level->hasFinerLevel()){
    return;  
  }

  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn  = Ghost::None;
  const MaterialSubset* react_matl = matl0->thisMaterial();  

  Task* t1 = scinew Task("MesoBurn::computeParticleVariables", this, 
                         &MesoBurn::computeParticleVariables, mi);

  printSchedule(level, cout_doing,"MesoBurn::scheduleComputeParticleVariables");  

  t1->requires( Task::OldDW, mi->delT_Label, level.get_rep());
  t1->requires(Task::OldDW, Mlb->pXLabel, react_matl, gn);
  t1->requires(Task::OldDW, Mlb->pMassLabel, react_matl, gn);
  t1->requires(Task::OldDW, Mlb->pTemperatureLabel, react_matl, gn);
  t1->requires(Task::OldDW, inductionTimePartLabel, react_matl, gn);
  t1->requires(Task::OldDW, timeInducedLabel, react_matl, gn);
  t1->computes(numPPCLabel,            react_matl);
  t1->computes(inductionTimeLabel,     react_matl);
  t1->computes(inductionTimePartLabel, react_matl);
  t1->computes(inducedLabel,           react_matl);
  t1->computes(inducedMassLabel,       react_matl);
  t1->computes(timeInducedLabel,       react_matl);

  sched->addTask(t1, level->eachPatch(), mymatls);


  //__________________________________
  Task* t = scinew Task("MesoBurn::computeModelSources", this, 
                        &MesoBurn::computeModelSources, mi);

  printSchedule(level,cout_doing,"MesoBurn::scheduleComputeModelSources");  
  t->requires( Task::OldDW, mi->delT_Label, level.get_rep());
  
  // define material subsets  
  const MaterialSet* all_matls = d_sharedState->allMaterials();
  const MaterialSubset* all_matls_sub = all_matls->getUnion();
  
  MaterialSubset* one_matl     = scinew MaterialSubset();
  one_matl->add(0);
  one_matl->addReference();
  
  Task::MaterialDomainSpec oms = Task::OutOfDomain;  //outside of mymatl set.

  t->requires(Task::OldDW, Ilb->temp_CCLabel,      all_matls_sub, oms, gac,1);
  t->requires(Task::NewDW, Ilb->vol_frac_CCLabel,  all_matls_sub, oms, gac,1);
  /*     Products     */
  /*     Reactants    */
  t->requires(Task::NewDW, Ilb->sp_vol_CCLabel,   react_matl, gn);
  t->requires(Task::NewDW, MIlb->vel_CCLabel,     react_matl, gn);
  t->requires(Task::NewDW, MIlb->cMassLabel,      react_matl, gn);
  t->requires(Task::NewDW, MIlb->gMassLabel,      react_matl, gac,1);
  t->requires(Task::NewDW, numPPCLabel,           react_matl, gac,1);
  t->requires(Task::NewDW, inducedLabel,          react_matl, gn);
  t->requires(Task::NewDW, inducedMassLabel,      react_matl, gn);
  /*     Misc      */
  t->requires(Task::NewDW,  Ilb->press_equil_CCLabel, one_matl, gac, 1);
  t->requires(Task::OldDW,  MIlb->NC_CCweightLabel,   one_matl, gac, 1);  
  
  t->modifies(mi->modelMass_srcLabel);
  t->modifies(mi->modelMom_srcLabel);
  t->modifies(mi->modelEng_srcLabel);
  t->modifies(mi->modelVol_srcLabel); 
  
  t->computes(BurningCellLabel, react_matl);
  t->computes(TsLabel,          react_matl);
     
  // Reduction variables
  if(d_saveConservedVars->mass ){
    t->computes(MesoBurn::totalMassBurnedLabel);
  }
  if(d_saveConservedVars->energy){
    t->computes(MesoBurn::totalHeatReleasedLabel);
  } 
  if(d_saveConservedVars->surfaceArea){
    t->computes(MesoBurn::totalSurfaceAreaLabel);
  } 
  
  sched->addTask(t, level->eachPatch(), mymatls);
  if(one_matl->removeReference())
    delete one_matl;
}

//__________________________________
void MesoBurn::scheduleModifyThermoTransportProperties(SchedulerP&, 
                                                       const LevelP&, 
                                                       const MaterialSet*){
  // do nothing      
}

//__________________________________
void MesoBurn::computeSpecificHeat(CCVariable<double>&, 
                                   const Patch*, 
                                   DataWarehouse*, 
                                   const int){
  //do nothing
}



/*
 ***************** Private Member Functions:******************************
 */
void MesoBurn::computeParticleVariables(const ProcessorGroup*, 
                                        const PatchSubset* patches,
                                        const MaterialSubset* /*matls*/,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw,
                                        const ModelInfo* mi)
{
  delt_vartype delT;
  old_dw->get(delT, mi->delT_Label,getLevel(patches));
    
  int m0 = matl0->getDWIndex(); /* reactant material */

  double pscale = 1.0/resolution;
    
  /* Patch Iteration */
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);  
    printTask(patches,patch,cout_doing,"Doing MesoBurn::computeParticleVariables");

    /* Indicating how many particles a cell contains */
    ParticleSubset* pset = old_dw->getParticleSubset(m0, patch);

    constParticleVariable<Point>  px;
    constParticleVariable<double> pTemp;
    constParticleVariable<double> pMass;
    constParticleVariable<double> inductionTimePartOld;
    constParticleVariable<double> timeInducedOld;
    old_dw->get(px, Mlb->pXLabel, pset);
    old_dw->get(pTemp, Mlb->pTemperatureLabel, pset);
    old_dw->get(pMass, Mlb->pMassLabel, pset);
    old_dw->get(inductionTimePartOld, inductionTimePartLabel, pset);
    old_dw->get(timeInducedOld, timeInducedLabel, pset);
      
    /* Indicating cells containing how many particles */
    CCVariable<double> pFlag;
    CCVariable<double> inductionTime;
    CCVariable<double> induced;
    CCVariable<double> inducedMass;
    ParticleVariable<double> inductionTimePart;
    ParticleVariable<double> timeInduced;
    

    new_dw->allocateAndPut(pFlag,         numPPCLabel,        m0, patch);
    pFlag.initialize(0.0);
    new_dw->allocateAndPut(inductionTime, inductionTimeLabel, m0, patch);
    inductionTime.initialize(ONE_YEAR_MICROSECONDS);
    new_dw->allocateAndPut(induced,       inducedLabel,       m0, patch);
    induced.initialize(0.0);
    new_dw->allocateAndPut(inducedMass,   inducedMassLabel,   m0, patch);
    inducedMass.initialize(0.0);
    new_dw->allocateAndPut(inductionTimePart, inductionTimePartLabel,  pset);
    new_dw->allocateAndPut(timeInduced,   timeInducedLabel,  pset);

    /* count how many reactant particles in each cell */
    for(ParticleSubset::iterator iter=pset->begin(), iter_end=pset->end();
                                                     iter != iter_end; iter++){
      particleIndex idx = *iter;
      IntVector c;
      patch->findCell(px[idx],c);
      pFlag[c] += 1.0;
      // newly computed induction time
      double inductionTimeNew = (pTemp[idx]*pTemp[idx]*Cv/(Ta*Qc))*exp(Ta/pTemp[idx])/k;
      // weighted average induction time
      inductionTimePart[idx] = (0.99*inductionTimePartOld[idx] + 0.01*inductionTimeNew); 
      // update the amount of time a particle has been inducing
      timeInduced[idx] = timeInducedOld[idx];
      if(pTemp[idx] > 470.0)
        timeInduced[idx] += delT;
      if(timeInduced[idx] > inductionTimePart[idx])
      {
        if(inductionTime[c] > inductionTimePart[idx])
          inductionTime[c] = inductionTimePart[idx];
        induced[c]     += pscale;
        inducedMass[c] += pMass[idx];
      }
    }    
    setBC(pFlag, "zeroNeumann", patch, d_sharedState, m0, new_dw);
  }
 
}
//__________________________________
//
void MesoBurn::computeModelSources(const ProcessorGroup*, 
                                   const PatchSubset* patches,
                                   const MaterialSubset* /*matls*/,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw,
                                   const ModelInfo* mi)
{
  delt_vartype delT;
  old_dw->get(delT, mi->delT_Label,getLevel(patches));
  
  //ASSERT(matls->size() == 2);
  int m0 = matl0->getDWIndex(); /* reactant material */
  int m1 = matl1->getDWIndex(); /* product material */
  double totalBurnedMass = 0.0;
  double totalHeatReleased = 0.0;
  double totalSurfaceArea = 0.0;
  
  Ghost::GhostType  gn  = Ghost::None;    
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gp;
  int ngc_p;
  d_sharedState->getParticleGhostLayer(gp, ngc_p);
  
  /* Patch Iteration */
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);  
    
    printTask(patches,patch,cout_doing,"Doing Steady::computeModelSources");
    CCVariable<double> mass_src_0, mass_src_1, mass_0;
    CCVariable<Vector> momentum_src_0, momentum_src_1;
    CCVariable<double> energy_src_0, energy_src_1;
    CCVariable<double> sp_vol_src_0, sp_vol_src_1;

    /* reactant */
    new_dw->getModifiable(mass_src_0,     mi->modelMass_srcLabel,  m0, patch);  
    new_dw->getModifiable(momentum_src_0, mi->modelMom_srcLabel,   m0, patch); 
    new_dw->getModifiable(energy_src_0,   mi->modelEng_srcLabel,   m0, patch);
    new_dw->getModifiable(sp_vol_src_0,   mi->modelVol_srcLabel,   m0, patch);

    /* product */
    new_dw->getModifiable(mass_src_1,     mi->modelMass_srcLabel,  m1, patch); 
    new_dw->getModifiable(momentum_src_1, mi->modelMom_srcLabel,   m1, patch); 
    new_dw->getModifiable(energy_src_1,   mi->modelEng_srcLabel,   m1, patch); 
    new_dw->getModifiable(sp_vol_src_1,   mi->modelVol_srcLabel,   m1, patch);
    
    constCCVariable<double>   press_CC, solidTemp, solidMass, solidSp_vol;
    constNCVariable<double>   NC_CCweight, NCsolidMass;
    constCCVariable<Vector>   vel_CC;
    constCCVariable<double>   pFlag;
    constCCVariable<double>   induced;
    constCCVariable<double>   inducedMass;

    /* Reactant data */
    old_dw->get(solidTemp,       MIlb->temp_CCLabel,    m0, patch, gac, 1);
    new_dw->get(solidMass,       MIlb->cMassLabel,      m0, patch, gn,  0);
    new_dw->get(solidSp_vol,     Ilb->sp_vol_CCLabel,   m0, patch, gn,  0);   
    new_dw->get(vel_CC,          MIlb->vel_CCLabel,     m0, patch, gn,  0);
    new_dw->get(NCsolidMass,     MIlb->gMassLabel,      m0, patch, gac, 1);
    new_dw->get(pFlag,           numPPCLabel,           m0, patch, gac, 1);
    new_dw->get(induced,         inducedLabel,          m0, patch, gn,  0);
    new_dw->get(inducedMass,     inducedMassLabel,      m0, patch, gn,  0);
    
    /* Product Data */   
    /* Misc */
    new_dw->get(press_CC,       Ilb->press_equil_CCLabel,    0, patch, gac, 1);
    old_dw->get(NC_CCweight,    MIlb->NC_CCweightLabel,      0, patch, gac, 1);

    CCVariable<double> BurningCell, surfTemp;
    new_dw->allocateAndPut(BurningCell, BurningCellLabel, m0, patch, gn, 0);
    new_dw->allocateAndPut(surfTemp,    TsLabel,          m0, patch, gn, 0);
    BurningCell.initialize(0.0);
    surfTemp.initialize(0.0);

    /* All Material Data */
    int numAllMatls = d_sharedState->getNumMatls();
    StaticArray<constCCVariable<double> >  vol_frac_CC(numAllMatls);
    StaticArray<constCCVariable<double> >  temp_CC(numAllMatls);
    for (int m = 0; m < numAllMatls; m++) {
      Material* matl = d_sharedState->getMaterial(m);
      int indx = matl->getDWIndex();
      old_dw->get(temp_CC[m],       MIlb->temp_CCLabel,    indx, patch, gac, 1);
      new_dw->get(vol_frac_CC[m],   Ilb->vol_frac_CCLabel, indx, patch, gac, 1);
    }

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
        MaxMass = std::max(MaxMass,
                             NC_CCweight[nodeIdx[nN]]*NCsolidMass[nodeIdx[nN]]);
        MinMass = std::min(MinMass,
                             NC_CCweight[nodeIdx[nN]]*NCsolidMass[nodeIdx[nN]]); 
      }

      /* test whether the current cell satisfies burning criteria */
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
                  if(induced[c] > 0.0 && temp_CC[m][cell] > 470.0){
                    burning = 1.0;
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
        
        Vector rhoGradVector = computeDensityGradientVector(nodeIdx,
                                                            NCsolidMass,
                                                            NC_CCweight,dx);
       
        double surfArea = computeSurfaceArea(rhoGradVector, dx); 
        double Tsurf = 850.0;  // initial guess for the surface temperature.
        
        double burnedMass = computeBurnedMass(Tzero, Tsurf, productPress,
                                              solidSp_vol[c], surfArea, delT,
                                              inducedMass[c]);

        surfTemp[c] = Tsurf;
        totalSurfaceArea += Tsurf;

        /* conservation of mass, momentum and energy   */
        mass_src_0[c]   -= burnedMass;
        mass_src_1[c]   += burnedMass;
        totalBurnedMass += burnedMass;

        Vector momX = vel_CC[c] * burnedMass;
        momentum_src_0[c]  -= momX;
        momentum_src_1[c]  += momX;

        double energyX   = Cp*solidTemp[c]*burnedMass; 
        double releasedHeat = burnedMass * (Qc + Qg);
        energy_src_0[c]   -= energyX;
        energy_src_1[c]   += energyX + releasedHeat;
        totalHeatReleased += releasedHeat;
        
        double createdVolx = burnedMass * solidSp_vol[c];
        sp_vol_src_0[c]  -= createdVolx;
        sp_vol_src_1[c]  += createdVolx;
      }  // if (cell is ignited)
    }  // cell iterator

    /*  set symetric BC  */
    setBC(mass_src_0, "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
    setBC(mass_src_1, "set_if_sym_BC",patch, d_sharedState, m1, new_dw); 
  }
  //__________________________________
  //save total quantities
  if(d_saveConservedVars->mass )
    new_dw->put(sum_vartype(totalBurnedMass),
                MesoBurn::totalMassBurnedLabel);
  if(d_saveConservedVars->energy)
    new_dw->put(sum_vartype(totalHeatReleased),
                MesoBurn::totalHeatReleasedLabel);
  if(d_saveConservedVars->surfaceArea)
    new_dw->put(sum_vartype(totalSurfaceArea),
                MesoBurn::totalSurfaceAreaLabel);
 
}

void MesoBurn::scheduleErrorEstimate(const LevelP&, 
                                     SchedulerP&){
  // Not implemented yet
}

void MesoBurn::scheduleTestConservation(SchedulerP&, 
                                        const PatchSet*, 
                                        const ModelInfo*){
  // Not implemented yet
}


/****************************************************************************/
/******************* Bisection Newton Solver ********************************/
/****************************************************************************/
double MesoBurn::computeBurnedMass(double To, 
                                   double& Ts, 
                                   double P, 
                                   double Vc, 
                                   double surfArea, 
                                   double delT, 
                                   double solidMass){  
  IterationVariables iterVar;
  UpdateConstants(To, P, Vc, &iterVar);
  Ts = BisectionNewton(Ts, &iterVar);
  double m =  m_Ts(Ts, &iterVar);
  double burnedMass = delT * surfArea * m;
  if (burnedMass + MIN_MASS_IN_A_CELL > solidMass) 
    burnedMass = solidMass - MIN_MASS_IN_A_CELL;  
  return burnedMass;
}

//______________________________________________________________________
void MesoBurn::UpdateConstants(double To, 
                               double P, 
                               double Vc, 
                               IterationVariables *iterVar){
  /* CC1 = Ac*R*Kc/Ec/Cp        */
  /* CC2 = Qc/Cp/2              */
  /* CC3 = 4*Kg*Bg*W*W/Cp/R/R;  */
  /* CC4 = Qc/Cp                */
  /* CC5 = Qg/Cp                */
  /* Vc = Condensed Phase Specific Volume */
  
  iterVar->C1 = CC1 / Vc; 
  iterVar->C2 = To + CC2; 
  iterVar->C3 = CC3 * P*P;
  iterVar->C4 = To + CC4; 
  iterVar->C5 = CC5 * iterVar->C3; 
  
  iterVar->Tmin = iterVar->C4;
  double Tsmax = Ts_max(iterVar);
  if (iterVar->Tmin < Tsmax)
    iterVar->Tmax =  F_Ts(Tsmax, iterVar);
  else
    iterVar->Tmax = F_Ts(iterVar->Tmin, iterVar);
  
  iterVar->IL = iterVar->Tmin;
  iterVar->IR = iterVar->Tmax;
}


/***   
 ***   Ts = F_Ts(Ts) = Ts_m(m_Ts(Ts))                                              
 ***   f_Ts(Ts) = C4 + C5/(sqrt(m^2+C3) + m)^2 
 ***
 ***   Solve for diff(f_Ts(Ts))=0 
 ***   Ts_max = C2 - Ec/2R + sqrt(4*R^2*C2^2+Ec^2)/2R
 ***   f_Ts_max = f_Ts(Ts_max)
 ***/
double MesoBurn::F_Ts(double Ts, 
                      IterationVariables *iterVar){
  return Ts_m(m_Ts(Ts, iterVar), iterVar);
}

double MesoBurn::m_Ts(double Ts, 
                      IterationVariables *iterVar){
  return sqrt( iterVar->C1*Ts*Ts/(Ts-iterVar->C2)*exp(-Ec/R/Ts) );
}

double MesoBurn::Ts_m(double m, 
                      IterationVariables *iterVar){
  double deno = sqrt(m*m+iterVar->C3)+m;
  return iterVar->C4 + iterVar->C5/(deno*deno);
}

/* the function value for the zero finding problem */
double MesoBurn::Func(double Ts, 
                      IterationVariables *iterVar){
  return Ts - F_Ts(Ts, iterVar);
}

/* dFunc/dTs */
double MesoBurn::Deri(double Ts, 
                      IterationVariables *iterVar){
  double m = m_Ts(Ts, iterVar);
  double K1 = Ts-iterVar->C2;
  double K2 = sqrt(m*m+iterVar->C3);
  double K3 = (R*Ts*(K1-iterVar->C2)+Ec*K1)*m*iterVar->C5;
  double K4 = (K2+m)*(K2+m)*K1*K2*R*Ts*Ts;
  return 1.0 + K3/K4;
}

/* F_Ts(Ts_max) is the max of F_Ts function */
double MesoBurn::Ts_max(IterationVariables *iterVar){
  return 0.5*(2.0*R*iterVar->C2 - Ec + sqrt(4.0*R*R*iterVar->C2*iterVar->C2+Ec*Ec))/R;
} 

void MesoBurn::SetInterval(double f, 
                           double Ts, 
                           IterationVariables *iterVar){  
  /* IL <= 0,  IR >= 0 */
  if(f < 0)  
    iterVar->IL = Ts;
  else if(f > 0)
    iterVar->IR = Ts;
  else if(f ==0){
    iterVar->IL = Ts;
    iterVar->IR = Ts; 
  }
}

/* Bisection - Newton Method */
double MesoBurn::BisectionNewton(double Ts, 
                                 IterationVariables *iterVar){  
  double y = 0;
  double df_dTs = 0;
  double delta_old = 0;
  double delta_new = 0;
  
  int iter = 0;
  if(Ts>iterVar->Tmax || Ts<iterVar->Tmin)
    Ts = (iterVar->Tmin+iterVar->Tmax)/2;
  
  while(1){
    iter++;
    y = Func(Ts, iterVar);
    SetInterval(y, Ts, iterVar);
    
    if(fabs(y)<EPSILON)
      return Ts;
    
    delta_new = 1e100;
    while(1){
      if(iter>100){
        cout<<"Not converging after 100 iterations in MesoBurn.cc."<<endl;
        exit(1);
      }

      df_dTs = Deri(Ts, iterVar);
      if(df_dTs==0) 
        break;

      delta_old = delta_new;
      delta_new = -y/df_dTs; //Newton Step
      Ts += delta_new;
      y = Func(Ts, iterVar);

      if(fabs(y)<EPSILON)
        return Ts;
      
      if(Ts<iterVar->IL || Ts>iterVar->IR || fabs(delta_new)>fabs(delta_old*0.7))
        break;

      iter++; 
      SetInterval(y, Ts, iterVar);  
    }
    
    Ts = (iterVar->IL+iterVar->IR)/2.0; //Bisection Step
  }
}
