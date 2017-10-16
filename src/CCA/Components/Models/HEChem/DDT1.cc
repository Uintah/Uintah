/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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


#include <CCA/Components/ICE/BoundaryCond.h>
#include <CCA/Components/ICE/ICEMaterial.h>
#include <CCA/Components/Models/HEChem/Common.h>
#include <CCA/Components/Models/HEChem/DDT1.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Labels/ICELabel.h>
#include <Core/Labels/MPMICELabel.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DebugStream.h>
#include <iostream>


using namespace Uintah;
using namespace std;
//__________________________________
//  setenv SCI_DEBUG "MODELS_DOING_COUT:+"
//  MODELS_DOING_COUT:   dumps when tasks are scheduled and performed
static DebugStream cout_doing("MODELS_DOING_COUT", false);

const double DDT1::d_EPSILON   = 1e-6;   /* stop epsilon for Bisection-Newton method */

DDT1::DDT1(const ProcessorGroup* myworld,
           ProblemSpecP& params,
           const ProblemSpecP& prob_spec)
  : ModelInterface(myworld), d_prob_spec(prob_spec), d_params(params)
{
  d_mymatls  = 0;
  d_one_matl = 0;
  Ilb  = scinew ICELabel();
  MIlb = scinew MPMICELabel();
  Mlb  = scinew MPMLabel();

  d_adj_IO_Press = scinew adj_IO();
  d_adj_IO_Det   = scinew adj_IO();
  //__________________________________
  //  diagnostic labels JWL++
  reactedFractionLabel   = VarLabel::create("F",
                                      CCVariable<double>::getTypeDescription());
                     
  delFLabel       = VarLabel::create("delF",
                                      CCVariable<double>::getTypeDescription());

  detLocalToLabel = VarLabel::create("detLocalTo",
                                      CCVariable<double>::getTypeDescription());

  detonatingLabel = VarLabel::create("detonating",
                                      CCVariable<double>::getTypeDescription());
  //__________________________________
  //  diagnostic labels   
  d_saveConservedVars = scinew saveConservedVars();
  
  onSurfaceLabel   = VarLabel::create("onSurface",
                                       CCVariable<double>::getTypeDescription());

  surfaceTempLabel = VarLabel::create("surfaceTemp",
                                       CCVariable<double>::getTypeDescription());
  
  inductionTimeLabel     = VarLabel::create("inductionTime",
                                       CCVariable<double>::getTypeDescription());
                                       
  countTimeLabel     = VarLabel::create("countTime",
                                       CCVariable<double>::getTypeDescription());
     
  BurningCriteriaLabel     = VarLabel::create("BurningCriteria",
                                       CCVariable<int>::getTypeDescription());
       
  numPPCLabel      = VarLabel::create("SteadyBurn.numPPC",
                                       CCVariable<double>::getTypeDescription());

  burningLabel     = VarLabel::create("burning",
                                       CCVariable<int>::getTypeDescription());

  crackedEnoughLabel   = VarLabel::create("crackedEnough",
                                          CCVariable<double>::getTypeDescription());
        
  totalMassBurnedLabel  = VarLabel::create( "totalMassBurned",
                                             sum_vartype::getTypeDescription() );
    
  totalHeatReleasedLabel= VarLabel::create( "totalHeatReleased",
                                             sum_vartype::getTypeDescription() );
                                             
  adjOutIntervalsLabel= VarLabel::create( "adjOutIntervals",
                                           max_vartype::getTypeDescription() );
}

DDT1::~DDT1()
{
  delete Ilb;
  delete MIlb;
  delete Mlb;
  delete d_saveConservedVars;
  delete d_adj_IO_Press;
  delete d_adj_IO_Det;

  // JWL++
  VarLabel::destroy(reactedFractionLabel);
  VarLabel::destroy(delFLabel);
  VarLabel::destroy(detLocalToLabel);
  VarLabel::destroy(detonatingLabel);

  VarLabel::destroy(BurningCriteriaLabel);
  VarLabel::destroy(surfaceTempLabel);
  VarLabel::destroy(onSurfaceLabel);
  VarLabel::destroy(burningLabel);
  VarLabel::destroy(crackedEnoughLabel);
  VarLabel::destroy(totalMassBurnedLabel);
  VarLabel::destroy(totalHeatReleasedLabel);
  VarLabel::destroy(numPPCLabel);
  VarLabel::destroy(inductionTimeLabel);
  VarLabel::destroy(countTimeLabel);
  VarLabel::destroy( adjOutIntervalsLabel );
  
  if(d_mymatls && d_mymatls->removeReference())
    delete d_mymatls;
    
  if (d_one_matl && d_one_matl->removeReference())
    delete d_one_matl;
}

bool DDT1::isDoubleEqual(double a, double b){
  return ( fabs(a-b) < DBL_EPSILON);
}

void DDT1::problemSetup(GridP&, SimulationStateP& sharedState, ModelSetup*, const bool isRestart)
{
  d_sharedState = sharedState;
  
  ProblemSpecP ddt_ps = d_params->findBlock("DDT1");
  // Required for JWL++
  ddt_ps->require("ThresholdPressureJWL",   d_threshold_press_JWL);
  ddt_ps->require("fromMaterial",fromMaterial);
  ddt_ps->require("toMaterial",  toMaterial);
  ddt_ps->getWithDefault("burnMaterial",  burnMaterial, toMaterial);
  ddt_ps->require("G",    d_G);
  ddt_ps->require("b",    d_b);
  ddt_ps->require("E0",   d_E0);
  ddt_ps->getWithDefault("ThresholdVolFrac",d_threshold_volFrac, 0.01);

  // Required for Simple Burn
  d_matl0 = sharedState->parseAndLookupMaterial(ddt_ps, "fromMaterial");
  d_matl1 = sharedState->parseAndLookupMaterial(ddt_ps, "toMaterial");
  d_matl2 = sharedState->parseAndLookupMaterial(ddt_ps, "burnMaterial");
  ddt_ps->require("IdealGasConst",     d_R );
  ddt_ps->require("PreExpCondPh",      d_Ac);
  ddt_ps->require("ActEnergyCondPh",   d_Ec);
  ddt_ps->require("PreExpGasPh",       d_Bg);
  ddt_ps->require("CondPhaseHeat",     d_Qc);
  ddt_ps->require("GasPhaseHeat",      d_Qg);
  ddt_ps->require("HeatConductGasPh",  d_Kg);
  ddt_ps->require("HeatConductCondPh", d_Kc);
  ddt_ps->require("SpecificHeatBoth",  d_Cp);
  ddt_ps->require("MoleWeightGasPh",   d_MW);
  ddt_ps->require("BoundaryParticles", d_BP);
  ddt_ps->require("IgnitionTemp",      d_ignitionTemp);
  ddt_ps->require("ThresholdPressureSB",d_thresholdPress_SB);
  ddt_ps->getWithDefault("useCrackModel",    d_useCrackModel, false); 
  ddt_ps->getWithDefault("useInductionTime", d_useInductionTime, false);
  
// Required for ignition time delay for burning propagation
  if(d_useInductionTime){
    ddt_ps->require("IgnitionConst",     d_IC);
    ddt_ps->require("PressureShift",     d_PS);
    ddt_ps->require("PreexpoConst",      d_Fb);
    ddt_ps->require("ExponentialConst",  d_Fc); 
  }
  
  if(d_useCrackModel){
    ddt_ps->require("Gcrack",           d_Gcrack);
    ddt_ps->getWithDefault("CrackVolThreshold",     d_crackVolThreshold, 1e-14 );
    ddt_ps->require("nCrack",           d_nCrack);
      
    pCrackRadiusLabel = VarLabel::find("p.crackRad");
    if(!pCrackRadiusLabel){
      ostringstream msg;
      msg << "\n ERROR:Model:DDT1: The constitutive model for the MPM reactant must be visco_scram in order to burn in cracks. \n";
      msg << " No other constitutive models are currently supported "; 
      throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
    }
  }
  
  //__________________________________
  //  Adjust the I/O intervals
  ProblemSpecP adj_ps = ddt_ps->findBlockWithOutAttribute( "adjust_IO_intervals" );
  
  if(adj_ps){
    ProblemSpecP PS_ps = adj_ps->findBlockWithOutAttribute( "PressureSwitch" );
    if( PS_ps ){
      d_adj_IO_Press->onOff     = true;
      PS_ps->require("PressureThreshold",     d_adj_IO_Press->pressThreshold );
      PS_ps->require("newOutputInterval",     d_adj_IO_Press->output_interval );  
      PS_ps->require("newCheckPointInterval", d_adj_IO_Press->chkPt_interval );
    }
  
    ProblemSpecP DS_ps = adj_ps->findBlockWithOutAttribute( "DetonationDetected" );
    if( DS_ps ){
      d_adj_IO_Det->onOff     = true;
      DS_ps->require("remainingTimesteps",    d_adj_IO_Det->timestepsLeft );
      DS_ps->require("newOutputInterval",     d_adj_IO_Det->output_interval );  
      DS_ps->require("newCheckPointInterval", d_adj_IO_Det->chkPt_interval );
    }
  }
  
  /* initialize constants */
  d_CC1 = d_Ac * d_R * d_Kc/d_Ec/d_Cp;        
  d_CC2 = d_Qc/d_Cp/2;              
  d_CC3 = 4*d_Kg*d_Bg*d_MW*d_MW/d_Cp/d_R/d_R;  
  d_CC4 = d_Qc/d_Cp;                
  d_CC5 = d_Qg/d_Cp;           
    
  //__________________________________
  //  define the materialSet
  d_mymatls = scinew MaterialSet();

  vector<int> m;
  m.push_back(0);                                 // needed for the pressure and NC_CCWeight
  m.push_back(d_matl0->getDWIndex());
  m.push_back(d_matl1->getDWIndex());
  m.push_back(d_matl2->getDWIndex());

  d_mymatls->addAll_unique(m);                    // elimiate duplicate entries
  d_mymatls->addReference();

  d_one_matl = scinew MaterialSubset();
  d_one_matl->add(0);
  d_one_matl->addReference();

  //__________________________________
  //  Are we saving the total burned mass and total burned energy
  ProblemSpecP DA_ps = d_prob_spec->findBlock("DataArchiver");
  for (ProblemSpecP child = DA_ps->findBlock("save"); child != nullptr; child = child->findNextBlock("save")) {
    map<string,string> var_attr;
    child->getAttributes(var_attr);
    if (var_attr["label"] == "totalMassBurned"){
      d_saveConservedVars->mass  = true;
    }
    if (var_attr["label"] == "totalHeatReleased"){
      d_saveConservedVars->energy = true;
    }
  }
  
  problemSetup_BulletProofing( d_prob_spec );
}

//______________________________________________________________________
//
void DDT1::problemSetup_BulletProofing(ProblemSpecP& ps)
{

  //__________________________________
  // The user can't specify a timestepInterval
  // and a interval for dynamic output
  bool usingOutputTimestepInterval = false;
  bool usingChkPtTimestepInterval  = false;
  int notUsedI;
  ProblemSpecP p = ps->findBlock( "DataArchiver" );
 
  if( p->get( "outputTimestepInterval", notUsedI ) ){
    usingOutputTimestepInterval = true;
  }

  ProblemSpecP checkpoint = p->findBlock( "checkpoint" );
  if( checkpoint != nullptr ) {
    map<string, string> attributes;
    
    attributes.clear();
    checkpoint->getAttributes(attributes);

    string attrib = attributes["timestepInterval"];
    if ( attrib != "" ){
      usingChkPtTimestepInterval = true;
    }
  }

  ProblemSpecP adj_ps = d_params->findBlockWithOutAttribute( "adjust_IO_intervals" );
  
  if(adj_ps){
    ProblemSpecP PS_ps = adj_ps->findBlockWithOutAttribute( "PressureSwitch" );
    ProblemSpecP DS_ps = adj_ps->findBlockWithOutAttribute( "DetonationDetected" );

    if( PS_ps || DS_ps ){

      if( usingOutputTimestepInterval || usingChkPtTimestepInterval ){
        ostringstream msg;
        msg << "\n ERROR:Model:DDT1: <adjust_IO_intervals>  You cannot specify: \n"
            << "      DataArchiver:outputTimestepInterval && adjust_IO_intervals:newOutputInterval \n"
            << "      Checkpoint:TimestepInterval && adjust_IO_intervals:newCheckPointInterval \n"
            << " you must be consistent";     
        
        throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
      }
    }
  }



  //__________________________________
  //
  ProblemSpecP root = ps->getRootNode();
  ProblemSpecP amr_ps = root->findBlock("AMR"); 
  if(amr_ps){     
    ProblemSpecP reg_ps = amr_ps->findBlock("Regridder");
    if (reg_ps) {

      string regridder;
      reg_ps->getAttribute( "type", regridder );

      if (regridder != "Tiled" && regridder != "SingleLevel" ) {
        ostringstream msg;
        msg << "\n ERROR:Model:DDT1: The (" << regridder << ") regridder will not work with this burn model. \n";
        msg << "The only regridder that works with this burn model is the \"Tiled\" regridder\n"; 
        throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
      }
    }
  }
}
//______________________________________________________________________
//
void DDT1::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute("type","DDT1");
  ProblemSpecP ddt_ps = model_ps->appendChild("DDT1");

  ddt_ps->appendElement("ThresholdPressureJWL",d_threshold_press_JWL);
  ddt_ps->appendElement("fromMaterial",fromMaterial);
  ddt_ps->appendElement("toMaterial",  toMaterial);
  ddt_ps->appendElement("burnMaterial",burnMaterial);
  ddt_ps->appendElement("G",    d_G);
  ddt_ps->appendElement("b",    d_b);
  ddt_ps->appendElement("E0",   d_E0);

  ddt_ps->appendElement("IdealGasConst",     d_R );
  ddt_ps->appendElement("PreExpCondPh",      d_Ac);
  ddt_ps->appendElement("ActEnergyCondPh",   d_Ec);
  ddt_ps->appendElement("PreExpGasPh",       d_Bg);
  ddt_ps->appendElement("CondPhaseHeat",     d_Qc);
  ddt_ps->appendElement("GasPhaseHeat",      d_Qg);
  ddt_ps->appendElement("HeatConductGasPh",  d_Kg);
  ddt_ps->appendElement("HeatConductCondPh", d_Kc);
  ddt_ps->appendElement("SpecificHeatBoth",  d_Cp);
  ddt_ps->appendElement("MoleWeightGasPh",   d_MW);
  ddt_ps->appendElement("BoundaryParticles", d_BP);
  ddt_ps->appendElement("ThresholdPressureSB", d_thresholdPress_SB);
  ddt_ps->appendElement("IgnitionTemp",      d_ignitionTemp);
 
  ddt_ps->appendElement("IgnitionConst",     d_IC );
  ddt_ps->appendElement("PressureShift",     d_PS );
  ddt_ps->appendElement("ExponentialConst",  d_Fc );
  ddt_ps->appendElement("PreexpoConst",      d_Fb );
  ddt_ps->appendElement("useInductionTime",  d_useInductionTime);
  
  //__________________________________
  // adjust output intervals
  if( d_adj_IO_Press->onOff || d_adj_IO_Det->onOff ){
    ProblemSpecP adj_ps = ddt_ps->appendChild( "adjust_IO_intervals" );

    if( d_adj_IO_Press->onOff ){
      ProblemSpecP PS_ps = adj_ps->appendChild( "PressureSwitch" );
      PS_ps->appendElement( "PressureThreshold",     d_adj_IO_Press->pressThreshold );
      PS_ps->appendElement( "PressureThreshold",     d_adj_IO_Press->pressThreshold );
      PS_ps->appendElement( "newOutputInterval",     d_adj_IO_Press->output_interval );  
      PS_ps->appendElement( "newCheckPointInterval", d_adj_IO_Press->chkPt_interval );
    }
  
    
    if( d_adj_IO_Det->onOff ){
      ProblemSpecP DS_ps = adj_ps->appendChild( "DetonationDetected" );
      DS_ps->appendElement( "remainingTimesteps",    d_adj_IO_Det->timestepsLeft );
      DS_ps->appendElement( "newOutputInterval",     d_adj_IO_Det->output_interval );  
      DS_ps->appendElement( "newCheckPointInterval", d_adj_IO_Det->chkPt_interval );
    }
  }
 
  if(d_useCrackModel){
    ddt_ps->appendElement("useCrackModel",     d_useCrackModel);
    ddt_ps->appendElement("Gcrack",            d_Gcrack);
    ddt_ps->appendElement("nCrack",            d_nCrack);
    ddt_ps->appendElement("CrackVolThreshold", d_crackVolThreshold);
  }
}

//______________________________________________________________________
//     
void DDT1::scheduleInitialize(SchedulerP& sched,
                              const LevelP& level,
                              const ModelInfo*)
{
  printSchedule(level,cout_doing,"DDT1::scheduleInitialize");
  Task* t = scinew Task("DDT1::initialize", this, &DDT1::initialize);
  const MaterialSubset* react_matl = d_matl0->thisMaterial();
  t->computes(reactedFractionLabel, react_matl);
  t->computes(burningLabel,         react_matl);
  t->computes(detLocalToLabel,      react_matl);
  t->computes(surfaceTempLabel,     react_matl);
  t->computes(BurningCriteriaLabel, react_matl);
  t->computes(inductionTimeLabel,   react_matl);
  t->computes(countTimeLabel,       react_matl);

  if( d_adj_IO_Press->onOff || d_adj_IO_Det->onOff ){
    t->computes( adjOutIntervalsLabel );
  }
  
  if(d_useCrackModel)
    t->computes(crackedEnoughLabel,   react_matl);
  sched->addTask(t, level->eachPatch(), d_mymatls);
}

//______________________________________________________________________
//
void DDT1::initialize(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* /*matls*/,
                      DataWarehouse*,
                      DataWarehouse* new_dw){
  int m0 = d_matl0->getDWIndex();
  
  double initTimestep = d_sharedState->getSimulationTime()->m_max_initial_delt;
  
 
  for(int p=0;p<patches->size();p++) {
    const Patch* patch = patches->get(p);
    printTask(patches,patch,cout_doing,"Doing DDT1::initialize");
    
    // This section is needed for outputting F and burn on each timestep
    CCVariable<double> F, Ts, det, crack, inductionTime, countTime, inductionTimeOld, countTimeOld;
    CCVariable<int> burningCellOld, BurningCriteriaOld;
    new_dw->allocateAndPut(F,                  reactedFractionLabel, m0, patch);
    new_dw->allocateAndPut(burningCellOld,     burningLabel,         m0, patch);
    new_dw->allocateAndPut(Ts,                 surfaceTempLabel,     m0, patch);
    new_dw->allocateAndPut(det,                detLocalToLabel,      m0, patch);
    new_dw->allocateAndPut(BurningCriteriaOld, BurningCriteriaLabel, m0, patch);
    new_dw->allocateAndPut(inductionTimeOld,   inductionTimeLabel,   m0, patch);
    new_dw->allocateAndPut(countTimeOld,       countTimeLabel,       m0, patch);
     
    if(d_useCrackModel)
    {
      new_dw->allocateAndPut(crack,crackedEnoughLabel,   m0, patch);
      crack.initialize(0.0);
    } 
    
    if( d_adj_IO_Press->onOff || d_adj_IO_Det->onOff ){
      new_dw->put( max_vartype( ZERO ), adjOutIntervalsLabel );
    }
    

    F.initialize(0.0);
    burningCellOld.initialize(0);
    det.initialize(0.0);
    Ts.initialize(0.0);
    BurningCriteriaOld.initialize(0);
    inductionTimeOld.initialize(initTimestep + 1e-20);
    countTimeOld.initialize(0.0);
  }
}

//______________________________________________________________________
//      
void DDT1::scheduleComputeStableTimestep(SchedulerP&,
                                         const LevelP&,
                                         const ModelInfo*)
{
  // None necessary...
}


//______________________________________________________________________
//      
void DDT1::scheduleComputeModelSources(SchedulerP& sched,
                                       const LevelP& level,
                                       const ModelInfo* mi)
{
  if (level->hasFinerLevel()){
    return;    // only schedule on the finest level
  }

  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn  = Ghost::None;
  const MaterialSubset* react_matl = d_matl0->thisMaterial();
  const MaterialSubset* prod_matl  = d_matl1->thisMaterial();
  const MaterialSubset* prod_matl2 = d_matl2->thisMaterial();

  const MaterialSubset* all_matls = d_sharedState->allMaterials()->getUnion();
  const MaterialSubset* ice_matls = d_sharedState->allICEMaterials()->getUnion();
  const MaterialSubset* mpm_matls = d_sharedState->allMPMMaterials()->getUnion();
  Task::MaterialDomainSpec oms = Task::OutOfDomain;

  proc0cout << "\nDDT1:scheduleComputeModelSources oneMatl " << *d_one_matl<< " react_matl " << *react_matl 
                                            << " det_matl "  << *prod_matl 
                                            << " burn_matl " << *prod_matl2 
                                            << " all_matls " << *all_matls 
                                            << " ice_matls " << *ice_matls << " mpm_matls " << *mpm_matls << "\n"<<endl;
  //__________________________________
  //
  // Task for computing the particles in a cell
  Task* t0 = scinew Task("DDT1::computeNumPPC", this, 
                         &DDT1::computeNumPPC, mi);
    
  printSchedule(level,cout_doing,"DDT1::scheduleComputeNumPPC");  
    
  t0->requires(Task::OldDW, Mlb->pXLabel,               react_matl, gn);
  t0->computes(numPPCLabel, react_matl);
  
  if(d_useCrackModel){  // Because there is a particle loop already in computeNumPPC, 
                        //  we will put crack threshold determination there as well
    t0->requires(Task::NewDW, Ilb->press_equil_CCLabel, d_one_matl, gac, 1);
    t0->requires(Task::OldDW, pCrackRadiusLabel,        react_matl, gn);
    t0->computes(crackedEnoughLabel,    react_matl);
  }
  sched->addTask(t0, level->eachPatch(), d_mymatls);
  
  
  //__________________________________
  //
  Task* t1 = scinew Task("DDT1::computeBurnLogic", this, 
                         &DDT1::computeBurnLogic, mi);    
    
  printSchedule(level,cout_doing,"DDT1::computeBurnLogic");  
  if(d_useCrackModel){  
    t1->requires(Task::NewDW, crackedEnoughLabel,        react_matl, gac,1);
  }
  //__________________________________
  // Requires
  //__________________________________
  t1->requires(Task::OldDW, mi->delT_Label,            level.get_rep());
  t1->requires(Task::OldDW, Ilb->temp_CCLabel,         ice_matls, oms, gac,1);
  t1->requires(Task::NewDW, MIlb->temp_CCLabel,        mpm_matls, oms, gac,1);
  t1->requires(Task::NewDW, Ilb->vol_frac_CCLabel,     all_matls, oms, gac,1);
  t1->requires(Task::OldDW, Mlb->pXLabel,              mpm_matls,  gn);
 
  
  //__________________________________
  // Products
  t1->requires(Task::NewDW,  Ilb->rho_CCLabel,         prod_matl,   gn); 
  t1->requires(Task::NewDW,  Ilb->rho_CCLabel,         prod_matl2,  gn); 
  t1->requires(Task::NewDW,  Ilb->press_equil_CCLabel, d_one_matl,  gac, 1);
  t1->requires(Task::OldDW,  MIlb->NC_CCweightLabel,   d_one_matl,  gac, 1);

  //__________________________________
  // Reactants
  t1->requires(Task::NewDW, Ilb->sp_vol_CCLabel,       react_matl, gn);
  t1->requires(Task::NewDW, MIlb->vel_CCLabel,         react_matl, gn);
  t1->requires(Task::NewDW, Ilb->rho_CCLabel,          react_matl, gn);
  t1->requires(Task::NewDW, Mlb->gMassLabel,           react_matl, gac,1);
  t1->requires(Task::NewDW, numPPCLabel,               react_matl, gac,1);
  t1->requires(Task::OldDW, burningLabel,              react_matl, gac,1);
  t1->requires(Task::OldDW, inductionTimeLabel,        react_matl, gn);
  t1->requires(Task::OldDW, countTimeLabel,            react_matl, gn);
  
  //__________________________________
  // Computes
  //__________________________________
  t1->computes(detLocalToLabel,         react_matl);
  t1->computes(detonatingLabel,         react_matl);
  t1->computes(BurningCriteriaLabel,    react_matl);
  t1->computes(inductionTimeLabel,      react_matl);
  t1->computes(countTimeLabel,          react_matl);
   
  // if detonation occurs change the output interval  
  if( d_adj_IO_Press->onOff || d_adj_IO_Det->onOff ){
    t1->requires( Task::OldDW, adjOutIntervalsLabel );
    t1->computes( adjOutIntervalsLabel );
    
    t1->computes( d_sharedState->get_outputInterval_label() );
    t1->computes( d_sharedState->get_checkpointInterval_label() );
    d_sharedState->updateOutputInterval( true );
    d_sharedState->updateCheckpointInterval( true ); 
  } 
  
  sched->addTask(t1, level->eachPatch(), d_mymatls);    
    
    
  //__________________________________
  //
  Task* t2 = scinew Task("DDT1::computeModelSources", this, 
                         &DDT1::computeModelSources, mi);
                        
  if(d_useCrackModel){  // Because there is a particle loop already in computeNumPPC, 
                        //  we will put crack threshold determination there as well
    t2->requires(Task::OldDW, Mlb->pXLabel,            mpm_matls,  gn);
    //t2->requires(Task::OldDW, pCrackRadiusLabel,       react_matl, gn);
   // t2->requires(Task::NewDW, crackedEnoughLabel,      react_matl, gac,1);
  }  
  
   
  //__________________________________
  // Requires
  //__________________________________
  t2->requires(Task::OldDW, mi->delT_Label,            level.get_rep());
  t2->requires(Task::OldDW, Ilb->temp_CCLabel,         ice_matls, oms, gac,1);
  t2->requires(Task::NewDW, MIlb->temp_CCLabel,        mpm_matls, oms, gac,1);
  t2->requires(Task::NewDW, Ilb->vol_frac_CCLabel,     all_matls, oms, gac,1);
  
  
  //__________________________________
  // Products
  t2->requires(Task::NewDW,  Ilb->rho_CCLabel,         prod_matl,   gn); 
  t2->requires(Task::NewDW,  Ilb->rho_CCLabel,         prod_matl2,  gn); 
  t2->requires(Task::NewDW,  Ilb->press_equil_CCLabel, d_one_matl,  gac, 1);
  t2->requires(Task::OldDW,  MIlb->NC_CCweightLabel,   d_one_matl,  gac, 1);

  //__________________________________
  // Reactants
  t2->requires(Task::NewDW, Ilb->sp_vol_CCLabel,       react_matl, gn);
  t2->requires(Task::NewDW, MIlb->vel_CCLabel,         react_matl, gn);
  t2->requires(Task::NewDW, Ilb->rho_CCLabel,          react_matl, gn);
  t2->requires(Task::NewDW, Mlb->gMassLabel,           react_matl, gac,1);
  t2->requires(Task::NewDW, numPPCLabel,               react_matl, gac,1);
  t2->requires(Task::NewDW, detonatingLabel,           react_matl, gn); 
  t2->requires(Task::NewDW, BurningCriteriaLabel,      react_matl, gn);
 
  //__________________________________
  // Computes
  //__________________________________
  t2->computes(reactedFractionLabel,    react_matl);
  t2->computes(delFLabel,               react_matl);
  t2->computes(burningLabel,            react_matl);
  t2->computes(onSurfaceLabel,          react_matl);
  t2->computes(surfaceTempLabel,        react_matl);  

  //__________________________________
  // Conserved Variables
  //__________________________________
  if(d_saveConservedVars->mass ){
      t2->computes(DDT1::totalMassBurnedLabel);
  }
  if(d_saveConservedVars->energy){
      t2->computes(DDT1::totalHeatReleasedLabel);
  }

  //__________________________________
  // Modifies  
  //__________________________________
  t2->modifies(mi->modelMass_srcLabel);
  t2->modifies(mi->modelMom_srcLabel);
  t2->modifies(mi->modelEng_srcLabel);
  t2->modifies(mi->modelVol_srcLabel); 

  sched->addTask(t2, level->eachPatch(), d_mymatls);
}
//______________________________________________________________________
//
void DDT1::computeNumPPC(const ProcessorGroup*, 
                         const PatchSubset* patches,
                         const MaterialSubset* /*matls*/,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw,
                         const ModelInfo* mi)
{
    int m0 = d_matl0->getDWIndex(); /* reactant material */
    Ghost::GhostType  gac = Ghost::AroundCells;
    
    /* Patch Iteration */
    for(int p=0;p<patches->size();p++){
        const Patch* patch = patches->get(p);  
        printTask(patches,patch,cout_doing,"Doing DDT1::computeNumPPC");
        
        /* Indicating how many particles a cell contains */
        ParticleSubset* pset = old_dw->getParticleSubset(m0, patch);
        
        constParticleVariable<Point>  px;
        old_dw->get(px, Mlb->pXLabel, pset);
        
        /* Indicating cells containing how many particles */
        CCVariable<double>  numPPC, crack;
        new_dw->allocateAndPut(numPPC,       numPPCLabel,        m0, patch);
        numPPC.initialize(0.0);
        
        // get cracked burning stuff
        constCCVariable<double> press_CC; 
        constParticleVariable<double> crackRad;  // store the level of cracking 
        if(d_useCrackModel){
          old_dw->get(crackRad,    pCrackRadiusLabel, pset); 
          new_dw->get(press_CC,    Ilb->press_equil_CCLabel, 0,  patch, gac, 1);
          new_dw->allocateAndPut(crack,       crackedEnoughLabel, m0, patch);
          crack.initialize(0.0);
        }

        /* count how many reactant particles in each cell */
        for(ParticleSubset::iterator iter=pset->begin(), iter_end=pset->end();
          iter != iter_end; iter++){
          particleIndex idx = *iter;
          IntVector c;
          patch->findCell(px[idx],c);
          numPPC[c] += 1.0;

          // if cracked burning is enabled register if crack threshold is exceeded
          if(d_useCrackModel){
            double crackWidthThreshold = sqrt(8.0e8/pow(press_CC[c],2.84));


            if(crackRad[*iter] > crackWidthThreshold) {
              crack[c] = 1;
            }
          } 
        }    
        setBC(numPPC, "zeroNeumann", patch, d_sharedState, m0, new_dw);
    }
}

//______________________________________________________________________
//
void DDT1::computeBurnLogic(const ProcessorGroup*, 
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw,
                            const ModelInfo* mi)
{
  delt_vartype delT;
  const Level* level = getLevel(patches);
  old_dw->get(delT, mi->delT_Label, level);

 
  int m0 = d_matl0->getDWIndex();
  int m1 = d_matl1->getDWIndex();
  int numAllMatls = d_sharedState->getNumMatls();

  for(int p=0;p<patches->size();p++){
    const Patch* patch   = patches->get(p);  
    ParticleSubset* pset = old_dw->getParticleSubset(m0, patch); 
    printTask(patches,patch,cout_doing,"Doing DDT1::computeBurnLogic");

    // Burning related
    CCVariable<double> inductionTime, countTime;
    constCCVariable<double>    crackedEnough;
    // Detonation Related
    CCVariable<double> Fr, delF;       
    // Diagnostics/Thresholds                     
    CCVariable<double> detonating, detLocalTo;
    CCVariable<int> BurningCriteria;
   
    
    // Old Reactant Quantities
    constCCVariable<double> cv_reactant, rctVolFrac;
    constCCVariable<double> rctTemp, rctRho, rctSpvol, rctFr, numPPC, inductionTimeOld,  countTimeOld;
    constCCVariable<int> burningCellOld;
    constCCVariable<Vector> rctvel_CC;
    constNCVariable<double> NC_CCweight, rctMass_NC;
    constParticleVariable<Point> px;
    // Old Product Quantities
    constCCVariable<double> prodRho, prodRho2;
    // Domain Wide Variables
    constCCVariable<double> press_CC; 
    std::vector<constCCVariable<double> > vol_frac_CC(numAllMatls);
    std::vector<constCCVariable<double> > temp_CC(numAllMatls);


    Vector dx = patch->dCell();
    double delta_x = (dx.x()+dx.y()+dx.z())/3; // average cell length
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    /* Gets and Computes */
    //__________________________________
    // Reactant data
    new_dw->get(rctMass_NC,    Mlb->gMassLabel,         m0, patch,gac,1);
    new_dw->get(rctVolFrac,    Ilb->vol_frac_CCLabel,   m0, patch,gac,1);
    new_dw->get(numPPC,        numPPCLabel,             m0, patch,gac,1);
    old_dw->get(inductionTimeOld, inductionTimeLabel,   m0, patch,gn, 0);
    old_dw->get(countTimeOld,  countTimeLabel,          m0, patch,gn, 0);
    old_dw->get(burningCellOld,burningLabel,            m0, patch,gac,1);
    
    
    if(d_useCrackModel){
      old_dw->get(px,          Mlb->pXLabel,      pset);
      new_dw->get(crackedEnough,       crackedEnoughLabel, m0, patch, gac, 1);
    }
    
    //__________________________________
    //   Misc.
    new_dw->get(press_CC,         Ilb->press_equil_CCLabel,0,  patch,gac,1);
    old_dw->get(NC_CCweight,      MIlb->NC_CCweightLabel,  0,  patch,gac,1);   
   
    // Temperature and Vol_frac 
    for(int m = 0; m < numAllMatls; m++) {
      Material*    matl     = d_sharedState->getMaterial(m);
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      int indx = matl->getDWIndex();
      if(ice_matl){
        old_dw->get(temp_CC[m],   MIlb->temp_CCLabel,    indx, patch,gac,1);
      }else {
        new_dw->get(temp_CC[m],   MIlb->temp_CCLabel,    indx, patch,gac,1);
      }
      new_dw->get(vol_frac_CC[m], Ilb->vol_frac_CCLabel, indx, patch,gac,1);
    }
    //__________________________________
    //  What is computed
    new_dw->allocateAndPut(detonating,      detonatingLabel,         m0,patch);
    new_dw->allocateAndPut(detLocalTo,      detLocalToLabel,         m0,patch);
    new_dw->allocateAndPut(BurningCriteria, BurningCriteriaLabel,    m0,patch);
    new_dw->allocateAndPut(inductionTime,   inductionTimeLabel,      m0,patch);
    new_dw->allocateAndPut(countTime,       countTimeLabel,          m0,patch);
   
    
   //might not need to initialize these again if all of them are being over rode. 
   //look at if each variable is changing
    detonating.initialize(0.);
    detLocalTo.initialize(0.);
    BurningCriteria.initialize(0);
    inductionTime.initialize(0.);

    countTime.initialize(0.);
  
    IntVector nodeIdx[8];
    
    bool press_switch_adj_IO  = false;  // switch based on pressure to adjust the I/O intervals
    bool det_switch_adj_IO    = false;  // switch based on detonation to adjust the I/O intervals    
    
    //__________________________________
    //  Loop over cells
    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      if (rctVolFrac[c] > 1e-10){ //only look at cells with reactant
        // Detonation model For explosions
        
        
        // check to see if we should adjust the output intervals based on pressure 
        if (press_CC[c] > d_adj_IO_Press->pressThreshold && numPPC[c] > 0){
          press_switch_adj_IO = true;
        }
        
        if (press_CC[c] > d_threshold_press_JWL && numPPC[c] > 0){

          detonating[c] = 1;   // Flag for detonating 
          det_switch_adj_IO = true;

        } else if(press_CC[c] < d_threshold_press_JWL && press_CC[c] > d_thresholdPress_SB) {
          // Steady Burn Model for deflagration
          patch->findNodesFromCell(c,nodeIdx);

          double MaxMass = d_SMALL_NUM;
          double MinMass = 1.0/d_SMALL_NUM; 
          for (int nN=0; nN<8; nN++){
            IntVector node = nodeIdx[nN];
            double nodeMass = NC_CCweight[node]*rctMass_NC[node];
            MaxMass = std::max(MaxMass, nodeMass);
            MinMass = std::min(MinMass, nodeMass); 
          }

          double minOverMax            = MinMass/MaxMass;

          /* test whether the current cell satisfies burning criteria */
          int    burning               = NOTDEFINED;    // flag that indicates whether surface burning is occuring
          double maxProductVolFrac     = -1.0;          // used for surface area calculation
          double productPress          = 0.0;           // the product pressure above the surface, used in WSB
          bool   temperatureExceeded   = false;         // tells whether the temperature in the cell exceeded the threshold regardless of surface burning

          /* near interface and containing particles */
          for(int i = -1; i<=1; i++){
            for(int j = -1; j<=1; j++){
              for(int k = -1; k<=1; k++){
                IntVector adjCell = c + IntVector(i,j,k);

                /* Search for pressure from max_vol_frac product adjCell */
                double temp_vf = vol_frac_CC[m1][adjCell];

                if( temp_vf > maxProductVolFrac ){
                   maxProductVolFrac = temp_vf;
                   productPress = press_CC[adjCell];
                 }

                if( burning == NOTDEFINED && numPPC[adjCell] <= d_BP ){
                  for (int m = 0; m < numAllMatls; m++){

                    if( vol_frac_CC[m][adjCell] > 0.2 && temp_CC[m][adjCell] > d_ignitionTemp ){
                     // Is the surface exposed for burning?
                      if( minOverMax < 0.7 && numPPC[c] > 0 ){        
                        burning = ONSURFACE;
                        break;
                      }
                    }
                  } 
                } //endif

                 // Used to prevent cracked burning right next to a detonating adjCell
                 //  as it has caused a speedup phenomenon at the front of the
                 //  detonation front
                 // Detect detonation within one adjCell
                if( press_CC[adjCell] > d_threshold_press_JWL) {
                  detLocalTo[c] = 1;
                } // end if detonation next to adjCell


                if(d_useCrackModel){

                 // make sure the temperature exceeded value is set
                  for (int m = 0; m < numAllMatls; m++){
                    if(temp_CC[m][adjCell] > d_ignitionTemp){
                      temperatureExceeded = true;
                      break;
                    } 
                  } 
                } 

              }  //end 3rd for (z)
            }  //end 2nd for (y)
          }  //end 1st for (x)

          //______________________________________________________________________

          if(d_useInductionTime){
            int ignitedFrom = NOTDEFINED;
            int ignitedFromHotSolidCell = NOTDEFINED;
            bool   inductionTimeExceeded    = false;    // tells whether the inductionTime has been exceeded
            bool   calculateInductionTime   = false;    // tells whether the cell should have an induction time when cell is on surface
            double largest_cos_theta_HotGas = -1;
            double largest_cos_thetaHotCell =-1;
            double theta_HotGas             = 0;
            double theta_HotCell            = 0;
            double A                        = 12345e100;
            double A_HotSolidCell           = 12345e100;
            double A_HotGasCell             = 12350e100;
            double phase_shift              = M_PI_2; //90 degrees or pi/2

            if(burningCellOld[c] == CONVECTIVE || burningCellOld[c] == CONDUCTIVE){  
              inductionTime[c]  = inductionTimeOld[c];
              countTime[c]      = countTimeOld[c] + delT;
              inductionTimeExceeded  = 1;
              ignitedFrom = burningCellOld[c];
             // continue;//make sure this is not caluculating induction time again
            }
            //__________________________________
            //On Surface Cells being ignited by hot gas
            else if( minOverMax<0.7 && numPPC[c]>0 ){ //determine if cell is on surface
            //__________________________________
            // Calculates induction time to slow down convective burning propagation
            //and on surface propagation 
              for(int i = -1; i<=1; i++){
                for(int j = -1; j<=1; j++){
                  for(int k = -1; k<=1; k++){
                    IntVector offset = IntVector(i,j,k);
                    IntVector adjcell = c + offset;
                    if(burning == ONSURFACE){
                      calculateInductionTime = true;

                      for (int m = 0; m < numAllMatls; m++){
                        if(vol_frac_CC[m][c] > 0.2 && temp_CC[m][c] > d_ignitionTemp){
                          theta_HotGas = 0.0;
                        }else if(vol_frac_CC[m][adjcell] > 0.2 && temp_CC[m][adjcell] > d_ignitionTemp){

                          Point hotcellCord = patch->getCellPosition(adjcell);
                          Point cellCord    = patch->getCellPosition(c);
                          double cos_theta  = 0;  
                          double computedTheta = computeInductionAngle(nodeIdx,rctMass_NC, NC_CCweight,dx, cos_theta, computedTheta, hotcellCord, cellCord);

                          if(cos_theta > largest_cos_theta_HotGas){
                            largest_cos_theta_HotGas = cos_theta;
                            theta_HotGas =computedTheta;
                          }
                        }

                        A_HotGasCell = ((1 + d_IC)/2) + ((1 - d_IC)/2)*sin( (2 * theta_HotGas) - phase_shift);//Constant used to speed up propagation for flame moving into a surface
                      }
                    }

                    //__________________________________
                    //On Surface Cells being ignited by surrounding cell that are already burning
                    if( (d_useCrackModel && crackedEnough[c]) && 
                        (burningCellOld[adjcell] == CONDUCTIVE ||  burningCellOld[adjcell] == CONVECTIVE) ){

                      calculateInductionTime = true;
                      //__________________________________
                      //  Determining vectors for direction of flame
                      for (int m = 0; m < numAllMatls; m++){ 
                        if(temp_CC[m][adjcell] > d_ignitionTemp){
                          Point hotcellCord = patch->getCellPosition(adjcell);
                          Point cellCord    = patch->getCellPosition(c);  
                          double cos_theta  = 0;
                          double computedTheta = computeInductionAngle(nodeIdx,rctMass_NC, NC_CCweight,dx, cos_theta, computedTheta, hotcellCord, cellCord);

                          if(cos_theta > largest_cos_thetaHotCell){
                            largest_cos_thetaHotCell = cos_theta;
                            theta_HotCell = computedTheta;

                            ignitedFromHotSolidCell = burningCellOld[adjcell];
                          }
                        }
                      }

                      A_HotSolidCell = (1 + d_IC)/2 + ((1 - d_IC)/2)*sin( 2 * theta_HotCell - phase_shift);//Constant used to speed up propagation for flame moving into a surface 

                    }
                    //__________________________________
                    //Determine A for on surface cells
                    if(calculateInductionTime){
                      A = min(A_HotSolidCell,A_HotGasCell);

                      if(A  == A_HotGasCell){
                        ignitedFrom = CONDUCTIVE;
                      }else {
                        ignitedFrom = ignitedFromHotSolidCell;
                      }
                     
                    }  
                  }//end 3rd for (z)
                }//end 2nd for (y)
              }//end 1st for (x)
            }else if(d_useCrackModel && minOverMax >= 0.7){
             for(int i = -1; i<=1; i++){
                for(int j = -1; j<=1; j++){
                  for(int k = -1; k<=1; k++){
                    IntVector offset = IntVector(i,j,k);
                    IntVector adjcell = c + offset;

                  //__________________________________
                  //Determine that the cell is not on the surface and that a surrounding cell is burning
                    if(crackedEnough[c] && ( burningCellOld[adjcell] == CONDUCTIVE || burningCellOld[adjcell] == CONVECTIVE) ){  
                      calculateInductionTime = true;
                      A = d_IC;
                      ignitedFrom = CONVECTIVE;

                    }//end crack burning if
                  }//end 3rd for (z)
                }//end 2nd for (y)
              }//end 1st for (x)
            }//end use crack model

            //__________________________________
            //Calculate Induction Time          
            if(calculateInductionTime){   
              double inductionTime_new;
              double P_new   = productPress/d_PS;
              double S_f_new = d_Fb*pow(P_new,d_Fc); //surface flame velocity dependent on current pressure
                                                 //From Son "Flame Spread Across Surfaces of PBX"
              inductionTime_new =  (delta_x*A)/S_f_new ;
              inductionTime[c] = inductionTime_new ;

              double initTimestep = d_sharedState->getSimulationTime()->m_max_initial_delt;

              if(inductionTimeOld[c] != ( initTimestep + 1e-20)){ //initializes induction time to the calculated indcutiontime on the first timestep. 
                inductionTime[c] = (0.2*inductionTime_new) + (0.8*inductionTimeOld[c]) ;
                

                
                
              } 
              if ((countTimeOld[c] + delT) > countTime[c]){ //to ensure count time is only added once even if more than one surrounding cell is burning
                countTime[c] = countTimeOld[c] + delT;
              } 
              if (countTime[c] >= inductionTime[c]){
                inductionTimeExceeded     = true;  
              }
            }          
            //__________________________________
            // Determine different burning criteria
            if(!detLocalTo[c] &&
              (burning == ONSURFACE && productPress >= d_thresholdPress_SB && inductionTimeExceeded &&ignitedFrom == CONDUCTIVE)){

              BurningCriteria[c] = CONDUCTIVE;
            } else if (!detLocalTo[c] &&    // escape early if there is a detonation next to the current cell 
              ( ( d_useCrackModel && crackedEnough[c] ) && temperatureExceeded && inductionTimeExceeded && ignitedFrom == CONVECTIVE )){

              BurningCriteria[c] = CONVECTIVE;
             }else if(countTime[c] < inductionTime[c] && countTime[c] > 0){
              BurningCriteria[c] = WARMINGUP;
            }
          }else{  // if(inductionTime)
          //__________________________________
          // Not using induction time
           if(!detLocalTo[c] && (burning == ONSURFACE && productPress >= d_thresholdPress_SB) ){

             BurningCriteria[c] = CONDUCTIVE;
            } else if (!detLocalTo[c] && (d_useCrackModel && crackedEnough[c] && temperatureExceeded)){   // escape early if there is a detonation next to the current cell 

             BurningCriteria[c] = CONVECTIVE;
            }
          }//end not using induction Model
        }//else if(press_CC[c] < d_threshold_press_JWL && press_CC[c] > d_thresholdPress_SB) 
      } // if rctVolFrac > 1e-10  
    }//Cell iterator
    

    //__________________________________
    // Update either the output and/or checkpoint intervals
    // pressure exceeding threshold detected
    if( d_adj_IO_Press->onOff || d_adj_IO_Det->onOff ){
    
      max_vartype me;
      old_dw->get( me,  adjOutIntervalsLabel );
      double hasSwitched = me; 
    
      // for readability
      const VarLabel* outIntervalLabel      = d_sharedState->get_outputInterval_label();
      const VarLabel* chkpointIntervalLabel = d_sharedState->get_checkpointInterval_label();
      
      //__________________________________
      // Pressure
      if ( press_switch_adj_IO && d_adj_IO_Press->onOff && isDoubleEqual( hasSwitched, ZERO) ){

        double newOUT  = d_adj_IO_Press->output_interval;
        double newCKPT = d_adj_IO_Press->chkPt_interval;
        hasSwitched = PRESSURE_EXCEEDED;
        
        cout << "\n__________________________________pressure exceeding threshold detected in a cell on patch: " << endl;
        cout << *patch << endl;
        cout << "    new outputInterval: " << newOUT << " new checkpoint Interval: " << newCKPT << "\n\n"<<  endl;

        new_dw->put( min_vartype( newOUT ),  outIntervalLabel );
        new_dw->put( min_vartype( newCKPT ), chkpointIntervalLabel );
      }             
      //__________________________________
      //  DETONATON
      else if ( det_switch_adj_IO && d_adj_IO_Det->onOff && isDoubleEqual(hasSwitched, PRESSURE_EXCEEDED) ){

        double newOUT  = d_adj_IO_Det->output_interval;
        double newCKPT = d_adj_IO_Det->chkPt_interval;
        hasSwitched = DETONATION_DETECTED;
        
        cout << "__________________________________ Detonation detected in a cell on patch:" << endl;
        cout << *patch << endl;
        cout << "    new outputInterval: " << newOUT << " new checkpoint Interval: " << newCKPT << "\n\n"<< endl;

        new_dw->put( min_vartype( newOUT ),  outIntervalLabel );
        new_dw->put( min_vartype( newCKPT ), chkpointIntervalLabel );
      }
      else {        
      //__________________________________
      //  DEFAULT
        min_vartype oldOUT;
        min_vartype oldCKPT;
        oldOUT.setBenignValue();
        oldCKPT.setBenignValue();

        new_dw->put( oldOUT,  outIntervalLabel );
        new_dw->put( oldCKPT, chkpointIntervalLabel );
      }
      new_dw->put( max_vartype(hasSwitched), adjOutIntervalsLabel );
    }
  }//End for{Patches}
}//end Task

//______________________________________________________________________
//
void DDT1::computeModelSources(const ProcessorGroup*, 
                               const PatchSubset* patches,
                               const MaterialSubset*,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw,
                               const ModelInfo* mi)
{
  delt_vartype delT;
  const Level* level = getLevel(patches);
  old_dw->get(delT, mi->delT_Label, level);

 
  int m0 = d_matl0->getDWIndex();
  int m1 = d_matl1->getDWIndex();
  int m2 = d_matl2->getDWIndex();
  double totalBurnedMass   = 0;
  double totalHeatReleased = 0;
  int numAllMatls = d_sharedState->getNumMatls();

  for(int p=0;p<patches->size();p++){
    const Patch* patch   = patches->get(p);
    printTask(patches,patch,cout_doing,"Doing DDT1::computeModelSources");
    

    /* Variable to modify or compute */
    // Sources and Sinks
    CCVariable<double> mass_src_0, mass_src_1, mass_src_2, mass_0;
    CCVariable<Vector> momentum_src_0, momentum_src_1, momentum_src_2;
    CCVariable<double> energy_src_0, energy_src_1, energy_src_2;
    CCVariable<double> sp_vol_src_0, sp_vol_src_1, sp_vol_src_2;
    
    // Burning related
    CCVariable<double> onSurface, surfTemp;
    
    // Detonation Related
    CCVariable<double> Fr, delF;       
    
    // Diagnostics/Thresholds                     
    CCVariable<int> burningCell;
    
    // Old Reactant Quantities
    constCCVariable<double> cv_reactant, rctVolFrac;
    constCCVariable<double> rctTemp, rctRho, rctSpvol, rctFr, detonating;
    constCCVariable<Vector> rctvel_CC;
    constCCVariable<int> BurningCriteria;
    constNCVariable<double> NC_CCweight, rctMass_NC;
    
    // Old Product Quantities
    constCCVariable<double> prodRho, prodRho2;
    
    // Domain Wide Variables
    constCCVariable<double> press_CC; 
    std::vector<constCCVariable<double> > vol_frac_CC(numAllMatls);
    std::vector<constCCVariable<double> > temp_CC(numAllMatls);

    Vector dx = patch->dCell();

    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    /* Gets and Computes */
    //__________________________________
    // Reactant data
    new_dw->get(rctTemp,       MIlb->temp_CCLabel,      m0,patch,gac, 1);
    new_dw->get(rctvel_CC,     MIlb->vel_CCLabel,       m0,patch,gn,  0);
    new_dw->get(rctRho,        Ilb->rho_CCLabel,        m0,patch,gn,  0);
    new_dw->get(rctSpvol,      Ilb->sp_vol_CCLabel,     m0,patch,gn,  0);
    new_dw->get(rctMass_NC,    Mlb->gMassLabel,         m0,patch,gac, 1);
    new_dw->get(rctVolFrac,    Ilb->vol_frac_CCLabel,   m0,patch,gac, 1);
    new_dw->get(detonating,    detonatingLabel,         m0,patch,gn,  0);
    new_dw->get(BurningCriteria,BurningCriteriaLabel,   m0,patch,gn,  0);
     
   
    //__________________________________
    // Product Data, 
    new_dw->get(prodRho,         Ilb->rho_CCLabel,   m1,patch,gn, 0);
    new_dw->get(prodRho2,        Ilb->rho_CCLabel,   m2,patch,gn, 0);
    
    //__________________________________
    //   Misc.
    new_dw->get(press_CC,         Ilb->press_equil_CCLabel,0,  patch,gac,1);
    old_dw->get(NC_CCweight,      MIlb->NC_CCweightLabel,  0,  patch,gac,1);   
   
    // Temperature and Vol_frac 
    for(int m = 0; m < numAllMatls; m++) {
      Material*    matl     = d_sharedState->getMaterial(m);
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      int indx = matl->getDWIndex();
      if(ice_matl){
        old_dw->get(temp_CC[m],   MIlb->temp_CCLabel,    indx, patch,gac,1);
      }else {
        new_dw->get(temp_CC[m],   MIlb->temp_CCLabel,    indx, patch,gac,1);
      }
      new_dw->get(vol_frac_CC[m], Ilb->vol_frac_CCLabel, indx, patch,gac,1);
    }
    //__________________________________
    //  What is computed
    new_dw->getModifiable(mass_src_0,    mi->modelMass_srcLabel,  m0,patch);
    new_dw->getModifiable(momentum_src_0,mi->modelMom_srcLabel,   m0,patch);
    new_dw->getModifiable(energy_src_0,  mi->modelEng_srcLabel,   m0,patch);
    new_dw->getModifiable(sp_vol_src_0,  mi->modelVol_srcLabel,   m0,patch);

    new_dw->getModifiable(mass_src_1,    mi->modelMass_srcLabel,  m1,patch);
    new_dw->getModifiable(momentum_src_1,mi->modelMom_srcLabel,   m1,patch);
    new_dw->getModifiable(energy_src_1,  mi->modelEng_srcLabel,   m1,patch);
    new_dw->getModifiable(sp_vol_src_1,  mi->modelVol_srcLabel,   m1,patch);
    
    new_dw->getModifiable(mass_src_2,    mi->modelMass_srcLabel,  m2,patch);
    new_dw->getModifiable(momentum_src_2,mi->modelMom_srcLabel,   m2,patch);
    new_dw->getModifiable(energy_src_2,  mi->modelEng_srcLabel,   m2,patch);
    new_dw->getModifiable(sp_vol_src_2,  mi->modelVol_srcLabel,   m2,patch);

    new_dw->allocateAndPut(burningCell,  burningLabel,            m0,patch);  
    new_dw->allocateAndPut(Fr,           reactedFractionLabel,    m0,patch);
    new_dw->allocateAndPut(delF,         delFLabel,               m0,patch);
    new_dw->allocateAndPut(onSurface,    onSurfaceLabel,          m0,patch);
    new_dw->allocateAndPut(surfTemp,     surfaceTempLabel,        m0,patch);
    
    
    Fr.initialize(0.);
    delF.initialize(0.);
    burningCell.initialize(0);
    onSurface.initialize(0.);
    surfTemp.initialize(0.);
   

    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m0);
    double cv_rct = mpm_matl->getSpecificHeat();
   
    double cell_vol = dx.x()*dx.y()*dx.z();
    double min_mass_in_a_cell = dx.x()*dx.y()*dx.z()*d_TINY_RHO;
    //__________________________________
    //  Loop over cells
    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter; 
      if (rctVolFrac[c] > 1e-10){ 
        double Tzero = 0.;
        double productPress = 0.;  
        double maxProductVolFrac  = -1.0;  // used for surface area calculation
        double maxReactantVolFrac = -1.0;  // used for surface area calculation
        double temp_vf = 0.0;


         IntVector nodeIdx[8];
         patch->findNodesFromCell(c,nodeIdx);
        //__________________________________
        // intermediate quantities
        /* near interface and containing particles */
       // for(CellIterator offset(IntVector(-1,-1,-1),IntVector(1,1,1) );!offset.done(); offset++){ 
       //   IntVector surCell = c + *offset;              
        for(int i = -1; i<=1; i++){
          for(int j = -1; j<=1; j++){
            for(int k = -1; k<=1; k++){
            IntVector adjcell = c + IntVector(i,j,k);

              /* Search for Tzero from max_vol_frac reactant cell */
              temp_vf = vol_frac_CC[m0][adjcell]; 
              if( temp_vf > maxReactantVolFrac ){
                maxReactantVolFrac = temp_vf;
                Tzero = rctTemp[adjcell];
              }

              /* Search for pressure from max_vol_frac product cell */
              temp_vf = vol_frac_CC[m1][adjcell]; 
              if( temp_vf > maxProductVolFrac ){
                maxProductVolFrac = temp_vf;
                productPress = press_CC[adjcell];
              }
            }
          }
        }

        //__________________________________
        // Detonation
        if(detonating[c] == 1){ 

          Fr[c] = prodRho[c]/(rctRho[c]+prodRho[c]);   
         // Use the JWL++ model for explosion
          if(Fr[c] >= 0. && Fr[c] < 1.0){
            delF[c] = d_G*pow(press_CC[c], d_b)*(1.0 - Fr[c]);
          }
          
          delF[c]*=delT;

          double rctMass    = rctRho[c]  * cell_vol;
          double prdMass    = prodRho[c] * cell_vol;
          double burnedMass = delF[c]*(prdMass+rctMass);

          burnedMass = min(burnedMass, rctMass);
          // 20 % burned mass is a hard limit based p. 55
          //   "JWL++: A Simple Reactive Flow Code Package for Detonation"
          burnedMass = min(burnedMass, .2*mpm_matl->getInitialDensity()*cell_vol);
          totalBurnedMass += burnedMass;

          //__________________________________
          // conservation of mass, momentum and energy                           
          mass_src_0[c] -= burnedMass;
          mass_src_1[c] += burnedMass;         

          Vector momX        = rctvel_CC[c] * burnedMass;
          momentum_src_0[c] -= momX;
          momentum_src_1[c] += momX;

          double energyX       = cv_rct * rctTemp[c] * burnedMass; 
          double releasedHeat  = burnedMass * d_E0;
          energy_src_0[c]     -= energyX;
          energy_src_1[c]     += energyX + releasedHeat;
          totalHeatReleased   += releasedHeat;

          double createdVolx  = burnedMass * rctSpvol[c];
          sp_vol_src_0[c]    -= createdVolx;
          sp_vol_src_1[c]    += createdVolx;
        }


        //__________________________________
        //  On Surface Burning
        if(BurningCriteria[c] == CONDUCTIVE)
        {
          burningCell[c]=CONDUCTIVE;
          Vector rhoGradVector = computeDensityGradientVector(nodeIdx,
                                                              rctMass_NC, NC_CCweight, dx);
          double surfArea = computeSurfaceArea(rhoGradVector, dx); 
          double Tsurf = 850.0;  // initial guess for the surface temperature.

          double solidMass  = rctRho[c]*rctVolFrac[c]*cell_vol;
          double burnedMass = 0.0;

          burnedMass = computeBurnedMass(Tzero, Tsurf, productPress,
                              rctSpvol[c], surfArea, delT,
                              solidMass, min_mass_in_a_cell);
          
          // Store debug variables
          onSurface[c] = surfArea;
          surfTemp[c]  = Tsurf;

          /* conservation of mass, momentum and energy   */
          mass_src_0[c]      -= burnedMass;
          mass_src_2[c]      += burnedMass;
          totalBurnedMass    += burnedMass;

          Vector momX         = rctvel_CC[c] * burnedMass;
          momentum_src_0[c]  -= momX;
          momentum_src_2[c]  += momX;

          double energyX      = cv_rct*rctTemp[c]*burnedMass; 
          double releasedHeat = burnedMass * (d_Qc + d_Qg);
          energy_src_0[c]    -= energyX;
          energy_src_2[c]    += energyX + releasedHeat;
          totalHeatReleased  += releasedHeat;

          double createdVolx  = burnedMass * rctSpvol[c];
          sp_vol_src_0[c]    -= createdVolx;
          sp_vol_src_2[c]    += createdVolx;


        //__________________________________
        //  Convective Burning 
        }  else if (BurningCriteria[c] == CONVECTIVE){
          burningCell[c]=CONVECTIVE;
          
          double surfArea = cell_vol/ 0.002; //divided by 2mm so burning will match what has already been run at 2mm and so the
                                            // burned mass is not depended on the resolution 
          if(surfArea < 1e-12)
            surfArea = 1e-12;
          double Tsurf = 850.0;  // initial guess for the surface temperature.

          double solidMass = rctRho[c]*rctVolFrac[c]*cell_vol;
          double burnedMass = 0.0;

          burnedMass = computeBurnedMass(Tzero, Tsurf, productPress,
                              rctSpvol[c], surfArea, delT,
                              solidMass, min_mass_in_a_cell);

          /* 
           // If cracking applies, add to mass
           if(d_useCrackModel && crackedEnough[c]){
             burnedMass += d_Gcrack*(1 - prodRho[c]/(rctRho[c]+prodRho[c]))
                         *pow(press_CC[c]/101325.0, d_nCrack);
           }

          */

          
          /* conservation of mass, momentum and energy   */
          mass_src_0[c]      -= burnedMass;
          mass_src_1[c]      += burnedMass;
          totalBurnedMass    += burnedMass;

          Vector momX         = rctvel_CC[c] * burnedMass;
          momentum_src_0[c]  -= momX;
          momentum_src_1[c]  += momX;

          double energyX      = cv_rct*rctTemp[c]*burnedMass;
          double releasedHeat = burnedMass * (d_Qc + d_Qg);
          energy_src_0[c]    -= energyX;
          energy_src_1[c]    += energyX + releasedHeat;
          totalHeatReleased  += releasedHeat;

          double createdVolx  = burnedMass * rctSpvol[c];
          sp_vol_src_0[c]    -= createdVolx;
          sp_vol_src_1[c]    += createdVolx;

        } else if(BurningCriteria[c] == WARMINGUP){
           burningCell[c] =WARMINGUP;
        }
      } //endif (rctVolFrac > 1e-10)  
    }  // cell iterator  

    //__________________________________
    //  set symetric BC
    setBC(mass_src_0, "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
    setBC(mass_src_1, "set_if_sym_BC",patch, d_sharedState, m1, new_dw);
    setBC(mass_src_2, "set_if_sym_BC",patch, d_sharedState, m2, new_dw);
    setBC(delF,       "set_if_sym_BC",patch, d_sharedState, m0, new_dw);  // I'm not sure you need these???? Todd
    setBC(Fr,         "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
  }
  //__________________________________
  //save total quantities
  if(d_saveConservedVars->mass ){
    new_dw->put(sum_vartype(totalBurnedMass),  DDT1::totalMassBurnedLabel);
  }
  if(d_saveConservedVars->energy){
    new_dw->put(sum_vartype(totalHeatReleased),DDT1::totalHeatReleasedLabel);
  }
}//End of Task

//______________________________________________________________________
//
void DDT1::scheduleRefine(const PatchSet* patches,
                          SchedulerP& sched)
{
  const Level* level = getLevel(patches);
  
  if(level->hasFinerLevel() == false){  // only on finest level
    printSchedule( patches ,cout_doing,"DDT1::scheduleRefine" );
    
    Task* t = scinew Task("DDT1::refine",this, &DDT1::refine);
    
    const MaterialSubset* react_matl = d_matl0->thisMaterial();
    t->computes( burningLabel,       react_matl );
    t->computes( countTimeLabel,     react_matl );
    t->computes( inductionTimeLabel, react_matl );
    
    sched->addTask(t, patches, d_mymatls);
  }
}
//__________________________________
// Initialize variables on the new fine level patches
// This only works with the tiled regridder.  With the other regridders
// it's possible to have a new patch that contains new cells and old cells.
// We don't want to overwrite the old cell data!
void DDT1::refine(const ProcessorGroup*,
                  const PatchSubset* patches,
                  const MaterialSubset* /*matls*/,
                  DataWarehouse* ,
                  DataWarehouse* new_dw)
{
  int m0 = d_matl0->getDWIndex();

  for(int p=0;p<patches->size();p++) {
    const Patch* patch = patches->get(p);
    printTask( patches,patch,cout_doing,"Doing DDT1::refine" );
    
    CCVariable<int>    burningCell;
    CCVariable<double> countTime;
    CCVariable<double> inductionTime;
    
    new_dw->allocateAndPut( burningCell,    burningLabel,       m0, patch );
    new_dw->allocateAndPut( countTime,      countTimeLabel,     m0, patch );
    new_dw->allocateAndPut( inductionTime,  inductionTimeLabel, m0, patch );
    
    burningCell.initialize( -9 );
    countTime.initialize( -9 );
    inductionTime.initialize( -9 );
  }
}

    
/****************************************************************************/
/******************* Bisection Newton Solver ********************************/    
/****************************************************************************/
double DDT1::computeBurnedMass(double To, double& Ts, double P, double Vc, double surfArea, 
                               double delT, double solidMass, const double min_mass_in_a_cell){  
  IterationVariables iterVar;
  UpdateConstants(To, P, Vc, &iterVar);
  Ts = BisectionNewton(Ts, &iterVar);
  double m =  m_Ts(Ts, &iterVar);
  double burnedMass = delT * surfArea * m;
 // Clamp burned mass to total convertable mass in cell
  if (burnedMass + min_mass_in_a_cell > solidMass){ 
      burnedMass = solidMass - min_mass_in_a_cell;  
      }
  return burnedMass;
  
}
//______________________________________________________________________
//
double DDT1::computeInductionAngle(IntVector *nodeIdx, 
                                  constNCVariable<double> &rctMass_NC, 
                                  constNCVariable<double> &NC_CCweight, 
                                  Vector &dx, 
                                  double& cos_theta, 
                                  double& computedTheta,
                                  Point hotcellCord, 
                                  Point cellCord ){  
                               
  Vector hotcellVector =Vector( (hotcellCord.x() - cellCord.x() ),
                      (hotcellCord.y() - cellCord.y() ),
                      (hotcellCord.z() - cellCord.z() )  );

  Vector rhoGradVector = computeDensityGradientVector(nodeIdx,rctMass_NC, NC_CCweight,dx);

  double massHotcell_dot = Dot(rhoGradVector, hotcellVector);
  cos_theta = abs(massHotcell_dot/(hotcellVector.length() * rhoGradVector.length()));
  
  if(cos_theta > 1){
    cos_theta = 1.0;
  }

  computedTheta = (acos( cos_theta));
  return computedTheta;
  return cos_theta;
}  
//______________________________________________________________________
void DDT1::UpdateConstants(double To, double P, double Vc, IterationVariables *iterVar){
  /* d_CC1 = Ac*R*Kc/Ec/Cp        */
  /* d_CC2 = Qc/Cp/2              */
  /* d_CC3 = 4*Kg*Bg*W*W/Cp/R/R;  */
  /* d_CC4 = Qc/Cp                */
  /* d_CC5 = Qg/Cp                */
  /* Vc = Condensed Phase Specific Volume */

  iterVar->C1 = d_CC1 / Vc; 
  iterVar->C2 = To + d_CC2; 
  iterVar->C3 = d_CC3 * P*P;
  iterVar->C4 = To + d_CC4; 
  iterVar->C5 = d_CC5 * iterVar->C3; 

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
double DDT1::F_Ts(double Ts, IterationVariables *iterVar){
  return Ts_m(m_Ts(Ts, iterVar), iterVar);
}

double DDT1::m_Ts(double Ts, IterationVariables *iterVar){
  return sqrt( iterVar->C1*Ts*Ts/(Ts-iterVar->C2)*exp(-d_Ec/d_R/Ts) );
}

double DDT1::Ts_m(double m, IterationVariables *iterVar){
  double deno = sqrt(m*m+iterVar->C3)+m;
  return iterVar->C4 + iterVar->C5/(deno*deno);
}

/* the function value for the zero finding problem */
double DDT1::Func(double Ts, IterationVariables *iterVar){
  return Ts - F_Ts(Ts, iterVar);
}

/* dFunc/dTs */
double DDT1::Deri(double Ts, IterationVariables *iterVar){
  double m = m_Ts(Ts, iterVar);
  double K1 = Ts-iterVar->C2;
  double K2 = sqrt( m * m + iterVar->C3 );
  double K3 = ( d_R * Ts * (K1-iterVar->C2) + d_Ec * K1) * m * iterVar->C5;
  double K4 = (K2 + m) * ( K2 + m ) * K1 * K2 * d_R * Ts * Ts;
  return 1.0 + K3/K4;
}

/* F_Ts(Ts_max) is the max of F_Ts function */
double DDT1::Ts_max(IterationVariables *iterVar){
  return 0.5*(2.0 * d_R * iterVar->C2 - d_Ec + sqrt(4.0 * d_R * d_R * iterVar->C2*iterVar->C2 + d_Ec * d_Ec))/d_R;
} 

void DDT1::SetInterval(double f, double Ts, IterationVariables *iterVar){  
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
double DDT1::BisectionNewton(double Ts, IterationVariables *iterVar){  
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

      if(fabs(y)<d_EPSILON)
          return Ts;

      delta_new = 1e100;
      while(1){
          if(iter>100){
              cout<<"Not converging after 100 iterations in DDT1.cc."<<endl;
              exit(1);
          }

          df_dTs = Deri(Ts, iterVar);
          if(df_dTs==0) 
              break;

          delta_old = delta_new;
          delta_new = -y/df_dTs; //Newton Step
          Ts += delta_new;
          y = Func(Ts, iterVar);

          if(fabs(y)< d_EPSILON)
              return Ts;

          if(Ts<iterVar->IL || Ts>iterVar->IR || fabs(delta_new)>fabs(delta_old*0.7))
              break;

          iter++; 
          SetInterval(y, Ts, iterVar);  
      }

      Ts = (iterVar->IL+iterVar->IR)/2.0; //Bisection Step
  }
}


//______________________________________________________________________
//
void DDT1::scheduleModifyThermoTransportProperties(SchedulerP&,
                                                   const LevelP&,
                                                   const MaterialSet*)
{
  // do nothing      
}
void DDT1::computeSpecificHeat(CCVariable<double>&,
                               const Patch*,   
                               DataWarehouse*, 
                               const int)      
{
  //do nothing
}
//______________________________________________________________________
//
void DDT1::scheduleErrorEstimate(const LevelP&,
                                 SchedulerP&)
{
  // Not implemented yet
}
//__________________________________
void DDT1::scheduleTestConservation(SchedulerP&,
                                    const PatchSet*,                      
                                    const ModelInfo*)                     
{
  // Not implemented yet
}
