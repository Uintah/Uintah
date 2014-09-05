#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/WestbrookDryer.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

//---------------------------------------------------------------------------
// Builder:
WestbrookDryerBuilder::WestbrookDryerBuilder(std::string srcName, 
                                         vector<std::string> reqLabelNames, 
                                         SimulationStateP& sharedState)
: SourceTermBuilder(srcName, reqLabelNames, sharedState)
{}

WestbrookDryerBuilder::~WestbrookDryerBuilder(){}

SourceTermBase*
WestbrookDryerBuilder::build(){
  return scinew WestbrookDryer( d_srcName, d_sharedState, d_requiredLabels );
}
// End Builder
//---------------------------------------------------------------------------

WestbrookDryer::WestbrookDryer( std::string srcName, SimulationStateP& sharedState,
                            vector<std::string> reqLabelNames ) 
: SourceTermBase(srcName, sharedState, reqLabelNames)
{ 

  d_extraLocalLabels.resize(3); 
  d_WDstrippingLabel = VarLabel::create( "WDstrip",  CCVariable<double>::getTypeDescription() ); 
  d_extraLocalLabels[0] = d_WDstrippingLabel; 
  
  d_WDextentLabel    = VarLabel::create( "WDextent", CCVariable<double>::getTypeDescription() ); 
  d_extraLocalLabels[1] = d_WDextentLabel; 

  d_WDO2Label        = VarLabel::create( "WDo2",     CCVariable<double>::getTypeDescription() ); 
  d_extraLocalLabels[2] = d_WDO2Label; 

}

WestbrookDryer::~WestbrookDryer()
{

  VarLabel::destroy( d_WDstrippingLabel ); 
  VarLabel::destroy( d_WDextentLabel ); 
  VarLabel::destroy( d_WDO2Label ); 

}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
WestbrookDryer::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 

  // Defaults to methane.
  db->getWithDefault("A",d_A, 1.3e8);     // Pre-exponential factor 
  db->getWithDefault("E_R",d_ER, 24.4e3); // Activation Temperature 
  db->getWithDefault("X", d_X, 1);        // C_xH_y
  db->getWithDefault("Y", d_Y, 4);        // C_xH_y
  db->getWithDefault("m", d_m, -0.3);     // [C_xH_y]^m 
  db->getWithDefault("n", d_n, 1.3 );     // [O_2]^n 
  db->require("MW_HydroCarbon",d_MW_HC);  // Molecular weight of C_xH_y (there may be an OH on the end)
  db->require("MF_HC_F1", d_MF_HC_f1);    // Mass fraction of C_xH_y when f=1
  db->require("MF_O2_F0", d_MF_O2_f0);    // Mass fraction of O2 when f=0

  // hard set some values...may want to change some of these to be inputs
  d_MW_O2 = 16.0; 
  d_R     = 8.314472; 
  d_Press = 101325; 

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
WestbrookDryer::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "WestbrookDryer::eval";
  Task* tsk = scinew Task(taskname, this, &WestbrookDryer::computeSource, timeSubStep);

  if (timeSubStep == 0 && !d_labelSchedInit) {
    // Every source term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;

    tsk->computes(d_srcLabel);
    tsk->computes(d_WDstrippingLabel); 
    tsk->computes(d_WDextentLabel); 
    tsk->computes(d_WDO2Label); 
  } else {
    tsk->modifies(d_srcLabel); 
    tsk->modifies(d_WDstrippingLabel); 
    tsk->modifies(d_WDextentLabel);   
    tsk->modifies(d_WDO2Label); 
  }

  for (vector<std::string>::iterator iter = d_requiredLabels.begin(); 
       iter != d_requiredLabels.end(); iter++) { 
    // HERE I WOULD REQUIRE ANY VARIABLES NEEDED TO COMPUTE THE SOURCe
    //tsk->requires( Task::OldDW, .... ); 
  }

  const VarLabel* temperatureLabel = VarLabel::find( "tempIN" ); 
  const VarLabel* fLabel = VarLabel::find( "scalarSP" ); 
  const VarLabel* mixMWLabel = VarLabel::find( "mixMW" ); 
  const VarLabel* denLabel = VarLabel::find( "densityCP" ); 
  // KLUDGE
  const VarLabel* hcMassFracLabel = VarLabel::find( "hcMassFrac" ); // this needs to me generalized

  tsk->requires( Task::OldDW, temperatureLabel, Ghost::None, 0 ); 
  tsk->requires( Task::OldDW, fLabel,           Ghost::None, 0 ); 
  tsk->requires( Task::OldDW, mixMWLabel,       Ghost::None, 0 ); 
  tsk->requires( Task::OldDW, hcMassFracLabel,  Ghost::None, 0 ); 
  tsk->requires( Task::OldDW, denLabel,         Ghost::None, 0 ); 

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
WestbrookDryer::computeSource( const ProcessorGroup* pc, 
                   const PatchSubset* patches, 
                   const MaterialSubset* matls, 
                   DataWarehouse* old_dw, 
                   DataWarehouse* new_dw, 
                   int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> CxHyRate; // rate source term  
    CCVariable<double> S; // stripping fraction 
    CCVariable<double> E; // extent of reaction 
    CCVariable<double> w_o2; // mass fraction of O2
    
    if ( new_dw->exists(d_srcLabel, matlIndex, patch ) ){
      new_dw->getModifiable( CxHyRate, d_srcLabel, matlIndex, patch ); 
      new_dw->getModifiable( S, d_WDstrippingLabel, matlIndex, patch ); 
      new_dw->getModifiable( E, d_WDextentLabel, matlIndex, patch ); 
      new_dw->getModifiable( w_o2, d_WDO2Label, matlIndex, patch ); 
      CxHyRate.initialize(0.0);
      S.initialize(0.0);
      E.initialize(0.0); 
      w_o2.initialize(0.0); 
    } else {
      new_dw->allocateAndPut( CxHyRate, d_srcLabel, matlIndex, patch );
      new_dw->allocateAndPut( S, d_WDstrippingLabel, matlIndex, patch );
      new_dw->allocateAndPut( E, d_WDextentLabel, matlIndex, patch );
      new_dw->allocateAndPut( w_o2, d_WDO2Label, matlIndex, patch ); 
    
      CxHyRate.initialize(0.0);
      S.initialize(0.0); 
      E.initialize(0.0); 
      w_o2.initialize(0.0); 
    } 

    for (vector<std::string>::iterator iter = d_requiredLabels.begin(); 
         iter != d_requiredLabels.end(); iter++) { 
      //CCVariable<double> temp; 
      //old_dw->get( *iter.... ); 
    }

    constCCVariable<double> T;     // temperature 
    constCCVariable<double> f;     // mixture fraction
    constCCVariable<double> den;   // mixture density
    constCCVariable<double> mixMW; // mixture molecular weight (from table is actually the inverse of the MW)
    constCCVariable<double> hcMF;  // mass fraction of the hydrocarbon

    const VarLabel* temperatureLabel = VarLabel::find( "tempIN" ); 
    const VarLabel* fLabel           = VarLabel::find( "scalarSP" ); 
    const VarLabel* mixMWLabel       = VarLabel::find( "mixMW" );
    const VarLabel* denLabel         = VarLabel::find( "densityCP" ); 
    const VarLabel* hcMassFracLabel  = VarLabel::find( "hcMassFrac" ); // this needs to me generalized

    old_dw->get( T, temperatureLabel, matlIndex, patch, Ghost::None, 0 ); 
    old_dw->get( f, fLabel,           matlIndex, patch, Ghost::None, 0 ); 
    old_dw->get( mixMW, mixMWLabel,   matlIndex, patch, Ghost::None, 0 ); 
    old_dw->get( den, denLabel,       matlIndex, patch, Ghost::None, 0 ); 
    old_dw->get( hcMF, hcMassFracLabel, matlIndex, patch, Ghost::None, 0 ); 

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 

      // Step 1: Compute the stripping fraction 
      double small =  1.0e-16; 
      S[c] = 0.0; 
      double hc_wo_rxn = f[c] * d_MF_HC_f1; //fuel as if no rxn occured (computed from mixture fraction)  
      
      if ( hcMF[c] > hc_wo_rxn ) 
        hc_wo_rxn = hcMF[c];   

      if ( hcMF[c] > small ) 
        S[c] = hcMF[c] / hc_wo_rxn; 

      double f_loc = hc_wo_rxn / d_MF_HC_f1; 

      // Step 2: Compute the extent of reaction 
      E[c] = 1.0 - S[c]; 

      // Step 3: Compute the mass fraction of O2
      w_o2[c] = ( 1.0 - f_loc ) * d_MF_O2_f0 - hc_wo_rxn  * E[c] * ( d_X + d_Y/4.0 ) * d_MW_O2 / d_MW_HC; 
      if (w_o2[c] < small)
        w_o2[c] = 0.0;

      // Step 4: Compute the concentration of O2
      double conc_O2 = w_o2[c] * 1.0/mixMW[c] * 1.0/d_MW_O2 * d_Press / ( d_R * T[c] ); 
      conc_O2 *= 1.0e-6; // to convert to gmol/cm^3

      // Step 5: Computes the concentration of the HydroCarbon
      double conc_HC = hcMF[c] * 1.0/mixMW[c] * 1.0/d_MW_HC * d_Press / ( d_R * T[c] ); 
      conc_HC *= 1.0e-6; // to convert to gmol/cm^3

      // Step 6: Compute the rate term 
      double my_exp = -1.0 * d_ER / T[c]; 

      double p_HC = 0.0; 
      if (conc_HC > small)
        p_HC = pow(conc_HC, d_m); 

      double rate = d_A * exp( my_exp ) * p_HC * pow(conc_O2, d_n); // gmol/cm^3/s

      CxHyRate[c] = - rate * d_MW_HC * mixMW[c] * d_R * T[c] / d_Press;
      CxHyRate[c] *= den[c] * 1.0e6; // to get [kg HC/time/vol]

      if (isnan(CxHyRate[c])) {
        CxHyRate[c] = 0.0; 
      }

    }
  }
}
//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
void
WestbrookDryer::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  string taskname = "WestbrookDryer::dummyInit"; 

  Task* tsk = scinew Task(taskname, this, &WestbrookDryer::dummyInit);

  tsk->computes(d_srcLabel);

  for (std::vector<const VarLabel*>::iterator iter = d_extraLocalLabels.begin(); iter != d_extraLocalLabels.end(); iter++){
    tsk->computes(*iter, d_sharedState->allArchesMaterials()->getUnion()); 
  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());

}
void 
WestbrookDryer::dummyInit( const ProcessorGroup* pc, 
                      const PatchSubset* patches, 
                      const MaterialSubset* matls, 
                      DataWarehouse* old_dw, 
                      DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> src;
    new_dw->allocateAndPut( src, d_srcLabel, matlIndex, patch ); 

    src.initialize(0.0); 

    for (std::vector<const VarLabel*>::iterator iter = d_extraLocalLabels.begin(); iter != d_extraLocalLabels.end(); iter++){
      const VarLabel* tempVL = *iter; 
      CCVariable<double> tempVar; 
      new_dw->allocateAndPut(tempVar, tempVL, matlIndex, patch ); 
      tempVar.initialize(0.0); 
    }
  }
}



