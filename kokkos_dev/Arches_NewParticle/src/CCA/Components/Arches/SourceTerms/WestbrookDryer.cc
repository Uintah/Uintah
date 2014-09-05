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

WestbrookDryer::WestbrookDryer( std::string src_name, SimulationStateP& shared_state,
                            vector<std::string> req_label_names ) 
: SourceTermBase(src_name, shared_state, req_label_names)
{ 

  _label_sched_init = false; 
  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() ); 

  _extra_local_labels.resize(4); 
  std::string tag = "WDstrip_" + src_name; 
  d_WDstrippingLabel = VarLabel::create( tag,  CCVariable<double>::getTypeDescription() ); 
  _extra_local_labels[0] = d_WDstrippingLabel; 
  
  tag = "WDextent_" + src_name; 
  d_WDextentLabel    = VarLabel::create( tag, CCVariable<double>::getTypeDescription() ); 
  _extra_local_labels[1] = d_WDextentLabel; 

  tag = "WDo2_" + src_name; 
  d_WDO2Label        = VarLabel::create( tag,     CCVariable<double>::getTypeDescription() ); 
  _extra_local_labels[2] = d_WDO2Label; 

  tag = "WDver_" + src_name; 
  d_WDverLabel = VarLabel::create( tag, CCVariable<double>::getTypeDescription() ); 
  _extra_local_labels[3] = d_WDverLabel; 

  _source_type = CC_SRC; 
}

WestbrookDryer::~WestbrookDryer()
{

  VarLabel::destroy( d_WDstrippingLabel ); 
  VarLabel::destroy( d_WDextentLabel ); 
  VarLabel::destroy( d_WDO2Label ); 
  VarLabel::destroy( d_WDverLabel ); 

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
  db->require("fuel_mass_fraction", d_MF_HC_f1);           // Mass fraction of C_xH_y when f=1
  db->require("oxidizer_O2_mass_fraction", d_MF_O2_f0);    // Mass fraction of O2 when f=0
  // labels: 
  db->require("mix_frac_label", d_mf_label);     // The name of the mixture fraction label
  db->getWithDefault("temperature_label", d_T_label, "temperature");     // The name of the mixture fraction label
  db->getWithDefault("density_label", d_rho_label, "density"); 
  db->require("hc_frac_label", d_hc_label);     // The name of the mixture fraction label
  db->require("mw_label", d_mw_label);           // The name of the MW label

  if ( db->findBlock("pos") ) { 
    d_sign = 1.0; 
  } else { 
    d_sign = -1.0; 
  }

  // hard set some values...may want to change some of these to be inputs
  d_MW_O2 = 32.0; 
  d_R     = 8.314472; 
  d_Press = 101325; 

  d_MW_HC = 12.0*d_X + 1.0*d_Y; // compute the molecular weight from input information

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
WestbrookDryer::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "WestbrookDryer::computeSource";
  Task* tsk = scinew Task(taskname, this, &WestbrookDryer::computeSource, timeSubStep);

  if (timeSubStep == 0 && !_label_sched_init) {
    // Every source term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    _label_sched_init = true;

    tsk->computes(_src_label);
    tsk->computes(d_WDstrippingLabel); 
    tsk->computes(d_WDextentLabel); 
    tsk->computes(d_WDO2Label); 
    tsk->computes(d_WDverLabel); 
  } else {
    tsk->modifies(_src_label); 
    tsk->modifies(d_WDstrippingLabel); 
    tsk->modifies(d_WDextentLabel);   
    tsk->modifies(d_WDO2Label); 
    tsk->modifies(d_WDverLabel); 
  }

  for (vector<std::string>::iterator iter = _required_labels.begin(); 
       iter != _required_labels.end(); iter++) { 
    // HERE I WOULD REQUIRE ANY VARIABLES NEEDED TO COMPUTE THE SOURCe
    //tsk->requires( Task::OldDW, .... ); 
  }

  _temperatureLabel = VarLabel::find( d_T_label );
  _fLabel           = VarLabel::find( d_mf_label );
  _mixMWLabel       = VarLabel::find( d_mw_label );
  _denLabel         = VarLabel::find( d_rho_label );
  _hcMassFracLabel  = VarLabel::find( d_hc_label );

  tsk->requires( Task::OldDW, _temperatureLabel, Ghost::None, 0 ); 
  tsk->requires( Task::OldDW, _fLabel,           Ghost::None, 0 ); 
  tsk->requires( Task::OldDW, _mixMWLabel,       Ghost::None, 0 ); 
  tsk->requires( Task::OldDW, _hcMassFracLabel,  Ghost::None, 0 ); 
  tsk->requires( Task::OldDW, _denLabel,         Ghost::None, 0 ); 

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials()); 

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
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> CxHyRate; // rate source term  
    CCVariable<double> S; // stripping fraction 
    CCVariable<double> E; // extent of reaction 
    CCVariable<double> w_o2; // mass fraction of O2
    CCVariable<double> ver; 

    if( timeSubStep == 0 ) {
      new_dw->allocateAndPut( CxHyRate , _src_label         , matlIndex , patch );
      new_dw->allocateAndPut( S        , d_WDstrippingLabel , matlIndex , patch );
      new_dw->allocateAndPut( E        , d_WDextentLabel    , matlIndex , patch );
      new_dw->allocateAndPut( w_o2     , d_WDO2Label        , matlIndex , patch );
      new_dw->allocateAndPut( ver      , d_WDverLabel       , matlIndex , patch );
    
      CxHyRate.initialize(0.0);
      S.initialize(0.0);
      E.initialize(0.0); 
      w_o2.initialize(0.0); 
      ver.initialize(0.0); 
    } else {
      new_dw->allocateAndPut( CxHyRate , _src_label         , matlIndex , patch );
      new_dw->allocateAndPut( S        , d_WDstrippingLabel , matlIndex , patch );
      new_dw->allocateAndPut( E        , d_WDextentLabel    , matlIndex , patch );
      new_dw->allocateAndPut( w_o2     , d_WDO2Label        , matlIndex , patch );
      new_dw->allocateAndPut( ver      , d_WDverLabel       , matlIndex , patch );
    
      CxHyRate.initialize(0.0);
      S.initialize(0.0);
      E.initialize(0.0); 
      w_o2.initialize(0.0); 
      ver.initialize(0.0); 
    }

    constCCVariable<double> T;     // temperature 
    constCCVariable<double> f;     // mixture fraction
    constCCVariable<double> den;   // mixture density
    constCCVariable<double> mixMW; // mixture molecular weight (assumes that the table value is an inverse)
    constCCVariable<double> hcMF;  // mass fraction of the hydrocarbon

    old_dw->get( T     , _temperatureLabel , matlIndex , patch , Ghost::None , 0 );
    old_dw->get( f     , _fLabel           , matlIndex , patch , Ghost::None , 0 );
    old_dw->get( mixMW , _mixMWLabel       , matlIndex , patch , Ghost::None , 0 );
    old_dw->get( den   , _denLabel         , matlIndex , patch , Ghost::None , 0 );
    old_dw->get( hcMF  , _hcMassFracLabel  , matlIndex , patch , Ghost::None , 0 );

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

      CxHyRate[c] = rate * d_MW_HC * mixMW[c] * d_R * T[c] / d_Press;
      CxHyRate[c] *= den[c] * 1.0e6; // to get [kg HC/time/vol]
      CxHyRate[c] *= d_sign; // pick the sign. 

      //if (isnan(CxHyRate[c])) {
      // Checking for NaN
      if ( CxHyRate[c] != CxHyRate[c] ) {
        CxHyRate[c] = 0.0; 
      }

      ver[c] = hc_wo_rxn * S[c]; 

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

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter, _shared_state->allArchesMaterials()->getUnion()); 
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

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
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> src;
    new_dw->allocateAndPut( src, _src_label, matlIndex, patch ); 

    src.initialize(0.0); 

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
      const VarLabel* tempVL = *iter; 
      CCVariable<double> tempVar; 
      new_dw->allocateAndPut(tempVar, tempVL, matlIndex, patch ); 
      tempVar.initialize(0.0); 
    }
  }
}



