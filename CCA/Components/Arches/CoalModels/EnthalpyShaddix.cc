#include <CCA/Components/Arches/CoalModels/EnthalpyShaddix.h>
#include <CCA/Components/Arches/CoalModels/CharOxidation.h>
#include <CCA/Components/Arches/CoalModels/Devolatilization.h>
#include <CCA/Components/Arches/CoalModels/PartVel.h>
#include <CCA/Components/Arches/ParticleModels/ParticleHelper.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <CCA/Components/Arches/CoalModels/fortran/rqpart_fort.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Parallel/Parallel.h>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace Uintah; 

//---------------------------------------------------------------------------
// Builder:
EnthalpyShaddixBuilder::EnthalpyShaddixBuilder( const std::string         & modelName,
                                                const vector<std::string> & reqICLabelNames,
                                                const vector<std::string> & reqScalarLabelNames,
                                                ArchesLabel               * fieldLabels,
                                                SimulationStateP          & sharedState,
                                                Properties                * props, 
                                                int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{
  d_props = props; 
}

EnthalpyShaddixBuilder::~EnthalpyShaddixBuilder(){}

ModelBase* EnthalpyShaddixBuilder::build() {
  return scinew EnthalpyShaddix( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_props, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

EnthalpyShaddix::EnthalpyShaddix( std::string modelName, 
                                  SimulationStateP& sharedState,
                                  ArchesLabel* fieldLabels,
                                  vector<std::string> icLabelNames, 
                                  vector<std::string> scalarLabelNames,
                                  Properties* props, 
                                  int qn ) 
: HeatTransfer(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  _sigma = 5.67e-8;   // [=] J/s/m^2/K^4 : Stefan-Boltzmann constant (from white book)
  _pi = acos(-1.0);  
  _Rgas = 8314.3; // J/K/kmol
  _Pr = 0.7; // 
  d_props = props; 
}

EnthalpyShaddix::~EnthalpyShaddix()
{
}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
EnthalpyShaddix::problemSetup(const ProblemSpecP& params, int qn)
{
  HeatTransfer::problemSetup( params, qn );

  ProblemSpecP db = params;
  const ProblemSpecP params_root = db->getRootNode();

  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();
  
  // check for particle enthalpy scaling constant
  std::string enthalpy_root = ParticleHelper::parse_for_role_to_label(db, "enthalpy"); 
  std::string enthalpyqn_name = ParticleHelper::append_qn_env( enthalpy_root, d_quadNode ); 
  EqnBase& temp_enthalpy_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(enthalpyqn_name);
  DQMOMEqn& enthalpy_eqn = dynamic_cast<DQMOMEqn&>(temp_enthalpy_eqn);
   _enthalpy_scaling_constant = enthalpy_eqn.getScalingConstant(d_quadNode);

  // check for particle temperature 
  std::string temperature_root = ParticleHelper::parse_for_role_to_label(db, "temperature"); 
  std::string temperature_name = ParticleHelper::append_env( temperature_root, d_quadNode ); 
  _particle_temperature_varlabel = VarLabel::find(temperature_name); 

  // check for length  
  std::string length_root = ParticleHelper::parse_for_role_to_label(db, "size"); 
  std::string length_name = ParticleHelper::append_env( length_root, d_quadNode ); 
  _length_varlabel = VarLabel::find(length_name); 

  // get weight and scaling constant
  std::string weightqn_name = ParticleHelper::append_qn_env("w", d_quadNode); 
  std::string weight_name = ParticleHelper::append_env("w", d_quadNode); 
  _weight_varlabel = VarLabel::find(weight_name); 
  EqnBase& temp_weight_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(weightqn_name);
  DQMOMEqn& weight_eqn = dynamic_cast<DQMOMEqn&>(temp_weight_eqn);
  _weight_small = weight_eqn.getSmallClipPlusTol();
  _weight_scaling_constant = weight_eqn.getScalingConstant(d_quadNode);

  // get computed rates from char oxidation model 
  CoalModelFactory& modelFactory = CoalModelFactory::self();
  CharOxiModelMap charoximodels_ = modelFactory.retrieve_charoxi_models();
  for( CharOxiModelMap::iterator iModel = charoximodels_.begin(); iModel != charoximodels_.end(); ++iModel ) {
    int modelNode = iModel->second->getquadNode();
    if( modelNode == d_quadNode) {
      _surfacerate_varlabel = iModel->second->getSurfaceRateLabel();
      _charoxiTemp_varlabel = iModel->second->getParticleTempSourceLabel();
      _chargas_varlabel = iModel->second->getGasSourceLabel();
    }
  }


  // get gas phase temperature label 
  if (VarLabel::find("temperature")) {
    _gas_temperature_varlabel = VarLabel::find("temperature");
  } else {
    throw InvalidValue("ERROR: EnthalpyShaddix: problemSetup(): can't find gas phase temperature.",__FILE__,__LINE__);
  }

  // get gas phase specific heat label 
  if (VarLabel::find("specificheat")) {
    _gas_cp_varlabel = VarLabel::find("specificheat"); 
  } else {
    throw InvalidValue("ERROR: EnthalpyShaddix: problemSetup(): can't find gas phase specificheat.",__FILE__,__LINE__);
  }

  std::string modelName;
  std::string baseNameAbskp;
  std::string baseNameAbskg;

  if (d_radiation ) {
    ProblemSpecP db_prop = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("PropertyModels");
    for ( ProblemSpecP db_model = db_prop->findBlock("model"); db_model != 0; 
        db_model = db_model->findNextBlock("model")){
      db_model->getAttribute("type", modelName);
      if (modelName=="radiation_properties"){
        if  (db_model->findBlock("calculator") == 0){
          if(qn ==0) {
            proc0cout <<"\n///-------------------------------------------///\n";
            proc0cout <<"WARNING: No radiation particle properties computed!\n";
            proc0cout <<"Particles will not interact with radiation!\n";
            proc0cout <<"///-------------------------------------------///\n";
          }
          d_radiation = false;
          break;
        }else if(db_model->findBlock("calculator")->findBlock("particles") == 0){
          if(qn ==0) {
            proc0cout <<"\n///-------------------------------------------///\n";
            proc0cout <<"WARNING: No radiation particle properties computed!\n";
            proc0cout <<"Particles will not interact with radiation!\n";
            proc0cout <<"///-------------------------------------------///\n";
          }
          d_radiation = false;
          break;
        }
        db_model->findBlock("calculator")->findBlock("particles")->findBlock("abskp")->getAttribute("label",baseNameAbskp);
        db_model->findBlock("calculator")->findBlock("abskg")->getAttribute("label",baseNameAbskg);
        break;
      }
      if  (db_model== 0){
          if(qn ==0) {
            proc0cout <<"\n///-------------------------------------------///\n";
            proc0cout <<"WARNING: No radiation particle properties computed!\n";
            proc0cout <<"Particles will not interact with radiation!\n";
            proc0cout <<"///-------------------------------------------///\n";
          }
        d_radiation = false;
        break;
      }
    }
    if (VarLabel::find("radiationVolq")) {
      _volq_varlabel  = VarLabel::find("radiationVolq"); 
    } else {
      throw InvalidValue("ERROR: EnthalpyShaddix: problemSetup(): can't find radiationVolq.",__FILE__,__LINE__);
    }
    std::string abskp_string = ParticleHelper::append_env(baseNameAbskp, d_quadNode);
    _abskp_varlabel = VarLabel::find(abskp_string);
    _abskg_varlabel = VarLabel::find(baseNameAbskg);
  }


  // get computed rates from devolatilization model 
  DevolModelMap devolmodels_ = modelFactory.retrieve_devol_models();
  for( DevolModelMap::iterator iModel = devolmodels_.begin(); iModel != devolmodels_.end(); ++iModel ) {
    int modelNode = iModel->second->getquadNode();
    if( modelNode == d_quadNode) {
      _devolgas_varlabel = iModel->second->getGasSourceLabel();
    }
  }

  // check for viscosity
  if (params_root->findBlock("PhysicalConstants")) {
    ProblemSpecP db_phys = params_root->findBlock("PhysicalConstants");
    db_phys->require("viscosity", _visc);
    if( _visc == 0 ) {
      throw InvalidValue("ERROR: EnthalpyShaddix: problemSetup(): Zero viscosity specified in <PhysicalConstants> section of input file.",__FILE__,__LINE__);
    }
  } else {
    throw InvalidValue("ERROR: EnthalpyShaddix: problemSetup(): Missing <PhysicalConstants> section in input file!",__FILE__,__LINE__);
  }

  // get coal properties 
  if (params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal")->findBlock("Properties")) {
    ProblemSpecP db_coal = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal")->findBlock("Properties");
    db_coal->require("raw_coal_enthalpy", _Hc0);
    db_coal->require("char_enthalpy", _Hh0);
    db_coal->getWithDefault( "ksi",_ksi,1); // Fraction of the heat released by char oxidation that goes to the particle
    ProblemSpecP db_ua = db_coal->findBlock("ultimate_analysis"); 
    CoalAnalysis coal; 
    db_ua->require("C",coal.C);
    db_ua->require("H",coal.H);
    db_ua->require("O",coal.O);
    db_ua->require("N",coal.N);
    db_ua->require("S",coal.S);
    db_ua->require("H2O",coal.H2O);
    db_ua->require("ASH",coal.ASH);
    db_ua->require("CHAR",coal.CHAR);
    total_rc=coal.C+coal.H+coal.O+coal.N+coal.S; // (C+H+O+N+S) dry ash free total
    yelem[0]=coal.C/total_rc; // C daf
    yelem[1]=coal.H/total_rc; // H daf
    yelem[2]=coal.N/total_rc; // N daf
    yelem[3]=coal.O/total_rc; // O daf
    yelem[4]=coal.S/total_rc; // S daf
  } else {
    throw InvalidValue("ERROR: EnthalpyShaddix: problemSetup(): Missing <CoalProperties> section in input file!",__FILE__,__LINE__);
  }

  double MW [5] = { 12., 1., 14., 16., 32.}; // Atomic weight of elements (C,H,N,O,S) - kg/kmol
  _MW_avg = 0.0; // Mean atomic weight of coal
  for(int i=0;i<5;i++){
    _MW_avg += yelem[i]/MW[i];
  }
  _MW_avg = 1/_MW_avg;

  //_RdC = _Rgas/12.0107;
  _RdC = _Rgas/12.0;
  _RdMW = _Rgas/_MW_avg; 

}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void 
EnthalpyShaddix::sched_initVars( const LevelP& level, SchedulerP& sched )
{
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
EnthalpyShaddix::initVars( const ProcessorGroup * pc, 
                              const PatchSubset    * patches, 
                              const MaterialSubset * matls, 
                              DataWarehouse        * old_dw, 
                              DataWarehouse        * new_dw )
{
}

//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model 
//---------------------------------------------------------------------------
void 
EnthalpyShaddix::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "EnthalpyShaddix::computeModel";
  Task* tsk = scinew Task(taskname, this, &EnthalpyShaddix::computeModel, timeSubStep);

  Ghost::GhostType  gn  = Ghost::None;

  Task::WhichDW which_dw; 

  if (timeSubStep == 0 ) { 
    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel); 
    tsk->computes(d_qconvLabel);
    tsk->computes(d_qradLabel);
    which_dw = Task::OldDW; 
  } else {
    tsk->modifies(d_modelLabel);
    tsk->modifies(d_gasLabel);  
    tsk->modifies(d_qconvLabel);
    tsk->modifies(d_qradLabel);
    which_dw = Task::NewDW; 
  }

  // require gas phase variables 
  tsk->requires( which_dw, _gas_temperature_varlabel, Ghost::None, 0);
  tsk->requires( which_dw, _gas_cp_varlabel, Ghost::None, 0);
  tsk->requires( which_dw, _devolgas_varlabel, Ghost::None, 0 );
  tsk->requires( which_dw, _chargas_varlabel, Ghost::None, 0 );
  tsk->requires( which_dw, d_fieldLabels->d_CCVelocityLabel, Ghost::None, 0);
  tsk->requires( which_dw, d_fieldLabels->d_densityCPLabel, Ghost::None, 0);
  if ( d_radiation ){ 
    tsk->requires( which_dw, _abskg_varlabel,  Ghost::None, 0);   
    tsk->requires( which_dw, _volq_varlabel, Ghost::None, 0);
    tsk->requires( which_dw, _abskp_varlabel, Ghost::None, 0);
  }

  // require particle phase variables
  tsk->requires( which_dw, _particle_temperature_varlabel, gn, 0 ); 
  tsk->requires( which_dw, _length_varlabel, gn, 0 ); 
  tsk->requires( which_dw, _weight_varlabel, gn, 0 ); 
  tsk->requires( which_dw, _surfacerate_varlabel, Ghost::None, 0 );
  tsk->requires( which_dw, _charoxiTemp_varlabel, Ghost::None, 0 );
  // require particle velocity
  ArchesLabel::PartVelMap::const_iterator i = d_fieldLabels->partVel.find(d_quadNode);
  tsk->requires( Task::NewDW, i->second, gn, 0 );

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}

//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
EnthalpyShaddix::computeModel( const ProcessorGroup * pc, 
                                  const PatchSubset    * patches, 
                                  const MaterialSubset * matls, 
                                  DataWarehouse        * old_dw, 
                                  DataWarehouse        * new_dw, 
                                  const int timeSubStep )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> heat_rate;
    CCVariable<double> gas_heat_rate; 
    CCVariable<double> qconv;
    CCVariable<double> qrad;
    DataWarehouse* which_dw; 
    if ( timeSubStep == 0 ){ 
      which_dw = old_dw;
      new_dw->allocateAndPut( heat_rate, d_modelLabel, matlIndex, patch );
      heat_rate.initialize(0.0);
      new_dw->allocateAndPut( gas_heat_rate, d_gasLabel, matlIndex, patch );
      gas_heat_rate.initialize(0.0);
      new_dw->allocateAndPut( qconv, d_qconvLabel, matlIndex, patch );
      qconv.initialize(0.0);
      new_dw->allocateAndPut( qrad, d_qradLabel, matlIndex, patch );
      qrad.initialize(0.0);
    } else { 
      which_dw = new_dw;
      new_dw->getModifiable( heat_rate, d_modelLabel, matlIndex, patch ); 
      new_dw->getModifiable( gas_heat_rate, d_gasLabel, matlIndex, patch ); 
      new_dw->getModifiable( qconv, d_qconvLabel, matlIndex, patch );
      new_dw->getModifiable( qrad, d_qradLabel, matlIndex, patch );
    }

    // get gas phase variables 
    constCCVariable<double> temperature;
    which_dw->get( temperature, _gas_temperature_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> specific_heat;
    which_dw->get( specific_heat, _gas_cp_varlabel, matlIndex, patch, gn, 0 );  // in J/kg/K
    constCCVariable<double> radiationVolqIN;
    constCCVariable<double> abskgIN;
    constCCVariable<double> abskp; 
    if ( d_radiation ){ 
      which_dw->get( radiationVolqIN, _volq_varlabel, matlIndex, patch, gn, 0);
      which_dw->get( abskgIN, _abskg_varlabel, matlIndex, patch, gn, 0);
      which_dw->get( abskp, _abskp_varlabel, matlIndex, patch, gn, 0);
    }
    constCCVariable<Vector> gasVel; 
    which_dw->get( gasVel, d_fieldLabels->d_CCVelocityLabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> den;
    which_dw->get( den, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> devol_gas_source;
    which_dw->get( devol_gas_source, _devolgas_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> chargas_source;
    which_dw->get( chargas_source, _chargas_varlabel, matlIndex, patch, gn, 0 );

    // get particle phase variables 
    constCCVariable<double> length;
    which_dw->get( length, _length_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> weight;
    which_dw->get( weight, _weight_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> particle_temperature;
    which_dw->get( particle_temperature, _particle_temperature_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> charoxi_temp_source;
    which_dw->get( charoxi_temp_source, _charoxiTemp_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> surface_rate;
    which_dw->get( surface_rate, _surfacerate_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<Vector> partVel;  
    ArchesLabel::PartVelMap::const_iterator iter = d_fieldLabels->partVel.find(d_quadNode);
    new_dw->get(partVel, iter->second, matlIndex, patch, gn, 0);
    
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 


      double temperatureph=temperature[c];
      double specific_heatph=specific_heat[c];
      //double radiationVolqINph=radiationVolqIN[c];
      //double abskgINph=abskgIN[c];
      double denph=den[c];
      double devol_gas_sourceph=devol_gas_source[c];
      double chargas_sourceph=chargas_source[c];
      double lengthph=length[c];
      double weightph=weight[c];
      double particle_temperatureph=particle_temperature[c];
      double charoxi_temp_sourceph=charoxi_temp_source[c];
      double surface_rateph=surface_rate[c];

      // velocities
      Vector gas_velocity = gasVel[c];
      Vector particle_velocity = partVel[c];

      //Verification
      //temperatureph=1206.4;
      //specific_heatph=18356.4;
      //radiationVolqINph=0;
      //abskgINph=0.666444;
      //denph=0.394622;
      //devol_gas_sourceph=7.14291e-08;
      //chargas_sourceph=9.14846e-05;
      //lengthph=2e-05;
      //weightph=1.40781e+09;
      //particle_temperatureph=536.954;
      //charoxi_temp_sourceph=842.663;
      //surface_rateph=5.72444e-05;
      //gas_velocity[0]=7.56321;
      //gas_velocity[1]=0.663992;
      //gas_velocity[2]=0.654003;
      //particle_velocity[0]=6.54863;
      //particle_velocity[1]=0.339306;
      //particle_velocity[2]=0.334942;

      double FSum = 0.0;

      double heat_rate_ = 0;
      double gas_heat_rate_ = 0;

      // intermediate calculation values
      double Re;
      double Nu; 
      double rkg;
      double Q_convection;
      double Q_radiation;
      double Q_reaction;

      if (weightph/_weight_scaling_constant < _weight_small) {
        heat_rate_ = 0.0;
        gas_heat_rate_ = 0.0;
        Q_convection = 0.0;
        Q_radiation = 0.0;
      } else {

        // Convection part: -----------------------
        // Reynolds number
        double delta_V =sqrt(pow(gas_velocity.x() - particle_velocity.x(),2.0) + pow(gas_velocity.y() - particle_velocity.y(),2.0)+pow(gas_velocity.z() - particle_velocity.z(),2.0));
        Re = delta_V*lengthph*denph/_visc;

        // Nusselt number
        Nu = 2.0 + 0.65*pow(Re,0.50)*pow(_Pr,(1.0/3.0)); 

        // Gas thermal conductivity
        rkg = props(temperatureph, particle_temperatureph); // [=] J/s/m/K

        // A BLOWING CORRECTION TO THE HEAT TRANSFER MODEL IS EMPLOYED
        kappa =  -surface_rateph*lengthph*specific_heatph/(2.0*rkg);
        if(std::abs(exp(kappa)-1) < 1e-16){
          blow = 1.0;
        } else {
          blow = kappa/(exp(kappa)-1.0);
        }
        // Q_convection (see Section 5.4 of LES_Coal document)
        Q_convection = Nu*_pi*blow*rkg*lengthph*(temperatureph - particle_temperatureph);

        // Radiation part: -------------------------
        Q_radiation = 0.0;
        if ( d_radiation) { 
          double Eb;
          if (_radiateAtGasTemp){
            Eb = 4.0*_sigma*pow(temperatureph,4.0); 
          }else{
            Eb = 4.0*_sigma*pow(particle_temperatureph,4.0); 
          }
          FSum = radiationVolqIN[c];    
          Q_radiation = abskp[c]*(FSum - Eb);
        } 

        double hint = -156.076 + 380/(-1 + exp(380 / particle_temperatureph)) + 3600/(-1 + exp(1800 / particle_temperatureph));
        double hc = _Hc0 + hint * _RdMW;
        double hh = _Hh0 + hint * _RdC;
        Q_reaction = charoxi_temp_sourceph;
                                             // This needs to be made consistant with lagrangian particles!!! - derek 12/14
        heat_rate_ = (Q_convection*weightph + Q_radiation + _ksi*Q_reaction - devol_gas_sourceph*hc - chargas_sourceph*hh)/
                     (_enthalpy_scaling_constant*_weight_scaling_constant);
        gas_heat_rate_ = -weightph*Q_convection + Q_radiation - _ksi*Q_reaction + devol_gas_sourceph*hc + chargas_sourceph*hh;
      }
  
      heat_rate[c] = heat_rate_;
      gas_heat_rate[c] = gas_heat_rate_;
      qconv[c] = Q_convection;
      qrad[c] = Q_radiation;

    }//end cell loop

  }//end patch loop
}



// ********************************************************
// Private methods:

double
EnthalpyShaddix::props(double Tg, double Tp){

  double tg0[10] = {300.,  400.,   500.,   600.,  700.,  800.,  900.,  1000., 1100., 1200. };
  double kg0[10] = {.0262, .03335, .03984, .0458, .0512, .0561, .0607, .0648, .0685, .07184};
  double T = (Tp+Tg)/2; // Film temperature

//   CALCULATE UG AND KG FROM INTERPOLATION OF TABLE VALUES FROM HOLMAN
//   FIND INTERVAL WHERE TEMPERATURE LIES. 

  double kg = 0.0;

  if( T > 1200.0 ) {
    kg = kg0[9] * pow( T/tg0[9], 0.58);

  } else if ( T < 300 ) {
    kg = kg0[0];
  
  } else {
    int J = -1;
    for ( int I=0; I < 9; I++ ) {
      if ( T > tg0[I] ) {
        J = J + 1;
      }
    }
    double FAC = ( tg0[J] - T ) / ( tg0[J] - tg0[J+1] );
    kg = ( -FAC*( kg0[J] - kg0[J+1] ) + kg0[J] );
  }

  return kg; // I believe this is in J/s/m/K, but not sure
}

