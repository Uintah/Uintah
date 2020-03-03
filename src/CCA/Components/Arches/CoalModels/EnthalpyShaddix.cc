/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <CCA/Components/Arches/CoalModels/EnthalpyShaddix.h>

#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <CCA/Components/Arches/CoalModels/CharOxidation.h>
#include <CCA/Components/Arches/CoalModels/Devolatilization.h>
#include <CCA/Components/Arches/CoalModels/PartVel.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/Exceptions/InvalidValue.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <iomanip>
#include <iostream>

using namespace std;
using namespace Uintah;

//---------------------------------------------------------------------------
// Builder:
EnthalpyShaddixBuilder::EnthalpyShaddixBuilder( const std::string         & modelName,
                                                const vector<std::string> & reqICLabelNames,
                                                const vector<std::string> & reqScalarLabelNames,
                                                ArchesLabel               * fieldLabels,
                                                MaterialManagerP          & materialManager,
                                                Properties                * props,
                                                int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, materialManager, qn )
{
  d_props = props;
}

EnthalpyShaddixBuilder::~EnthalpyShaddixBuilder(){}

ModelBase* EnthalpyShaddixBuilder::build() {
  return scinew EnthalpyShaddix( d_modelName, d_materialManager, d_fieldLabels, d_icLabels, d_scalarLabels, d_props, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

EnthalpyShaddix::EnthalpyShaddix( std::string modelName,
                                  MaterialManagerP& materialManager,
                                  ArchesLabel* fieldLabels,
                                  vector<std::string> icLabelNames,
                                  vector<std::string> scalarLabelNames,
                                  Properties* props,
                                  int qn )
: HeatTransfer(modelName, materialManager, fieldLabels, icLabelNames, scalarLabelNames, qn)
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
  std::string enthalpy_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_ENTHALPY);
  std::string enthalpyqn_name = ArchesCore::append_qn_env( enthalpy_root, d_quadNode );
  EqnBase& temp_enthalpy_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(enthalpyqn_name);
  DQMOMEqn& enthalpy_eqn = dynamic_cast<DQMOMEqn&>(temp_enthalpy_eqn);
   _enthalpy_scaling_constant = enthalpy_eqn.getScalingConstant(d_quadNode);

  // check for particle temperature
  std::string temperature_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_TEMPERATURE);
  std::string temperature_name = ArchesCore::append_env( temperature_root, d_quadNode );
  _particle_temperature_varlabel = VarLabel::find(temperature_name);

  // check for length
  std::string length_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_SIZE);
  std::string length_name = ArchesCore::append_env( length_root, d_quadNode );
  _length_varlabel = VarLabel::find(length_name);

  // create raw coal mass var label
  std::string rcmass_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_RAWCOAL);
  std::string rcmass_name = ArchesCore::append_env( rcmass_root, d_quadNode );
  _rcmass_varlabel = VarLabel::find(rcmass_name);

  // check for char mass and get scaling constant
  std::string char_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_CHAR);
  std::string char_name = ArchesCore::append_env( char_root, d_quadNode );
  _char_varlabel = VarLabel::find(char_name);

  // get weight and scaling constant
  std::string weightqn_name = ArchesCore::append_qn_env("w", d_quadNode);
  std::string weight_name = ArchesCore::append_env("w", d_quadNode);
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

  if (d_radiation ) {
    ProblemSpecP db_propV2 = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("PropertyModelsV2");
    bool foundParticleRadPropertyModel=false;
    for ( ProblemSpecP db_model = db_propV2->findBlock("model"); db_model != nullptr;
        db_model = db_model->findNextBlock("model")){
      db_model->getAttribute("type", modelName);
      if (modelName=="partRadProperties"){
        db_model->getAttribute("label",baseNameAbskp);

        foundParticleRadPropertyModel=true;
        break;
      }
    }
    if  (foundParticleRadPropertyModel== 0 && d_radiation){
      throw InvalidValue("ERROR: EnthalpyShaddix.cc can't find particle absorption coefficient model.",__FILE__,__LINE__);
    }


    if (VarLabel::find("radiationVolq")) {
      _volq_varlabel  = VarLabel::find("radiationVolq");
    }
    else {
      throw InvalidValue("ERROR: EnthalpyShaddix: problemSetup(): can't find radiationVolq.",__FILE__,__LINE__);
    }
    std::string abskp_string = ArchesCore::append_env(baseNameAbskp, d_quadNode);
    _abskp_varlabel = VarLabel::find(abskp_string);
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
  if (params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")) {
    ProblemSpecP db_coal = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties");
    std::string particleType;
    db_coal->getAttribute("type",particleType);
    if (particleType != "coal"){
      throw InvalidValue("ERROR: EnthalpyShaddix: Can't transport enthalpy of particles of type: "+particleType,__FILE__,__LINE__);
    }
    db_coal->require("raw_coal_enthalpy", _Hc0);
    db_coal->require("char_enthalpy", _Hh0);
    db_coal->require("density",_rhop_o);
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
    db_coal->require("diameter_distribution", _sizes);
    //double coal_daf = coal.C + coal.H + coal.O + coal.N + coal.S; //dry ash free coal
    double coal_dry = coal.C + coal.H + coal.O + coal.N + coal.S + coal.ASH + coal.CHAR; //moisture free coal
    //double raw_coal_mf = coal_daf / coal_dry;
    //double char_mf = coal.CHAR / coal_dry;
    double ash_mf = coal.ASH / coal_dry;
    _init_ash.clear();
    for ( unsigned int i = 0; i < _sizes.size(); i++ ){
      double mass_dry = (_pi/6.0) * std::pow(_sizes[i],3.0) * _rhop_o;     // kg/particle
      _init_ash.push_back(mass_dry  * ash_mf);                      // kg_ash/particle (initial)
    }
  } else {
    throw InvalidValue("ERROR: EnthalpyShaddix: problemSetup(): Missing <CoalProperties> section in input file!",__FILE__,__LINE__);
  }

  double MW [5] = { 12., 1., 14., 16., 32.}; // Atomic weight of elements (C,H,N,O,S) - kg/kmol
  _MW_avg = 0.0; // Mean atomic weight of coal
  for(int i=0;i<5;i++){
    _MW_avg += yelem[i]/MW[i];
  }
  _MW_avg = 1.0/_MW_avg;

  //_RdC = _Rgas/12.0107;
  _RdC = _Rgas/12.0;
  _RdMW = _Rgas/_MW_avg;
  _radiationOn = d_radiation;
  _nQuadNode=d_quadNode;

}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void
EnthalpyShaddix::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  string taskname = "EnthalpyShaddix::initVars";
  Task* tsk = scinew Task(taskname, this, &EnthalpyShaddix::initVars);

  tsk->computes(d_modelLabel);
  tsk->computes(d_gasLabel);
  tsk->computes(d_qconvLabel);
  tsk->computes(d_qradLabel);

  sched->addTask(tsk, level->eachPatch(), d_materialManager->allMaterials( "Arches" ));
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
  //patch loop
  for (int p=0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<double> heat_rate;
    CCVariable<double> gas_heat_rate;
    CCVariable<double> qconv;
    CCVariable<double> qrad;

    new_dw->allocateAndPut( heat_rate, d_modelLabel, matlIndex, patch );
    heat_rate.initialize(0.0);
    new_dw->allocateAndPut( gas_heat_rate, d_gasLabel, matlIndex, patch );
    gas_heat_rate.initialize(0.0);
    new_dw->allocateAndPut( qconv, d_qconvLabel, matlIndex, patch );
    qconv.initialize(0.0);
    new_dw->allocateAndPut( qrad, d_qradLabel, matlIndex, patch );
    qrad.initialize(0.0);

  }
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
  }
  else {
    tsk->modifies(d_modelLabel);
    tsk->modifies(d_gasLabel);
    tsk->modifies(d_qconvLabel);
    tsk->modifies(d_qradLabel);
    which_dw = Task::NewDW;
  }

  // require gas phase variables
  tsk->requires( which_dw, _gas_temperature_varlabel, Ghost::None, 0);
  tsk->requires( which_dw, _gas_cp_varlabel, Ghost::None, 0);
  tsk->requires( Task::NewDW, _devolgas_varlabel, Ghost::None, 0 );
  tsk->requires( Task::NewDW, _chargas_varlabel, Ghost::None, 0 );
  tsk->requires( which_dw, d_fieldLabels->d_CCVelocityLabel, Ghost::None, 0);
  tsk->requires( which_dw, d_fieldLabels->d_densityCPLabel, Ghost::None, 0);
  if ( d_radiation ){
    tsk->requires( which_dw, _volq_varlabel, Ghost::None, 0);

    tsk->requires( which_dw, _abskp_varlabel, Ghost::None, 0);
  }
  tsk->requires( Task::OldDW, d_fieldLabels->d_delTLabel);

  // require particle phase variables
  tsk->requires( which_dw, _rcmass_varlabel, gn, 0 );
  tsk->requires( which_dw, _char_varlabel, gn, 0 );
  tsk->requires( which_dw, _particle_temperature_varlabel, gn, 0 );
  tsk->requires( which_dw, _length_varlabel, gn, 0 );
  tsk->requires( which_dw, _weight_varlabel, gn, 0 );
  tsk->requires( Task::NewDW, _surfacerate_varlabel, Ghost::None, 0 );
  tsk->requires( Task::NewDW, _charoxiTemp_varlabel, Ghost::None, 0 );
  // require particle velocity
  ArchesLabel::PartVelMap::const_iterator i = d_fieldLabels->partVel.find(d_quadNode);
  tsk->requires( Task::NewDW, i->second, gn, 0 );

  sched->addTask(tsk, level->eachPatch(), d_materialManager->allMaterials( "Arches" ));
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
    int matlIndex = d_fieldLabels->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    delt_vartype DT;
    old_dw->get(DT, d_fieldLabels->d_delTLabel);
    double dt = DT;

    CCVariable<double> heat_rate;
    CCVariable<double> gas_heat_rate;
    CCVariable<double> qconv;
    CCVariable<double> qrad;
    DataWarehouse* which_dw;
    
    if ( timeSubStep == 0 ){
      which_dw = old_dw;
      new_dw->allocateAndPut( heat_rate,     d_modelLabel, matlIndex, patch );
      new_dw->allocateAndPut( gas_heat_rate, d_gasLabel,   matlIndex, patch );
      new_dw->allocateAndPut( qconv,         d_qconvLabel, matlIndex, patch );
      new_dw->allocateAndPut( qrad,          d_qradLabel,  matlIndex, patch );
      
      heat_rate.initialize(0.0);
      gas_heat_rate.initialize(0.0);
      qconv.initialize(0.0);
      qrad.initialize(0.0);
    }
    else {
      which_dw = new_dw;
      new_dw->getModifiable( heat_rate,     d_modelLabel, matlIndex, patch );
      new_dw->getModifiable( gas_heat_rate, d_gasLabel,   matlIndex, patch );
      new_dw->getModifiable( qconv,         d_qconvLabel, matlIndex, patch );
      new_dw->getModifiable( qrad,          d_qradLabel,  matlIndex, patch );
    }

    // get gas phase variables
    constCCVariable<double> temperature;
    constCCVariable<double> specific_heat;
    which_dw->get( temperature,   _gas_temperature_varlabel, matlIndex, patch, gn, 0 );
    which_dw->get( specific_heat, _gas_cp_varlabel,           matlIndex, patch, gn, 0 );  // in J/kg/K
    
    constCCVariable<double> radiationVolqIN;
    constCCVariable<double> abskp;
    constCCVariable<double> rad_particle_temperature;
    
    if ( d_radiation ){
      which_dw->get( radiationVolqIN, _volq_varlabel,  matlIndex, patch, gn, 0);
      which_dw->get( abskp,           _abskp_varlabel, matlIndex, patch, gn, 0);
      
      if (_radiateAtGasTemp){
        which_dw->get( rad_particle_temperature, _gas_temperature_varlabel, matlIndex, patch, gn, 0 );
      }else{
        which_dw->get( rad_particle_temperature, _particle_temperature_varlabel, matlIndex, patch, gn, 0 );
      }
    }
    
    constCCVariable<Vector> gasVel;
    constCCVariable<double> den;
    constCCVariable<double> devol_gas_source;
    constCCVariable<double> chargas_source;
    
    which_dw->get( gasVel,        d_fieldLabels->d_CCVelocityLabel, matlIndex, patch, gn, 0 );
    which_dw->get( den,           d_fieldLabels->d_densityCPLabel,  matlIndex, patch, gn, 0 );
    new_dw->get( devol_gas_source, _devolgas_varlabel, matlIndex, patch, gn, 0 );
    new_dw->get( chargas_source,    _chargas_varlabel, matlIndex, patch, gn, 0 );

    // get particle phase variables
    constCCVariable<double> length;
    constCCVariable<double> weight;
    constCCVariable<double> rawcoal_mass;
    constCCVariable<double> char_mass;
    constCCVariable<double> particle_temperature;
    
    which_dw->get( length,        _length_varlabel, matlIndex, patch, gn, 0 );
    which_dw->get( weight,        _weight_varlabel, matlIndex, patch, gn, 0 );
    which_dw->get( rawcoal_mass,  _rcmass_varlabel, matlIndex, patch, gn, 0 );
    which_dw->get( char_mass,     _char_varlabel,   matlIndex, patch, gn, 0 );
    which_dw->get( particle_temperature, _particle_temperature_varlabel, matlIndex, patch, gn, 0 );

    constCCVariable<double> charoxi_temp_source;
    constCCVariable<double> surface_rate;
    constCCVariable<Vector> partVel;
    
    new_dw->get( charoxi_temp_source, _charoxiTemp_varlabel, matlIndex, patch, gn, 0 );
    new_dw->get( surface_rate,        _surfacerate_varlabel, matlIndex, patch, gn, 0 );
    
    ArchesLabel::PartVelMap::const_iterator iter = d_fieldLabels->partVel.find(d_quadNode);
    new_dw->get(partVel, iter->second, matlIndex, patch, gn, 0);
    //______________________________________________________________________
    //
    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
    
    Uintah::parallel_for( range, [&](int i, int j, int k) {
      double heatRate;
      double gas_heatRate;
      double Q_convection;
      double Q_radiation;
      double Q_reaction;

      if (weight(i,j,k)/_weight_scaling_constant < _weight_small) {
        heatRate     = 0.0;
        gas_heatRate = 0.0;
        Q_convection = 0.0;
        Q_radiation  = 0.0;
      } else {

        double rawcoal_massph = rawcoal_mass(i,j,k);
        double char_massph = char_mass(i,j,k);
        double temperatureph = temperature(i,j,k);
        double specific_heatph = specific_heat(i,j,k);
        double denph = den(i,j,k);
        double devol_gas_sourceph = devol_gas_source(i,j,k);
        double chargas_sourceph = chargas_source(i,j,k);
        double lengthph = length(i,j,k);
        double weightph = weight(i,j,k);
        double particle_temperatureph = particle_temperature(i,j,k);
        double charoxi_temp_sourceph = charoxi_temp_source(i,j,k);
        double surface_rateph = surface_rate(i,j,k);

        // velocities
        Vector gas_velocity      = gasVel(i,j,k);
        Vector particle_velocity = partVel(i,j,k);

        // intermediate calculation values
      
        // Convection part: -----------------------
        // Reynolds number
        double delta_V =sqrt( std::pow(gas_velocity.x() - particle_velocity.x(),2.0) + 
                              std::pow(gas_velocity.y() - particle_velocity.y(),2.0) + 
                              std::pow(gas_velocity.z() - particle_velocity.z(),2.0));
                              
        double Re = delta_V * lengthph * denph/_visc;

        // Nusselt number
        double Nu = 2.0 + 0.65*std::pow(Re,0.50)*std::pow(_Pr,(1.0/3.0));

        // Gas thermal conductivity
        double rkg = props(temperatureph, particle_temperatureph); // [=] J/s/m/K

        // A BLOWING CORRECTION TO THE HEAT TRANSFER MODEL IS EMPLOYED
        double kappa =  -surface_rateph*lengthph*specific_heatph/(2.0*rkg);

        double blow;
        if(std::abs(exp(kappa)-1.0) < 1e-16){
          blow = 1.0;
        } else {
          blow = kappa/(exp(kappa)-1.0);
        }

        Q_convection = Nu * _pi * blow * rkg * lengthph * (temperatureph - particle_temperatureph); // J/(#.s)
        //clip convection term if timesteps are too large
        double deltaT   = temperatureph - particle_temperatureph;
        double alpha_rc = (rawcoal_massph + char_massph);
        double alpha_cp = cp_c(particle_temperatureph) * alpha_rc + cp_ash(particle_temperatureph) * _init_ash[_nQuadNode];
        double max_Q_convection= alpha_cp*(deltaT/dt);
        
        if (std::abs(Q_convection) > std::abs(max_Q_convection)){
          Q_convection = max_Q_convection;
        }
        
        Q_convection = Q_convection*weightph;
        
        // Radiation part: -------------------------
        Q_radiation = 0.0;
        
        if ( _radiationOn) {

          double Eb   = 4.0*_sigma*std::pow( rad_particle_temperature(i,j,k), 4.0);
          double FSum = radiationVolqIN(i,j,k);
          Q_radiation = abskp(i,j,k)*(FSum - Eb);
          
          double Q_radMax=(std::pow( radiationVolqIN(i,j,k) / (4.0 * _sigma ), 0.25) - rad_particle_temperature(i,j,k))/(dt)*alpha_cp*weightph ;
          
          if (std::abs(Q_radMax) < std::abs(Q_radiation)){
            Q_radiation=Q_radMax;
          }
        }
        
        double hint = -156.076 + 380/(-1 + exp(380 / particle_temperatureph)) + 3600/(-1 + exp(1800 / particle_temperatureph));
        double hc = _Hc0 + hint * _RdMW;
        Q_reaction = charoxi_temp_sourceph;
        
        // This needs to be made consistant with lagrangian particles!!! - derek 12/14
        heatRate = (Q_convection + Q_radiation + _ksi*Q_reaction - (devol_gas_sourceph + chargas_sourceph)*hc)/
                     (_enthalpy_scaling_constant*_weight_scaling_constant);
        
        gas_heatRate = -Q_convection - Q_radiation - _ksi*Q_reaction + (devol_gas_sourceph+chargas_sourceph)*hc;
      }
      
      
      heat_rate(i,j,k)     = heatRate;
      gas_heat_rate(i,j,k) = gas_heatRate;
      qconv(i,j,k)         = Q_convection; // W/m^3
      qrad(i,j,k)          = Q_radiation;  // W/m^3
    } );


  }//end patch loop

}



// ********************************************************
// Private methods:

double
EnthalpyShaddix::g2( double z ){
  double sol = exp(z)/std::pow((exp(z)-1.0)/z,2.0);
  return sol;
}

double
EnthalpyShaddix::cp_c( double Tp){
  double z1 = 380.0/Tp;
  double z2 = 1800.0/Tp;
  double cp = (_RdMW)*(g2(z1)+2*g2(z2));
  return cp;
}
double
EnthalpyShaddix::cp_ash( double Tp){
  double cp = 754.0 + 0.586*Tp;
  return cp;
}
double
EnthalpyShaddix::cp_h( double Tp){
  double z1 = 380.0/Tp;
  double z2 = 1800.0/Tp;
  double cp = (_RdC)*(g2(z1)+2*g2(z2));
  return cp;
}


double
EnthalpyShaddix::props(double Tg, double Tp){

  double tg0[10] = {300.,  400.,   500.,   600.,  700.,  800.,  900.,  1000., 1100., 1200. };
  double kg0[10] = {.0262, .03335, .03984, .0458, .0512, .0561, .0607, .0648, .0685, .07184};
  double T = (Tp+Tg)/2; // Film temperature

//   CALCULATE UG AND KG FROM INTERPOLATION OF TABLE VALUES FROM HOLMAN
//   FIND INTERVAL WHERE TEMPERATURE LIES.

  double kg = 0.0;

  if( T > 1200.0 ) {
    kg = kg0[9] * std::pow( T/tg0[9], 0.58);

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
