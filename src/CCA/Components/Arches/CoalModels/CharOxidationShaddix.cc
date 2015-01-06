#include <CCA/Components/Arches/CoalModels/CharOxidationShaddix.h>
#include <CCA/Components/Arches/CoalModels/Devolatilization.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/PropertyModelsV2/PropertyHelper.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>
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
CharOxidationShaddixBuilder::CharOxidationShaddixBuilder( const std::string         & modelName,
                                                      const vector<std::string> & reqICLabelNames,
                                                      const vector<std::string> & reqScalarLabelNames,
                                                      ArchesLabel         * fieldLabels,
                                                      SimulationStateP          & sharedState,
                                                      int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{
}

CharOxidationShaddixBuilder::~CharOxidationShaddixBuilder(){}

ModelBase* CharOxidationShaddixBuilder::build() {
  return scinew CharOxidationShaddix( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

CharOxidationShaddix::CharOxidationShaddix( std::string modelName, 
                                        SimulationStateP& sharedState,
                                        ArchesLabel* fieldLabels,
                                        vector<std::string> icLabelNames, 
                                        vector<std::string> scalarLabelNames,
                                        int qn ) 
: CharOxidation(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  // Set constants
  _pi = acos(-1.0);
  _R = 8.314; // J/K/mol
  _WC = 12.0e-3;  //kg/mol
  _WO2 = 32.0; //g/mol
  _WCO2 = 44.0; //g/mol
  _WH2O = 18.0; //g/mol
  _WN2 = 28.0; //g/mol
}

CharOxidationShaddix::~CharOxidationShaddix()
{
}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
CharOxidationShaddix::problemSetup(const ProblemSpecP& params, int qn)
{

  ProblemSpecP db = params;
  const ProblemSpecP params_root = db->getRootNode();

  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();
  ProblemSpecP db_coal_props = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal")->findBlock("Properties");
  
  // create raw coal mass var label 
  std::string rcmass_root = PropertyHelper::parse_for_role_to_label(db, "raw_coal"); 
  std::string rcmass_name = PropertyHelper::append_env( rcmass_root, d_quadNode ); 
  _rcmass_varlabel = VarLabel::find(rcmass_name);
 
  // check for char mass and get scaling constant
  std::string char_root = PropertyHelper::parse_for_role_to_label(db, "char"); 
  std::string char_name = PropertyHelper::append_env( char_root, d_quadNode ); 
  std::string charqn_name = PropertyHelper::append_qn_env( char_root, d_quadNode ); 
  _char_varlabel = VarLabel::find(char_name);
  EqnBase& temp_char_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(charqn_name);
  DQMOMEqn& char_eqn = dynamic_cast<DQMOMEqn&>(temp_char_eqn);
  _char_scaling_constant = char_eqn.getScalingConstant();

  // check for particle temperature 
  std::string temperature_root = PropertyHelper::parse_for_role_to_label(db, "temperature"); 
  std::string temperature_name = PropertyHelper::append_env( temperature_root, d_quadNode ); 
  _particle_temperature_varlabel = VarLabel::find(temperature_name);
  if(_particle_temperature_varlabel == 0){
    throw ProblemSetupException("Error: Unable to find coal temperature label!!!! Looking for name: "+temperature_name, __FILE__, __LINE__); 
  }
 
  // check for length  
  std::string length_root = PropertyHelper::parse_for_role_to_label(db, "size"); 
  std::string length_name = PropertyHelper::append_env( length_root, d_quadNode ); 
  _length_varlabel = VarLabel::find(length_name);
  
  // get model coefficients  
  if (db_coal_props->findBlock("Shaddix_char_coefficients")) {
    db_coal_props->require("Shaddix_char_coefficients", Shaddix_char_coefficients);
   
    _D1 = Shaddix_char_coefficients[0]; // Binary diffusion coef. O2/CO2
    _D2 = Shaddix_char_coefficients[1];  // Binary diffusion coef. O2/H2O
    _D3 = Shaddix_char_coefficients[2]; // Binary diffusion coef. O2/N2
    _T0 = Shaddix_char_coefficients[3];
    // Eastern bituminous coal, non-linear regression
    _As = Shaddix_char_coefficients[4];  // mol/s.m^2.atm^n
    _Es = Shaddix_char_coefficients[5]; // J/mol
    _n = Shaddix_char_coefficients[6];
    // Enthalpy of formation (J/mol)
    _HF_CO2 = Shaddix_char_coefficients[7];
    _HF_CO  = Shaddix_char_coefficients[8];
  } else { 
    throw ProblemSetupException("Error: Shaddix_char_coefficients missing in <CoalProperties>.", __FILE__, __LINE__); 
  }

  // get weight scaling constant
  std::string weightqn_name = PropertyHelper::append_qn_env("w", d_quadNode); 
  std::string weight_name = PropertyHelper::append_env("w", d_quadNode); 
  _weight_varlabel = VarLabel::find(weight_name); 
  EqnBase& temp_weight_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(weightqn_name);
  DQMOMEqn& weight_eqn = dynamic_cast<DQMOMEqn&>(temp_weight_eqn);
  _weight_small = weight_eqn.getSmallClipCriteria();
  _weight_scaling_constant = weight_eqn.getScalingConstant();

  // get Char source term label from the devolatilization model
  CoalModelFactory& modelFactory = CoalModelFactory::self(); 
  DevolModelMap devolmodels_ = modelFactory.retrieve_devol_models();
  for( DevolModelMap::iterator iModel = devolmodels_.begin(); iModel != devolmodels_.end(); ++iModel ) {
    int modelNode = iModel->second->getquadNode();
    if( modelNode == d_quadNode) {
      _devolCharLabel = iModel->second->getCharSourceLabel();
    }
  }

  // get gas phase temperature label 
  if (VarLabel::find("temperature")) {
    _gas_temperature_varlabel = VarLabel::find("temperature");
  } else {
    throw InvalidValue("ERROR: CharOxidationShaddix: problemSetup(): can't find gas phase temperature.",__FILE__,__LINE__);
  }
  // get gas phase O2 label 
  if (VarLabel::find("O2")) {
    _O2_varlabel = VarLabel::find("O2");
  } else {
    throw InvalidValue("ERROR: CharOxidationShaddix: problemSetup(): can't find gas phase O2.",__FILE__,__LINE__);
  }
  // get gas phase CO2 label 
  if (VarLabel::find("CO2")) {
    _CO2_varlabel = VarLabel::find("CO2");
  } else {
    throw InvalidValue("ERROR: CharOxidationShaddix: problemSetup(): can't find gas phase CO2.",__FILE__,__LINE__);
  }
  // get gas phase H2O label 
  if (VarLabel::find("H2O")) {
    _H2O_varlabel = VarLabel::find("H2O");
  } else {
    throw InvalidValue("ERROR: CharOxidationShaddix: problemSetup(): can't find gas phase H2O.",__FILE__,__LINE__);
  }
  // get gas phase N2 label 
  if (VarLabel::find("N2")) {
    _N2_varlabel = VarLabel::find("N2");
  } else {
    throw InvalidValue("ERROR: CharOxidationShaddix: problemSetup(): can't find gas phase N2.",__FILE__,__LINE__);
  }
  // get gas phase mixture_molecular_weight label 
  if (VarLabel::find("mixture_molecular_weight")) {
    _MW_varlabel = VarLabel::find("mixture_molecular_weight");
  } else {
    throw InvalidValue("ERROR: CharOxidationShaddix: problemSetup(): can't find gas phase mixture_molecular_weight.",__FILE__,__LINE__);
  }

}


//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void 
CharOxidationShaddix::sched_initVars( const LevelP& level, SchedulerP& sched )
{
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
CharOxidationShaddix::initVars( const ProcessorGroup * pc, 
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
CharOxidationShaddix::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "CharOxidationShaddix::sched_computeModel";
  Task* tsk = scinew Task(taskname, this, &CharOxidationShaddix::computeModel, timeSubStep );

  Ghost::GhostType  gn  = Ghost::None;

  Task::WhichDW which_dw; 

  if (timeSubStep == 0 ) { 
    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel);
    tsk->computes(d_particletempLabel);
    tsk->computes(d_surfacerateLabel);
    tsk->computes(d_PO2surfLabel);
    which_dw = Task::OldDW; 
  } else {
    tsk->modifies(d_modelLabel);
    tsk->modifies(d_gasLabel);  
    tsk->modifies(d_particletempLabel);
    tsk->modifies(d_surfacerateLabel);
    tsk->modifies(d_PO2surfLabel);
    which_dw = Task::NewDW; 
  }
  tsk->requires( which_dw, _particle_temperature_varlabel, gn, 0 ); 
  tsk->requires( which_dw, _rcmass_varlabel, gn, 0 ); 
  tsk->requires( which_dw, _char_varlabel, gn, 0 );
  tsk->requires( which_dw, _length_varlabel, gn, 0 ); 
  tsk->requires( which_dw, _weight_varlabel, gn, 0 ); 
  tsk->requires( which_dw, _gas_temperature_varlabel, gn, 0);
  tsk->requires( which_dw, _O2_varlabel, gn, 0 );
  tsk->requires( which_dw, _CO2_varlabel, gn, 0 );
  tsk->requires( which_dw, _H2O_varlabel, gn, 0 );
  tsk->requires( which_dw, _N2_varlabel, gn, 0 );
  tsk->requires( which_dw, _MW_varlabel, gn, 0 );
  tsk->requires( Task::OldDW, d_fieldLabels->d_sharedState->get_delt_label()); 
  
  tsk->requires( which_dw, d_fieldLabels->d_densityCPLabel, gn, 0);
  tsk->requires( which_dw, _devolCharLabel, gn, 0);

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}

//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
CharOxidationShaddix::computeModel( const ProcessorGroup * pc, 
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

    delt_vartype DT;
    old_dw->get(DT, d_fieldLabels->d_sharedState->get_delt_label());
    double dt = DT;

    CCVariable<double> char_rate;
    CCVariable<double> gas_char_rate; 
    CCVariable<double> particle_temp_rate;
    CCVariable<double> surface_rate;
    CCVariable<double> PO2surf_;
     
    DataWarehouse* which_dw; 
    if ( timeSubStep == 0 ){ 
      which_dw = old_dw;
      new_dw->allocateAndPut( char_rate, d_modelLabel, matlIndex, patch );
      char_rate.initialize(0.0);
      new_dw->allocateAndPut( gas_char_rate, d_gasLabel, matlIndex, patch );
      gas_char_rate.initialize(0.0);
      new_dw->allocateAndPut( particle_temp_rate, d_particletempLabel, matlIndex, patch );
      particle_temp_rate.initialize(0.0);
      new_dw->allocateAndPut(surface_rate, d_surfacerateLabel, matlIndex, patch );
      surface_rate.initialize(0.0);
      new_dw->allocateAndPut(PO2surf_, d_PO2surfLabel, matlIndex, patch );
      PO2surf_.initialize(0.0);
    } else { 
      which_dw = new_dw;
      new_dw->getModifiable( char_rate, d_modelLabel, matlIndex, patch ); 
      new_dw->getModifiable( gas_char_rate, d_gasLabel, matlIndex, patch ); 
      new_dw->getModifiable( particle_temp_rate, d_particletempLabel, matlIndex, patch );
      new_dw->getModifiable( surface_rate, d_surfacerateLabel, matlIndex, patch );
      new_dw->getModifiable( PO2surf_, d_PO2surfLabel, matlIndex, patch );
    }

    constCCVariable<double> den;
    which_dw->get( den, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0 ); 
    constCCVariable<double> temperature;
    which_dw->get( temperature , _gas_temperature_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> particle_temperature;
    which_dw->get( particle_temperature , _particle_temperature_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> length;
    which_dw->get( length, _length_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> rawcoal_mass;
    which_dw->get( rawcoal_mass, _rcmass_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> char_mass;
    which_dw->get( char_mass, _char_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> weight;
    which_dw->get( weight, _weight_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> O2;
    which_dw->get( O2, _O2_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> CO2;
    which_dw->get( CO2, _CO2_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> H2O;
    which_dw->get( H2O, _H2O_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> N2;
    which_dw->get( N2, _N2_varlabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> MWmix;
    which_dw->get( MWmix, _MW_varlabel, matlIndex, patch, gn, 0 );  // in kmol/kg_mix ?
    constCCVariable<double> devolChar;
    which_dw->get( devolChar, _devolCharLabel, matlIndex, patch, gn, 0 );
    
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 
   
      if (weight[c]/_weight_scaling_constant < _weight_small) {
        char_production_rate_ = devolChar[c];
        char_rate[c] = char_production_rate_/(_char_scaling_constant*_weight_scaling_constant);
        gas_char_rate[c] = 0.0;
        particle_temp_rate[c] = 0.0;
        surface_rate[c] = 0.0;
      } else {
        double small = 1e-16;

        double denph=den[c];
        double temperatureph=temperature[c];
        double particle_temperatureph=particle_temperature[c];
        double lengthph=length[c];
        double rawcoal_massph=rawcoal_mass[c];
        double char_massph=char_mass[c];
        double weightph=weight[c];
        double O2ph=O2[c];
        double CO2ph=CO2[c];
        double H2Oph=H2O[c];
        double N2ph=N2[c];
        double MWmixph=MWmix[c];
        double devolCharph=devolChar[c];
        
        //VERIFICATION
        //denph=0.394622;
        //temperatureph=1206.4;
        //particle_temperatureph=530.79;
        //lengthph=2e-5;
        //rawcoal_massph=4.99019e-12;
        //char_massph=-7.1393e-18;
        //weightph=1.40781e+09;
        //O2ph=0.0146663;
        //CO2ph=0.898772;
        //H2Oph=0.00334152;
        //N2ph=0.00186698;
        //MWmixph=0.025599;
        //devolCharph=5.67408e-08;


        PO2_inf = O2ph/_WO2/MWmixph;
        if((PO2_inf < 1e-10) || ((rawcoal_massph+char_massph) < small)) {
          PO2_surf = 0.0;
          CO2CO = 0.0;
          q = 0.0;
        } else {
          char_reaction_rate_ = 0.0;
          char_production_rate_ = 0.0;
          particle_temp_rate_ = 0.0;
          PO2_surf = PO2_inf;
          d_totIter = 8;
          NIter = 3;
          delta = PO2_inf/30.0;
          d_tol = 1e-15;
          f1 = 1.0;
          icount = 0;
          // Calculate O2 diffusion coefficient
          DO2 = (CO2ph/_WCO2 + H2Oph/_WH2O + N2ph/_WN2)/(CO2ph/(_WCO2*_D1) + H2Oph/(_WH2O*_D2) + N2ph/(_WN2*_D3))*
                (pow((temperatureph/_T0),1.5));
          // Concentration C = P/RT
          Conc = MWmixph*denph*1000.0;

          // Solving diffusion of O2:
          // Newton_Raphson method - faster does not always converge 
          for ( int iter = 0; iter < NIter; iter++) {
            icount++;
            CO2CO = 0.02*(pow(PO2_surf,0.21))*exp(-3070.0/particle_temperatureph);
            OF = 0.5*(1.0 + CO2CO*(1+CO2CO));
            gamma = -(1.0-OF);
            ks = _As*exp(-_Es/(_R*particle_temperatureph));
            q = ks*(pow(PO2_surf,_n));
            f1 = PO2_surf - gamma - (PO2_inf-gamma)*exp(-(q*lengthph)/(2*Conc*DO2));
            if (std::abs(f1) < d_tol)
              break;
            PO2_surf += delta;
            CO2CO = 0.02*(pow(PO2_surf,0.21))*exp(-3070.0/particle_temperatureph);
            OF = 0.5*(1.0 + CO2CO*(1+CO2CO));
            gamma = -(1.0-OF);
            ks = _As*exp(-_Es/(_R*particle_temperatureph));
            q = ks*(pow(PO2_surf,_n));
            f2 = PO2_surf - gamma - (PO2_inf-gamma)*exp(-(q*lengthph)/(2*Conc*DO2));
            PO2_surf -= delta + f1*delta/(f2-f1);
            PO2_surf = min(PO2_inf,max(0.0,PO2_surf));
          }
          
          if(std::abs(f1) > d_tol || isnan(f1)){ //switching to bisection technique
            lower_bound = 0.0;
            upper_bound = PO2_inf;
            for ( int iter = 0; iter < d_totIter; iter++) {
              icount++;
              PO2_surf = lower_bound;
              CO2CO = 0.02*(pow(PO2_surf,0.21))*exp(-3070.0/particle_temperatureph);
              OF = 0.5*(1.0 + CO2CO*(1+CO2CO));
              gamma = -(1.0-OF);
              ks = _As*exp(-_Es/(_R*particle_temperatureph));
              q = ks*(pow(PO2_surf,_n));
              f3 = PO2_surf - gamma - (PO2_inf-gamma)*exp(-(q*lengthph)/(2*Conc*DO2));
              PO2_surf = upper_bound;
              CO2CO = 0.02*(pow(PO2_surf,0.21))*exp(-3070.0/particle_temperatureph);
              OF = 0.5*(1.0 + CO2CO*(1+CO2CO));
              gamma = -(1.0-OF);
              ks = _As*exp(-_Es/(_R*particle_temperatureph));
              q = ks*(pow(PO2_surf,_n));
              f2 = PO2_surf - gamma - (PO2_inf-gamma)*exp(-(q*lengthph)/(2*Conc*DO2));
              if (std::abs(f2) < d_tol){
                break;
              }
              PO2_surf = 0.5*(lower_bound+upper_bound);
              CO2CO = 0.02*(pow(PO2_surf,0.21))*exp(-3070.0/particle_temperatureph);
              OF = 0.5*(1.0 + CO2CO*(1+CO2CO));
              gamma = -(1.0-OF);
              ks = _As*exp(-_Es/(_R*particle_temperatureph));
              q = ks*(pow(PO2_surf,_n));
              f1 = PO2_surf - gamma - (PO2_inf-gamma)*exp(-(q*lengthph)/(2*Conc*DO2));

              if (std::abs(f1) < d_tol){
                break;
              }
              if(icount > d_totIter+NIter-1) {
                PO2_surf = PO2_inf*2.0/3.0;;
                CO2CO = 0.02*(pow(PO2_surf,0.21))*exp(-3070.0/particle_temperatureph);
                OF = 0.5*(1.0 + CO2CO*(1+CO2CO));
                gamma = -(1.0-OF);
                ks = _As*exp(-_Es/(_R*particle_temperatureph));
                q = ks*(pow(PO2_surf,_n));
                break;
              }

              if(f2*f1<0){
                 lower_bound = PO2_surf;
              } else if(f1*f3<0){
                 upper_bound = PO2_surf;
              } else {
                PO2_surf = PO2_inf*2.0/3.0;
                CO2CO = 0.02*(pow(PO2_surf,0.21))*exp(-3070.0/particle_temperatureph);
                OF = 0.5*(1.0 + CO2CO*(1+CO2CO));
                gamma = -(1.0-OF);
                ks = _As*exp(-_Es/(_R*particle_temperatureph));
                q = ks*(pow(PO2_surf,_n));
                break;
              }
            }

          }              
        }

        char_production_rate_ = devolCharph;
        char_reaction_rate_ = - 1.0 * min( _pi*( pow(lengthph,2.0))*_WC*q , char_massph+rawcoal_massph+char_production_rate_/weightph*dt );
        particle_temp_rate_ = char_reaction_rate_/_WC/(1.0+CO2CO)*(CO2CO*_HF_CO2 + _HF_CO); // in J/(s.#)
          
        char_rate[c] = (char_reaction_rate_*weightph + char_production_rate_)/(_char_scaling_constant*_weight_scaling_constant);
        gas_char_rate[c] = -char_reaction_rate_*weightph;
        particle_temp_rate[c] = particle_temp_rate_*weightph;
        surface_rate[c] = _WC*q;  // in kg/s/m^2
        PO2surf_[c] = PO2_surf;
      }
    }//end cell loop
  }//end patch loop
}
