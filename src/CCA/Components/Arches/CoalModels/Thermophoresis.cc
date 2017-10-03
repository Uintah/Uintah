#include <CCA/Components/Arches/CoalModels/Thermophoresis.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/Directives.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Parallel/Parallel.h>

#include <boost/math/special_functions/erf.hpp>
//===========================================================================

using namespace std;
using namespace Uintah;

//---------------------------------------------------------------------------
// Builder:
ThermophoresisBuilder::ThermophoresisBuilder( const std::string         & modelName,
                                                            const vector<std::string> & reqICLabelNames,
                                                            const vector<std::string> & reqScalarLabelNames,
                                                            ArchesLabel         * fieldLabels,
                                                            SimulationStateP          & sharedState,
                                                            int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{
}

ThermophoresisBuilder::~ThermophoresisBuilder(){}

ModelBase* ThermophoresisBuilder::build() {
  return scinew Thermophoresis( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

Thermophoresis::Thermophoresis( std::string modelName,
                                              SimulationStateP& sharedState,
                                              ArchesLabel* fieldLabels,
                                              vector<std::string> icLabelNames,
                                              vector<std::string> scalarLabelNames,
                                              int qn )
: ModelBase(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  // Create a label for this model
  d_modelLabel = VarLabel::create( modelName, CCVariable<double>::getTypeDescription() );
  // Create the gas phase source term associated with this model
  std::string gasSourceName = modelName + "_gasSource";
  d_gasLabel = VarLabel::create( gasSourceName, CCVariable<double>::getTypeDescription() );
  //constants
  _pi = acos(-1.0);
  _C_tm = 0.461; // [=] m/s/Kelvin
  _C_t = 3.32; // [=] dimensionless
  _C_m = 1.19; // [=] dimensionless
  _Adep = 2.4; // [=] dimensionless
}

Thermophoresis::~Thermophoresis()
{
}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
  void
Thermophoresis::problemSetup(const ProblemSpecP& params, int qn)
{

  ProblemSpecP db = params;
  const ProblemSpecP params_root = db->getRootNode();

  std::string coord;
  db->require("direction",coord);

  if ( coord == "x" || coord == "X" ){
    _cell_minus = IntVector(-1,0,0);
    _cell_plus = IntVector(1,0,0);
    _dir = 0;
  } else if ( coord == "y" || coord == "Y" ){
    _cell_minus = IntVector(0,-1,0);
    _cell_plus = IntVector(0,1,0);
    _dir = 1;
  } else {
    _cell_minus = IntVector(0,0,-1);
    _cell_plus = IntVector(0,0,1);
    _dir = 2;
  }

  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();

  ProblemSpecP db_coal_props = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties");

  // Need velocity scaling constant
  std::string vel_root;
  if ( _dir == 0 ){
    vel_root = ParticleTools::parse_for_role_to_label(db, "uvel");
  } else if ( _dir == 1){
    vel_root = ParticleTools::parse_for_role_to_label(db, "vvel");
  } else {
    vel_root = ParticleTools::parse_for_role_to_label(db, "wvel");
  }

  vel_root = ParticleTools::append_qn_env( vel_root, d_quadNode );
  EqnBase& temp_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(vel_root);
  DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(temp_current_eqn);
  _vel_scaling_constant = current_eqn.getScalingConstant(d_quadNode);

  // Need a size IC:
  std::string length_root = ParticleTools::parse_for_role_to_label(db, "size");
  std::string length_name = ParticleTools::append_env( length_root, d_quadNode );
  _length_varlabel = VarLabel::find(length_name);

  // Need a density
  std::string density_root = ParticleTools::parse_for_role_to_label(db, "density");
  std::string density_name = ParticleTools::append_env( density_root, d_quadNode );
  _particle_density_varlabel = VarLabel::find(density_name);

  // create particle temperature label
  std::string temperature_root = ParticleTools::parse_for_role_to_label(db, "temperature");
  std::string temperature_name = ParticleTools::append_env( temperature_root, d_quadNode );
  _particle_temperature_varlabel = VarLabel::find(temperature_name);

  // get weight scaling constant
  std::string weightqn_name = ParticleTools::append_qn_env("w", d_quadNode);
  std::string weight_name = ParticleTools::append_env("w", d_quadNode);
  _weight_scaled_varlabel = VarLabel::find(weightqn_name);
  EqnBase& temp_weight_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(weightqn_name);
  DQMOMEqn& weight_eqn = dynamic_cast<DQMOMEqn&>(temp_weight_eqn);
  _weight_small = weight_eqn.getSmallClipPlusTol();
  _weight_scaling_constant = weight_eqn.getScalingConstant(d_quadNode);

  std::string vol_frac= "volFraction";
  _volFraction_varlabel = VarLabel::find(vol_frac);

  // get gas phase temperature label
  if (VarLabel::find("temperature")) {
    _gas_temperature_varlabel = VarLabel::find("temperature");
  } else {
    throw InvalidValue("ERROR: Thermophoresis: problemSetup(): can't find gas phase temperature.",__FILE__,__LINE__);
  }

  // get coal thermoconductivity
  if (db_coal_props->findBlock("particle_thermal_conductivity")) {
    db_coal_props->getWithDefault("particle_thermal_conductivity",_rkp,0.5);
  } else {
    throw ProblemSetupException("Error: Thermophoresis - particle_thermal_conductivity missing in <ParticleProperties>.", __FILE__, __LINE__);
  }


  if (params_root->findBlock("PhysicalConstants")) {
    ProblemSpecP db_phys = params_root->findBlock("PhysicalConstants");
    db_phys->require("viscosity", _visc);
    if( _visc == 0 ) {
      throw InvalidValue("ERROR: Thermophoresis: problemSetup(): Zero viscosity specified in <PhysicalConstants> section of input file.",__FILE__,__LINE__);
    }
  } else {
    throw InvalidValue("ERROR: Thermophoresis: problemSetup(): Missing <PhysicalConstants> section in input file!",__FILE__,__LINE__);
  }

}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void
Thermophoresis::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  string taskname = "Thermophoresis::initVars";
  Task* tsk = scinew Task(taskname, this, &Thermophoresis::initVars);

  tsk->computes(d_modelLabel);

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
Thermophoresis::initVars( const ProcessorGroup * pc,
                              const PatchSubset    * patches,
                              const MaterialSubset * matls,
                              DataWarehouse        * old_dw,
                              DataWarehouse        * new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    CCVariable<double> thp_rate;

    new_dw->allocateAndPut( thp_rate, d_modelLabel, matlIndex, patch );
    thp_rate.initialize(0.0);

  }
}

//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model
//---------------------------------------------------------------------------
void
Thermophoresis::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "Thermophoresis::computeModel";
  Task* tsk = scinew Task(taskname, this, &Thermophoresis::computeModel, timeSubStep);

  Ghost::GhostType gn = Ghost::None;
  Ghost::GhostType  gac = Ghost::AroundCells;

  Task::WhichDW which_dw;

  if (timeSubStep == 0 ) {
    tsk->computes(d_modelLabel);
    which_dw = Task::OldDW;
  } else {
    tsk->modifies(d_modelLabel);
    which_dw = Task::NewDW;
  }
  tsk->requires( which_dw, _particle_temperature_varlabel, gn, 0 );
  tsk->requires( which_dw, _particle_density_varlabel, gn, 0 );
  tsk->requires( Task::OldDW, _volFraction_varlabel, gac, 1 );
  tsk->requires( which_dw, _gas_temperature_varlabel, gac, 1 );
  tsk->requires( which_dw, _length_varlabel, gn, 0 );
  tsk->requires( which_dw, _weight_scaled_varlabel, gn, 0 );
  tsk->requires( which_dw, d_fieldLabels->d_densityCPLabel, gn, 0 );

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());

}

//---------------------------------------------------------------------------
// Method: Actually compute the source term
//---------------------------------------------------------------------------
void
Thermophoresis::computeModel( const ProcessorGroup * pc,
                                     const PatchSubset    * patches,
                                     const MaterialSubset * matls,
                                     DataWarehouse        * old_dw,
                                     DataWarehouse        * new_dw,
                                     const int timeSubStep )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    Vector Dx = patch->dCell();
    double delta_n = 0.0;
    if (_dir==0) {
       delta_n=Dx.x();
    } else if (_dir==1) {
       delta_n=Dx.y();
    } else {
       delta_n=Dx.z();
    }
    CCVariable<double> thp_rate;
    DataWarehouse* which_dw;

    if ( timeSubStep == 0 ){
      which_dw = old_dw;
      new_dw->allocateAndPut( thp_rate, d_modelLabel, matlIndex, patch );
      thp_rate.initialize(0.0);
    } else {
      which_dw = new_dw;
      new_dw->getModifiable( thp_rate, d_modelLabel, matlIndex, patch );
    }
    constCCVariable<double> gas_density;
    which_dw->get(gas_density, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0 );
    constCCVariable<double> gasT;
    which_dw->get( gasT , _gas_temperature_varlabel , matlIndex , patch , gac , 1 );
    constCCVariable<double> volFraction;
    old_dw->get( volFraction , _volFraction_varlabel , matlIndex , patch , gac , 1 );
    constCCVariable<double> pT;
    which_dw->get( pT , _particle_temperature_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> particle_density;
    which_dw->get( particle_density , _particle_density_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> diam;
    which_dw->get( diam    , _length_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> scaled_weight;
    which_dw->get( scaled_weight , _weight_scaled_varlabel , matlIndex , patch , gn , 0 );

    double dT_dn=0.0;
    double f_T_m=0.0; // face temperature on minus side
    double f_T_p=0.0; // face temperature on plus side
    double Ft=0.0; // thermophoretic force
    double at=0.0; // thermophoretic acceleration
    double mu=0.0; //
    double rkp=0.0; //
    double rkg=0.0; //
    double umol=0.0; //
    double fpl=0.0; //
    double K_n=0.0; //
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      if (volFraction[c]<1.0){
        thp_rate[c]=0.0;
      } else {
        IntVector cp = c + _cell_plus;
        IntVector cm = c + _cell_minus;
        // for computing dT/dn we have 4 scenarios:
        // (1) both neigbor cells are flow cells:
        //         gasT[cp] - gasT[cm]
        // dT_dn = __________________
        //              2 * delta_n
        // (2) cp is a wall:
        //         gasT[cp]          gasT[c] + gasT[cm]
        // dT_dn = _________   _      ________________
        //          delta_n             2 * delta_n
        // (3) cm is a wall:
        //         gasT[cp] + gasT[c]              gasT[cm]
        // dT_dn = __________________     _      _____________
        //          2 * delta_n                     delta_n
        // (4) cm and cp are walls:
        //         gasT[cp] - gasT[cm]
        // dT_dn = __________________
        //             delta_n
        // Note: Surface temperature of the wall is stored at the cell center.
        if (volFraction[cp] < 1.0){
          f_T_p=gasT[cp];
        } else {
          f_T_p = (gasT[cp]+gasT[c])/2;
        }
        if (volFraction[cm] < 1.0){
          f_T_m=gasT[cm];
        } else {
          f_T_m = (gasT[c]+gasT[cm])/2;
        }
        dT_dn = (f_T_p - f_T_m) / delta_n;
        // The thermophoretic force can be computed using the following equation:
        //      3 * pi * mu * d^2 * K_n * C_tm * [( kg/kp + C_t*K_n)*(1+1.3333*Adep*C_m*K_n)-1.33*Adep*C_m*K_n]*grad_T
        // Ft = __________________________________________________________________________________________________
        //                              (1+3*C_m*K_n)*(1+2*kg/kp+2*C_t*K_n)
        // Note: This model was taken from Phil's RANS code. I could not find the model in the literature but it compares
        // relatively well with results from Derjaguin et. al. (1976)
        mu = _visc; // dynamic viscosity of the gas kg/(m s)
        rkp = _rkp; // thermal conductvity of the particles [=] W / (m K)
        rkg = props(gasT[c], pT[c]); // [=] J/s/m/K
        umol = pow(8*8.314*gasT[c]/_pi/0.03,0.5); // [=] m/s assuming the average molar mass(0.03 kg/mol)
        fpl = 2.0*mu/(gas_density[c]*umol); // [=] m
        K_n = 2.0*fpl/diam[c]; // [=] m/m
        Ft = - ( 3 * _pi * mu * pow(diam[c],2) * K_n * _C_tm  * (( rkg/rkp + _C_t*K_n)*(1+1.3333*_Adep*_C_m*K_n)-1.3333*_Adep*_C_m*K_n) * dT_dn )/
            ( (1+3*_C_m*K_n)*(1+2*rkg/rkp+2*_C_t*K_n) ); // kg m / s^2
        at = Ft / (particle_density[c]*(_pi/6)*pow(diam[c],3.0));// thermophoretic acceleration [=] m/(s^2)
        thp_rate[c]=scaled_weight[c]*at/_vel_scaling_constant; // [=] #/m^3 * m/(s^2) = #/(m^2 * s^2)
      }
    }//end cell loop

  }//end patch loop
}

double
Thermophoresis::props(double Tg, double Tp){

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
