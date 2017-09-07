#include <CCA/Components/Arches/PropertyModelsV2/CO.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>

namespace Uintah{

CO::CO( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {

  _Rgas = 1.9872041; // kcol/K/kmol
  _MW_CO = 28.01; // g/mol
  _MW_H2O = 18.01528; // g/mol
  _MW_O2 = 31.9988; // g/mol
  _st_O2 = (1.0/0.5)*(_MW_CO / _MW_O2); // stoichiometric g CO / g O2
  _st_H2O = (1.0/1.0)*(_MW_CO / _MW_H2O); // stoichiometric g CO / g H2O
  _disc = 0;
  _boundary_condition = scinew BoundaryCondition_new( matl_index );

}

CO::~CO(){
  delete _disc;
  delete _boundary_condition;
}

void
CO::problemSetup( ProblemSpecP& db ){

  _disc = scinew Discretization_new();

  db->getAttribute("label", _CO_model_name);
  _CO_diff_name=_CO_model_name+"_diff";
  _CO_conv_name=_CO_model_name+"_conv";
  db->require("CO_defect_label",_defect_name);
  db->getWithDefault("CO_rate_label",_rate_name,_CO_model_name+"_rate");
  db->getWithDefault("density_label",_rho_table_name,"density");
  db->getWithDefault("temperature_label",_temperature_table_name,"temperature");
  db->getWithDefault("CO_label",_CO_table_name,"CO");
  db->getWithDefault("H2O_label",_H2O_table_name,"H2O");
  db->getWithDefault("O2_label",_O2_table_name,"O2");
  db->getWithDefault("MW_label",_MW_table_name,"mixture_molecular_weight");
  db->getWithDefault("a",_a,1.0);
  db->getWithDefault("b",_b,0.5);
  db->getWithDefault("c",_c,0.25);
  db->getWithDefault("A",_A,2.61e12);// kmole/m3/s
  db->getWithDefault("Ea",_Ea,45566.0);// kcal/kmole
  db->getWithDefault("Tcrit",_T_crit,1150);

  if ( db->findBlock("conv_scheme")){
    db->require("conv_scheme",_conv_scheme);
  } else {
    throw ProblemSetupException("Error: Convection scheme not specified for CO property model.",__FILE__,__LINE__);
  }

  db->getWithDefault("Pr", _prNo, 0.4);

  _u_vel = "uVelocitySPBC";
  _v_vel = "vVelocitySPBC";
  _w_vel = "wVelocitySPBC";
  _area_frac= "areaFraction";
  _turb_visc= "turb_viscosity";
  _vol_frac= "volFraction";

  _boundary_condition->problemSetup( db, _CO_model_name );
  // Warning! When setting CO boundary conditions you have to set
  // the boundary conditions using Dirichlet or Neumann, you cannot
  // use tabulated boundary conditions.
}

void
CO::create_local_labels(){

  register_new_variable<CCVariable<double> >(_CO_model_name);
  register_new_variable<CCVariable<double> >(_defect_name);
  register_new_variable<CCVariable<double> >(_rate_name);
  register_new_variable<CCVariable<double> >(_CO_diff_name);
  register_new_variable<CCVariable<double> >(_CO_conv_name);

}

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void
CO::register_initialize( VIVec& variable_registry , const bool pack_tasks){

  register_variable( _CO_model_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _defect_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _rate_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _CO_diff_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _CO_conv_name, ArchesFieldContainer::COMPUTES, variable_registry );

}



void
CO::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>* vCO       = tsk_info->get_uintah_field<CCVariable<double> >( _CO_model_name );
  CCVariable<double>* vCO_diff  = tsk_info->get_uintah_field<CCVariable<double> >( _CO_diff_name );
  CCVariable<double>* vCO_conv  = tsk_info->get_uintah_field<CCVariable<double> >( _CO_conv_name );
  CCVariable<double>* vd        = tsk_info->get_uintah_field<CCVariable<double> >( _defect_name );
  CCVariable<double>* vrate     = tsk_info->get_uintah_field<CCVariable<double> >( _rate_name );

  CCVariable<double>& CO = *vCO;
  CCVariable<double>& CO_diff = *vCO_diff;
  CCVariable<double>& CO_conv = *vCO_conv;
  CCVariable<double>& d = *vd;
  CCVariable<double>& rate = *vrate;

  CO.initialize(0.0);
  CO_diff.initialize(0.0);
  CO_conv.initialize(0.0);
  d.initialize(0.0);
  rate.initialize(0.0);

  _boundary_condition->checkForBC( 0, patch , _CO_model_name);
  _boundary_condition->setScalarValueBC( 0, patch, CO, _CO_model_name );
}

//--------------------------------------------------------------------------------------------------
void CO::register_restart_initialize( VIVec& variable_registry , const bool packed_tasks){

}

//--------------------------------------------------------------------------------------------------
void CO::restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

}

//--------------------------------------------------------------------------------------------------
void CO::register_timestep_init( VIVec& variable_registry , const bool packed_tasks){

  register_variable( _CO_model_name , ArchesFieldContainer::COMPUTES, variable_registry);
  register_variable( _CO_model_name , ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );
  register_variable( _defect_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _defect_name , ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );
  register_variable( _CO_diff_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _CO_conv_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _rate_name, ArchesFieldContainer::COMPUTES, variable_registry );

}

//--------------------------------------------------------------------------------------------------
void CO::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& CO          = *(tsk_info->get_uintah_field<CCVariable<double>>( _CO_model_name ));
  constCCVariable<double>& CO_old = *(tsk_info->get_const_uintah_field<constCCVariable<double>>( _CO_model_name ));
  CCVariable<double>& defect      = *(tsk_info->get_uintah_field<CCVariable<double>>( _defect_name ));
  constCCVariable<double>& defect_old = *(tsk_info->get_const_uintah_field<constCCVariable<double>>( _defect_name ));
  CCVariable<double>& CO_diff     = *(tsk_info->get_uintah_field<CCVariable<double>>(_CO_diff_name));
  CCVariable<double>& CO_conv     = *(tsk_info->get_uintah_field<CCVariable<double>>(_CO_conv_name));
  CCVariable<double>& rate        = *(tsk_info->get_uintah_field<CCVariable<double>>(_rate_name));

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
  Uintah::parallel_for( range, [&](int i, int j, int k){
    CO(i,j,k) = CO_old(i,j,k);
    defect(i,j,k)  = defect_old(i,j,k);
    CO_diff(i,j,k) = 0.0; // this is needed because all conv and diff computations are +=
    CO_conv(i,j,k) = 0.0;
    rate(i,j,k) = 0.0;
  });

}


//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

void
CO::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){

  // computed variables
  register_variable( _CO_model_name, ArchesFieldContainer::MODIFIES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  register_variable( _defect_name, ArchesFieldContainer::MODIFIES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  register_variable( _rate_name, ArchesFieldContainer::MODIFIES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  register_variable( _CO_diff_name, ArchesFieldContainer::MODIFIES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  register_variable( _CO_conv_name, ArchesFieldContainer::MODIFIES, 0, ArchesFieldContainer::NEWDW,  variable_registry, time_substep );

  // OLDDW variables
  register_variable( _CO_model_name, ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::OLDDW, variable_registry, time_substep );
  register_variable( _defect_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry, time_substep );
  register_variable( _CO_table_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry, time_substep );
  register_variable( _H2O_table_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry, time_substep );
  register_variable( _O2_table_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry, time_substep );
  register_variable( _rho_table_name , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::OLDDW , variable_registry, time_substep );
  register_variable( _MW_table_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry, time_substep );
  register_variable( _temperature_table_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry, time_substep );
  register_variable( _u_vel, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW, variable_registry, time_substep );
  register_variable( _v_vel, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW, variable_registry, time_substep );
  register_variable( _w_vel, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW, variable_registry, time_substep );
  register_variable( _area_frac, ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::OLDDW, variable_registry, time_substep );
  register_variable( _turb_visc, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW, variable_registry, time_substep );
  register_variable( _vol_frac, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry, time_substep );

  // NEWDW variables
  register_variable( _CO_table_name , ArchesFieldContainer::REQUIRES, 0 , ArchesFieldContainer::NEWDW , variable_registry, time_substep );
  register_variable( _rho_table_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

}

void
CO::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){


  /* This model computes carbon monoxide as a sum of the equilibrum CO and a defect CO.
  CO = CO_equil + defect
  y = ym + d
  or d = y - ym
  accordingly,
  d rho*d
  ________ = RHS_y/V + r_y - (RHS_ym/V + r_ym)
    dt

    the reaction rate for CO "r_y" is defined as follows:

                              [rho*ym]^(t+1) - [rho*y]^(t)
           { for T > T_crit    ___________________________    - RHS_y/V
           {                               dt
    r_y =  {
           { for T < T_crit   r_a

           where r_a = A * CO^a * H2O^b * O2^c * exp( -E/(RT) )

  One can then compute the update for d (and consequently y) as follows:
  [rho*d]^(t+1) - [rho*d]^(t)                              D rho*ym
  ___________________________  = [RHS_y]^t/V + [r_y]^t -   ________
             dt                                               Dt
                                    (S1)        (S2)         (S3)

                         [rho*ym]^(t+1) - [rho*ym]^(t)
           where S3 =      ___________________________
                                      dt
  Then we get:
                1
  [d]^(t+1) = ______ * ( [rho*d]^(t) + dt * ( S1 + S2 - S3))
           [rho]^(t+1)

  [y]^(t+1) = [ym]^(t+1) + [d]^(t+1)

  It is trivial to show mathematically that when T > Tcrit
  [d]^(t+1) = 0
  and [y]^(t+1) = [ym]^(t+1)
  Otherwise S1, S2, and S2 need to be computed.

  The following algorithm is used to update y and d:
  if T > T_crit
    d=0 , y=0
  else
    Step 1: compute S1 (all variables are at time t)
    Step 2: Compute S2 (all variables are at time t)
    Step 3: compute S3 (variables at t and t+1)
    Step 4: update d to time t+1
    Step 5: update y to time t+1
  */

  const double dt = tsk_info->get_dt();
  constCCVariable<double>* ym_new          = tsk_info->get_const_uintah_field<constCCVariable<double> >( _CO_table_name, ArchesFieldContainer::NEWDW );
  constCCVariable<double>* rho_new         = tsk_info->get_const_uintah_field<constCCVariable<double> >( _rho_table_name, ArchesFieldContainer::NEWDW );
  constCCVariable<double>* y_old           = tsk_info->get_const_uintah_field<constCCVariable<double> >( _CO_model_name, ArchesFieldContainer::OLDDW );
  constCCVariable<double>* d_old           = tsk_info->get_const_uintah_field<constCCVariable<double> >( _defect_name, ArchesFieldContainer::OLDDW );
  constCCVariable<double>* ym_old          = tsk_info->get_const_uintah_field<constCCVariable<double> >( _CO_table_name, ArchesFieldContainer::OLDDW );
  constCCVariable<double>* rho_old         = tsk_info->get_const_uintah_field<constCCVariable<double> >( _rho_table_name, ArchesFieldContainer::OLDDW );
  constCCVariable<double>* H2O_old         = tsk_info->get_const_uintah_field<constCCVariable<double> >( _H2O_table_name, ArchesFieldContainer::OLDDW );
  constCCVariable<double>* O2_old          = tsk_info->get_const_uintah_field<constCCVariable<double> >( _O2_table_name, ArchesFieldContainer::OLDDW );
  constCCVariable<double>* invMW_old       = tsk_info->get_const_uintah_field<constCCVariable<double> >( _MW_table_name, ArchesFieldContainer::OLDDW );
  constCCVariable<double>* temperature_old = tsk_info->get_const_uintah_field<constCCVariable<double> >( _temperature_table_name, ArchesFieldContainer::OLDDW );
  constCCVariable<double>* mu_t            = tsk_info->get_const_uintah_field<constCCVariable<double> >( _turb_visc, ArchesFieldContainer::OLDDW );
  constCCVariable<double>* volFraction     = tsk_info->get_const_uintah_field<constCCVariable<double> >( _vol_frac, ArchesFieldContainer::OLDDW );
  constCCVariable<Vector>* areaFraction    = tsk_info->get_const_uintah_field<constCCVariable<Vector> >( _area_frac, ArchesFieldContainer::OLDDW );
  constSFCXVariable<double>* uVel          = tsk_info->get_const_uintah_field<constSFCXVariable<double> >( _u_vel, ArchesFieldContainer::OLDDW );
  constSFCYVariable<double>* vVel          = tsk_info->get_const_uintah_field<constSFCYVariable<double> >( _v_vel, ArchesFieldContainer::OLDDW );
  constSFCZVariable<double>* wVel          = tsk_info->get_const_uintah_field<constSFCZVariable<double> >( _w_vel, ArchesFieldContainer::OLDDW );

  CCVariable<double>* y_new    = tsk_info->get_uintah_field<CCVariable<double> >( _CO_model_name );
  CCVariable<double>* d_new    = tsk_info->get_uintah_field<CCVariable<double> >( _defect_name );
  CCVariable<double>* rate_new = tsk_info->get_uintah_field<CCVariable<double> >( _rate_name );
  CCVariable<double>* y_diff   = tsk_info->get_uintah_field<CCVariable<double> >( _CO_diff_name );
  CCVariable<double>* y_conv   = tsk_info->get_uintah_field<CCVariable<double> >( _CO_conv_name );

  Vector Dx = patch->dCell();
  double vol = Dx.x()*Dx.y()*Dx.z();
  double const_mol_D = 0.0;

  _disc->computeConv( patch, *y_conv, *y_old, *uVel, *vVel, *wVel, *rho_old, *areaFraction, _conv_scheme );
  _disc->computeDiff( patch, *y_diff, *y_old, *mu_t, const_mol_D, *rho_old, *areaFraction, _prNo );

  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

    IntVector c = *iter;

    if ( (*temperature_old)[c] > _T_crit || (*volFraction)[c] < 1e-10 ) { // the volFraction condition is needed to avoid nan in intrusion cells.

      (*d_new)[c] = 0.0;
      (*y_new)[c] = (*ym_new)[c];
      (*rate_new)[c] = 0.0;

    } else {
      double S1 =  (*y_diff)[c] - (*y_conv)[c]; // kg/s
      double S3 = ( (*rho_new)[c] * (*ym_new)[c] - (*rho_old)[c] * (*ym_old)[c] ) / dt; // kg/(m^3 s)

      double MW_old = 1.0 / (*invMW_old)[c];// g mix / mole mix
      double CO_mole = (*y_old)[c] * MW_old / _MW_CO;// moles CO / moles mix
      double H2O_mole = (*H2O_old)[c] * MW_old / _MW_H2O;// moles H2O / moles mix
      double O2_mole = (*O2_old)[c] * MW_old / _MW_O2;// moles O2 / moles mix
      double S2 = - _A * pow(CO_mole,_a) * pow(H2O_mole,_b) * pow(O2_mole,_c) * exp(-_Ea / (_Rgas * (*temperature_old)[c])) * MW_old; // kg/(m^3 s)
      // rate clipping for safety
      // (1) fuel (CO) limited
      // (2) O2 limited (CO + 0.5O2 > CO2)
      // (3) H2O limited (water-gas shift reaction, CO + H2O > CO2 + H2)
      // (4) rate < 0.0

      // (1)
      double min_fuel_S2 = ( - (*rho_new)[c] * (*ym_new)[c] - (*rho_old)[c] * (*d_old)[c] ) / dt - S1/vol + S3;// kgCO/(m^3/s) to consume all available CO
      // (2) (3) and (4)
      double lim_reac_O2_H2O = std::min(_st_O2 * (*O2_old)[c] , _st_H2O * (*H2O_old)[c]);// kg of CO / kg mix to burn avaialbe O2 or H2O
      double lim_reac = std::max(   -(*rho_old)[c] / dt * lim_reac_O2_H2O , min_fuel_S2 ); // slowest rate of destruction for  kgCO/(m^3/s)
      S2 = std::min( 0.0, std::max( lim_reac, S2) );// make sure S2 is slower than the limiting reactant rate, and not positive.

      (*d_new)[c] = (1.0 / (*rho_new)[c]) * ( (*rho_old)[c] * (*d_old)[c] + dt * ( S1/vol + S2 - S3 ) ); // kg d / kg mix
      (*y_new)[c] = std::max( 0.0, ( (*ym_new)[c] + (*d_new)[c] )); // kg CO / kg mix
      (*rate_new)[c] = S2;
    }
  }

  _boundary_condition->setScalarValueBC( 0, patch, *y_new, _CO_model_name );

}
} //namespace Uintah
