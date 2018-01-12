#include <CCA/Components/Arches/PropertyModelsV2/CO.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>

namespace Uintah{

typedef ArchesFieldContainer AFC;

//--------------------------------------------------------------------------------------------------
CO::CO( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {

  m_st_O2 = (1.0/0.5)*(m_MW_CO / m_MW_O2); // stoichiometric g CO / g O2
  m_st_H2O = (1.0/1.0)*(m_MW_CO / m_MW_H2O); // stoichiometric g CO / g H2O
  m_disc = 0;
  m_boundary_condition = scinew BoundaryCondition_new( matl_index );

}

//--------------------------------------------------------------------------------------------------
CO::~CO(){

  delete m_disc;
  delete m_boundary_condition;

}

//--------------------------------------------------------------------------------------------------
void
CO::problemSetup( ProblemSpecP& db ){

  m_disc = scinew Discretization_new();

  db->getAttribute("label", m_CO_model_name);
  m_CO_diff_name=m_CO_model_name+"_diff";
  m_CO_conv_name=m_CO_model_name+"_conv";

  db->require("CO_defect_label",m_defect_name);

  db->getWithDefault("CO_rate_label"     , m_rate_name              , m_CO_model_name+"_rate");
  db->getWithDefault("density_label"     , m_rho_table_name         , "density");
  db->getWithDefault("temperature_label" , m_temperature_table_name , "temperature");
  db->getWithDefault("CO_label"          , m_CO_table_name          , "CO");
  db->getWithDefault("H2O_label"         , m_H2O_table_name         , "H2O");
  db->getWithDefault("O2_label"          , m_O2_table_name          , "O2");
  db->getWithDefault("MW_label"          , m_MW_table_name          , "mixture_molecular_weight");
  db->getWithDefault("a"                 , m_a                      , 1.0);
  db->getWithDefault("b"                 , m_b                      , 0.5);
  db->getWithDefault("c"                 , m_c                      , 0.25);
  db->getWithDefault("A"                 , m_A                      , 2.61e12);// kmole/m3/s
  db->getWithDefault("Ea"                , m_Ea                     , 45566.0);// kcal/kmole
  db->getWithDefault("Tcrit"             , m_T_crit                 , 1150);

  if ( db->findBlock("conv_scheme")){
    db->require("conv_scheme",m_conv_scheme);
  } else {
    throw ProblemSetupException("Error: Convection scheme not specified for CO property model.",__FILE__,__LINE__);
  }

  db->getWithDefault("Pr", m_prNo, 0.4);

  m_u_vel     = "uVelocitySPBC";
  m_v_vel     = "vVelocitySPBC";
  m_w_vel     = "wVelocitySPBC";
  m_area_frac = "areaFraction";
  m_turb_visc = "turb_viscosity";
  m_vol_frac  = "volFraction";

  m_boundary_condition->problemSetup( db, m_CO_model_name );
  // Warning! When setting CO boundary conditions you have to set
  // the boundary conditions using Dirichlet or Neumann, you cannot
  // use tabulated boundary conditions.
}

//--------------------------------------------------------------------------------------------------
void
CO::create_local_labels(){

  register_new_variable<CCVariable<double> >(m_CO_model_name);
  register_new_variable<CCVariable<double> >(m_defect_name);
  register_new_variable<CCVariable<double> >(m_rate_name);
  register_new_variable<CCVariable<double> >(m_CO_diff_name);
  register_new_variable<CCVariable<double> >(m_CO_conv_name);

}

//--------------------------------------------------------------------------------------------------
void
CO::register_initialize( VIVec& variable_registry , const bool pack_tasks){

  register_variable( m_CO_model_name, AFC::COMPUTES, variable_registry );
  register_variable( m_defect_name, AFC::COMPUTES, variable_registry );
  register_variable( m_rate_name, AFC::COMPUTES, variable_registry );
  register_variable( m_CO_diff_name, AFC::COMPUTES, variable_registry );
  register_variable( m_CO_conv_name, AFC::COMPUTES, variable_registry );

}

//--------------------------------------------------------------------------------------------------
void
CO::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& CO      = tsk_info->get_uintah_field_add<CCVariable<double> >( m_CO_model_name );
  CCVariable<double>& CO_diff = tsk_info->get_uintah_field_add<CCVariable<double> >( m_CO_diff_name );
  CCVariable<double>& CO_conv = tsk_info->get_uintah_field_add<CCVariable<double> >( m_CO_conv_name );
  CCVariable<double>& d       = tsk_info->get_uintah_field_add<CCVariable<double> >( m_defect_name );
  CCVariable<double>& rate    = tsk_info->get_uintah_field_add<CCVariable<double> >( m_rate_name );

  CO.initialize(0.0);
  CO_diff.initialize(0.0);
  CO_conv.initialize(0.0);
  d.initialize(0.0);
  rate.initialize(0.0);

  m_boundary_condition->checkForBC( 0, patch , m_CO_model_name);
  m_boundary_condition->setScalarValueBC( 0, patch, CO, m_CO_model_name );

}

//--------------------------------------------------------------------------------------------------
void CO::register_timestep_init( VIVec& variable_registry , const bool packed_tasks){

  register_variable( m_CO_model_name , AFC::COMPUTES, variable_registry);
  register_variable( m_CO_model_name , AFC::REQUIRES, 0, AFC::OLDDW, variable_registry );
  register_variable( m_defect_name, AFC::COMPUTES, variable_registry );
  register_variable( m_defect_name , AFC::REQUIRES, 0, AFC::OLDDW, variable_registry );
  register_variable( m_CO_diff_name, AFC::COMPUTES, variable_registry );
  register_variable( m_CO_conv_name, AFC::COMPUTES, variable_registry );
  register_variable( m_rate_name, AFC::COMPUTES, variable_registry );

}

//--------------------------------------------------------------------------------------------------
void CO::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& CO          = tsk_info->get_uintah_field_add<CCVariable<double>>( m_CO_model_name );
  constCCVariable<double>& CO_old = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_CO_model_name );
  CCVariable<double>& defect      = tsk_info->get_uintah_field_add<CCVariable<double>>( m_defect_name );
  constCCVariable<double>& defect_old = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_defect_name );
  CCVariable<double>& CO_diff     = tsk_info->get_uintah_field_add<CCVariable<double>>(m_CO_diff_name);
  CCVariable<double>& CO_conv     = tsk_info->get_uintah_field_add<CCVariable<double>>(m_CO_conv_name);
  CCVariable<double>& rate        = tsk_info->get_uintah_field_add<CCVariable<double>>(m_rate_name);

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
  Uintah::parallel_for( range, [&](int i, int j, int k){
    CO(i,j,k)      = CO_old(i,j,k);
    defect(i,j,k)  = defect_old(i,j,k);
    CO_diff(i,j,k) = 0.0; // this is needed because all conv and diff computations are + =
    CO_conv(i,j,k) = 0.0;
    rate(i,j,k)    = 0.0;
  });

}

//--------------------------------------------------------------------------------------------------
void
CO::register_timestep_eval( std::vector<AFC::VariableInformation>& variable_registry,
                            const int time_substep , const bool packed_tasks){

  // computed variables
  register_variable( m_CO_model_name , AFC::MODIFIES , 0 , AFC::NEWDW , variable_registry , time_substep );
  register_variable( m_defect_name   , AFC::MODIFIES , 0 , AFC::NEWDW , variable_registry , time_substep );
  register_variable( m_rate_name     , AFC::MODIFIES , 0 , AFC::NEWDW , variable_registry , time_substep );
  register_variable( m_CO_diff_name  , AFC::MODIFIES , 0 , AFC::NEWDW , variable_registry , time_substep );
  register_variable( m_CO_conv_name  , AFC::MODIFIES , 0 , AFC::NEWDW , variable_registry , time_substep );

  // OLDDW variables
  register_variable( m_CO_model_name          , AFC::REQUIRES , 2 , AFC::OLDDW , variable_registry , time_substep );
  register_variable( m_defect_name            , AFC::REQUIRES , 0 , AFC::OLDDW , variable_registry , time_substep );
  register_variable( m_CO_table_name          , AFC::REQUIRES , 0 , AFC::OLDDW , variable_registry , time_substep );
  register_variable( m_H2O_table_name         , AFC::REQUIRES , 0 , AFC::OLDDW , variable_registry , time_substep );
  register_variable( m_O2_table_name          , AFC::REQUIRES , 0 , AFC::OLDDW , variable_registry , time_substep );
  register_variable( m_rho_table_name         , AFC::REQUIRES , 1 , AFC::OLDDW , variable_registry , time_substep );
  register_variable( m_MW_table_name          , AFC::REQUIRES , 0 , AFC::OLDDW , variable_registry , time_substep );
  register_variable( m_temperature_table_name , AFC::REQUIRES , 0 , AFC::OLDDW , variable_registry , time_substep );
  register_variable( m_u_vel                  , AFC::REQUIRES , 1 , AFC::OLDDW , variable_registry , time_substep );
  register_variable( m_v_vel                  , AFC::REQUIRES , 1 , AFC::OLDDW , variable_registry , time_substep );
  register_variable( m_w_vel                  , AFC::REQUIRES , 1 , AFC::OLDDW , variable_registry , time_substep );
  register_variable( m_area_frac              , AFC::REQUIRES , 2 , AFC::OLDDW , variable_registry , time_substep );
  register_variable( m_turb_visc              , AFC::REQUIRES , 1 , AFC::OLDDW , variable_registry , time_substep );
  register_variable( m_vol_frac               , AFC::REQUIRES , 0 , AFC::OLDDW , variable_registry , time_substep );

  // NEWDW variables
  register_variable( m_CO_table_name  , AFC::REQUIRES , 0 , AFC::NEWDW , variable_registry , time_substep );
  register_variable( m_rho_table_name , AFC::REQUIRES , 0 , AFC::NEWDW , variable_registry , time_substep );

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
  constCCVariable<double>& ym_new          = tsk_info->get_const_uintah_field_add<constCCVariable<double> >( m_CO_table_name, AFC::NEWDW );
  constCCVariable<double>& rho_new         = tsk_info->get_const_uintah_field_add<constCCVariable<double> >( m_rho_table_name, AFC::NEWDW );
  constCCVariable<double>& y_old           = tsk_info->get_const_uintah_field_add<constCCVariable<double> >( m_CO_model_name, AFC::OLDDW );
  constCCVariable<double>& d_old           = tsk_info->get_const_uintah_field_add<constCCVariable<double> >( m_defect_name, AFC::OLDDW );
  constCCVariable<double>& ym_old          = tsk_info->get_const_uintah_field_add<constCCVariable<double> >( m_CO_table_name, AFC::OLDDW );
  constCCVariable<double>& rho_old         = tsk_info->get_const_uintah_field_add<constCCVariable<double> >( m_rho_table_name, AFC::OLDDW );
  constCCVariable<double>& H2O_old         = tsk_info->get_const_uintah_field_add<constCCVariable<double> >( m_H2O_table_name, AFC::OLDDW );
  constCCVariable<double>& O2_old          = tsk_info->get_const_uintah_field_add<constCCVariable<double> >( m_O2_table_name, AFC::OLDDW );
  constCCVariable<double>& invMW_old       = tsk_info->get_const_uintah_field_add<constCCVariable<double> >( m_MW_table_name, AFC::OLDDW );
  constCCVariable<double>& temperature_old = tsk_info->get_const_uintah_field_add<constCCVariable<double> >( m_temperature_table_name, AFC::OLDDW );
  constCCVariable<double>& mu_t            = tsk_info->get_const_uintah_field_add<constCCVariable<double> >( m_turb_visc, AFC::OLDDW );
  constCCVariable<double>& volFraction     = tsk_info->get_const_uintah_field_add<constCCVariable<double> >( m_vol_frac, AFC::OLDDW );
  constCCVariable<Vector>& areaFraction    = tsk_info->get_const_uintah_field_add<constCCVariable<Vector> >( m_area_frac, AFC::OLDDW );
  constSFCXVariable<double>& uVel          = tsk_info->get_const_uintah_field_add<constSFCXVariable<double> >( m_u_vel, AFC::OLDDW );
  constSFCYVariable<double>& vVel          = tsk_info->get_const_uintah_field_add<constSFCYVariable<double> >( m_v_vel, AFC::OLDDW );
  constSFCZVariable<double>& wVel          = tsk_info->get_const_uintah_field_add<constSFCZVariable<double> >( m_w_vel, AFC::OLDDW );

  CCVariable<double>& y_new    = tsk_info->get_uintah_field_add<CCVariable<double> >( m_CO_model_name );
  CCVariable<double>& d_new    = tsk_info->get_uintah_field_add<CCVariable<double> >( m_defect_name );
  CCVariable<double>& rate_new = tsk_info->get_uintah_field_add<CCVariable<double> >( m_rate_name );
  CCVariable<double>& y_diff   = tsk_info->get_uintah_field_add<CCVariable<double> >( m_CO_diff_name );
  CCVariable<double>& y_conv   = tsk_info->get_uintah_field_add<CCVariable<double> >( m_CO_conv_name );

  Vector Dx = patch->dCell();
  double vol = Dx.x()*Dx.y()*Dx.z();
  double const_mol_D = 0.0;

  m_disc->computeConv( patch, y_conv, y_old, uVel, vVel, wVel,
                       rho_old, areaFraction, m_conv_scheme );

  m_disc->computeDiff( patch, y_diff, y_old, mu_t, const_mol_D,
                       rho_old, areaFraction, m_prNo );

  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

    IntVector c = *iter;

    if ( temperature_old[c] > m_T_crit || volFraction[c] < 1e-10 ) { // the volFraction condition is needed to avoid nan in intrusion cells.

      d_new[c] = 0.0;
      y_new[c] = ym_new[c];
      rate_new[c] = 0.0;

    } else {

      const double S1 =  y_diff[c] - y_conv[c]; // kg/s
      const double S3 = ( rho_new[c] * ym_new[c] - rho_old[c] * ym_old[c] ) / dt; // kg/(m^3 s)

      const double MW_old = 1.0 / invMW_old[c]; // g mix / mole mix
      const double CO_mole = y_old[c] * MW_old / m_MW_CO;// moles CO / moles mix
      const double H2O_mole = H2O_old[c] * MW_old / m_MW_H2O;// moles H2O / moles mix
      const double O2_mole = O2_old[c] * MW_old / m_MW_O2;// moles O2 / moles mix

      double S2 = - m_A * std::pow(CO_mole,m_a) * std::pow(H2O_mole,m_b) * std::pow(O2_mole,m_c) *
                    std::exp(-m_Ea / (m_Rgas * temperature_old[c])) * MW_old; // kg/(m^3 s)

      // ****************************
      // rate clipping for safety
      // (1) fuel (CO) limited
      // (2) O2 limited (CO + 0.5O2 > CO2)
      // (3) H2O limited (water-gas shift reaction, CO + H2O > CO2 + H2)
      // (4) rate < 0.0

      // (1)
      const double min_fuel_S2 = ( - rho_new[c] * ym_new[c] - rho_old[c] * d_old[c] ) / dt
                                  - S1/vol + S3;// kgCO/(m^3/s) to consume all available CO
      // (2) (3) and (4)
      // kg of CO / kg mix to burn avaialbe O2 or H2O
      const double lim_reac_O2_H2O = std::min(m_st_O2 * O2_old[c] , m_st_H2O * H2O_old[c]);
      // slowest rate of destruction for  kgCO/(m^3/s)
      const double lim_reac = std::max(   - rho_old[c] / dt * lim_reac_O2_H2O , min_fuel_S2 );
      // make sure S2 is slower than the limiting reactant rate, and not positive.
      S2 = std::min( 0.0, std::max( lim_reac, S2) );

      d_new[c] = ( 1.0 / rho_new[c] ) * ( rho_old[c] * d_old[c] + dt * ( S1/vol + S2 - S3 ) ); // kg d / kg mix
      y_new[c] = std::max( 0.0, ( ym_new[c] + d_new[c] )); // kg CO / kg mix

      rate_new[c] = S2;
    }
  }

  m_boundary_condition->setScalarValueBC( 0, patch, y_new, m_CO_model_name );

}
} //namespace Uintah
