#include <CCA/Components/Arches/PropertyModelsV2/OneDWallHT.h>
#include <CCA/Components/Arches/KokkosTools.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>


namespace Uintah{


//--------------------------------------------------------------------------------------------------
OneDWallHT::OneDWallHT( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
  _sigma_constant =  5.670e-8;
}

//--------------------------------------------------------------------------------------------------
OneDWallHT::~OneDWallHT(){
}

//--------------------------------------------------------------------------------------------------
void
OneDWallHT::problemSetup( ProblemSpecP& db ){

  if (db->findBlock("incident_hf_label")) {
    db->require("incident_hf_label",_incident_hf_label);
  } else {
    throw InvalidValue("ERROR: OneDWallHT::problemSetup() Missing <incident_hf_label> in input file!",__FILE__,__LINE__);
  }
  if (db->findBlock("emissivity_label")) {
    db->require("emissivity_label",_emissivity_label);
  } else {
    throw InvalidValue("ERROR: OneDWallHT::problemSetup() Missing <emissivity_label> in input file!",__FILE__,__LINE__);
  }
  if (db->findBlock("Tshell_label")) {
    db->require("Tshell_label",_Tshell_label);
  } else {
    throw InvalidValue("ERROR: OneDWallHT::problemSetup() Missing <Tshell_label> in input file!",__FILE__,__LINE__);
  }
  if (db->findBlock("wall_resistance_label")) {
    db->require("wall_resistance_label",_wall_resistance_label);
  } else {
    throw InvalidValue("ERROR: OneDWallHT::problemSetup() Missing <wall_resistance_label> in input file!",__FILE__,__LINE__);
  }
}

void
OneDWallHT::create_local_labels(){
  register_new_variable<CCVariable<double> >( "Twall" );
}

//--------------------------------------------------------------------------------------------------
void
OneDWallHT::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>&
                                 variable_registry, const bool packed_tasks ){

  register_variable( "Twall", ArchesFieldContainer::COMPUTES, variable_registry );

}

//--------------------------------------------------------------------------------------------------
void
OneDWallHT::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){


  CCVariable<double>& Twall = *(tsk_info->get_uintah_field<CCVariable<double> >("Twall"));
  KOKKOS_INITIALIZE_TO_CONSTANT_EXTRA_CELL( Twall, 300.0 );

}

//--------------------------------------------------------------------------------------------------
void
OneDWallHT::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>&
                                    variable_registry, const bool packed_tasks ){

  register_variable( "Twall", ArchesFieldContainer::COMPUTES, variable_registry );

}

//--------------------------------------------------------------------------------------------------
void
OneDWallHT::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& Twall = *(tsk_info->get_uintah_field<CCVariable<double> >("Twall"));
  KOKKOS_INITIALIZE_TO_CONSTANT_EXTRA_CELL( Twall, 300.0 );

}

//--------------------------------------------------------------------------------------------------
void
OneDWallHT::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>&
                                    variable_registry, const int time_substep,
                                    const bool packed_tasks ){

  register_variable( "Twall",                ArchesFieldContainer::MODIFIES ,  variable_registry, time_substep );
  register_variable( _incident_hf_label,     ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
  register_variable( _emissivity_label,      ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
  register_variable( _Tshell_label,          ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
  register_variable( _wall_resistance_label, ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
  register_variable( "wall_HF_area",         ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::LATEST , variable_registry , time_substep );

}

//--------------------------------------------------------------------------------------------------
void
OneDWallHT::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& Twall = *(tsk_info->get_uintah_field<CCVariable<double> >( "Twall"));
  constCCVariable<double>& rad_q = *(tsk_info->get_const_uintah_field<constCCVariable<double > >( _incident_hf_label ));
  constCCVariable<double>& emissivity = *(tsk_info->get_const_uintah_field<constCCVariable<double > >( _emissivity_label ));
  constCCVariable<double>& Tsh = *(tsk_info->get_const_uintah_field<constCCVariable<double > >( _Tshell_label ));
  constCCVariable<double>& R_tot = *(tsk_info->get_const_uintah_field<constCCVariable<double > >( _wall_resistance_label ));
  constCCVariable<double>& wall_HF_area = *(tsk_info->get_const_uintah_field<constCCVariable<double > >( "wall_HF_area" ));
  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
    if ( wall_HF_area(i,j,k) > 0.0 ){
      newton_solve( Twall(i,j,k), Tsh(i,j,k), rad_q(i,j,k), emissivity(i,j,k), R_tot(i,j,k) );
    }
  });
}
//----------------------------------
void
OneDWallHT::newton_solve( double& Twall, const double& Tsh, const double& rad_q, const double& emissivity, const double& R_tot)
{
  // solver constants
  double net_q;
  double d_tol    = 1e-15;
  double delta    = 1;
  int NIter       = 15;
  double f0       = 0.0;
  double f1       = 0.0;
  double T_max    = pow( rad_q/_sigma_constant, 0.25); // if k = 0.0;
  double Twall_guess, Twall_tmp, Twall_old;

  //required variables

  // new solve
  Twall_guess = Twall;
  Twall_old = Twall_guess-delta;
  net_q = rad_q - _sigma_constant * pow( Twall_old, 4 );
  net_q = net_q > 0 ? net_q : 0;
  net_q *= emissivity;
  f0 = - Twall_old + Tsh + net_q * R_tot;
  Twall = Twall_guess+delta;
  net_q = rad_q - _sigma_constant * pow( Twall, 4 );
  net_q *= emissivity;
  net_q = net_q>0 ? net_q : 0;
  f1 = - Twall + Tsh + net_q * R_tot;
  for ( int iterT=0; iterT < NIter; iterT++) {
    Twall_tmp = Twall_old;
    Twall_old = Twall;
    Twall = Twall_tmp - ( Twall - Twall_tmp )/( f1 - f0 ) * f0;
    Twall = std::max( Tsh , std::min( T_max, Twall ) );
    if (std::abs(Twall-Twall_old) < d_tol){
      net_q =  rad_q - _sigma_constant * pow( Twall, 4 );
      net_q =  net_q > 0 ? net_q : 0;
      net_q *= emissivity;
      break;
    }
    f0    =  f1;
    net_q =  rad_q - _sigma_constant * pow( Twall, 4 );
    net_q =  net_q>0 ? net_q : 0;
    net_q *= emissivity;
    f1    = - Twall + Tsh + net_q * R_tot;
  }
}
} //namespace Uintah
