#include <CCA/Components/Arches/TurbulenceModels/WallConstSmag.h>
#include <CCA/Components/Arches/GridTools.h>
#include <CCA/Components/Arches/TurbulenceModels/DynamicSmagorinskyHelper.h>

#define c    i,   j,   k
#define xm   i-1, j,   k
#define xp   i+1, j,   k
#define ym   i,   j-1, k
#define yp   i,   j+1, k
#define zm   i  , j,   k-1
#define zp   i  , j,   k+1
#define xmym i-1, j-1, k
#define xmyp i-1, j+1, k
#define xmzm i-1, j  , k-1
#define xmzp i-1, j  , k+1
#define xpym i+1, j-1, k
#define xpzm i+1, j  , k-1
#define ymzm i  , j-1, k-1
#define ymzp i  , j-1, k+1
#define ypzm i  , j+1, k-1
namespace Uintah{


typedef ArchesFieldContainer AFC;

//---------------------------------------------------------------------------------
WallConstSmag::WallConstSmag( std::string task_name, int matl_index, const ProblemSpecP db_turb_parent ) :
TaskInterface( task_name, matl_index ), m_db_turb_parent(db_turb_parent)
{}

//---------------------------------------------------------------------------------
WallConstSmag::~WallConstSmag()
{}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace WallConstSmag::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace WallConstSmag::loadTaskInitializeFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace WallConstSmag::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &WallConstSmag::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &WallConstSmag::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &WallConstSmag::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace WallConstSmag::loadTaskTimestepInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace WallConstSmag::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//---------------------------------------------------------------------------------
void
WallConstSmag::problemSetup( ProblemSpecP& db ){

  using namespace Uintah::ArchesCore;

  m_u_vel_name = parse_ups_for_role( UVELOCITY_ROLE, db, "uVelocity" );
  m_v_vel_name = parse_ups_for_role( VVELOCITY_ROLE, db, "vVelocity" );
  m_w_vel_name = parse_ups_for_role( WVELOCITY_ROLE, db, "wVelocity" );
  m_density_name     = parse_ups_for_role( DENSITY_ROLE, db, "density" );

  //Which turb model is going to supply the strain rate mag?
  std::string which_model = "NotSet";
  bool found_the_model = false;
  db->findBlock("momentum_closure_model")->getAttribute("label", which_model);
  //Now parse through the closure models and get the label for the strain rate magnitude:
  for ( ProblemSpecP db_model = m_db_turb_parent->findBlock("model"); db_model != nullptr;
        db_model=db_model->findNextBlock("model")){
    std::string label;
    db_model->getAttribute("label",label);

    if ( label == which_model ){
      found_the_model = true;
    }
  }

  m_IsI_name = "strainMagnitude";
  if ( found_the_model ){
    m_IsI_name += "_" + which_model;
  } else {
    throw ProblemSetupException("Error: Could not match wall closure model with the turbulence model: "+which_model, __FILE__, __LINE__);
  }

  m_sigma_t_names.resize(3);
  m_sigma_t_names[0] = "sigma12";
  m_sigma_t_names[1] = "sigma13";
  m_sigma_t_names[2] = "sigma23";
  db->getWithDefault("Cs", m_Cs, 0.17);
  db->getWithDefault("standoff", m_standoff, 1);

  const ProblemSpecP params_root = db->getRootNode();
  if (params_root->findBlock("PhysicalConstants")) {
      params_root->findBlock("PhysicalConstants")->require("viscosity", m_molecular_visc);
    if( m_molecular_visc == 0 ) {
      std::stringstream msg;
      msg << "Error: Constant WallConstSmag::problemSetup() - Zero viscosity specified \n"
          << "       in <PhysicalConstants> section of input file." << std::endl;
      throw InvalidValue(msg.str(),__FILE__,__LINE__);
    }
  } else {
    std::stringstream msg;
    msg << "ERROR: Constant WallConstSmag: problemSetup(): Missing <PhysicalConstants> \n"
        << "       section in input file!" << std::endl;
    throw InvalidValue(msg.str(),__FILE__,__LINE__);
  }
}

//---------------------------------------------------------------------------------
void
WallConstSmag::create_local_labels(){


}

//---------------------------------------------------------------------------------
void
WallConstSmag::register_initialize( std::vector<AFC::VariableInformation>&
                                       variable_registry , const bool packed_tasks){


}

//---------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void WallConstSmag::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){


}

//---------------------------------------------------------------------------------
void
WallConstSmag::register_timestep_eval( std::vector<AFC::VariableInformation>&
                              variable_registry, const int time_substep , const bool packed_tasks){

  register_variable( m_u_vel_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep);
  register_variable( m_v_vel_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep);
  register_variable( m_w_vel_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep);

  register_variable( m_density_name,     AFC::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  register_variable( m_IsI_name,         AFC::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  register_variable( m_volFraction_name, AFC::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

  register_variable( m_sigma_t_names[0], ArchesFieldContainer::MODIFIES, variable_registry, time_substep );
  register_variable( m_sigma_t_names[1], ArchesFieldContainer::MODIFIES, variable_registry, time_substep );
  register_variable( m_sigma_t_names[2], ArchesFieldContainer::MODIFIES, variable_registry, time_substep );

}

//---------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void WallConstSmag::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  constSFCXVariable<double>& uVel = tsk_info->get_field<constSFCXVariable<double> >(m_u_vel_name);
  constSFCYVariable<double>& vVel = tsk_info->get_field<constSFCYVariable<double> >(m_v_vel_name);
  constSFCZVariable<double>& wVel = tsk_info->get_field<constSFCZVariable<double> >(m_w_vel_name);


  constCCVariable<double>& IsI = tsk_info->get_field< constCCVariable<double> >(m_IsI_name);
  constCCVariable<double>& rho = tsk_info->get_field<constCCVariable<double> >(m_density_name);
  constCCVariable<double>& eps = tsk_info->get_field<constCCVariable<double> >(m_volFraction_name);

  CCVariable<double>& sigma12 = tsk_info->get_field<CCVariable<double> >(m_sigma_t_names[0]);
  CCVariable<double>& sigma13 = tsk_info->get_field<CCVariable<double> >(m_sigma_t_names[1]);
  CCVariable<double>& sigma23 = tsk_info->get_field<CCVariable<double> >(m_sigma_t_names[2]);


  const Vector Dx = patch->dCell();
  const double delta = pow(Dx.x()*Dx.y()*Dx.z(),1./3.);
  const double dx = Dx.x();
  const double dy = Dx.y();
  const double dz = Dx.z();

   //
  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );

  Uintah::parallel_for( range, [&](int i, int j, int k){

    //apply u-mom bc -
    if ( eps(xm) * eps(c) > .5 ){
      // Y- sigma 12
      if ( eps(ym) * eps(xmym) < .5 ){
        const double i_so    = ( eps(i,j+m_standoff,k) * eps(i-1,j+m_standoff,k) > .5 ) ? m_standoff : 0;
        const double ISI     = 0.5 * ( IsI(i,j+i_so,k) + IsI(i-1,j+i_so,k) );
        const double rho_int = 0.5 *(rho(c)+rho(xm));
        const double mu_t    = pow( m_Cs * delta, 2.0 ) * rho_int * ISI;
        const double dudy    =  ( uVel(c) - uVel(ym)) / dy;
        // sigma12
        sigma12(c) +=  ( mu_t + m_molecular_visc ) * dudy;

      }
      // Y+ sigma 12
      if ( eps(yp) * eps(xmyp) < .5 ){
        const double i_so = ( eps(i,j-m_standoff,k) * eps(i-1,j-m_standoff,k) > .5 ) ?
                             m_standoff :
                             0;
        const double ISI     = 0.5*(IsI(i,j-i_so,k) + IsI(i-1,j-i_so,k));
        const double rho_int = 0.5 *(rho(c)+rho(xm));
        const double mu_t    = pow( m_Cs*delta, 2.0 ) * rho_int * ISI;

        const double dudy   =  ( uVel(yp) - uVel(c)) / dy;
        // sigma 12
        sigma12(c) +=  ( mu_t + m_molecular_visc )* dudy;

      }

        // Z- sigma 13
      if ( eps(zm) * eps(xmzm) < .5 ){
        const double i_so = ( eps(i,j,k+m_standoff) * eps(i-1,j,k+m_standoff) > .5 ) ?
                             m_standoff :
                             0;
        const double ISI     = 0.5 * ( IsI(i,j,k+i_so) + IsI(i-1,j,k+i_so) );
        const double rho_int = 0.5 *(rho(c)+rho(xm));
        const double mu_t    = pow( m_Cs * delta, 2.0 ) * rho_int * ISI;
        const double dudz    =   ( uVel(c)-uVel(zm) )/ dz;
        // sigma 13
        sigma13(c) += ( mu_t + m_molecular_visc ) * dudz;

      }

      // Z+ sigma 13
      if ( eps(zp) * eps(xmzp) < .5 ){
        const double i_so = ( eps(i,j,k-m_standoff) * eps(i-1,j,k-m_standoff) > .5 ) ?
                             m_standoff :
                             0;
        const double ISI     = 0.5*(IsI(i,j,k-i_so) + IsI(i-1,j,k-i_so));
        const double rho_int = 0.5 *(rho(c)+rho(xm));
        const double mu_t    = pow( m_Cs*delta, 2.0 ) * rho_int * ISI;
        const double dudz    =   ( uVel(zp) - uVel(c))/ dz;
        // sigma 13
        sigma13(c) +=  ( mu_t + m_molecular_visc ) * dudz;

      }
    }
    //apply v-mom bc -
    if ( eps(ym) * eps(c) > 0.5 ) {
      // X-
      if ( eps(xm) * eps(xmym) < .5 ){
        const double i_so = ( eps(i+m_standoff,j,k) * eps(i+m_standoff,j-1,k) > .5 ) ?
                             m_standoff :
                             0;
        const double ISI     = 0.5 * ( IsI(i+i_so,j,k) + IsI(i+i_so,j-1,k) );
        const double rho_int = 0.5 *(rho(c)+rho(ym));
        const double mu_t    = pow( m_Cs * delta, 2.0 ) * rho_int * ISI;
        const double dvdx    =   ( vVel(c) - vVel(xm))/ dx;
        // sigma 12
        sigma12(c) +=  ( mu_t + m_molecular_visc ) * dvdx ;

      }
      // X+
      if ( eps(xp) * eps(xpym) < .5 ){
        const double i_so = ( eps(i-m_standoff,j,k) * eps(i-m_standoff,j-1,k) > .5 ) ?
                             m_standoff :
                             0;
        const double ISI     = 0.5 * ( IsI(i-i_so,j,k) + IsI(i-i_so,j-1,k));
        const double rho_int = 0.5 *(rho(c)+rho(ym));
        const double mu_t    = pow( m_Cs * delta, 2.0 ) * rho_int * ISI;
        const double dvdx    = (vVel(xp)- vVel(c) )/ dx;

        // sigma 12
        sigma12(c) +=  ( mu_t + m_molecular_visc ) * dvdx;

      }
      // Z-
      if ( eps(zm) * eps(ymzm) < .5 ){
        const double i_so = ( eps(i,j,k+m_standoff) * eps(i,j-1,k+m_standoff) > .5 ) ?
                             m_standoff :
                             0;
        const double ISI     = 0.5 * ( IsI(i,j,k+i_so)+ IsI(i,j-1,k+i_so) );
        const double rho_int = 0.5 *(rho(c)+rho(ym));
        const double mu_t    = pow( m_Cs * delta, 2.0 ) * rho_int * ISI;
        const double dvdz    = (vVel(c)- vVel(zm) )/ dz;

        // sigma 23
        sigma23(c) += ( mu_t + m_molecular_visc ) * dvdz;

      }
      // Z+
      if ( eps(zp) * eps(ymzp) < .5 ){
        const double i_so = ( eps(i,j,k-m_standoff) * eps(i,j-1,k-m_standoff) > .5 ) ?
                             m_standoff :
                             0;
        const double ISI     = 0.5 * ( IsI(i,j,k-i_so) + IsI(i,j-1,k-i_so) );
        const double rho_int = 0.5 *(rho(c)+rho(ym));
        const double mu_t    = pow( m_Cs * delta, 2.0 ) * rho_int * ISI;
        const double dvdz    = (vVel(zp)- vVel(c) )/ dz;

        // sigma 23
        sigma23(c) +=  ( mu_t + m_molecular_visc ) * dvdz ;

      }
    }

      //apply w-mom bc -
    if ( eps(zm) * eps(c) > 0.5 ) {
      // X-
      if ( eps(xm) * eps(xmzm) < .5 ){
        const double i_so = ( eps(i+m_standoff,j,k) * eps(i+m_standoff,j,k-1) > .5 ) ?
                             m_standoff :
                             0;
        const double ISI     = 0.5 * ( IsI(i+i_so,j,k) + IsI(i+i_so,j,k-1));
        const double rho_int = 0.5 *(rho(c)+rho(zm));
        const double mu_t    = pow( m_Cs * delta, 2.0 ) * rho_int * ISI;
        const double dwdx    = (wVel(c)- wVel(xm) )/ dx;

        // sigma 13
        sigma13(c) +=  ( mu_t + m_molecular_visc ) * dwdx ;

      }
      // X+
      if ( eps(xp) * eps(xpzm) < .5 ){
        const double i_so = ( eps(i-m_standoff,j,k) * eps(i-m_standoff,j,k-1) > .5 ) ?
                             m_standoff :
                             0;
        const double ISI     = 0.5 * ( IsI(i-i_so,j,k) + IsI(i-i_so,j,k-1) );
        const double rho_int = 0.5 *(rho(c)+rho(zm));
        const double mu_t    = pow( m_Cs * delta, 2.0 ) * rho_int * ISI;
        const double dwdx    = (wVel(xp)- wVel(c) )/ dx;

        // sigma 13
        sigma13(c) += ( mu_t + m_molecular_visc ) * dwdx ;

      }
      // Y-
      if ( eps(ym) * eps(ymzm) < .5 ){
        const double i_so = ( eps(i,j+m_standoff,k) * eps(i,j+m_standoff,k-1) > .5 ) ?
                             m_standoff :
                             0;
        const double ISI     = 0.5 * ( IsI(i,j+i_so,k) + IsI(i,j+i_so,k-1));
        const double rho_int = 0.5 *(rho(c)+rho(zm));
        const double mu_t    = pow( m_Cs * delta, 2.0 ) * rho_int * ISI;
        const double dwdy    = (wVel(c)- wVel(ym) )/ dy;

        // sigma 23
        sigma23(c) += ( mu_t + m_molecular_visc ) * dwdy ;

      }
        // Y+
      if ( eps(yp) * eps(ypzm) < .5 ){
        const double i_so = ( eps(i,j-m_standoff,k) * eps(i,j-m_standoff,k-1) > .5 ) ?
                             m_standoff :
                             0;
        const double ISI     = 0.5*(IsI(i,j-i_so,k) + IsI(i,j-i_so,k-1));
        const double rho_int = 0.5 *(rho(c)+rho(zm));
        const double mu_t    = pow( m_Cs*delta, 2.0 ) * rho_int * ISI;
        const double dwdy    = (wVel(yp)- wVel(c) )/ dy;

        // sigma 23
        sigma23(c) += ( mu_t + m_molecular_visc ) * dwdy ;

      }
      }
  });


}
} //namespace Uintah
