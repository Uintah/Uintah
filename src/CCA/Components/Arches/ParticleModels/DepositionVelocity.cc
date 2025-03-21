#include <CCA/Components/Arches/ParticleModels/DepositionVelocity.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>
#include <CCA/Components/Arches/ParticleModels/CoalHelper.h>

using namespace Uintah;

DepositionVelocity::DepositionVelocity( std::string task_name, int matl_index, const int N, MaterialManagerP materialManager  ) :
TaskInterface( task_name, matl_index ), _Nenv(N),_materialManager(materialManager) {

}

DepositionVelocity::~DepositionVelocity(){
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DepositionVelocity::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DepositionVelocity::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &DepositionVelocity::initialize<UINTAH_CPU_TAG>               // Task supports non-Kokkos builds
                                     //, &DepositionVelocity::initialize<KOKKOS_OPENMP_TAG>          // Task supports Kokkos::OpenMP builds
                                     //, &DepositionVelocity::initialize<KOKKOS_DEFAULT_HOST_TAG>    // Task supports Kokkos::DefaultHostExecutionSpace builds
                                     //, &DepositionVelocity::initialize<KOKKOS_DEFAULT_DEVICE_TAG>  // Task supports Kokkos::DefaultExecutionSpace builds
                                     //, &DepositionVelocity::initialize<KOKKOS_DEFAULT_DEVICE_TAG>            // Task supports Kokkos builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DepositionVelocity::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &DepositionVelocity::eval<UINTAH_CPU_TAG>               // Task supports non-Kokkos builds
                                     //, &DepositionVelocity::eval<KOKKOS_OPENMP_TAG>          // Task supports Kokkos::OpenMP builds
                                     //, &DepositionVelocity::eval<KOKKOS_DEFAULT_HOST_TAG>    // Task supports Kokkos::DefaultHostExecutionSpace builds
                                     //, &DepositionVelocity::eval<KOKKOS_DEFAULT_DEVICE_TAG>  // Task supports Kokkos::DefaultExecutionSpace builds
                                     //, &DepositionVelocity::eval<KOKKOS_DEFAULT_DEVICE_TAG>            // Task supports Kokkos builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DepositionVelocity::loadTaskTimestepInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DepositionVelocity::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void
DepositionVelocity::problemSetup( ProblemSpecP& db ){

  const ProblemSpecP db_root = db->getRootNode();
  CoalHelper& coal_helper = CoalHelper::self();

  double p_void0;
  _ash_mass_src="AshMassSrc";
  _cellType_name="cellType";
  _impact_velocity_name = m_task_name + "_impact";
  _impact_mass_src = "MassImpactLoss";
  _impact_org_wall_flow ="OrganicFlowImpLossWall";
  _impact_ash_wall_flow ="AshFlowImpLossWall";

  _ratedepx_name="RateDepositionX";
  _ratedepy_name="RateDepositionY";
  _ratedepz_name="RateDepositionZ";

  _rateImpactx_name="RateImpactLossX";
  _rateImpacty_name="RateImpactLossY";
  _rateImpactz_name="RateImpactLossZ";

  if ( db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("BoundaryConditions")->findBlock("WallHT") ){
    ProblemSpecP wallht_db = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("BoundaryConditions")->findBlock("WallHT");
    wallht_db->getWithDefault("relaxation_coef",_relaxation_coe,1.0);
    bool error_flag = true;
    std::string ht_model_type;
    for ( ProblemSpecP db_wall_model = wallht_db->findBlock( "model" ); db_wall_model != nullptr; db_wall_model = db_wall_model->findNextBlock( "model" ) ){
      db_wall_model->getAttribute("type",ht_model_type);
      if (ht_model_type == "coal_region_ht"){
        db_wall_model->getWithDefault( "sb_deposit_porosity",p_void0,0.6); // note here we are using the sb layer to estimate the wall density no the enamel layer.
        error_flag = false;
      }
    }
    if (error_flag)
      throw InvalidValue("Error: DepositionVelocity model requires WallHT model of type coal_region_ht.", __FILE__, __LINE__);
  } else {
    throw InvalidValue("Error: DepositionVelocity model requires WallHT model for relaxation coefficient.", __FILE__, __LINE__);
  }

  double rho_ash_bulk;
  if ( db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")){
    ProblemSpecP db_part_properties =  db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties");
    db_part_properties->getWithDefault( "rho_ash_bulk",rho_ash_bulk,2300.0);
  } else {
    throw InvalidValue("Error: DepositionVelocity model requires ParticleProperties to be specified.", __FILE__, __LINE__);
  }

  _user_specified_rho = rho_ash_bulk * (1 - p_void0);

  _d.push_back(IntVector(1,0,0)); // cell center located +x
  _d.push_back(IntVector(-1,0,0)); // cell center located -x
  _d.push_back(IntVector(0,1,0)); // cell center located +y
  _d.push_back(IntVector(0,-1,0)); // cell center located -y
  _d.push_back(IntVector(0,0,1)); // cell center located +z
  _d.push_back(IntVector(0,0,-1)); // cell center located -z
  _fd.push_back(IntVector(1,0,0)); // +x face
  _fd.push_back(IntVector(0,0,0)); // -x face
  _fd.push_back(IntVector(0,1,0)); // +y face
  _fd.push_back(IntVector(0,0,0)); // -y face
  _fd.push_back(IntVector(0,0,1)); // +z face
  _fd.push_back(IntVector(0,0,0)); // -z face
  _diameter_base_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_SIZE);

  // Need a density
  _density_base_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_DENSITY);
  double init_particle_density = ArchesCore::get_inlet_particle_density( db );
  double ash_mass_frac = coal_helper.get_coal_db().ash_mf;
  double initial_diameter = 0.0;
  double p_volume = 0.0;
  for ( int n = 0; n < _Nenv; n++ ){
    initial_diameter = ArchesCore::get_inlet_particle_size( db, n );
    p_volume = M_PI/6.*initial_diameter*initial_diameter*initial_diameter; // particle volme [m^3]
    _mass_ash.push_back(p_volume*init_particle_density*ash_mass_frac);
  }
  double retained_value = 1.0;
  if(db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources")){
    const ProblemSpecP sources = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources");
    for ( ProblemSpecP sourceBlock = sources->findBlock("src"); sourceBlock != nullptr;
    sourceBlock = sourceBlock->findNextBlock("src") ) {
      std::string tempTypeName;
      sourceBlock->getAttribute("type",tempTypeName);
      if (tempTypeName == "coal_gas_devol"){
        std::string tempTypeName;
        if(sourceBlock->findBlock("devol_BirthDeath")){
          retained_value = 0.0;
        }
      }
    }
  } else {
    throw ProblemSetupException("Error: cannot find the Sources block which requries the devol source for obtaining retained_deposit_factor in arches block.",__FILE__,__LINE__);
  }
  _retained_deposit_factor = retained_value;
}

void
DepositionVelocity::create_local_labels(){

  register_new_variable<CCVariable<double> >( m_task_name );
  register_new_variable<CCVariable<double> >( _ash_mass_src );
  register_new_variable<CCVariable<double> >( _impact_velocity_name);
  register_new_variable<CCVariable<double> >( _impact_mass_src );
  register_new_variable<CCVariable<double> > (_impact_org_wall_flow);
  register_new_variable<CCVariable<double> > (_impact_ash_wall_flow);

  for ( int n = 0; n < _Nenv; n++ ){
    const std::string d_vol_ave_num_s = ArchesCore::append_env("d_vol_ave_num",n);
    const std::string d_vol_ave_den_s = ArchesCore::append_env("d_vol_ave_den",n);
    register_new_variable<CCVariable<double> >( d_vol_ave_num_s );
    register_new_variable<CCVariable<double> >( d_vol_ave_den_s );
  }
}

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void
DepositionVelocity::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

  register_variable( m_task_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable(_impact_velocity_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable(_impact_mass_src, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _ash_mass_src, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _impact_org_wall_flow, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _impact_ash_wall_flow, ArchesFieldContainer::COMPUTES, variable_registry );

  for ( int n = 0; n < _Nenv; n++ ){
    const std::string d_vol_ave_num_s = ArchesCore::append_env("d_vol_ave_num",n);
    const std::string d_vol_ave_den_s = ArchesCore::append_env("d_vol_ave_den",n);
    register_variable( d_vol_ave_num_s, ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( d_vol_ave_den_s, ArchesFieldContainer::COMPUTES, variable_registry );
  }
}

template <typename ExecSpace, typename MemSpace>
void DepositionVelocity::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  CCVariable<double>& deposit_velocity = tsk_info->get_field<CCVariable<double> >(m_task_name);
  CCVariable<double>& impact_velocity  = tsk_info->get_field<CCVariable<double> >(_impact_velocity_name);
  CCVariable<double>& impact_mass_src  = tsk_info->get_field<CCVariable<double> >(_impact_mass_src);
  CCVariable<double>& flow_org_wall    = tsk_info->get_field<CCVariable<double> >(_impact_org_wall_flow);
  CCVariable<double>& flow_ash_wall    = tsk_info->get_field<CCVariable<double> >(_impact_ash_wall_flow);
  CCVariable<double>& ash_mass_src     = tsk_info->get_field<CCVariable<double> >(_ash_mass_src);
  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
    ash_mass_src(i,j,k)     = 0.0;
    deposit_velocity(i,j,k) = 0.0;
    impact_velocity(i,j,k)  = 0.0;
    impact_mass_src(i,j,k)  = 0.0;
    flow_org_wall(i,j,k)    = 0.0 ;
    flow_ash_wall(i,j,k)    = 0.0 ;
  });
  for ( int n = 0; n < _Nenv; n++ ){
    const std::string d_vol_ave_num_s = ArchesCore::append_env("d_vol_ave_num",n);
    const std::string d_vol_ave_den_s = ArchesCore::append_env("d_vol_ave_den",n);
    CCVariable<double>& d_vol_ave_num = tsk_info->get_field<CCVariable<double> >(d_vol_ave_num_s);
    CCVariable<double>& d_vol_ave_den = tsk_info->get_field<CCVariable<double> >(d_vol_ave_den_s);
    Uintah::parallel_for( range, [&](int i, int j, int k){
      d_vol_ave_num(i,j,k)=0.0;
      d_vol_ave_den(i,j,k)=0.0;
    });
  }
}

//--------------------------------------------------------------------------------------------------
void
DepositionVelocity::register_timestep_eval(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const int time_substep, const bool packed_tasks ){

  register_variable( _cellType_name, ArchesFieldContainer::NEEDSLABEL, 1, ArchesFieldContainer::OLDDW , variable_registry );
  register_variable( m_task_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_task_name, ArchesFieldContainer::NEEDSLABEL, 1, ArchesFieldContainer::OLDDW, variable_registry  );
  register_variable( _ash_mass_src, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _impact_mass_src, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _impact_org_wall_flow, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _impact_ash_wall_flow, ArchesFieldContainer::COMPUTES, variable_registry );


  register_variable( _impact_velocity_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _impact_velocity_name, ArchesFieldContainer::NEEDSLABEL, 1, ArchesFieldContainer::OLDDW, variable_registry  );

  for ( int n = 0; n < _Nenv; n++ ){
    const std::string d_vol_ave_num_s = ArchesCore::append_env("d_vol_ave_num",n);
    const std::string d_vol_ave_den_s = ArchesCore::append_env("d_vol_ave_den",n);
    register_variable( d_vol_ave_num_s, ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( d_vol_ave_num_s, ArchesFieldContainer::NEEDSLABEL, 0, ArchesFieldContainer::OLDDW, variable_registry );
    register_variable( d_vol_ave_den_s, ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( d_vol_ave_den_s, ArchesFieldContainer::NEEDSLABEL, 0, ArchesFieldContainer::OLDDW, variable_registry );

    const std::string RateDepositionX = get_env_name(n, _ratedepx_name);
    const std::string RateDepositionY = get_env_name(n, _ratedepy_name);
    const std::string RateDepositionZ = get_env_name(n, _ratedepz_name);
    const std::string diameter_name = get_env_name(n, _diameter_base_name );
    const std::string density_name = get_env_name(n, _density_base_name );
    register_variable( RateDepositionX, ArchesFieldContainer::NEEDSLABEL, 1, ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( RateDepositionY, ArchesFieldContainer::NEEDSLABEL, 1, ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( RateDepositionZ, ArchesFieldContainer::NEEDSLABEL, 1, ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( diameter_name , ArchesFieldContainer::NEEDSLABEL, 1, ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( density_name , ArchesFieldContainer::NEEDSLABEL, 1, ArchesFieldContainer::NEWDW , variable_registry );

    const std::string RateImpactX = get_env_name(n, _rateImpactx_name);
    const std::string RateImpactY = get_env_name(n, _rateImpacty_name);
    const std::string RateImpactZ = get_env_name(n, _rateImpactz_name);

    register_variable( RateImpactX, ArchesFieldContainer::NEEDSLABEL, 1, ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( RateImpactY, ArchesFieldContainer::NEEDSLABEL, 1, ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( RateImpactZ, ArchesFieldContainer::NEEDSLABEL, 1, ArchesFieldContainer::NEWDW , variable_registry );



  }

}

template <typename ExecSpace, typename MemSpace>
void DepositionVelocity::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  const int FLOW = -1;
  Vector Dx = patch->dCell(); // cell spacing
  double DxDy = Dx.x() * Dx.y();
  double DxDz = Dx.x() * Dx.z();
  double DyDz = Dx.y() * Dx.z();
  double Vcell = Dx.x() * Dx.y() * Dx.z();
  std::vector<double> area_face;
  area_face.push_back(DyDz);
  area_face.push_back(DyDz);
  area_face.push_back(DxDz);
  area_face.push_back(DxDz);
  area_face.push_back(DxDy);
  area_face.push_back(DxDy);
  std::vector<int> container_flux_ind;
  int total_flux_ind = 0;
  double total_area_face = 0.0;
  double flux = 0.0;
  double flux_impact = 0.0;
  double mp = 0.0;
  double marr = 0.0;
  double ash_frac = 0.0;
  double rhoi = 0.0;
  double d_mass = 0.0;
  double d_mass_impact     = 0.0;
  double d_mass_impact_org = 0.0;
  double d_mass_impact_ash = 0.0;
  double d_flow = 0.0;
  double d_flow_impact = 0.0;
  double mo = 0.0;
  double d_velocity = 0.0;
  double i_velocity = 0.0;
  double env_flow_rate = 0.0;
  double env_flow_rate_d = 0.0;
  double particle_volume = 0.0;
  double particle_diameter = 0.0;
  double particle_density = 0.0;

  //double vel_i_ave = 0.0;
  IntVector lowPindex = patch->getCellLowIndex();
  IntVector highPindex = patch->getCellHighIndex();
  //Pad for ghosts
  lowPindex -= IntVector(1,1,1);
  highPindex += IntVector(1,1,1);

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );

  CCVariable<double>& deposit_velocity = tsk_info->get_field<CCVariable<double> >(m_task_name);
  deposit_velocity.initialize(0.0);

  CCVariable<double>& impact_velocity = tsk_info->get_field<CCVariable<double> >(_impact_velocity_name);
  impact_velocity.initialize(0.0);

  CCVariable<double>& ash_mass_src = tsk_info->get_field<CCVariable<double> >(_ash_mass_src);
  ash_mass_src.initialize(0.0);
  CCVariable<double>& flow_org_wall = tsk_info->get_field<CCVariable<double> >(_impact_org_wall_flow);
  CCVariable<double>& flow_ash_wall = tsk_info->get_field<CCVariable<double> >(_impact_ash_wall_flow);
  flow_org_wall.initialize(0.0);
  flow_ash_wall.initialize(0.0);

  CCVariable<double>& impact_mass_src = tsk_info->get_field<CCVariable<double> >(_impact_mass_src);
  impact_mass_src.initialize(0.0);

  constCCVariable<double>& deposit_velocity_old = tsk_info->get_field<constCCVariable<double> >(m_task_name);
  constCCVariable<double>& impact_velocity_old  = tsk_info->get_field<constCCVariable<double> >(_impact_velocity_name);
  constCCVariable<int>& celltype = tsk_info->get_field<constCCVariable<int> >(_cellType_name);

  for( int n = 0; n < _Nenv; n++ ){

    const std::string d_vol_ave_num_s = ArchesCore::append_env("d_vol_ave_num",n);
    const std::string d_vol_ave_den_s = ArchesCore::append_env("d_vol_ave_den",n);
    CCVariable<double>& d_vol_ave_num = tsk_info->get_field<CCVariable<double> >(d_vol_ave_num_s);
    d_vol_ave_num.initialize(0.0);
    CCVariable<double>& d_vol_ave_den = tsk_info->get_field<CCVariable<double> >(d_vol_ave_den_s);
    d_vol_ave_den.initialize(0.0);
    constCCVariable<double>& d_vol_ave_num_old = tsk_info->get_field<constCCVariable<double> >(d_vol_ave_num_s);
    constCCVariable<double>& d_vol_ave_den_old = tsk_info->get_field<constCCVariable<double> >(d_vol_ave_den_s);

    const std::string RateDepositionX = get_env_name(n, _ratedepx_name);
    const std::string RateDepositionY = get_env_name(n, _ratedepy_name);
    const std::string RateDepositionZ = get_env_name(n, _ratedepz_name);
    const std::string diameter_name  = get_env_name(n, _diameter_base_name );
    const std::string density_name  = get_env_name(n, _density_base_name );
    constSFCXVariable<double>& dep_x = tsk_info->get_field<constSFCXVariable<double> >(RateDepositionX);
    constSFCYVariable<double>& dep_y = tsk_info->get_field<constSFCYVariable<double> >(RateDepositionY);
    constSFCZVariable<double>& dep_z = tsk_info->get_field<constSFCZVariable<double> >(RateDepositionZ);
    constCCVariable<double>& dp = tsk_info->get_field<constCCVariable<double> >( diameter_name );
    constCCVariable<double>& rhop = tsk_info->get_field<constCCVariable<double> >( density_name );

    const std::string RateImpactX = get_env_name(n, _rateImpactx_name);
    const std::string RateImpactY = get_env_name(n, _rateImpacty_name);
    const std::string RateImpactZ = get_env_name(n, _rateImpactz_name);
    constSFCXVariable<double>& imp_x = tsk_info->get_field<constSFCXVariable<double> >( RateImpactX );
    constSFCYVariable<double>& imp_y = tsk_info->get_field<constSFCYVariable<double> >( RateImpactY );
    constSFCZVariable<double>& imp_z = tsk_info->get_field<constSFCZVariable<double> >( RateImpactZ );

    for (CellIterator iter=patch->getExtraCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;

      container_flux_ind.clear();
      // first determine if the current cell a wall or intrusion cell?
      if ( celltype[c] == BoundaryCondition_new::WALL || celltype[c] == BoundaryCondition_new::INTRUSION ){
        total_flux_ind = 0;
        // Now loop over all surrounding faces to determine WALL or INTRUSION cell is exposed to a flow cells
        for ( int face = 0; face < 6; face++ ){
          // if we are at a wall (i.e. -1,0,0) we want to make sure we aren't searching beyond (i.e. -2,0,0).
          if ( patch->containsIndex(lowPindex, highPindex, c + _d[face] ) )
          {
            // if the neighbor in the current direction is a flow cell, then c is now active.
            if ( celltype[c + _d[face]] == FLOW )
            {
              container_flux_ind.push_back(face);
              total_flux_ind+=1;
            }
          }
        }
        if ( total_flux_ind>0 )
        {
          total_area_face = 0;
          d_mass = 0;
          d_flow = 0;
          d_velocity = 0;
          env_flow_rate = 0;
          env_flow_rate_d = 0;
          // impact
          i_velocity      = 0;
          d_flow_impact   = 0;
          d_mass_impact   = 0;
          d_mass_impact_org = 0;
          d_mass_impact_ash = 0;

          for ( int pp = 0; pp < total_flux_ind; pp++ ){
            if (container_flux_ind[pp]==0) {
                    particle_diameter = dp[c+_d[container_flux_ind[pp]]];//m
                    particle_volume = M_PI/6.*particle_diameter*particle_diameter*particle_diameter;//m^3
              particle_density = rhop[c+_d[container_flux_ind[pp]]];
              flux = std::abs(dep_x[c+_fd[container_flux_ind[pp]]]);
              //
              flux_impact = std::abs(imp_x[c+_fd[container_flux_ind[pp]]]);
            } else if (container_flux_ind[pp]==1) {
                    particle_diameter = dp[c+_d[container_flux_ind[pp]]];//m
                    particle_volume = M_PI/6.*particle_diameter*particle_diameter*particle_diameter;//m^3
              particle_density = rhop[c+_d[container_flux_ind[pp]]];
              flux = std::abs(dep_x[c+_fd[container_flux_ind[pp]]]);
              //
              flux_impact = std::abs(imp_x[c+_fd[container_flux_ind[pp]]]);
            } else if (container_flux_ind[pp]==2) {
                    particle_diameter = dp[c+_d[container_flux_ind[pp]]];//m
                    particle_volume = M_PI/6.*particle_diameter*particle_diameter*particle_diameter;//m^3
              particle_density = rhop[c+_d[container_flux_ind[pp]]];
              flux = std::abs(dep_y[c+_fd[container_flux_ind[pp]]]);
              //
              flux_impact = std::abs(imp_y[c+_fd[container_flux_ind[pp]]]);
            } else if (container_flux_ind[pp]==3) {
                    particle_diameter = dp[c+_d[container_flux_ind[pp]]];//m
                    particle_volume = M_PI/6.*particle_diameter*particle_diameter*particle_diameter;//m^3
              particle_density = rhop[c+_d[container_flux_ind[pp]]];
              flux = std::abs(dep_y[c+_fd[container_flux_ind[pp]]]);
              //
              flux_impact = std::abs(imp_y[c+_fd[container_flux_ind[pp]]]);
            } else if (container_flux_ind[pp]==4) {
                    particle_diameter = dp[c+_d[container_flux_ind[pp]]];//m
                    particle_volume = M_PI/6.*particle_diameter*particle_diameter*particle_diameter;//m^3
              particle_density = rhop[c+_d[container_flux_ind[pp]]];
              flux = std::abs(dep_z[c+_fd[container_flux_ind[pp]]]);
              //
              flux_impact = std::abs(imp_z[c+_fd[container_flux_ind[pp]]]);
            } else {
                    particle_diameter = dp[c+_d[container_flux_ind[pp]]];//m
                    particle_volume = M_PI/6.*particle_diameter*particle_diameter*particle_diameter;//m^3
              particle_density = rhop[c+_d[container_flux_ind[pp]]];
              flux = std::abs(dep_z[c+_fd[container_flux_ind[pp]]]);
              //
              flux_impact = std::abs(imp_z[c+_fd[container_flux_ind[pp]]]);
            }
            mp = particle_volume*particle_density; // current mass of the particle [kg]
            mo = mp - _mass_ash[n]; // current mass of organic in the particle [kg]
            marr = mo * _retained_deposit_factor + _mass_ash[n]; // mass arriving at the wall [kg].
            ash_frac = marr/mp; // correction factor to flux
            flux = flux*ash_frac;// arriving mass flux [kg/m^2/s]
            rhoi = _user_specified_rho;
            d_mass += flux*area_face[container_flux_ind[pp]];// [kg/s] ash
            // volumetric flow rate for particle i:
            d_flow = (flux/rhoi) * area_face[container_flux_ind[pp]]+1e-100;// [m^3/s] ash
            d_velocity += d_flow;
            // The total cell surface area exposed to radiation:
            total_area_face += area_face[container_flux_ind[pp]];
                  env_flow_rate += d_flow; // m^3 / s * dp
                  env_flow_rate_d += d_flow * particle_diameter; // m^3 / s

            // particle that reach the walls but not are deposited
            const double ash_frac2 = _mass_ash[n]/mp; //fraction of ash in particle without  _retained_deposit_factor
            d_mass_impact += flux_impact*area_face[container_flux_ind[pp]];// [kg/s] ash+organic
            d_mass_impact_org += flux_impact*area_face[container_flux_ind[pp]]*(1.-ash_frac2); // [kg/s] organic
            d_mass_impact_ash += flux_impact*area_face[container_flux_ind[pp]]*ash_frac2; // [kg/s] ash

            d_flow_impact = (flux_impact/particle_density) * area_face[container_flux_ind[pp]]+1e-100;// [m^3/s] ash + organic
            i_velocity +=  d_flow_impact;

          }
          // compute the current deposit velocity for each particle: d_vel [m3/s * 1/m2 = m/s]
          d_velocity /= total_area_face; // area weighted incoming velocity to the cell for particle i.
          d_vol_ave_num[c] = (1.0-_relaxation_coe)*d_vol_ave_num_old[c] + _relaxation_coe*env_flow_rate_d; // this is the time-averaged flow rate * d for environment i
          d_vol_ave_den[c] = (1.0-_relaxation_coe)*d_vol_ave_den_old[c] + _relaxation_coe*env_flow_rate; // this is the time-averaged flow rate for the environment i
          deposit_velocity[c] += d_velocity; // add the contribution for the deposition
          ash_mass_src[c] += d_mass/Vcell; // [kg/s/m^3] for particle mass balance.
          // overall we are trying to achieve:  v_hat = (1-alpha)*v_hat_old + alpha*v_new. We initialize v_hat to v_hat_old*(1-alpha) in timestep_init (first term).
          // we handle the second term using the summation alpha*v_new = alpha * (v1 + v2 + v3):
          // v_hat += alpha*v1 + alpha*v2 + alpha*v3

          i_velocity /= total_area_face;
          impact_velocity[c] += i_velocity;
          impact_mass_src[c] += d_mass_impact/Vcell; // [kg/s/m^3] for particle mass balance.
          flow_org_wall[c]   = d_mass_impact_org;   // [kg/s] of organic mass that will leave the computation domain
          flow_ash_wall[c]   = d_mass_impact_ash; // [kg/s] of ash

        }// if there is a deposition flux
      } // wall or intrusion cell-type
    } // cell loop
  } // environment loop
  Uintah::parallel_for( range, [&](int i, int j, int k){
    deposit_velocity(i,j,k) = (1.0 - _relaxation_coe) * deposit_velocity_old(i,j,k) + _relaxation_coe * deposit_velocity(i,j,k); // time-average the deposition rate.
    impact_velocity(i,j,k) = (1.0 - _relaxation_coe) * impact_velocity_old(i,j,k) + _relaxation_coe * impact_velocity(i,j,k); // time-average the deposition rate.

    // note here that when the flux is away from the wall the volume average size calculation can be negative.. so we simply set the particle size to 1e-8.
  });
}
