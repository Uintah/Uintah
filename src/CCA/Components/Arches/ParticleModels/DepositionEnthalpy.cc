#include <CCA/Components/Arches/ParticleModels/DepositionEnthalpy.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/ParticleModels/CoalHelper.h>

using namespace Uintah;

DepositionEnthalpy::DepositionEnthalpy( std::string task_name, int matl_index, const int N, MaterialManagerP materialManager  ) :
TaskInterface( task_name, matl_index ), _Nenv(N),_materialManager(materialManager) {

}

DepositionEnthalpy::~DepositionEnthalpy(){
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DepositionEnthalpy::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DepositionEnthalpy::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &DepositionEnthalpy::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &DepositionEnthalpy::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &DepositionEnthalpy::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DepositionEnthalpy::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &DepositionEnthalpy::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &DepositionEnthalpy::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &DepositionEnthalpy::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DepositionEnthalpy::loadTaskTimestepInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DepositionEnthalpy::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void
DepositionEnthalpy::problemSetup( ProblemSpecP& db ){

  const ProblemSpecP db_root = db->getRootNode();
  CoalHelper& coal_helper = CoalHelper::self();

  _gasT_name="temperature";
  _cellType_name="cellType";
  _ratedepx_name="RateDepositionX";
  _ratedepy_name="RateDepositionY";
  _ratedepz_name="RateDepositionZ";
  _ash_enthalpy_src="AshEnthalpySrc";


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
  _temperature_base_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_TEMPERATURE);

  // Need a density
  _density_base_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_DENSITY);
  double init_particle_density = ArchesCore::get_inlet_particle_density( db );
  double ash_mass_frac = coal_helper.get_coal_db().ash_mf;
  double initial_diameter = 0.0;
  double p_volume = 0.0;
  for ( int i = 0; i < _Nenv; i++ ){
    initial_diameter = ArchesCore::get_inlet_particle_size( db, i );
    p_volume = M_PI/6.*initial_diameter*initial_diameter*initial_diameter; // particle volme [m^3]
    _mass_ash.push_back(p_volume*init_particle_density*ash_mass_frac);
  }

  if ( db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties") ){
    ProblemSpecP db_coal_props = db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties");
    db_coal_props->require("ash_enthalpy", _Ha0);
  } else {
    throw ProblemSetupException("Error: <Coal> is missing the <Properties> section.", __FILE__, __LINE__);
  }

}

void
DepositionEnthalpy::create_local_labels(){
  register_new_variable<CCVariable<double> >( m_task_name );
  register_new_variable<CCVariable<double> >( _ash_enthalpy_src );
}

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void
DepositionEnthalpy::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

  register_variable( m_task_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _ash_enthalpy_src, ArchesFieldContainer::COMPUTES, variable_registry );
}

template <typename ExecSpace, typename MemSpace>
void DepositionEnthalpy::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  CCVariable<double>& ash_enthalpy_flux = tsk_info->get_field<CCVariable<double> >(m_task_name);
  CCVariable<double>& ash_enthalpy_src = tsk_info->get_field<CCVariable<double> >(_ash_enthalpy_src);
  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
    ash_enthalpy_flux(i,j,k)=0.0;
    ash_enthalpy_src(i,j,k)=0.0;
  });

}

//--------------------------------------------------------------------------------------------------
void
DepositionEnthalpy::register_timestep_eval(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const int time_substep, const bool packed_tasks ){

  register_variable( _cellType_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW , variable_registry );
  register_variable( m_task_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _ash_enthalpy_src, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _gasT_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );

  for ( int i = 0; i < _Nenv; i++ ){
    const std::string RateDepositionX = get_env_name(i, _ratedepx_name);
    const std::string RateDepositionY = get_env_name(i, _ratedepy_name);
    const std::string RateDepositionZ = get_env_name(i, _ratedepz_name);
    const std::string diameter_name = get_env_name( i, _diameter_base_name );
    const std::string temperature_name = get_env_name( i, _temperature_base_name );
    const std::string density_name = get_env_name( i, _density_base_name );

    register_variable( RateDepositionX, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( RateDepositionY, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( RateDepositionZ, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( diameter_name , ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( temperature_name , ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( density_name , ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW , variable_registry );
  }

}

template <typename ExecSpace, typename MemSpace>
void DepositionEnthalpy::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

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
  double mp = 0.0;
  double marr = 0.0;
  double ash_frac = 0.0;
  double d_energy = 0.0;
  double d_energy_ash = 0.0;
  double delta_H = 0.0;
  double H_arriving = 0.0;
  double H_deposit = 0.0;
  double particle_volume = 0.0;
  double particle_diameter = 0.0;
  double particle_temperature = 0.0;
  double particle_density = 0.0;

  //double vel_i_ave = 0.0;
  IntVector lowPindex = patch->getCellLowIndex();
  IntVector highPindex = patch->getCellHighIndex();
  //Pad for ghosts
  lowPindex -= IntVector(1,1,1);
  highPindex += IntVector(1,1,1);

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );

  CCVariable<double>& ash_enthalpy_flux = tsk_info->get_field<CCVariable<double> >(m_task_name);
  ash_enthalpy_flux.initialize(0.0);
  CCVariable<double>& ash_enthalpy_src = tsk_info->get_field<CCVariable<double> >(_ash_enthalpy_src);
  ash_enthalpy_src.initialize(0.0);
  constCCVariable<int>& celltype = tsk_info->get_field<constCCVariable<int> >(_cellType_name);
  constCCVariable<double>& gasT = tsk_info->get_field<constCCVariable<double> >( _gasT_name );

  for( int i = 0; i < _Nenv; i++ ){

    const std::string RateDepositionX = get_env_name(i, _ratedepx_name);
    const std::string RateDepositionY = get_env_name(i, _ratedepy_name);
    const std::string RateDepositionZ = get_env_name(i, _ratedepz_name);
    const std::string diameter_name  = get_env_name( i, _diameter_base_name );
    const std::string temperature_name  = get_env_name( i, _temperature_base_name );
    const std::string density_name  = get_env_name( i, _density_base_name );
    constSFCXVariable<double>& dep_x = tsk_info->get_field<constSFCXVariable<double> >(RateDepositionX);
    constSFCYVariable<double>& dep_y = tsk_info->get_field<constSFCYVariable<double> >(RateDepositionY);
    constSFCZVariable<double>& dep_z = tsk_info->get_field<constSFCZVariable<double> >(RateDepositionZ);
    constCCVariable<double>& dp = tsk_info->get_field<constCCVariable<double> >( diameter_name );
    constCCVariable<double>& pT = tsk_info->get_field<constCCVariable<double> >( temperature_name );
    constCCVariable<double>& rhop = tsk_info->get_field<constCCVariable<double> >( density_name );

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
          d_energy = 0;
          d_energy_ash = 0;
          for ( int pp = 0; pp < total_flux_ind; pp++ ){
            if (container_flux_ind[pp]==0) {
	            particle_diameter = dp[c+_d[container_flux_ind[pp]]];//m
	            particle_temperature = pT[c+_d[container_flux_ind[pp]]];//K
	            particle_volume = M_PI/6.*particle_diameter*particle_diameter*particle_diameter;//m^3
              particle_density = rhop[c+_d[container_flux_ind[pp]]];
              flux = std::abs(dep_x[c+_fd[container_flux_ind[pp]]]);
            } else if (container_flux_ind[pp]==1) {
	            particle_diameter = dp[c+_d[container_flux_ind[pp]]];//m
	            particle_temperature = pT[c+_d[container_flux_ind[pp]]];//K
	            particle_volume = M_PI/6.*particle_diameter*particle_diameter*particle_diameter;//m^3
              particle_density = rhop[c+_d[container_flux_ind[pp]]];
              flux = std::abs(dep_x[c+_fd[container_flux_ind[pp]]]);
            } else if (container_flux_ind[pp]==2) {
	            particle_diameter = dp[c+_d[container_flux_ind[pp]]];//m
	            particle_temperature = pT[c+_d[container_flux_ind[pp]]];//K
	            particle_volume = M_PI/6.*particle_diameter*particle_diameter*particle_diameter;//m^3
              particle_density = rhop[c+_d[container_flux_ind[pp]]];
              flux = std::abs(dep_y[c+_fd[container_flux_ind[pp]]]);
            } else if (container_flux_ind[pp]==3) {
	            particle_diameter = dp[c+_d[container_flux_ind[pp]]];//m
	            particle_temperature = pT[c+_d[container_flux_ind[pp]]];//K
	            particle_volume = M_PI/6.*particle_diameter*particle_diameter*particle_diameter;//m^3
              particle_density = rhop[c+_d[container_flux_ind[pp]]];
              flux = std::abs(dep_y[c+_fd[container_flux_ind[pp]]]);
            } else if (container_flux_ind[pp]==4) {
	            particle_diameter = dp[c+_d[container_flux_ind[pp]]];//m
	            particle_temperature = pT[c+_d[container_flux_ind[pp]]];//K
	            particle_volume = M_PI/6.*particle_diameter*particle_diameter*particle_diameter;//m^3
              particle_density = rhop[c+_d[container_flux_ind[pp]]];
              flux = std::abs(dep_z[c+_fd[container_flux_ind[pp]]]);
            } else {
	            particle_diameter = dp[c+_d[container_flux_ind[pp]]];//m
	            particle_temperature = pT[c+_d[container_flux_ind[pp]]];//K
	            particle_volume = M_PI/6.*particle_diameter*particle_diameter*particle_diameter;//m^3
              particle_density = rhop[c+_d[container_flux_ind[pp]]];
              flux = std::abs(dep_z[c+_fd[container_flux_ind[pp]]]);
            }
            // energy release/gain from persepective of the particle:
            // dH = integral(cp dT)
            // from Merrick 1981: for ash integral(cp dT) = pT * (593. + pT * 0.293)
            H_arriving = -202849.0 + _Ha0 + particle_temperature * (593. + particle_temperature * 0.293); // [J/kg]
            H_deposit = -202849.0 + _Ha0 + gasT[c] * (593. + gasT[c] * 0.293); // [J/kg]
            delta_H = H_arriving - H_deposit; // J/kg

            mp = particle_volume*particle_density; // current mass of the particle [kg]
            marr = _mass_ash[i]; // mass of ash in the particle[kg].
            ash_frac = marr/mp; // fraction of flux that is ash.
            flux = flux*ash_frac;// arriving mass flux [kg/m^2/s]
            d_energy += flux*delta_H*area_face[container_flux_ind[pp]]; // [J/s]
            d_energy_ash += flux*H_arriving*area_face[container_flux_ind[pp]]; // [J/s]
            total_area_face += area_face[container_flux_ind[pp]];
          }
          // compute the current deposit velocity for each particle: d_vel [m3/s * 1/m2 = m/s]
          d_energy /= total_area_face; // energy flux
          ash_enthalpy_flux[c] += d_energy; // [J/s/m^2] add the contribution
          ash_enthalpy_src[c] += d_energy_ash/Vcell; // [J/s/m^3] for particle energy balance.
        }// if there is a deposition flux
      } // wall or intrusion cell-type
    } // cell loop
  } // environment loop
}
