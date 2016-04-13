#include <CCA/Components/Arches/ParticleModels/DepositionVelocity.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>

#include <spatialops/structured/FVStaggered.h>

using namespace Uintah;

DepositionVelocity::DepositionVelocity( std::string task_name, int matl_index, const int N, SimulationStateP shared_state  ) :
TaskInterface( task_name, matl_index ), _Nenv(N),_shared_state(shared_state) {

}

DepositionVelocity::~DepositionVelocity(){
}

void
DepositionVelocity::problemSetup( ProblemSpecP& db ){

  const ProblemSpecP db_root = db->getRootNode();

  _cellType_name="cellType";
  _ratedepx_name="RateDepositionX";
  _ratedepy_name="RateDepositionY";
  _ratedepz_name="RateDepositionZ";
  _rhoP_name  = ParticleTools::parse_for_role_to_label(db,"density");
  _dep_vel_rs_name= "DepositVelRunningSum";
  _dep_vel_rs_start_name= "DepositVelRunningSumStart";
  _new_time_name= "current_interval_time";
  if ( db->findBlock("t_interval")){
  db->require("t_interval",_t_interval);
  } else {
    throw ProblemSetupException("Error: DepositionVelocity.cc time-averaging start time not specified.", __FILE__, __LINE__);
  }
  db->getWithDefault("ash_density",_user_specified_rho,-1.0);
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
  db->getWithDefault("current_time_in_interval",_new_time,0.0);
}

void
DepositionVelocity::create_local_labels(){

  register_new_variable<CCVariable<double> >( _task_name );
  register_new_variable<CCVariable<double> >( _new_time_name );
  for ( int i = 0; i < _Nenv; i++ ){
    const std::string dep_vel_rs = get_env_name(i, _dep_vel_rs_name);
    const std::string dep_vel_rs_start = get_env_name(i, _dep_vel_rs_start_name);
    register_new_variable<CCVariable<double> >( dep_vel_rs );
    register_new_variable<CCVariable<double> >( dep_vel_rs_start );

  }
}

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void
DepositionVelocity::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

  register_variable( _task_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _new_time_name, ArchesFieldContainer::COMPUTES, variable_registry );
  for ( int i = 0; i < _Nenv; i++ ){
    const std::string dep_vel_rs = get_env_name(i, _dep_vel_rs_name);
    const std::string dep_vel_rs_start = get_env_name(i, _dep_vel_rs_start_name);
    register_variable( dep_vel_rs, ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( dep_vel_rs_start, ArchesFieldContainer::COMPUTES, variable_registry );
  }

}

void
DepositionVelocity::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
    SpatialOps::OperatorDatabase& opr ){

  CCVariable<double>* vdeposit_velocity = tsk_info->get_uintah_field<CCVariable<double> >(_task_name);
  CCVariable<double>& deposit_velocity = *vdeposit_velocity;
  CCVariable<double>* vnew_time = tsk_info->get_uintah_field<CCVariable<double> >(_new_time_name);
  CCVariable<double>& new_time = *vnew_time;
  for (CellIterator iter=patch->getExtraCellIterator(); !iter.done(); iter++){
    IntVector c = *iter;
    deposit_velocity[c]=0.0;
    new_time[c]=_new_time;
  }
  for ( int i = 0; i < _Nenv; i++ ){
    const std::string dep_vel_rs = get_env_name(i, _dep_vel_rs_name);
    const std::string dep_vel_rs_start = get_env_name(i, _dep_vel_rs_start_name);
    CCVariable<double>* vd_velocity_rs = tsk_info->get_uintah_field<CCVariable<double> >(dep_vel_rs);
    CCVariable<double>& d_velocity_rs = *vd_velocity_rs;
    d_velocity_rs.initialize(0.0);
    CCVariable<double>* vd_velocity_rs_start = tsk_info->get_uintah_field<CCVariable<double> >(dep_vel_rs_start);
    CCVariable<double>& d_velocity_rs_start = *vd_velocity_rs_start;
    d_velocity_rs_start.initialize(0.0);
    for (CellIterator iter=patch->getExtraCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      d_velocity_rs[c]=0.0;
      d_velocity_rs_start[c]=0.0;
    }
  }
}

//
//------------------------------------------------
//------------- TIMESTEP INIT --------------------
//------------------------------------------------
//
void
DepositionVelocity::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

  register_variable( _task_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _task_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW, variable_registry  ); 
  register_variable( _new_time_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW, variable_registry  ); 
  register_variable( _new_time_name, ArchesFieldContainer::COMPUTES, variable_registry );
  for ( int i = 0; i < _Nenv; i++ ){
    const std::string dep_vel_rs = get_env_name(i, _dep_vel_rs_name);
    const std::string dep_vel_rs_start = get_env_name(i, _dep_vel_rs_start_name);
    register_variable( dep_vel_rs, ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( dep_vel_rs_start, ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( dep_vel_rs, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW, variable_registry  ); 
    register_variable( dep_vel_rs_start, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW, variable_registry  ); 
  }

}

void
DepositionVelocity::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
    SpatialOps::OperatorDatabase& opr ){

  CCVariable<double>* vdeposit_velocity = tsk_info->get_uintah_field<CCVariable<double> >(_task_name);
  CCVariable<double>& deposit_velocity = *vdeposit_velocity;
  constCCVariable<double>* vdeposit_velocity_old = tsk_info->get_const_uintah_field<constCCVariable<double> >(_task_name);
  constCCVariable<double>& deposit_velocity_old = *vdeposit_velocity_old;
  constCCVariable<double>* vnew_time_old = tsk_info->get_const_uintah_field<constCCVariable<double> >(_new_time_name);
  constCCVariable<double>& new_time_old = *vnew_time_old;
  CCVariable<double>* vnew_time = tsk_info->get_uintah_field<CCVariable<double> >(_new_time_name);
  CCVariable<double>& new_time = *vnew_time;

  const double delta_t = tsk_info->get_dt();
  for (CellIterator iter=patch->getExtraCellIterator(); !iter.done(); iter++){
    IntVector c = *iter;
    new_time[c] = new_time_old[c] + delta_t; // this is required for determining when to reset running sums for time-averaging.
    if (new_time[c] > _t_interval){
      new_time[c] = 0.0; // this is required for determining when to reset running sums for time-averaging.
      deposit_velocity[c]=0.0;
    } else {
      deposit_velocity[c]=deposit_velocity_old[c];
    }
  }
  
  for ( int i = 0; i < _Nenv; i++ ){
    const std::string dep_vel_rs = get_env_name(i, _dep_vel_rs_name);
    const std::string dep_vel_rs_start = get_env_name(i, _dep_vel_rs_start_name);
    CCVariable<double>* vd_velocity_rs = tsk_info->get_uintah_field<CCVariable<double> >(dep_vel_rs);
    CCVariable<double>& d_velocity_rs = *vd_velocity_rs;
    constCCVariable<double>* vd_velocity_rs_old = tsk_info->get_const_uintah_field<constCCVariable<double> >(dep_vel_rs);
    constCCVariable<double>& d_velocity_rs_old = *vd_velocity_rs_old;
    CCVariable<double>* vd_velocity_rs_start = tsk_info->get_uintah_field<CCVariable<double> >(dep_vel_rs_start);
    CCVariable<double>& d_velocity_rs_start = *vd_velocity_rs_start;
    constCCVariable<double>* vd_velocity_rs_start_old = tsk_info->get_const_uintah_field<constCCVariable<double> >(dep_vel_rs_start);
    constCCVariable<double>& d_velocity_rs_start_old = *vd_velocity_rs_start_old;
    for (CellIterator iter=patch->getExtraCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      d_velocity_rs[c] = d_velocity_rs_old[c];
      d_velocity_rs_start[c] = d_velocity_rs_start_old[c];
    }
  }
}
//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

void
DepositionVelocity::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){

  register_variable( _cellType_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW , variable_registry );
  register_variable( _task_name, ArchesFieldContainer::MODIFIES, variable_registry );
  register_variable( _new_time_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::LATEST, variable_registry );
  
  for ( int i = 0; i < _Nenv; i++ ){
    const std::string RateDepositionX = get_env_name(i, _ratedepx_name);
    const std::string RateDepositionY = get_env_name(i, _ratedepy_name);
    const std::string RateDepositionZ = get_env_name(i, _ratedepz_name);
    const std::string rho_name = get_env_name(i, _rhoP_name);
    const std::string dep_vel_rs = get_env_name(i, _dep_vel_rs_name);
    const std::string dep_vel_rs_start = get_env_name(i, _dep_vel_rs_start_name);
    
    register_variable( RateDepositionX, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::LATEST , variable_registry );
    register_variable( RateDepositionY, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::LATEST , variable_registry );
    register_variable( RateDepositionZ, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::LATEST , variable_registry );
    register_variable( rho_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::LATEST, variable_registry );
    register_variable( dep_vel_rs, ArchesFieldContainer::MODIFIES, variable_registry );
    register_variable( dep_vel_rs_start, ArchesFieldContainer::MODIFIES, variable_registry );
  }

}

void
DepositionVelocity::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
    SpatialOps::OperatorDatabase& opr ){
  const int FLOW = -1;
  double current_time = _shared_state->getElapsedTime();  
  Vector Dx = patch->dCell(); // cell spacing
  double DxDy = Dx.x() * Dx.y();
  double DxDz = Dx.x() * Dx.z();
  double DyDz = Dx.y() * Dx.z();
  std::vector<double> area_face;
  area_face.push_back(DyDz);
  area_face.push_back(DyDz);
  area_face.push_back(DxDz);
  area_face.push_back(DxDz);
  area_face.push_back(DxDy);
  area_face.push_back(DxDy);
  const double delta_t = tsk_info->get_dt();
  std::vector<int> container_flux_ind;
  int total_flux_ind = 0; 
  double total_area_face = 0.0; 
  double flux = 0.0; 
  double rhoi = 0.0; 
  double d_velocity = 0.0; 
  double vel_i_ave = 0.0; 
  IntVector lowPindex = patch->getCellLowIndex();
  IntVector highPindex = patch->getCellHighIndex();
  //Pad for ghosts
  lowPindex -= IntVector(1,1,1);
  highPindex += IntVector(1,1,1);

  CCVariable<double>* vdeposit_velocity = tsk_info->get_uintah_field<CCVariable<double> >(_task_name);
  CCVariable<double>& deposit_velocity = *vdeposit_velocity;
  constCCVariable<int>* vcelltype = tsk_info->get_const_uintah_field<constCCVariable<int> >(_cellType_name);
  constCCVariable<int>& celltype = *vcelltype;
  constCCVariable<double>* vnew_time = tsk_info->get_const_uintah_field<constCCVariable<double> >(_new_time_name);
  constCCVariable<double>& new_time = *vnew_time;
  
  for( int i = 0; i < _Nenv; i++ ){
   
    const std::string RateDepositionX = get_env_name(i, _ratedepx_name);
    const std::string RateDepositionY = get_env_name(i, _ratedepy_name);
    const std::string RateDepositionZ = get_env_name(i, _ratedepz_name);
    const std::string rho_name = get_env_name(i, _rhoP_name);
    const std::string dep_vel_rs = get_env_name(i, _dep_vel_rs_name);
    const std::string dep_vel_rs_start = get_env_name(i, _dep_vel_rs_start_name);
    constSFCXVariable<double>* vdep_x = tsk_info->get_const_uintah_field<constSFCXVariable<double> >(RateDepositionX);
    constSFCXVariable<double>& dep_x = *vdep_x;
    constSFCYVariable<double>* vdep_y = tsk_info->get_const_uintah_field<constSFCYVariable<double> >(RateDepositionY);
    constSFCYVariable<double>& dep_y = *vdep_y;
    constSFCZVariable<double>* vdep_z = tsk_info->get_const_uintah_field<constSFCZVariable<double> >(RateDepositionZ);
    constSFCZVariable<double>& dep_z = *vdep_z;
    constCCVariable<double>* vrhop = tsk_info->get_const_uintah_field<constCCVariable<double> >(rho_name);
    constCCVariable<double>& rhop = *vrhop;
    CCVariable<double>* vd_velocity_rs = tsk_info->get_uintah_field<CCVariable<double> >(dep_vel_rs);
    CCVariable<double>& d_velocity_rs = *vd_velocity_rs;
    CCVariable<double>* vd_velocity_rs_start = tsk_info->get_uintah_field<CCVariable<double> >(dep_vel_rs_start);
    CCVariable<double>& d_velocity_rs_start = *vd_velocity_rs_start;
  
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
          d_velocity = 0;
          for ( int pp = 0; pp < total_flux_ind; pp++ ){
            if (container_flux_ind[pp]==0) {
              flux = std::abs(dep_x[c+_fd[container_flux_ind[pp]]]);
              rhoi = rhop[c+_d[container_flux_ind[pp]]];
            } else if (container_flux_ind[pp]==1) {
              flux = std::abs(dep_x[c+_fd[container_flux_ind[pp]]]);
              rhoi = rhop[c+_d[container_flux_ind[pp]]];
            } else if (container_flux_ind[pp]==2) {
              flux = std::abs(dep_y[c+_fd[container_flux_ind[pp]]]);
              rhoi = rhop[c+_d[container_flux_ind[pp]]];
            } else if (container_flux_ind[pp]==3) {
              flux = std::abs(dep_y[c+_fd[container_flux_ind[pp]]]);
              rhoi = rhop[c+_d[container_flux_ind[pp]]];
            } else if (container_flux_ind[pp]==4) {
              flux = std::abs(dep_z[c+_fd[container_flux_ind[pp]]]);
              rhoi = rhop[c+_d[container_flux_ind[pp]]];
            } else {
              flux = std::abs(dep_z[c+_fd[container_flux_ind[pp]]]);
              rhoi = rhop[c+_d[container_flux_ind[pp]]];
            }
            if (_user_specified_rho > 0) {
              rhoi = _user_specified_rho;
            }
            // volumetric flow rate for particle i:
            d_velocity += (flux/rhoi) * area_face[container_flux_ind[pp]];
            // The total cell surface area exposed to radiation:
            total_area_face += area_face[container_flux_ind[pp]];
          }
          // compute the current deposit velocity for each particle: d_vel [m3/s * 1/m2 = m/s] 
          d_velocity /= total_area_face; // area weighted incoming velocity to the cell for particle i.
          // update the running-sum for time-averaging the deposit velocity for particle i.
          d_velocity_rs[c]=d_velocity_rs[c] + d_velocity*delta_t; // during timestep init d_velocity_rs[c] is set to the old values..
          if (new_time[c] == 0.0){
            vel_i_ave = (d_velocity_rs[c] - d_velocity_rs_start[c] ) / _t_interval;
            deposit_velocity[c] = deposit_velocity[c] + vel_i_ave; // add the contribution per particle.
            d_velocity_rs_start[c]=d_velocity_rs[c];
          }
        }// if there is a deposition flux 
      } // wall or intrusion cell-type
    } // cell loop
  } // environment loop
}
