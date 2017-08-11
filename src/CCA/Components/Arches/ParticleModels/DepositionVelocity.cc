#include <CCA/Components/Arches/ParticleModels/DepositionVelocity.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>

using namespace Uintah;

DepositionVelocity::DepositionVelocity( std::string task_name, int matl_index, const int N, SimulationStateP shared_state  ) :
TaskInterface( task_name, matl_index ), _Nenv(N),_shared_state(shared_state) {

}

DepositionVelocity::~DepositionVelocity(){
}

void
DepositionVelocity::problemSetup( ProblemSpecP& db ){

  const ProblemSpecP db_root = db->getRootNode();

  double p_void0;
  _cellType_name="cellType";
  _ratedepx_name="RateDepositionX";
  _ratedepy_name="RateDepositionY";
  _ratedepz_name="RateDepositionZ";
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

  _user_specified_rho = rho_ash_bulk * p_void0;

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
  _diameter_base_name = ParticleTools::parse_for_role_to_label(db, "size");
}

void
DepositionVelocity::create_local_labels(){

  register_new_variable<CCVariable<double> >( _task_name );
  register_new_variable<CCVariable<double> >( "d_vol_ave_num" );
  register_new_variable<CCVariable<double> >( "d_vol_ave_den" );
  register_new_variable<CCVariable<double> >( "d_vol_ave" );
}

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void
DepositionVelocity::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

  register_variable( _task_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( "d_vol_ave_num", ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( "d_vol_ave_den", ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( "d_vol_ave", ArchesFieldContainer::COMPUTES, variable_registry );
}

void
DepositionVelocity::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& deposit_velocity = tsk_info->get_uintah_field_add<CCVariable<double> >(_task_name);
  CCVariable<double>& d_vol_ave_num = tsk_info->get_uintah_field_add<CCVariable<double> >("d_vol_ave_num");
  CCVariable<double>& d_vol_ave_den = tsk_info->get_uintah_field_add<CCVariable<double> >("d_vol_ave_den");
  CCVariable<double>& d_vol_ave = tsk_info->get_uintah_field_add<CCVariable<double> >("d_vol_ave");
  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
    deposit_velocity(i,j,k)=0.0;
    d_vol_ave_num(i,j,k)=0.0;
    d_vol_ave_den(i,j,k)=0.0;
    d_vol_ave(i,j,k)=0.0;
  });

}

//--------------------------------------------------------------------------------------------------
void
DepositionVelocity::register_timestep_eval(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const int time_substep, const bool packed_tasks ){

  register_variable( _cellType_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW , variable_registry );
  register_variable( _task_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _task_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW, variable_registry  );
  register_variable( "d_vol_ave_num", ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( "d_vol_ave_den", ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( "d_vol_ave", ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( "d_vol_ave", ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );

  for ( int i = 0; i < _Nenv; i++ ){
    const std::string RateDepositionX = get_env_name(i, _ratedepx_name);
    const std::string RateDepositionY = get_env_name(i, _ratedepy_name);
    const std::string RateDepositionZ = get_env_name(i, _ratedepz_name);
    const std::string diameter_name = get_env_name( i, _diameter_base_name );

    register_variable( RateDepositionX, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( RateDepositionY, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( RateDepositionZ, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( diameter_name , ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW , variable_registry );
  }

}

void
DepositionVelocity::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  const int FLOW = -1;
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
  std::vector<int> container_flux_ind;
  int total_flux_ind = 0;
  double total_area_face = 0.0;
  double flux = 0.0;
  double rhoi = 0.0;
  double d_velocity = 0.0;
  double d_diam_num = 0.0;
  double d_diam_den = 0.0;
  double particle_area = 0.0;

  //double vel_i_ave = 0.0;
  IntVector lowPindex = patch->getCellLowIndex();
  IntVector highPindex = patch->getCellHighIndex();
  //Pad for ghosts
  lowPindex -= IntVector(1,1,1);
  highPindex += IntVector(1,1,1);

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );

  CCVariable<double>& deposit_velocity = tsk_info->get_uintah_field_add<CCVariable<double> >(_task_name);
  deposit_velocity.initialize(0.0);
  constCCVariable<double>& deposit_velocity_old = *(tsk_info->get_const_uintah_field<constCCVariable<double> >(_task_name));
  CCVariable<double>* vd_vol_ave_num = tsk_info->get_uintah_field<CCVariable<double> >("d_vol_ave_num");
  CCVariable<double>& d_vol_ave_num = *vd_vol_ave_num;
  d_vol_ave_num.initialize(0.0);
  CCVariable<double>* vd_vol_ave_den = tsk_info->get_uintah_field<CCVariable<double> >("d_vol_ave_den");
  CCVariable<double>& d_vol_ave_den = *vd_vol_ave_den;
  d_vol_ave_den.initialize(0.0);
  CCVariable<double>* vd_vol_ave = tsk_info->get_uintah_field<CCVariable<double> >("d_vol_ave");
  CCVariable<double>& d_vol_ave = *vd_vol_ave;
  d_vol_ave.initialize(0.0);
  constCCVariable<double>& d_vol_ave_old = *(tsk_info->get_const_uintah_field<constCCVariable<double> >("d_vol_ave"));
  constCCVariable<int>* vcelltype = tsk_info->get_const_uintah_field<constCCVariable<int> >(_cellType_name);
  constCCVariable<int>& celltype = *vcelltype;

  for( int i = 0; i < _Nenv; i++ ){

    const std::string RateDepositionX = get_env_name(i, _ratedepx_name);
    const std::string RateDepositionY = get_env_name(i, _ratedepy_name);
    const std::string RateDepositionZ = get_env_name(i, _ratedepz_name);
    const std::string diameter_name  = get_env_name( i, _diameter_base_name );
    constSFCXVariable<double>* vdep_x = tsk_info->get_const_uintah_field<constSFCXVariable<double> >(RateDepositionX);
    constSFCXVariable<double>& dep_x = *vdep_x;
    constSFCYVariable<double>* vdep_y = tsk_info->get_const_uintah_field<constSFCYVariable<double> >(RateDepositionY);
    constSFCYVariable<double>& dep_y = *vdep_y;
    constSFCZVariable<double>* vdep_z = tsk_info->get_const_uintah_field<constSFCZVariable<double> >(RateDepositionZ);
    constSFCZVariable<double>& dep_z = *vdep_z;
    constCCVariable<double>& dp = *(tsk_info->get_const_uintah_field<constCCVariable<double> >( diameter_name ));

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
	        particle_area = _pi*dp[c]*dp[c];
          total_area_face = 0;
          d_velocity = 0;
          d_diam_num = 0;
          d_diam_den = 0;
          for ( int pp = 0; pp < total_flux_ind; pp++ ){
            if (container_flux_ind[pp]==0) {
              flux = std::abs(dep_x[c+_fd[container_flux_ind[pp]]]);
            } else if (container_flux_ind[pp]==1) {
              flux = std::abs(dep_x[c+_fd[container_flux_ind[pp]]]);
            } else if (container_flux_ind[pp]==2) {
              flux = std::abs(dep_y[c+_fd[container_flux_ind[pp]]]);
            } else if (container_flux_ind[pp]==3) {
              flux = std::abs(dep_y[c+_fd[container_flux_ind[pp]]]);
            } else if (container_flux_ind[pp]==4) {
              flux = std::abs(dep_z[c+_fd[container_flux_ind[pp]]]);
            } else {
              flux = std::abs(dep_z[c+_fd[container_flux_ind[pp]]]);
            }
            rhoi = _user_specified_rho;
            // volumetric flow rate for particle i:
            d_velocity += (flux/rhoi) * area_face[container_flux_ind[pp]];
            // The total cell surface area exposed to radiation:
            total_area_face += area_face[container_flux_ind[pp]];
	          d_diam_num += (flux/rhoi)*particle_area*dp[c];
	          d_diam_den += (flux/rhoi)*particle_area;
          }
          // compute the current deposit velocity for each particle: d_vel [m3/s * 1/m2 = m/s]
          d_velocity /= total_area_face; // area weighted incoming velocity to the cell for particle i.
          d_vol_ave_num[c] += d_diam_num;
	        d_vol_ave_den[c] += d_diam_den;
          deposit_velocity[c] += d_velocity; // add the contribution for the deposition
          // overall we are trying to achieve:  v_hat = (1-alpha)*v_hat_old + alpha*v_new. We initialize v_hat to v_hat_old*(1-alpha) in timestep_init (first term).
          // we handle the second term using the summation alpha*v_new = alpha * (v1 + v2 + v3):
          // v_hat += alpha*v1 + alpha*v2 + alpha*v3
        }// if there is a deposition flux
      } // wall or intrusion cell-type
    } // cell loop
  } // environment loop
  Uintah::parallel_for( range, [&](int i, int j, int k){
    deposit_velocity(i,j,k) = (1.0 - _relaxation_coe) * deposit_velocity_old(i,j,k) + _relaxation_coe * deposit_velocity(i,j,k); // time-average the deposition rate.
    d_vol_ave(i,j,k)= (1.0 - _relaxation_coe) * d_vol_ave_old(i,j,k) +
      _relaxation_coe * std::min(std::max(d_vol_ave_num(i,j,k)/(d_vol_ave_den(i,j,k)+1.0e-100),1e-8),0.1); // this is the time-averaged volume-averaged arriving particle size [m]
      // note here that when the flux is away from the wall the volume average size calculation can be negative.. so we simply set the particle size to 1e-8.
  });
}
