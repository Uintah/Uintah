#include <CCA/Components/Arches/PropertyModelsV2/ForFilterDensity.h>
#include <CCA/Components/Arches/UPSHelper.h>

using namespace Uintah;
using namespace ArchesCore;

//--------------------------------------------------------------------------------------------------
ForFilterDensity::ForFilterDensity( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ){}

//--------------------------------------------------------------------------------------------------
ForFilterDensity::~ForFilterDensity(){}

//--------------------------------------------------------------------------------------------------
void ForFilterDensity::problemSetup( ProblemSpecP& db ){

  using namespace Uintah::ArchesCore;
  m_volfraction_name = "volFraction"; 
  m_fm_name = "filter_cells"; 
  m_density_name = parse_ups_for_role( DENSITY, db, "density" );
  m_ff_density_name = "density_for_filter";
}

//--------------------------------------------------------------------------------------------------
void ForFilterDensity::create_local_labels(){

  register_new_variable<CCVariable<double> >(m_ff_density_name);
  register_new_variable<CCVariable<double> >(m_fm_name);

}

//--------------------------------------------------------------------------------------------------
void ForFilterDensity::register_initialize( AVarInfo& variable_registry , const bool pack_tasks){

//  typedef ArchesFieldContainer AFC;

//  register_variable( m_density_name, AFC::REQUIRES, 0, AFC::NEWDW, variable_registry, _task_name );
  register_variable( m_ff_density_name, AFC::COMPUTES, variable_registry, _task_name );
  register_variable( m_volfraction_name, AFC::REQUIRES, 0, AFC::NEWDW, variable_registry, _task_name );
  register_variable( m_fm_name, AFC::COMPUTES, variable_registry, _task_name );

}

//--------------------------------------------------------------------------------------------------
void ForFilterDensity::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  constCCVariable<double>& volFraction  = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_volfraction_name);
  CCVariable<double>& ff_density = tsk_info->get_uintah_field_add<CCVariable<double> >(m_ff_density_name);
  CCVariable<double>& fm = tsk_info->get_uintah_field_add<CCVariable<double> >(m_fm_name);
  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
    ff_density(i,j,k) = 0.0;
    fm(i,j,k) = 0.0;
  });
//  compute_ff_density( patch, tsk_info );

  IntVector low = patch->getCellLowIndex() ;
  IntVector high =  patch->getCellHighIndex();
  Uintah::BlockRange range2(low , high );
  Uintah::parallel_for( range2, [&](int i, int j, int k){
  
  if ( volFraction(i,j,k) > 0.0 ) {

    for ( int m = -1; m <= 1; m++ ){
      for ( int n = -1; n <= 1; n++ ){
        for ( int l = -1; l <= 1; l++ ){
          if (volFraction(i+m,j+n,k+l) < 1.0 ) {
            fm(i+m,j+n,k+l) = 1.0;
          }
        }
      }
    }

  }
  });

}

//--------------------------------------------------------------------------------------------------
void ForFilterDensity::register_timestep_init( AVarInfo& variable_registry , const bool pack_tasks){

  typedef ArchesFieldContainer AFC;

  register_variable( m_ff_density_name, AFC::COMPUTES, variable_registry, _task_name );
  register_variable( m_ff_density_name, AFC::REQUIRES, 0, AFC::OLDDW, variable_registry, _task_name );
  register_variable( m_fm_name, AFC::COMPUTES, variable_registry, _task_name );
  register_variable( m_fm_name, AFC::REQUIRES, 0, AFC::OLDDW, variable_registry, _task_name );
  
}

//--------------------------------------------------------------------------------------------------
void ForFilterDensity::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  constCCVariable<double>& old_ff_density = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_ff_density_name);
  CCVariable<double>& ff_density = tsk_info->get_uintah_field_add<CCVariable<double> >(m_ff_density_name);

  constCCVariable<double>& old_fm = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_fm_name);
  CCVariable<double>& fm = tsk_info->get_uintah_field_add<CCVariable<double> >(m_fm_name);

  ff_density.copy(old_ff_density);
  fm.copy(old_fm);

}

//--------------------------------------------------------------------------------------------------
void ForFilterDensity::register_timestep_eval( VIVec& variable_registry, const int time_substep , const bool packed_tasks){

  typedef ArchesFieldContainer AFC;

  register_variable( m_density_name, AFC::REQUIRES, 1, AFC::LATEST, variable_registry, time_substep ,_task_name );
  register_variable( m_volfraction_name, AFC::REQUIRES, 1, AFC::LATEST, variable_registry, time_substep ,_task_name );
  register_variable( m_ff_density_name, AFC::MODIFIES, variable_registry, time_substep , _task_name );
  register_variable( m_fm_name, AFC::REQUIRES,1, AFC::OLDDW,variable_registry, time_substep , _task_name );

}

//--------------------------------------------------------------------------------------------------
void ForFilterDensity::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  compute_ff_density( patch, tsk_info );

}

//--------------------------------------------------------------------------------------------------
void ForFilterDensity::compute_ff_density( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  constCCVariable<double>& density  = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_density_name);
  constCCVariable<double>& volFraction  = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_volfraction_name);
  CCVariable<double>& ff_density = tsk_info->get_uintah_field_add<CCVariable<double> >(m_ff_density_name);
  constCCVariable<double>& fm = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_fm_name);

  //ff_density.copyData(density);
  //fm.copyData(volFraction);
  
  IntVector low = patch->getCellLowIndex() ;
  IntVector high =  patch->getCellHighIndex();
  Uintah::BlockRange range2(low , high );
  Uintah::parallel_for( range2, [&](int i, int j, int k){
  
  if ( fm(i,j,k) > 0.0 ) {
    
    // x-dir 
    double sum_den = 0;
    int count =0;
    {
    STENCIL3_1D(0);
    if (volFraction(IJK_M_) > 0.0 ) {
       sum_den += density(IJK_M_);
       count += 1;
    }

    if (volFraction(IJK_P_) > 0.0 ) {
       sum_den += density(IJK_P_);
       count += 1;
    }
    }

    // y-dir 
    {
    STENCIL3_1D(1);
    if (volFraction(IJK_M_) > 0.0 ) {
       sum_den += density(IJK_M_);
       count += 1;
    }

    if (volFraction(IJK_P_) > 0.0 ) {
       sum_den += density(IJK_P_);
       count += 1;
    }
    }
    // z-dir 
    {
    STENCIL3_1D(2);
    if (volFraction(IJK_M_) > 0.0 ) {
       sum_den += density(IJK_M_);
       count += 1;
    }

    if (volFraction(IJK_P_) > 0.0 ) {
       sum_den += density(IJK_P_);
       count += 1;
    }
    }
    if (count > 0) {
    ff_density(i,j,k) = sum_den/count; 
    }
  }
  });
}
