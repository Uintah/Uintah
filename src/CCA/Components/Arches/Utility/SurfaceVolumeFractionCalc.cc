#include <CCA/Components/Arches/Utility/SurfaceVolumeFractionCalc.h>
#include <CCA/Components/Arches/GridTools.h>

using namespace Uintah;

//--------------------------------------------------------------------------------------------------
void
SurfaceVolumeFractionCalc::create_local_labels(){

  // CC fields
  register_new_variable<CCVariable<double> >( "cc_volume_fraction" );

  // FX fields
  register_new_variable<SFCXVariable<double> >( "fx_volume_fraction" );

  // FY fields
  register_new_variable<SFCYVariable<double> >( "fy_volume_fraction" );

  // FZ fields
  register_new_variable<SFCZVariable<double> >( "fz_volume_fraction" );

  m_var_names.push_back( "cc_volume_fraction" );
  m_var_names.push_back( "fx_volume_fraction" );
  m_var_names.push_back( "fy_volume_fraction" );
  m_var_names.push_back( "fz_volume_fraction" );

}

//--------------------------------------------------------------------------------------------------
void
SurfaceVolumeFractionCalc::register_initialize( ArchesVIVector& variable_registry , const bool packed_tasks){

  for ( auto i = m_var_names.begin(); i != m_var_names.end(); i++ ){
    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry, _task_name );
  }
}

//--------------------------------------------------------------------------------------------------
void
SurfaceVolumeFractionCalc::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& cc_vol_frac = *(tsk_info->get_uintah_field<CCVariable<double> >("cc_volume_fraction"));
  SFCXVariable<double>& fx_vol_frac = *(tsk_info->get_uintah_field<SFCXVariable<double> >("fx_volume_fraction"));
  SFCYVariable<double>& fy_vol_frac = *(tsk_info->get_uintah_field<SFCYVariable<double> >("fy_volume_fraction"));
  SFCZVariable<double>& fz_vol_frac = *(tsk_info->get_uintah_field<SFCZVariable<double> >("fz_volume_fraction"));

  Uintah::BlockRange range( patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
    cc_vol_frac(i,j,k) = 1.0;
    fx_vol_frac(i,j,k) = 1.0;
    fy_vol_frac(i,j,k) = 1.0;
    fz_vol_frac(i,j,k) = 1.0;
  });

}

//--------------------------------------------------------------------------------------------------
void
SurfaceVolumeFractionCalc::register_timestep_init( ArchesVIVector& variable_registry , const bool packed_tasks){

  for ( auto i = m_var_names.begin(); i != m_var_names.end(); i++ ){
    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry, _task_name );
    register_variable( *i, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry, _task_name );
  }
}

//--------------------------------------------------------------------------------------------------
void
SurfaceVolumeFractionCalc::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& cc_vol_frac = *(tsk_info->get_uintah_field<CCVariable<double> >("cc_volume_fraction"));
  SFCXVariable<double>& fx_vol_frac = *(tsk_info->get_uintah_field<SFCXVariable<double> >("fx_volume_fraction"));
  SFCYVariable<double>& fy_vol_frac = *(tsk_info->get_uintah_field<SFCYVariable<double> >("fy_volume_fraction"));
  SFCZVariable<double>& fz_vol_frac = *(tsk_info->get_uintah_field<SFCZVariable<double> >("fz_volume_fraction"));

  constCCVariable<double>&   old_cc_vol_frac = *(tsk_info->get_const_uintah_field<constCCVariable<double> >("cc_volume_fraction"));
  constSFCXVariable<double>& old_fx_vol_frac = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("fx_volume_fraction"));
  constSFCYVariable<double>& old_fy_vol_frac = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >("fy_volume_fraction"));
  constSFCZVariable<double>& old_fz_vol_frac = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >("fz_volume_fraction"));

  Uintah::BlockRange range( patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
    cc_vol_frac(i,j,k) = old_cc_vol_frac(i,j,k);
    fx_vol_frac(i,j,k) = old_fx_vol_frac(i,j,k);
    fy_vol_frac(i,j,k) = old_fy_vol_frac(i,j,k);
    fz_vol_frac(i,j,k) = old_fz_vol_frac(i,j,k);
  });
}
