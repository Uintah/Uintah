#include <CCA/Components/Arches/Utility/SurfaceNormals.h>
#include <CCA/Components/Arches/GridTools.h>
#include <math.h>

using namespace Uintah;

//helper
typedef std::vector<ArchesFieldContainer::VariableInformation> VIVec;

//--------------------------------------------------------------------------------------------------
SurfaceNormals::SurfaceNormals( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

SurfaceNormals::~SurfaceNormals(){
}

//--------------------------------------------------------------------------------------------------
void
SurfaceNormals::problemSetup( ProblemSpecP& db ){
}

//--------------------------------------------------------------------------------------------------
void
SurfaceNormals::create_local_labels(){

  //outward facing normals
  //
  //    gas    <--|--   solid
  //
  register_new_variable<SFCXVariable<double> >( "surf_out_normX" );
  register_new_variable<SFCYVariable<double> >( "surf_out_normY" );
  register_new_variable<SFCZVariable<double> >( "surf_out_normZ" );

  //inward facing normals
  //
  //    gas    --|-->   solid
  //
  register_new_variable<SFCXVariable<double> >( "surf_in_normX" );
  register_new_variable<SFCYVariable<double> >( "surf_in_normY" );
  register_new_variable<SFCZVariable<double> >( "surf_in_normZ" );

}

//--------------------------------------------------------------------------------------------------
void
SurfaceNormals::register_initialize( VIVec& variable_registry , const bool packed_tasks){

  register_variable( "surf_out_normX" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_out_normY" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_out_normZ" , ArchesFieldContainer::COMPUTES , variable_registry );

  register_variable( "surf_in_normX" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_in_normY" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_in_normZ" , ArchesFieldContainer::COMPUTES , variable_registry );

  register_variable( "volFraction", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW,
                      variable_registry );

}

void
SurfaceNormals::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  GET_EXTRACELL_FX_BUFFERED_PATCH_RANGE(0,1)
  GET_EXTRACELL_FY_BUFFERED_PATCH_RANGE(0,1)
  GET_EXTRACELL_FZ_BUFFERED_PATCH_RANGE(0,1)

  constCCVariable<double>& vol_fraction = *(tsk_info->get_const_uintah_field<constCCVariable<double> >("volFraction"));

  SFCXVariable<double>& n_in_x = *(tsk_info->get_uintah_field<SFCXVariable<double> >("surf_in_normX"));
  SFCYVariable<double>& n_in_y = *(tsk_info->get_uintah_field<SFCYVariable<double> >("surf_in_normY"));
  SFCZVariable<double>& n_in_z = *(tsk_info->get_uintah_field<SFCZVariable<double> >("surf_in_normZ"));

  SFCXVariable<double>& n_out_x = *(tsk_info->get_uintah_field<SFCXVariable<double> >("surf_out_normX"));
  SFCYVariable<double>& n_out_y = *(tsk_info->get_uintah_field<SFCYVariable<double> >("surf_out_normY"));
  SFCZVariable<double>& n_out_z = *(tsk_info->get_uintah_field<SFCZVariable<double> >("surf_out_normZ"));

  Uintah::BlockRange full_range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
  Uintah::parallel_for(full_range, [&](int i, int j, int k){
    n_in_x(i,j,k) = 0.0;
    n_out_x(i,j,k) = 0.0;
    n_in_y(i,j,k) = 0.0;
    n_out_y(i,j,k) = 0.0;
    n_in_z(i,j,k) = 0.0;
    n_out_z(i,j,k) = 0.0;
  });

  const double noise = 1e-10;
  //X
  Uintah::parallel_for(Uintah::BlockRange(low_fx_patch_range, high_fx_patch_range), [&](int i, int j, int k){
    n_out_x(i,j,k) = ( vol_fraction(i,j,k) - vol_fraction(i-1,j,k) )
                     / std::abs( vol_fraction(i,j,k) - vol_fraction(i-1,j,k) + noise);
    n_in_x(i,j,k) = ( vol_fraction(i-1,j,k) - vol_fraction(i,j,k) ) /
                    std::abs( vol_fraction(i-1,j,k) - vol_fraction(i,j,k) + noise);
  });
  //Y
  Uintah::parallel_for(Uintah::BlockRange(low_fy_patch_range, high_fy_patch_range), [&](int i, int j, int k){
    n_out_y(i,j,k) = ( vol_fraction(i,j,k) - vol_fraction(i,j-1,k) )
                     / std::abs( vol_fraction(i,j,k) - vol_fraction(i,j-1,k) + noise);
    n_in_y(i,j,k) = ( vol_fraction(i,j-1,k) - vol_fraction(i,j,k) ) /
                    std::abs( vol_fraction(i,j,k) - vol_fraction(i,j-1,k) + noise);
  });
  //Z
  Uintah::parallel_for(Uintah::BlockRange(low_fz_patch_range, high_fz_patch_range), [&](int i, int j, int k){
    n_out_z(i,j,k) = ( vol_fraction(i,j,k) - vol_fraction(i,j,k-1) )
                     / std::abs( vol_fraction(i,j,k) - vol_fraction(i,j,k-1) + noise);
    n_in_z(i,j,k) = ( vol_fraction(i,j,k-1) - vol_fraction(i,j,k) ) /
                    std::abs( vol_fraction(i,j,k-1) - vol_fraction(i,j,k) + noise);
  });

}

//--------------------------------------------------------------------------------------------------
void
SurfaceNormals::register_timestep_init( VIVec& variable_registry, const bool packed_tasks){

  register_variable( "surf_out_normX" , ArchesFieldContainer::REQUIRES , 0,
                      ArchesFieldContainer::OLDDW, variable_registry );
  register_variable( "surf_out_normY" , ArchesFieldContainer::REQUIRES , 0,
                      ArchesFieldContainer::OLDDW, variable_registry );
  register_variable( "surf_out_normZ" , ArchesFieldContainer::REQUIRES , 0,
                      ArchesFieldContainer::OLDDW, variable_registry );

  register_variable( "surf_in_normX" , ArchesFieldContainer::REQUIRES, 0,
                     ArchesFieldContainer::OLDDW , variable_registry );
  register_variable( "surf_in_normY" , ArchesFieldContainer::REQUIRES, 0,
                     ArchesFieldContainer::OLDDW , variable_registry );
  register_variable( "surf_in_normZ" , ArchesFieldContainer::REQUIRES, 0,
                     ArchesFieldContainer::OLDDW , variable_registry );

  register_variable( "surf_out_normX" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_out_normY" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_out_normZ" , ArchesFieldContainer::COMPUTES , variable_registry );

  register_variable( "surf_in_normX" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_in_normY" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_in_normZ" , ArchesFieldContainer::COMPUTES , variable_registry );

}

void
SurfaceNormals::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  SFCXVariable<double>& n_in_x = *(tsk_info->get_uintah_field<SFCXVariable<double> >("surf_in_normX"));
  SFCYVariable<double>& n_in_y = *(tsk_info->get_uintah_field<SFCYVariable<double> >("surf_in_normY"));
  SFCZVariable<double>& n_in_z = *(tsk_info->get_uintah_field<SFCZVariable<double> >("surf_in_normZ"));
  constSFCXVariable<double>& old_n_in_x = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("surf_in_normX"));
  constSFCYVariable<double>& old_n_in_y = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >("surf_in_normY"));
  constSFCZVariable<double>& old_n_in_z = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >("surf_in_normZ"));

  SFCXVariable<double>& n_out_x = *(tsk_info->get_uintah_field<SFCXVariable<double> >("surf_out_normX"));
  SFCYVariable<double>& n_out_y = *(tsk_info->get_uintah_field<SFCYVariable<double> >("surf_out_normY"));
  SFCZVariable<double>& n_out_z = *(tsk_info->get_uintah_field<SFCZVariable<double> >("surf_out_normZ"));
  constSFCXVariable<double>& old_n_out_x = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("surf_out_normX"));
  constSFCYVariable<double>& old_n_out_y = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >("surf_out_normY"));
  constSFCZVariable<double>& old_n_out_z = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >("surf_out_normZ"));

  Uintah::BlockRange full_range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
  Uintah::parallel_for(full_range, [&](int i, int j, int k){
    n_in_x(i,j,k)  = old_n_in_x(i,j,k);
    n_out_x(i,j,k) = old_n_out_x(i,j,k);
    n_in_y(i,j,k)  = old_n_in_y(i,j,k);
    n_out_y(i,j,k) = old_n_out_y(i,j,k);
    n_in_z(i,j,k)  = old_n_in_z(i,j,k);
    n_out_z(i,j,k) = old_n_out_z(i,j,k);
  });
}
