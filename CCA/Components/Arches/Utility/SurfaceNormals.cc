#include <CCA/Components/Arches/Utility/SurfaceNormals.h>
#include <CCA/Components/Arches/Operators/Operators.h>

#include <spatialops/structured/FVStaggered.h>

using namespace Uintah;

//field types
typedef SpatialOps::SSurfXField SurfX;
typedef SpatialOps::SSurfYField SurfY;
typedef SpatialOps::SSurfZField SurfZ;
typedef SpatialOps::SVolField   SVolF;
typedef SpatialOps::XVolField   XVolF;
typedef SpatialOps::YVolField   YVolF;
typedef SpatialOps::ZVolField   ZVolF;
typedef SpatialOps::SpatFldPtr<SVolF> SVolFP;
typedef SpatialOps::SpatFldPtr<XVolF> XVolFP;
typedef SpatialOps::SpatFldPtr<YVolF> YVolFP;
typedef SpatialOps::SpatFldPtr<ZVolF> ZVolFP;
//operators
typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, SVolF, SpatialOps::XVolField >::type GradX;
typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, SVolF, SpatialOps::YVolField >::type GradY;
typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, SVolF, SpatialOps::ZVolField >::type GradZ;

SurfaceNormals::SurfaceNormals( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

SurfaceNormals::~SurfaceNormals(){
}

void
SurfaceNormals::problemSetup( ProblemSpecP& db ){
}

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

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void
SurfaceNormals::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

  register_variable( "surf_out_normX" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_out_normY" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_out_normZ" , ArchesFieldContainer::COMPUTES , variable_registry );

  register_variable( "surf_in_normX" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_in_normY" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_in_normZ" , ArchesFieldContainer::COMPUTES , variable_registry );

  register_variable( "volFraction", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry );

}

void
SurfaceNormals::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                      SpatialOps::OperatorDatabase& opr ){

  using namespace SpatialOps;
  using SpatialOps::operator *;

  SVolFP vol_fraction = tsk_info->get_const_so_field<SVolF>("volFraction");

  XVolFP n_in_x = tsk_info->get_so_field<XVolF>("surf_in_normX");
  YVolFP n_in_y = tsk_info->get_so_field<YVolF>("surf_in_normY");
  ZVolFP n_in_z = tsk_info->get_so_field<ZVolF>("surf_in_normZ");

  XVolFP n_out_x = tsk_info->get_so_field<XVolF>("surf_out_normX");
  YVolFP n_out_y = tsk_info->get_so_field<YVolF>("surf_out_normY");
  ZVolFP n_out_z = tsk_info->get_so_field<ZVolF>("surf_out_normZ");

  const GradX* const gradx = opr.retrieve_operator<GradX>();
  const GradY* const grady = opr.retrieve_operator<GradY>();
  const GradZ* const gradz = opr.retrieve_operator<GradZ>();

  *n_in_x <<= ( *gradx )( *vol_fraction )/SpatialOps::abs((*gradx)(*vol_fraction));
  *n_in_y <<= ( *grady )( *vol_fraction )/SpatialOps::abs((*grady)(*vol_fraction));
  *n_in_z <<= ( *gradz )( *vol_fraction )/SpatialOps::abs((*gradz)(*vol_fraction));

  *n_out_x <<= ( *gradx )( 1. - *vol_fraction )/SpatialOps::abs( (*gradx)( 1. - *vol_fraction ) );
  *n_out_y <<= ( *grady )( 1. - *vol_fraction )/SpatialOps::abs( (*grady)( 1. - *vol_fraction ) );
  *n_out_z <<= ( *gradz )( 1. - *vol_fraction )/SpatialOps::abs( (*gradz)( 1. - *vol_fraction ) );

}

//
//------------------------------------------------
//------------- TIMESTEP INIT --------------------
//------------------------------------------------
//
void
SurfaceNormals::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

  register_variable( "surf_out_normX" , ArchesFieldContainer::REQUIRES , 1, ArchesFieldContainer::OLDDW, variable_registry );
  register_variable( "surf_out_normY" , ArchesFieldContainer::REQUIRES , 1, ArchesFieldContainer::OLDDW, variable_registry );
  register_variable( "surf_out_normZ" , ArchesFieldContainer::REQUIRES , 1, ArchesFieldContainer::OLDDW, variable_registry );

  register_variable( "surf_in_normX" , ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW , variable_registry );
  register_variable( "surf_in_normY" , ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW , variable_registry );
  register_variable( "surf_in_normZ" , ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW , variable_registry );

  register_variable( "surf_out_normX" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_out_normY" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_out_normZ" , ArchesFieldContainer::COMPUTES , variable_registry );

  register_variable( "surf_in_normX" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_in_normY" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_in_normZ" , ArchesFieldContainer::COMPUTES , variable_registry );
}

void
SurfaceNormals::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                          SpatialOps::OperatorDatabase& opr ){

  using namespace SpatialOps;
  using SpatialOps::operator *;

  XVolFP n_in_x = tsk_info->get_so_field<XVolF>("surf_in_normX");
  YVolFP n_in_y = tsk_info->get_so_field<YVolF>("surf_in_normY");
  ZVolFP n_in_z = tsk_info->get_so_field<ZVolF>("surf_in_normZ");

  XVolFP n_out_x = tsk_info->get_so_field<XVolF>("surf_out_normX");
  YVolFP n_out_y = tsk_info->get_so_field<YVolF>("surf_out_normY");
  ZVolFP n_out_z = tsk_info->get_so_field<ZVolF>("surf_out_normZ");

  XVolFP old_n_in_x = tsk_info->get_const_so_field<XVolF>("surf_in_normX");
  YVolFP old_n_in_y = tsk_info->get_const_so_field<YVolF>("surf_in_normY");
  ZVolFP old_n_in_z = tsk_info->get_const_so_field<ZVolF>("surf_in_normZ");

  XVolFP old_n_out_x = tsk_info->get_const_so_field<XVolF>("surf_out_normX");
  YVolFP old_n_out_y = tsk_info->get_const_so_field<YVolF>("surf_out_normY");
  ZVolFP old_n_out_z = tsk_info->get_const_so_field<ZVolF>("surf_out_normZ");

  *n_in_x <<= *old_n_in_x;
  *n_in_y <<= *old_n_in_y;
  *n_in_z <<= *old_n_in_z;

  *n_out_x <<= *old_n_out_x;
  *n_out_y <<= *old_n_out_y;
  *n_out_z <<= *old_n_out_z;

}
//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

void
SurfaceNormals::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){

}

void
SurfaceNormals::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                SpatialOps::OperatorDatabase& opr ){
}
