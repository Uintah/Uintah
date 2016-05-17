#include <CCA/Components/Arches/Utility/SurfaceNormals.h>
#include <CCA/Components/Arches/Operators/Operators.h>

#include <spatialops/structured/FVStaggered.h>

using namespace Uintah;

using namespace SpatialOps;
using SpatialOps::operator *;
//field types
typedef SSurfXField SurfX;
typedef SSurfYField SurfY;
typedef SSurfZField SurfZ;
typedef SVolField   SVolF;
typedef XVolField   XVolF;
typedef YVolField   YVolF;
typedef ZVolField   ZVolF;
typedef SpatFldPtr<SVolF> SVolFP;
typedef SpatFldPtr<XVolF> XVolFP;
typedef SpatFldPtr<YVolF> YVolFP;
typedef SpatFldPtr<ZVolF> ZVolFP;
//operators
typedef OperatorTypeBuilder< SpatialOps::Gradient, SVolF, SpatialOps::XVolField >::type GradX;
typedef OperatorTypeBuilder< SpatialOps::Gradient, SVolF, SpatialOps::YVolField >::type GradY;
typedef OperatorTypeBuilder< SpatialOps::Gradient, SVolF, SpatialOps::ZVolField >::type GradZ;

typedef OperatorTypeBuilder< SpatialOps::Interpolant, XVolF, SVolF >::type InterpTX;
typedef OperatorTypeBuilder< SpatialOps::Interpolant, YVolF, SVolF >::type InterpTY;
typedef OperatorTypeBuilder< SpatialOps::Interpolant, ZVolF, SVolF >::type InterpTZ;


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

  //to trace out the solid/gas interface
  // This marks the first solid cell
  register_new_variable<CCVariable<double> >("gas_solid_interface");

}

//--------------------------------------------------------------------------------------------------
void
SurfaceNormals::register_initialize( VIVec& variable_registry ){

  register_variable( "surf_out_normX" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_out_normY" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_out_normZ" , ArchesFieldContainer::COMPUTES , variable_registry );

  register_variable( "surf_in_normX" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_in_normY" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_in_normZ" , ArchesFieldContainer::COMPUTES , variable_registry );

  register_variable( "gas_solid_interface" , ArchesFieldContainer::COMPUTES , variable_registry );

  register_variable( "volFraction", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW,
                      variable_registry );

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

  SpatialOps::SpatFldPtr<SVolF> ccnormal = SpatialFieldStore::get<SVolF>(*vol_fraction);
  *ccnormal <<= 0.0;

  SVolFP gas_solid_interface = tsk_info->get_so_field<SVolF>("gas_solid_interface");

  const GradX* const gradx = opr.retrieve_operator<GradX>();
  const GradY* const grady = opr.retrieve_operator<GradY>();
  const GradZ* const gradz = opr.retrieve_operator<GradZ>();

  const InterpTX* const interpx = opr.retrieve_operator<InterpTX>();
  const InterpTY* const interpy = opr.retrieve_operator<InterpTY>();
  const InterpTZ* const interpz = opr.retrieve_operator<InterpTZ>();

  double noise  = 1e-10;

  *n_out_x <<= ( *gradx )( *vol_fraction )/SpatialOps::abs((*gradx)(*vol_fraction)+noise);
  *n_out_y <<= ( *grady )( *vol_fraction )/SpatialOps::abs((*grady)(*vol_fraction)+noise);
  *n_out_z <<= ( *gradz )( *vol_fraction )/SpatialOps::abs((*gradz)(*vol_fraction)+noise);

  *n_in_x <<= ( *gradx )( 1. - *vol_fraction ) /
    SpatialOps::abs( (*gradx)( 1. - *vol_fraction )+noise);
  *n_in_y <<= ( *grady )( 1. - *vol_fraction ) /
    SpatialOps::abs( (*grady)( 1. - *vol_fraction )+noise);
  *n_in_z <<= ( *gradz )( 1. - *vol_fraction ) /
    SpatialOps::abs( (*gradz)( 1. - *vol_fraction )+noise);

  *ccnormal <<= ((*interpx)(abs(*n_in_x)))
                  + ((*interpy)(abs(*n_in_y)))
                  + ((*interpz)(abs(*n_in_z)));

  *gas_solid_interface <<= 0.0;

  *gas_solid_interface <<= cond( *ccnormal != 0 && *vol_fraction < .5, 1. )
                                ( 0. );

}

//--------------------------------------------------------------------------------------------------
void
SurfaceNormals::register_timestep_init( VIVec& variable_registry ){

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

  register_variable( "gas_solid_interface" , ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( "gas_solid_interface" , ArchesFieldContainer::REQUIRES, 0,
                     ArchesFieldContainer::OLDDW, variable_registry );
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

  SVolFP old_surf = tsk_info->get_const_so_field<SVolF>("gas_solid_interface");
  SVolFP new_surf = tsk_info->get_so_field<SVolF>("gas_solid_interface");

  *n_in_x <<= *old_n_in_x;
  *n_in_y <<= *old_n_in_y;
  *n_in_z <<= *old_n_in_z;

  *n_out_x <<= *old_n_out_x;
  *n_out_y <<= *old_n_out_y;
  *n_out_z <<= *old_n_out_z;

  *new_surf <<= *old_surf;

}
