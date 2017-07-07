#include <CCA/Components/Arches/Utility/GridInfo.h>
#include <Core/Grid/Box.h>

using namespace Uintah;

typedef ArchesFieldContainer AFC;

//--------------------------------------------------------------------------------------------------
void
GridInfo::create_local_labels(){

  register_new_variable<CCVariable<double> >( "gridX" );
  register_new_variable<CCVariable<double> >( "gridY" );
  register_new_variable<CCVariable<double> >( "gridZ" );
  register_new_variable<CCVariable<double> >( "ucellX" );
  register_new_variable<CCVariable<double> >( "vcellY" );
  register_new_variable<CCVariable<double> >( "wcellZ" );

}

//--------------------------------------------------------------------------------------------------
void
GridInfo::register_initialize( std::vector<AFC::VariableInformation>& variable_registry , const bool packed_tasks){

  register_variable( "gridX", AFC::COMPUTES, variable_registry );
  register_variable( "gridY", AFC::COMPUTES, variable_registry );
  register_variable( "gridZ", AFC::COMPUTES, variable_registry );
  register_variable( "ucellX", AFC::COMPUTES, variable_registry );
  register_variable( "vcellY", AFC::COMPUTES, variable_registry );
  register_variable( "wcellZ", AFC::COMPUTES, variable_registry );

}

//--------------------------------------------------------------------------------------------------
void
GridInfo::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& gridX = tsk_info->get_uintah_field_add<CCVariable<double> >( "gridX" );
  CCVariable<double>& gridY = tsk_info->get_uintah_field_add<CCVariable<double> >( "gridY" );
  CCVariable<double>& gridZ = tsk_info->get_uintah_field_add<CCVariable<double> >( "gridZ" );
  CCVariable<double>& ucellX = tsk_info->get_uintah_field_add<CCVariable<double> >( "ucellX" );
  CCVariable<double>& vcellY = tsk_info->get_uintah_field_add<CCVariable<double> >( "vcellY" );
  CCVariable<double>& wcellZ = tsk_info->get_uintah_field_add<CCVariable<double> >( "wcellZ" );

  Vector Dx = patch->dCell();
  const double dx = Dx.x();
  const double dy = Dx.y();
  const double dz = Dx.z();
  const double dx2 = Dx.x()/2.;
  const double dy2 = Dx.y()/2.;
  const double dz2 = Dx.z()/2.;

  const Level* lvl = patch->getLevel();
  IntVector min; IntVector max;
  lvl->getGrid()->getLevel(0)->findCellIndexRange(min,max);
  IntVector period_bc = IntVector(1,1,1) - lvl->getPeriodicBoundaries();
  Box domainBox = lvl->getBox(min+period_bc, max-period_bc);
  const double lowx = domainBox.lower().x();
  const double lowy = domainBox.lower().y();
  const double lowz = domainBox.lower().z();

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
  Uintah::parallel_for( range, [&](int i, int j, int k){
    gridX(i,j,k) = lowx + i * dx + dx2;
    gridY(i,j,k) = lowy + j * dy + dy2;
    gridZ(i,j,k) = lowz + k * dz + dz2;

    ucellX(i,j,k) = gridX(i,j,k) - dx2;
    vcellY(i,j,k) = gridY(i,j,k) - dy2;
    wcellZ(i,j,k) = gridZ(i,j,k) - dz2;
  });
}

//--------------------------------------------------------------------------------------------------
void
GridInfo::register_timestep_init( std::vector<AFC::VariableInformation>& variable_registry , const bool packed_tasks){

  register_variable( "gridX" , AFC::COMPUTES, variable_registry );
  register_variable( "gridY" , AFC::COMPUTES, variable_registry );
  register_variable( "gridZ" , AFC::COMPUTES, variable_registry );
  register_variable( "ucellX" , AFC::COMPUTES, variable_registry );
  register_variable( "vcellY" , AFC::COMPUTES, variable_registry );
  register_variable( "wcellZ" , AFC::COMPUTES, variable_registry );

  register_variable( "gridX" , AFC::REQUIRES , 0 , AFC::OLDDW , variable_registry );
  register_variable( "gridY" , AFC::REQUIRES , 0 , AFC::OLDDW , variable_registry );
  register_variable( "gridZ" , AFC::REQUIRES , 0 , AFC::OLDDW , variable_registry );
  register_variable( "ucellX" , AFC::REQUIRES , 0 , AFC::OLDDW , variable_registry );
  register_variable( "vcellY" , AFC::REQUIRES , 0 , AFC::OLDDW , variable_registry );
  register_variable( "wcellZ" , AFC::REQUIRES , 0 , AFC::OLDDW , variable_registry );

}

//--------------------------------------------------------------------------------------------------
void
GridInfo::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& gridX = tsk_info->get_uintah_field_add<CCVariable<double>>( "gridX" );
  CCVariable<double>& gridY = tsk_info->get_uintah_field_add<CCVariable<double>>( "gridY" );
  CCVariable<double>& gridZ = tsk_info->get_uintah_field_add<CCVariable<double>>( "gridZ" );
  CCVariable<double>& ucellX = tsk_info->get_uintah_field_add<CCVariable<double>>( "ucellX" );
  CCVariable<double>& vcellY = tsk_info->get_uintah_field_add<CCVariable<double>>( "vcellY" );
  CCVariable<double>& wcellZ = tsk_info->get_uintah_field_add<CCVariable<double>>( "wcellZ" );

  constCCVariable<double>& old_gridX = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( "gridX" );
  constCCVariable<double>& old_gridY = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( "gridY" );
  constCCVariable<double>& old_gridZ = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( "gridZ" );
  constCCVariable<double>& old_ucellX = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( "ucellX" );
  constCCVariable<double>& old_vcellY = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( "vcellY" );
  constCCVariable<double>& old_wcellZ = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( "wcellZ" );

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
  Uintah::parallel_for( range, [&](int i, int j, int k){
    gridX(i,j,k) = old_gridX(i,j,k);
    gridY(i,j,k) = old_gridY(i,j,k);
    gridZ(i,j,k) = old_gridZ(i,j,k);
    ucellX(i,j,k) = old_ucellX(i,j,k);
    vcellY(i,j,k) = old_vcellY(i,j,k);
    wcellZ(i,j,k) = old_wcellZ(i,j,k);
  });
}
