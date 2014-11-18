#include <CCA/Components/Arches/Utility/GridInfo.h>
#include <CCA/Components/Arches/Operators/Operators.h>

#include <spatialops/structured/FVStaggered.h>

using namespace Uintah;

GridInfo::GridInfo( std::string task_name, int matl_index ) : 
TaskInterface( task_name, matl_index ) { 
}

GridInfo::~GridInfo(){ 
}

void 
GridInfo::problemSetup( ProblemSpecP& db ){ 
}

void 
GridInfo::create_local_labels(){ 

  register_new_variable( "gridX", CC_DOUBLE ); 
  register_new_variable( "gridY", CC_DOUBLE ); 
  register_new_variable( "gridZ", CC_DOUBLE ); 

}

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void 
GridInfo::register_initialize( std::vector<VariableInformation>& variable_registry ){ 

  //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
  register_variable( "gridX" , CC_DOUBLE , COMPUTES , 0 , NEWDW , variable_registry );
  register_variable( "gridY" , CC_DOUBLE , COMPUTES , 0 , NEWDW , variable_registry );
  register_variable( "gridZ" , CC_DOUBLE , COMPUTES , 0 , NEWDW , variable_registry );

}

void 
GridInfo::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                      SpatialOps::OperatorDatabase& opr ){ 

  CCVariable<double>* gridX = tsk_info->get_uintah_field<CCVariable<double> >( "gridX" ); 
  CCVariable<double>* gridY = tsk_info->get_uintah_field<CCVariable<double> >( "gridY" ); 
  CCVariable<double>* gridZ = tsk_info->get_uintah_field<CCVariable<double> >( "gridZ" ); 

  for ( CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++ ){ 

    IntVector c = *iter; 
    Point P = patch->getCellPosition(c); 
    (*gridX)[c] = P.x(); 
    (*gridY)[c] = P.y(); 
    (*gridZ)[c] = P.z(); 


  }
}

//
//------------------------------------------------
//------------- TIMESTEP INIT --------------------
//------------------------------------------------
//
void 
GridInfo::register_timestep_init( std::vector<VariableInformation>& variable_registry ){ 

  //carry forward the old values
  //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
  register_variable( "gridX" , CC_DOUBLE , COMPUTES , 0 , NEWDW , variable_registry );
  register_variable( "gridY" , CC_DOUBLE , COMPUTES , 0 , NEWDW , variable_registry );
  register_variable( "gridZ" , CC_DOUBLE , COMPUTES , 0 , NEWDW , variable_registry );

  register_variable( "gridX" , CC_DOUBLE , REQUIRES , 0 , OLDDW , variable_registry );
  register_variable( "gridY" , CC_DOUBLE , REQUIRES , 0 , OLDDW , variable_registry );
  register_variable( "gridZ" , CC_DOUBLE , REQUIRES , 0 , OLDDW , variable_registry );

}

void 
GridInfo::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                          SpatialOps::OperatorDatabase& opr ){ 

  using namespace SpatialOps;
  using SpatialOps::operator *; 
  typedef SpatialOps::SVolField   SVolF;
  typedef SpatialOps::SpatFldPtr<SVolF> SVolFP; 

  SVolFP gridX = tsk_info->get_so_field<SVolF>( "gridX" ); 
  SVolFP gridY = tsk_info->get_so_field<SVolF>( "gridY" ); 
  SVolFP gridZ = tsk_info->get_so_field<SVolF>( "gridZ" ); 

  SVolFP const old_gridX = tsk_info->get_const_so_field<SVolF>( "gridX" ); 
  SVolFP const old_gridY = tsk_info->get_const_so_field<SVolF>( "gridY" ); 
  SVolFP const old_gridZ = tsk_info->get_const_so_field<SVolF>( "gridZ" ); 

  //carry forward  the grid information.                   
  *gridX <<= *old_gridX;                   
  *gridY <<= *old_gridY;                   
  *gridZ <<= *old_gridZ;                   

}
//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

void 
GridInfo::register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ){ 

}

void 
GridInfo::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                SpatialOps::OperatorDatabase& opr ){ 
}
