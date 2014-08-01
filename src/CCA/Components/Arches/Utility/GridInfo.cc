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

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void 
GridInfo::register_initialize( std::vector<VariableInformation>& variable_registry ){ 

  //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
  register_variable( "gridX",             CC_DOUBLE, LOCAL_COMPUTES,       0, NEWDW,  variable_registry ); 
  register_variable( "gridY",             CC_DOUBLE, LOCAL_COMPUTES,       0, NEWDW,  variable_registry ); 
  register_variable( "gridZ",             CC_DOUBLE, LOCAL_COMPUTES,       0, NEWDW,  variable_registry ); 

}

void 
GridInfo::initialize( const Patch* patch, FieldCollector* field_collector, 
                      SpatialOps::OperatorDatabase& opr ){ 


  CCVariable<double>* gridX  = field_collector->get_uintah_field<CCVariable<double> >("gridX", NEWDW );
  CCVariable<double>* gridY  = field_collector->get_uintah_field<CCVariable<double> >("gridY", NEWDW );
  CCVariable<double>* gridZ  = field_collector->get_uintah_field<CCVariable<double> >("gridZ", NEWDW );

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
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

void 
GridInfo::register_all_variables( std::vector<VariableInformation>& variable_registry, const int time_substep ){ 

  //carry forward the old values
  //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
  register_variable( "gridX",             CC_DOUBLE, COMPUTES,       0, NEWDW,  variable_registry, time_substep ); 
  register_variable( "gridY",             CC_DOUBLE, COMPUTES,       0, NEWDW,  variable_registry, time_substep ); 
  register_variable( "gridZ",             CC_DOUBLE, COMPUTES,       0, NEWDW,  variable_registry, time_substep ); 

  register_variable( "gridX",             CC_DOUBLE, REQUIRES,       0, OLDDW,  variable_registry, time_substep ); 
  register_variable( "gridY",             CC_DOUBLE, REQUIRES,       0, OLDDW,  variable_registry, time_substep ); 
  register_variable( "gridZ",             CC_DOUBLE, REQUIRES,       0, OLDDW,  variable_registry, time_substep ); 

}

void 
GridInfo::eval( const Patch* patch, FieldCollector* field_collector, 
                SpatialOps::OperatorDatabase& opr, 
                SchedToTaskInfo& info ){ 

    
  using namespace SpatialOps;
  using SpatialOps::operator *; 
  typedef SpatialOps::SVolField   SVol;

  SVol* const gridX = field_collector->get_so_field<SVol>( "gridX", NEWDW ); 
  SVol* const gridY = field_collector->get_so_field<SVol>( "gridY", NEWDW ); 
  SVol* const gridZ = field_collector->get_so_field<SVol>( "gridZ", NEWDW ); 

  SVol* const old_gridX = field_collector->get_so_field<SVol>( "gridX", OLDDW ); 
  SVol* const old_gridY = field_collector->get_so_field<SVol>( "gridY", OLDDW ); 
  SVol* const old_gridZ = field_collector->get_so_field<SVol>( "gridZ", OLDDW ); 

  //carry forward  the grid information.                   
  *gridX <<= *old_gridX;                   
  *gridY <<= *old_gridY;                   
  *gridZ <<= *old_gridZ;                   

}
