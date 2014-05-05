#include <CCA/Components/Arches/Task/SampleTask.h>

#include <spatialops/Nebo.h>


using namespace Uintah;

SampleTask::SampleTask( std::string task_name, int matl_index ) : 
TaskInterface( task_name, matl_index ) { 
}

SampleTask::~SampleTask(){ 
}

//Register all variables both local and those needed from elsewhere that are required for this task. 
void 
SampleTask::register_all_variables( std::vector<VariableInformation>& variable_registry ){ 

  //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
  register_variable( "a_sample_variable", CC_DOUBLE, LOCAL_COMPUTES, 0, NEWDW,  variable_registry );
  register_variable( "a_result_variable", CC_DOUBLE, LOCAL_COMPUTES, 0, NEWDW,  variable_registry );
  register_variable( "uVelocitySPBC",     FACEX,     REQUIRES,       1, LATEST, variable_registry );
  register_variable( "vVelocitySPBC",     FACEY,     REQUIRES,       2, LATEST, variable_registry );
  register_variable( "wVelocitySPBC",     FACEZ,     REQUIRES,       2, LATEST, variable_registry );

}

//This is the work for the task.  First, get the variables. Second, do the work! 
void 
SampleTask::eval( const Patch* patch, UintahVarMap& var_map, ConstUintahVarMap& const_var_map ){ 

  using namespace SpatialOps;
  using SpatialOps::operator *; 

  typedef SpatialOps::structured::SVolField   SVol;
  typedef SpatialOps::structured::SSurfXField SurfX;
  typedef SpatialOps::structured::SSurfYField SurfY;
  typedef SpatialOps::structured::SSurfZField SurfZ;

  //CCVariable<double>*      temp = get_uintah_grid_var<CCVariable<double> >("a_sample_variable", var_map);
  //CCVariable<double>*     temp2 = get_uintah_grid_var<CCVariable<double> >("a_result_variable", var_map); 
  //constSFCXVariable<double>*  u = get_uintah_grid_var<constSFCXVariable<double> >("uVelocitySPBC", const_var_map); 
  //constSFCYVariable<double>*  v = get_uintah_grid_var<constSFCYVariable<double> >("vVelocitySPBC", const_var_map); 

  SVol* const field = get_so_field<SVol>( "a_sample_variable", var_map, patch, 0, *this ); 
  SVol* const result = get_so_field<SVol>( "a_result_variable", var_map, patch, 0, *this ); 
  SurfX* const u = get_so_field<SurfX>("uVelocitySPBC", const_var_map, patch, 0, *this ); 
  SurfY* const v = get_so_field<SurfY>("vVelocitySPBC", const_var_map, patch, 0, *this ); 
  SurfZ* const w = get_so_field<SurfZ>("wVelocitySPBC", const_var_map, patch, 0, *this ); 


  *field <<= 2.0;

  *result <<= (*field)*(*field); 

}
