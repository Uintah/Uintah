#include <CCA/Components/Arches/Task/SampleTask.h>
#include <CCA/Components/Arches/Operators/Operators.h>

#include <spatialops/Nebo.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>


using namespace Uintah;

SampleTask::SampleTask( std::string task_name, int matl_index ) : 
TaskInterface( task_name, matl_index ) { 
}

SampleTask::~SampleTask(){ 
}

void 
SampleTask::problemSetup( ProblemSpecP& db ){ 

  _value = 1.0;
  db->findBlock("sample_task")->getAttribute("value",_value); 

  std::cout << " FOUND A VALUE = " << _value << std::endl; 

}

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void 
SampleTask::register_initialize( std::vector<VariableInformation>& variable_registry ){ 

  //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
  register_variable( "a_sample_variable", CC_DOUBLE, LOCAL_COMPUTES, 0, NEWDW,  variable_registry );
  register_variable( "a_result_variable", CC_DOUBLE, LOCAL_COMPUTES, 0, NEWDW,  variable_registry );

}

void 
SampleTask::initialize( const Patch* patch, UintahVarMap& var_map, 
                        ConstUintahVarMap& const_var_map, SpatialOps::OperatorDatabase& opr ){ 


  using namespace SpatialOps;
  using SpatialOps::operator *; 

  typedef SpatialOps::structured::SVolField   SVol;

  SVol* const field = get_so_field<SVol>( "a_sample_variable", var_map, patch, 0, *this ); 
  SVol* const result = get_so_field<SVol>( "a_result_variable", var_map, patch, 0, *this ); 

  *field  <<= 1.1; 
  *result <<= 2.1; 

}

//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

//Register all variables both local and those needed from elsewhere that are required for this task. 
void 
SampleTask::register_all_variables( std::vector<VariableInformation>& variable_registry ){ 

  //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
  register_variable( "old_sample_v",      CC_DOUBLE, REQUIRES,       0, OLDDW,  variable_registry ); 
  register_variable( "a_sample_variable", CC_DOUBLE, LOCAL_COMPUTES, 0, NEWDW,  variable_registry );
  register_variable( "a_result_variable", CC_DOUBLE, LOCAL_COMPUTES, 0, NEWDW,  variable_registry );
  register_variable( "density",           CC_DOUBLE, REQUIRES,       1, LATEST, variable_registry );
  register_variable( "uVelocitySPBC",     FACEX,     REQUIRES,       1, LATEST, variable_registry );
  register_variable( "vVelocitySPBC",     FACEY,     REQUIRES,       2, LATEST, variable_registry );
  register_variable( "wVelocitySPBC",     FACEZ,     REQUIRES,       2, LATEST, variable_registry );

}

//This is the work for the task.  First, get the variables. Second, do the work! 
void 
SampleTask::eval( const Patch* patch, UintahVarMap& var_map, 
                  ConstUintahVarMap& const_var_map, SpatialOps::OperatorDatabase& opr, const int time_substep ){ 

  using namespace SpatialOps;
  using SpatialOps::operator *; 

  typedef SpatialOps::structured::SVolField   SVol;
  typedef SpatialOps::structured::SSurfXField SurfX;
  typedef SpatialOps::structured::SSurfYField SurfY;
  typedef SpatialOps::structured::SSurfZField SurfZ;

  typedef SpatialOps::structured::BasicOpTypes<SVol>::GradX GradX;
  const GradX* const gradx = opr.retrieve_operator<GradX>();

  //Get uintah fields for work: 
  //CCVariable<double>*      field  = get_uintah_grid_var<CCVariable<double> >("a_sample_variable", var_map);
  //CCVariable<double>*      result = get_uintah_grid_var<CCVariable<double> >("a_result_variable", var_map); 
  //constSFCXVariable<double>*    u = get_uintah_grid_var<constSFCXVariable<double> >("uVelocitySPBC", const_var_map); 
  //constSFCYVariable<double>*    v = get_uintah_grid_var<constSFCYVariable<double> >("vVelocitySPBC", const_var_map); 

  //Get spatialops variables for work: 
  SVol* const field = get_so_field<SVol>( "a_sample_variable", var_map, patch, 0, *this ); 
  SVol* const result = get_so_field<SVol>( "a_result_variable", var_map, patch, 0, *this ); 
  SVol* const density = get_so_field<SVol>( "density", const_var_map, patch,  1, *this ); 
  SurfX* const u = get_so_field<SurfX>("uVelocitySPBC", const_var_map, patch, 1, *this ); 
  SurfY* const v = get_so_field<SurfY>("vVelocitySPBC", const_var_map, patch, 2, *this ); 
  SurfZ* const w = get_so_field<SurfZ>("wVelocitySPBC", const_var_map, patch, 2, *this ); 

  *field <<= _value*(*density);

  *result <<= (*field)*(*field); 

}
