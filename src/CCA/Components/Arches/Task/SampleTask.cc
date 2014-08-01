#include <CCA/Components/Arches/Task/SampleTask.h>
#include <CCA/Components/Arches/Operators/Operators.h>

#include <spatialops/Nebo.h>
#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>


using namespace Uintah;

SampleTask::SampleTask( std::string task_name, int matl_index ) : 
TaskInterface( task_name, matl_index ) { 
}

SampleTask::~SampleTask(){ 
}

void 
SampleTask::problemSetup( ProblemSpecP& db ){ 

  _value = 1.0;
  //db->findBlock("sample_task")->getAttribute("value",_value); 

}

//
//------------------------------------------------
//------------- TIMESTEP INIT --------------------
//------------------------------------------------
//
void 
SampleTask::register_timestep_init( std::vector<VariableInformation>& variable_registry ){ 
}

void 
SampleTask::timestep_init( const Patch* patch, FieldCollector* field_collector, 
                          SpatialOps::OperatorDatabase& opr ){ 

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
SampleTask::initialize( const Patch* patch, FieldCollector* field_collector, 
                        SpatialOps::OperatorDatabase& opr ){ 


  using namespace SpatialOps;
  using SpatialOps::operator *; 

  typedef SpatialOps::SVolField   SVol;

  SVol* const field = field_collector->get_so_field<SVol>( "a_sample_variable", NEWDW ); 
  SVol* const result = field_collector->get_so_field<SVol>( "a_result_variable", NEWDW ); 

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
SampleTask::register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ){ 

  //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
  register_variable( "a_sample_variable", CC_DOUBLE, COMPUTES,       0, NEWDW,  variable_registry, time_substep );
  register_variable( "a_result_variable", CC_DOUBLE, COMPUTES,       0, NEWDW,  variable_registry, time_substep );
  register_variable( "density",           CC_DOUBLE, REQUIRES,       1, LATEST, variable_registry, time_substep );
  register_variable( "uVelocitySPBC",     FACEX,     REQUIRES,       1, LATEST, variable_registry, time_substep );
  register_variable( "vVelocitySPBC",     FACEY,     REQUIRES,       2, LATEST, variable_registry, time_substep );
  register_variable( "wVelocitySPBC",     FACEZ,     REQUIRES,       2, LATEST, variable_registry, time_substep );

}

//This is the work for the task.  First, get the variables. Second, do the work! 
void 
SampleTask::eval( const Patch* patch, FieldCollector* field_collector, 
                  SpatialOps::OperatorDatabase& opr ){

  using namespace SpatialOps;
  using SpatialOps::operator *; 

  typedef SpatialOps::SVolField   SVol;
  typedef SpatialOps::SSurfXField SurfX;
  typedef SpatialOps::SSurfYField SurfY;
  typedef SpatialOps::SSurfZField SurfZ;

  typedef SpatialOps::BasicOpTypes<SVol>::GradX GradX;
  const GradX* const gradx = opr.retrieve_operator<GradX>();

  //Get uintah fields for work: 
  //CCVariable<double>*      field  = get_uintah_grid_var<CCVariable<double> >("a_sample_variable", var_map);
  //CCVariable<double>*      result = get_uintah_grid_var<CCVariable<double> >("a_result_variable", var_map); 
  //constSFCXVariable<double>*    u = get_uintah_grid_var<constSFCXVariable<double> >("uVelocitySPBC", const_var_map); 
  //constSFCYVariable<double>*    v = get_uintah_grid_var<constSFCYVariable<double> >("vVelocitySPBC", const_var_map); 

  //Get spatialops variables for work: 
  SVol* const field = field_collector->get_so_field<SVol>( "a_sample_variable", NEWDW ); 
  SVol* const result = field_collector->get_so_field<SVol>( "a_result_variable", NEWDW ); 
  SVol* const density = field_collector->get_so_field<SVol>( "density", LATEST ); 
  SurfX* const u = field_collector->get_so_field<SurfX>("uVelocitySPBC", LATEST ); 
  SurfY* const v = field_collector->get_so_field<SurfY>("vVelocitySPBC", LATEST ); 
  SurfZ* const w = field_collector->get_so_field<SurfZ>("wVelocitySPBC", LATEST ); 

  *field <<= _value*(*density);

  *result <<= (*field)*(*field); 

}
