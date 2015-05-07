#include <CCA/Components/Arches/Task/SampleTask.h>
#include <CCA/Components/Arches/Operators/Operators.h>

#include <spatialops/Nebo.h>
#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>


using namespace Uintah;
using namespace SpatialOps;
using SpatialOps::operator *; 
typedef SVolField   SVolF;
typedef SSurfXField SurfX;
typedef SSurfYField SurfY;
typedef SSurfZField SurfZ;
typedef SpatialOps::SpatFldPtr<SVolF> SVolFP; 
typedef SpatialOps::SpatFldPtr<SurfX> SurfXP; 
typedef SpatialOps::SpatFldPtr<SurfY> SurfYP; 
typedef SpatialOps::SpatFldPtr<SurfZ> SurfZP; 

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

void 
SampleTask::create_local_labels(){ 

  register_new_variable("a_sample_variable", CC_DOUBLE); 
  register_new_variable("a_result_variable", CC_DOUBLE); 

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
SampleTask::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
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
  register_variable( "a_sample_variable", CC_DOUBLE, COMPUTES, 0, NEWDW,  variable_registry );
  register_variable( "a_result_variable", CC_DOUBLE, COMPUTES, 0, NEWDW,  variable_registry );

}

void 
SampleTask::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                        SpatialOps::OperatorDatabase& opr ){ 


  using namespace SpatialOps;
  using SpatialOps::operator *; 

  SVolFP field  = tsk_info->get_so_field<SVolF>( "a_sample_variable" );
  SVolFP result = tsk_info->get_so_field<SVolF>( "a_result_variable" );

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
SampleTask::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                  SpatialOps::OperatorDatabase& opr ){

  using namespace SpatialOps;
  using SpatialOps::operator *; 

  //typedef SpatialOps::BasicOpTypes<SVolF>::GradX GradX;
  //const GradX* const gradx = opr.retrieve_operator<GradX>();

  //Get uintah fields for work: 
  //CCVariable<double>*      field  = get_uintah_grid_var<CCVariable<double> >("a_sample_variable", var_map);
  //CCVariable<double>*      result = get_uintah_grid_var<CCVariable<double> >("a_result_variable", var_map); 
  //constSFCXVariable<double>*    u = get_uintah_grid_var<constSFCXVariable<double> >("uVelocitySPBC", const_var_map); 
  //constSFCYVariable<double>*    v = get_uintah_grid_var<constSFCYVariable<double> >("vVelocitySPBC", const_var_map); 

  //Get spatialops variables for work: 
  SVolFP field   = tsk_info->get_so_field<SVolF>( "a_sample_variable" );
  SVolFP result  = tsk_info->get_so_field<SVolF>( "a_result_variable" );
  SVolFP const density = tsk_info->get_so_field<SVolF>( "density" );
  SurfXP const u      = tsk_info->get_so_field<SurfX>("uVelocitySPBC" );
  SurfYP const v      = tsk_info->get_so_field<SurfY>("vVelocitySPBC" );
  SurfZP const w      = tsk_info->get_so_field<SurfZ>("wVelocitySPBC" );

  *field <<= _value*(*density);

  *result <<= (*field)*(*field); 

}
//
//------------------------------------------------
//------------- BOUNDARY CONDITIONS --------------
//------------------------------------------------
//

void 
SampleTask::register_compute_bcs( std::vector<VariableInformation>& variable_registry, const int time_substep ){ 
}

void 
SampleTask::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                        SpatialOps::OperatorDatabase& opr ){ 

}

