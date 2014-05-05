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

}

//This is the work for the task.  First, get the variables. Second, do the work! 
void 
SampleTask::eval( const Patch* patch, UintahVarMap& var_map, ConstUintahVarMap& const_var_map ){ 

  using namespace SpatialOps;
  using SpatialOps::operator *; //becuase MPM is opening up SCIRun namespace which is causing clashes -- may need to divorce Arhces completely from mpmarches. 

  typedef SpatialOps::structured::SVolField   SVol;
  typedef SpatialOps::structured::SSurfXField SurfX;

  //CCVariable<double>*      temp = get_uintah_grid_var<CCVariable<double> >("a_sample_variable", var_map);
  //CCVariable<double>*     temp2 = get_uintah_grid_var<CCVariable<double> >("a_result_variable", var_map); 
  //constSFCXVariable<double>*  u = get_uintah_grid_var<constSFCXVariable<double> >("uVelocitySPBC", const_var_map); 
  constSFCYVariable<double>*  v = get_uintah_grid_var<constSFCYVariable<double> >("vVelocitySPBC", const_var_map); 
  SVol* const so_field = get_sos_grid_var<SVol>("a_sample_variable", var_map, patch, 0 ); 
  SVol* const re_field = get_sos_grid_var<SVol>("a_result_variable", var_map, patch, 0 ); 
  SurfX* const u_1 = get_sos_grid_var<SurfX>("uVelocitySPBC", var_map, patch, 0 ); 

 // temp->initialize(4.5); 
 //
  *so_field <<= 2.0;


  *re_field <<= (*so_field)/(*so_field); 

  *re_field <<= 2.0*(*so_field); 

}
