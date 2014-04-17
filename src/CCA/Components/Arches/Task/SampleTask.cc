#include <CCA/Components/Arches/Task/SampleTask.h>

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
  register_variable( "uVelocitySPBC",     FACEX,     REQUIRES,       1, LATEST, variable_registry );
  register_variable( "vVelocitySPBC",     FACEY,     REQUIRES,       2, LATEST, variable_registry );

}

//This is the work for the task.  First, get the variables. Second, do the work! 
void 
SampleTask::eval( const Patch* patch, UintahVarMap& var_map, ConstUintahVarMap& const_var_map ){ 

  CCVariable<double>*      temp = get_var<CCVariable<double> >("a_sample_variable", var_map);
  constSFCXVariable<double>*  u = get_const_var<constSFCXVariable<double> >("uVelocitySPBC", const_var_map); 
  constSFCYVariable<double>*  v = get_const_var<constSFCYVariable<double> >("vVelocitySPBC", const_var_map); 

  temp->initialize(3.0); 

}
