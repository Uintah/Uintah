#include <CCA/Components/Arches/Task/SampleTask.h>

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
SampleTask::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
}

void
SampleTask::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}


//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void
SampleTask::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

  //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
  register_variable( "a_sample_field", ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW,  variable_registry );
  register_variable( "a_result_field", ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW,  variable_registry );

}

void
SampleTask::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& field  = *(tsk_info->get_uintah_field<CCVariable<double> >( "a_sample_field" ));
  CCVariable<double>& result = *(tsk_info->get_uintah_field<CCVariable<double> >( "a_result_field" ));

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
    field(i,j,k) = 1.1;
    result(i,j,k) = 2.1;
  });
}

//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

//Register all variables both local and those needed from elsewhere that are required for this task.
void
SampleTask::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){

  //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
  register_variable( "a_sample_field", ArchesFieldContainer::COMPUTES,       0, ArchesFieldContainer::NEWDW,  variable_registry, time_substep );
  register_variable( "a_result_field", ArchesFieldContainer::COMPUTES,       0, ArchesFieldContainer::NEWDW,  variable_registry, time_substep );
  register_variable( "density",           ArchesFieldContainer::REQUIRES,       1, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  register_variable( "uVelocitySPBC",     ArchesFieldContainer::REQUIRES,       1, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  register_variable( "vVelocitySPBC",     ArchesFieldContainer::REQUIRES,       2, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  register_variable( "wVelocitySPBC",     ArchesFieldContainer::REQUIRES,       2, ArchesFieldContainer::LATEST, variable_registry, time_substep );

}

//This is the work for the task.  First, get the variables. Second, do the work!
void
SampleTask::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& field   = *(tsk_info->get_uintah_field<CCVariable<double> >( "a_sample_field" ));
  CCVariable<double>& result  = *(tsk_info->get_uintah_field<CCVariable<double> >( "a_result_field" ));
  CCVariable<double>& density = *(tsk_info->get_uintah_field<CCVariable<double> >( "density" ));
  constSFCXVariable<double>& u = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("uVelocitySPBC" ));
  constSFCXVariable<double>& v = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("vVelocitySPBC" ));
  constSFCXVariable<double>& w = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("wVelocitySPBC" ));

  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
    field(i,j,k) = _value * ( density(i,j,k));
    result(i,j,k)= field(i,j,k)*field(i,j,k);
  });
}
