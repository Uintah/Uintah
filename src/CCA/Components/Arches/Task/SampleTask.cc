#include <CCA/Components/Arches/Task/SampleTask.h>

using namespace Uintah;

//--------------------------------------------------------------------------------------------------
SampleTask::SampleTask( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

//--------------------------------------------------------------------------------------------------
SampleTask::~SampleTask(){
}

//--------------------------------------------------------------------------------------------------
void
SampleTask::problemSetup( ProblemSpecP& db ){

  _value = 1.0;
  //db->findBlock("sample_task")->getAttribute("value",_value);

}

//--------------------------------------------------------------------------------------------------
void
SampleTask::register_timestep_init(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const bool packed_tasks ){
}

//--------------------------------------------------------------------------------------------------
void
SampleTask::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

//--------------------------------------------------------------------------------------------------
void
SampleTask::register_initialize(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const bool packed_tasks ){

  typedef ArchesFieldContainer AFC;

  //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
  register_variable( "a_sample_field", AFC::REQUIRES, 0, AFC::NEWDW,  variable_registry );
  register_variable( "a_result_field", AFC::COMPUTES, 0, AFC::NEWDW,  variable_registry );

}

//--------------------------------------------------------------------------------------------------
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

//--------------------------------------------------------------------------------------------------
//Register all variables both local and those needed from elsewhere that are required for this task.
void
SampleTask::register_timestep_eval(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const int time_substep, const bool packed_tasks ){

  typedef ArchesFieldContainer AFC;

  //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
  register_variable( "a_sample_field", AFC::COMPUTES, variable_registry, time_substep );
  register_variable( "a_result_field", AFC::COMPUTES, variable_registry, time_substep );
  register_variable( "density",        AFC::REQUIRES, 1, AFC::LATEST, variable_registry, time_substep );

  register_variable( "A", AFC::COMPUTES, variable_registry, time_substep );

}

//--------------------------------------------------------------------------------------------------
//This is the work for the task.  First, get the variables. Second, do the work!
void
SampleTask::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& field   = tsk_info->get_uintah_field_add<CCVariable<double> >( "a_sample_field" );
  CCVariable<double>& result  = tsk_info->get_uintah_field_add<CCVariable<double> >( "a_result_field" );
  CCVariable<double>& density = tsk_info->get_uintah_field_add<CCVariable<double> >( "density" );

  //Three ways to get variables:
  // Note that there are 'const' versions of these access calls to tsk_info as well. Just use a
  // tsk_info->get_const_*
  // By reference
  //CCVariable<double>& A_ref = tsk_info->get_uintah_field_add<CCVariable<double> >("A");
  // Pointer
  //CCVariable<double>* A_ptr = tsk_info->get_uintah_field<CCVariable<double> >("A");
  // Traditional Uintah Style
  // But, in this case you lose some of the convenient feature of the Arches Task Interface
  // which may or may not be important to you.
  //CCVariable<double> A_trad;
  //tsk_info->get_unmanaged_uintah_field<CCVariable<double> >("A", A_trad);

  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
    field(i,j,k) = _value * ( density(i,j,k));
    result(i,j,k) = field(i,j,k)*field(i,j,k);
  });
}
