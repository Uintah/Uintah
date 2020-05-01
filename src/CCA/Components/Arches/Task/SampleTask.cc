#include <CCA/Components/Arches/Task/SampleTask.h>

using namespace Uintah;

//--------------------------------------------------------------------------------------------------
SampleTask::SampleTask( std::string task_name, int matl_index )
  : TaskInterface( task_name, matl_index )
{
}

//--------------------------------------------------------------------------------------------------
SampleTask::~SampleTask()
{
}

//--------------------------------------------------------------------------------------------------
void
SampleTask::problemSetup( ProblemSpecP& db )
{
  _value = 1.0;
  //db->findBlock("sample_task")->getAttribute("value",_value);
}

//--------------------------------------------------------------------------------------------------
void
SampleTask::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks )
{
  // Register all data warehouse variables used in task SampleTask::initialize
  register_variable( "a_sample_field", ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );
  register_variable( "a_result_field", ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );

  // NOTES:
  // * Pass underlying strings into register_variable where possible to improve searchability (e.g., "a_sample_field")
  // * Uintah infrastructure uses underlying strings for debugging output and exceptions
  // * Supported parameter lists can be found in src/CCA/Components/Arches/Task/TaskVariableTools.cc
}

//--------------------------------------------------------------------------------------------------
void
SampleTask::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info )
{
  // Get all data warehouse variables used in SampleTask::initialize
  CCVariable<double>& field  = tsk_info->get_field<CCVariable<double> >( "a_sample_field" );
  CCVariable<double>& result = tsk_info->get_field<CCVariable<double> >( "a_result_field" );

  // Initialize data warehouse variables
  field.initialize( 1.1 );
  result.initialize( 2.1 );

  // NOTES:
  // * Non-portable get_field calls require 1 template parameter: (1) legacy Uintah type
  // * Pass underlying strings into get_field where possible to improve searchability (e.g., "a_sample_field")
  // * Uintah infrastructure uses underlying strings for debugging output and exceptions
}

//--------------------------------------------------------------------------------------------------
void
SampleTask::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep, const bool packed_tasks )
{
  // Register all data warehouse variables used in SampleTask::eval
  register_variable( "a_sample_field", ArchesFieldContainer::COMPUTES, /* Ghost Cell Quantity, Data Warehouse, */            variable_registry, time_substep, m_task_name );
  register_variable( "a_result_field", ArchesFieldContainer::COMPUTES, /* Ghost Cell Quantity, Data Warehouse, */            variable_registry, time_substep, m_task_name );
  register_variable( "density",        ArchesFieldContainer::REQUIRES, 1,                      ArchesFieldContainer::LATEST, variable_registry, time_substep, m_task_name );

  // NOTES:
  // * Pass underlying strings into register_variable where possible to improve searchability (e.g., "a_sample_field")
  // * Uintah infrastructure uses underlying strings for debugging output and exceptions
  // * Supported parameter lists can be found in src/CCA/Components/Arches/Task/TaskVariableTools.cc
}

//--------------------------------------------------------------------------------------------------
void
SampleTask::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info )
{
  // Get all data warehouse variables used in SampleTask::eval
  CCVariable<double>& field   = tsk_info->get_field<CCVariable<double> >( "a_sample_field" );
  CCVariable<double>& result  = tsk_info->get_field<CCVariable<double> >( "a_result_field" );
  CCVariable<double>& density = tsk_info->get_field<CCVariable<double> >( "density" );

  // Setup the range of cells to iterate over
  Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );

  // Setup the loop that iterates over cells
  Uintah::parallel_for( range, [&]( int i, int j, int k ){
    field(i,j,k)  = _value * density(i,j,k);
    result(i,j,k) = field(i,j,k) * field(i,j,k);
  });

  // NOTES:
  // * Non-portable get_field calls require 1 template parameter: (1) legacy Uintah type
  // * Pass underlying strings into get_field where possible to improve searchability (e.g., "a_sample_field")
  // * Uintah infrastructure uses underlying strings for debugging output and exceptions
  // * Non-portable Uintah::parallel_for calls do not pass execObj and are executed serially
}
