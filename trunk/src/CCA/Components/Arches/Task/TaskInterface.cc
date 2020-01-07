#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <Core/Grid/Variables/VarTypes.h>

//Uintah Includes:

using namespace Uintah;

typedef ArchesFieldContainer::WHICH_DW WHICH_DW;
typedef ArchesFieldContainer::VAR_DEPEND VAR_DEPEND;
typedef ArchesFieldContainer::VariableRegistry VariableRegistry;

TaskInterface::TaskInterface( std::string task_name, int matl_index ) :
  m_task_name(task_name),
  m_matl_index(matl_index)
{
}

TaskInterface::~TaskInterface()
{
  //destroy local labels
  for ( auto ilab = m_local_labels.begin(); ilab != m_local_labels.end(); ilab++ ){
    VarLabel::destroy(*ilab);
  }
}
