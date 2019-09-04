#include <CCA/Components/Arches/Task/AtomicTaskInterface.h>

using namespace Uintah;

typedef ArchesFieldContainer::WHICH_DW WHICH_DW;
typedef ArchesFieldContainer::VAR_DEPEND VAR_DEPEND;
typedef ArchesFieldContainer::VariableRegistry VariableRegistry;

AtomicTaskInterface::AtomicTaskInterface( std::string task_name, int matl_index ) :
  TaskInterface( task_name, matl_index )
{
}

AtomicTaskInterface::~AtomicTaskInterface()
{
  //destroy local labels
  for ( auto ilab = m_local_labels.begin(); ilab != m_local_labels.end(); ilab++ ){
    VarLabel::destroy( *ilab );
  }
}
