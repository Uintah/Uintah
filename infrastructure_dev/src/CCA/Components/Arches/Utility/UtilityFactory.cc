#include <CCA/Components/Arches/Utility/UtilityFactory.h>
#include <CCA/Components/Arches/Utility/GridInfo.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>

using namespace Uintah; 

UtilityFactory::UtilityFactory()
{}

UtilityFactory::~UtilityFactory()
{}

void 
UtilityFactory::register_all_tasks( ProblemSpecP& db )
{ 

  //GRID INFORMATION
  std::string tname = "grid_info"; 
  TaskInterface::TaskBuilder* tsk = new GridInfo::Builder(tname,0); 
  register_task(tname, tsk); 

  _active_tasks.push_back(tname); 


}

void 
UtilityFactory::build_all_tasks( ProblemSpecP& db )
{ 

  typedef std::vector<std::string> SV; 

  for ( SV::iterator i = _active_tasks.begin(); i != _active_tasks.end(); i++){ 

    TaskInterface* tsk = retrieve_task(*i); 
    tsk->problemSetup( db ); 

    tsk->create_local_labels(); 

  }

}
