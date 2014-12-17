#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <CCA/Components/Arches/ArchesParticlesHelper.h>

using namespace Uintah; 

TaskFactoryBase::TaskFactoryBase()
{
}

TaskFactoryBase::~TaskFactoryBase()
{
  for (BuildMap::iterator i = _builders.begin(); i != _builders.end(); i++ ){ 
    delete i->second; 
  }
  for (TaskMap::iterator i = _tasks.begin(); i != _tasks.end(); i++ ){ 
    delete i->second; 
  }
}


void 
TaskFactoryBase::register_task(std::string task_name, 
                               TaskInterface::TaskBuilder* builder ){ 

  ASSERT(builder != NULL); 

  BuildMap::iterator i = _builders.find(task_name); 
  if ( i == _builders.end() ){ 

    _builders.insert(std::make_pair(task_name, builder )); 

  } else {

    throw InvalidValue("Error: Attemping to load a duplicate builder: "+task_name,__FILE__,__LINE__); 

  }
}

TaskInterface* 
TaskFactoryBase::retrieve_task( const std::string task_name ){ 

  BuildMap::iterator i = _builders.find(task_name); 

  if ( i != _builders.end() ){ 

    TaskInterface::TaskBuilder* b = i->second; 
    TaskInterface* t = b->build(); 

    _tasks.insert(std::make_pair(task_name,t)); 

    TaskMap::iterator itsk = _tasks.find(task_name); 

    //don't return t. must return the element in the map to 
    //have the correct object intitiated
    return itsk->second; 
  
  } else { 

    throw InvalidValue("Error: Cannot find task named: "+task_name,__FILE__,__LINE__); 

  }
}

void 
TaskFactoryBase::create_varlabels(){ 

  for ( std::vector<std::string>::iterator i = _active_tasks.begin(); 
        i != _active_tasks.end(); i++){ 

    TaskInterface* tsk = retrieve_task(*i); 
    tsk->create_local_labels(); 

  }
}
