#include <CCA/Components/Arches/Task/TaskFactoryBase.h>

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

    return t; 
  
  } else { 

    throw InvalidValue("Error: Cannot find task named: "+task_name,__FILE__,__LINE__); 

  }
}
