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

  ASSERT(builder != nullptr);

  BuildMap::iterator i = _builders.find(task_name);
  if ( i == _builders.end() ){

    _builders.insert(std::make_pair(task_name, builder ));
    //If we are creating a builder, we assume the task is "active"
    _active_tasks.push_back(task_name);

  } else {

    throw InvalidValue("Error: Attemping to load a duplicate builder: "+task_name,__FILE__,__LINE__);

  }
}

TaskInterface*
TaskFactoryBase::retrieve_task( const std::string task_name ){

  TaskMap::iterator itsk = _tasks.find(task_name);

  if ( itsk != _tasks.end() ){

    return itsk->second;

  } else {

    BuildMap::iterator ibuild = _builders.find(task_name);

    if ( ibuild != _builders.end() ){

      TaskInterface::TaskBuilder* b = ibuild->second;
      TaskInterface* t = b->build();

      _tasks.insert(std::make_pair(task_name,t));

      TaskMap::iterator itsk_new = _tasks.find(task_name);

      //don't return t. must return the element in the map to
      //have the correct object intitiated
      return itsk_new->second;

    } else {

      throw InvalidValue("Error: Cannot find task named: "+task_name,__FILE__,__LINE__);

    }

  }

}
