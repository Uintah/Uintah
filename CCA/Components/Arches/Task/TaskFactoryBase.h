#ifndef UT_TaskFactoryBase_h
#define UT_TaskFactoryBase_h

#include <Core/Grid/SimulationState.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <string>

namespace Uintah{

  class ArchesParticlesHelper;

  class TaskFactoryBase{

  public:

    TaskFactoryBase();
    virtual ~TaskFactoryBase();

    typedef std::map< std::string, TaskInterface*>              TaskMap;
    typedef std::map< std::string, TaskInterface::TaskBuilder*> BuildMap;
    typedef std::map<std::string, std::vector<std::string> >    TypeToTaskMap;

    //May need to overload for the builders.
    /** @brief Parse the input file and create builders for all tasks listed in the UPS file **/
    virtual void register_all_tasks( ProblemSpecP& db ) = 0;

    /** @brief Actually build and call problemSetups for each tasks **/
    virtual void build_all_tasks( ProblemSpecP& db ) = 0;

    /** @brief Register tasks in a convnient container **/
    void register_task(std::string task_name,
                       TaskInterface::TaskBuilder* builder );

    /** @brief Retrieve a subset (collection) of tasks given the subset name **/
    virtual std::vector<std::string> retrieve_task_subset(const std::string subset) = 0;

    /** @brief Retrieve a task by its name **/
    TaskInterface* retrieve_task( const std::string task_name );

    /** @brief Set the particle helper **/
    void set_particle_helper( ArchesParticlesHelper* part_helper ){ _part_helper = part_helper; }

    /** @brief Set the shared state **/
    void set_shared_state( SimulationStateP shared_state ){ _shared_state = shared_state; }

    TaskMap   _tasks;             ///< Task map
    TaskMap& retrieve_all_tasks(){ return _tasks; };

    const TypeToTaskMap& retrieve_type_to_tasks(){ return _type_to_tasks; }

    /** @brief Retrieve a set of tasks by their type **/
    const std::vector<std::string> retrieve_tasks_by_type( std::string type ){

      TypeToTaskMap::iterator iter = _type_to_tasks.find(type);
      if ( iter != _type_to_tasks.end() ){
        return iter->second;
      }

      //else return an empty vector
      std::vector<std::string> empty;
      empty.clear();
      return empty;

    }

    /** @brief Insert a task with its type group **/
    void assign_task_to_type_storage( std::string name, std::string type ){

      TypeToTaskMap::iterator iter = _type_to_tasks.find(type);

      if ( iter == _type_to_tasks.end() ){
        std::vector<std::string> names;
        names.push_back(name);
        _type_to_tasks.insert(std::make_pair(type,names));
        return;
      }

      if ( std::find( iter->second.begin(), iter->second.end(), name ) != iter->second.end() ){
        throw InvalidValue("Error: Found a task with the same name as another task: "+name, __FILE__, __LINE__);
      } else {
        iter->second.push_back(name);
        return;
      }

    }



  protected:

    BuildMap  _builders;                          ///< Builder map
    std::vector<std::string> _active_tasks;       ///< Task which are active
    TypeToTaskMap _type_to_tasks;                 ///< Collects all tasks of a common type
    SimulationStateP _shared_state;               ///< Uintah SharedState

  private:

    ArchesParticlesHelper* _part_helper;          ///< Particle Helper

  };
}
#endif
