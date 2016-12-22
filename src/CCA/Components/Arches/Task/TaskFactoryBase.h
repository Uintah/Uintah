#ifndef UT_TaskFactoryBase_h
#define UT_TaskFactoryBase_h

#include <Core/Grid/SimulationState.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Task/AtomicTaskInterface.h>
#include <CCA/Components/Arches/WBCHelper.h>
#include <string>
#include <iomanip>

namespace Uintah{

  class ArchesParticlesHelper;

  class TaskFactoryBase{

  public:

    TaskFactoryBase();
    virtual ~TaskFactoryBase();

    typedef std::map< std::string, TaskInterface*>              TaskMap;
    typedef std::map< std::string, TaskInterface::TaskBuilder*> BuildMap;
    typedef std::map< std::string, AtomicTaskInterface*>              ATaskMap;
    typedef std::map< std::string, AtomicTaskInterface::AtomicTaskBuilder*> ABuildMap;
    typedef std::map<std::string, std::vector<std::string> >    TypeToTaskMap;

    //May need to overload for the builders.
    /** @brief Parse the input file and create builders for all tasks listed in the UPS file **/
    virtual void register_all_tasks( ProblemSpecP& db ) = 0;

    /** @brief Actually build and call problemSetups for each tasks **/
    virtual void build_all_tasks( ProblemSpecP& db ) = 0;

    /** @brief Register tasks in a convenient container **/
    void register_task(std::string task_name,
                       TaskInterface::TaskBuilder* builder );

    /** @brief Register tasks in a convenient container **/
    void register_atomic_task( std::string task_name,
                               AtomicTaskInterface::AtomicTaskBuilder* builder );

    /** @brief Retrieve a subset (collection) of tasks given the subset name **/
    virtual std::vector<std::string> retrieve_task_subset(const std::string subset) = 0;

    /** @brief Retrieve a task by name **/
    TaskInterface* retrieve_task( const std::string task_name );

    /** @brief Retrieve an atomic task by name **/
    AtomicTaskInterface* retrieve_atomic_task( const std::string task_name );

    /** @brief Set the particle helper **/
    void set_particle_helper( ArchesParticlesHelper* part_helper ){ _part_helper = part_helper; }

    /** @brief Set the shared state **/
    void set_shared_state( SimulationStateP shared_state ){ _shared_state = shared_state; }

    TaskMap   _tasks;             ///< Task map
    ATaskMap   _atomic_tasks;      ///< Atomic Task map
    TaskMap& retrieve_all_tasks(){ return _tasks; }
    ATaskMap& retrieve_all_atomic_tasks(){ return _atomic_tasks; }

    /** @brief Retrieve the map of types -> tasks. **/
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

    /** @brief Test to see if a task exists **/
    const bool has_task( const std::string name ){

        std::vector<std::string>::const_iterator i_full_tasks
          = std::find(_active_tasks.begin(), _active_tasks.end(), name);
        std::vector<std::string>::const_iterator i_atomic_tasks
          = std::find(_active_atomic_tasks.begin(), _active_atomic_tasks.end(), name);

        if ( i_full_tasks == _active_tasks.end() && i_atomic_tasks == _active_atomic_tasks.end() ){
          return false;
        }

        return true;

    }

    virtual void add_task( ProblemSpecP& db ){
      //The derived class must do this for each specific factory.
      throw InvalidValue("Error: This factory currently cannot add tasks on the fly.", __FILE__,__LINE__);
    }

    void set_bcHelper( WBCHelper* helper ){
      m_bcHelper = helper;

      //assign all tasks a copy of the bcHelper
      for ( auto i = _tasks.begin(); i != _tasks.end(); i++ ){
        i->second->set_bcHelper( helper );
      }
      for ( auto i = _atomic_tasks.begin(); i != _atomic_tasks.end(); i++ ){
        i->second->set_bcHelper( helper );
      }

    }

    /** @brief Allow the factory to specify the order of scheduling for its tasks for initialization**/
    virtual void schedule_initialization( const LevelP& level,
                                  SchedulerP& sched,
                                  const MaterialSet* matls,
                                  bool doing_restart ){
      throw InvalidValue("Error: Task factory specific initialization for this factory is not implemented", __FILE__, __LINE__);
    }

  protected:

    BuildMap  _builders;                           ///< Builder map
    ABuildMap  _atomic_builders;                    ///< Builders for atomic tasks
    std::vector<std::string> _active_tasks;        ///< Active tasks
    std::vector<std::string> _active_atomic_tasks; ///< Active atomic tasks
    std::vector<std::string> m_task_init_order;    ///< Allows a factory to set an execution order for the tasks
    TypeToTaskMap _type_to_tasks;                  ///< Collects all tasks of a common type
    SimulationStateP _shared_state;                ///< Uintah SharedState

    WBCHelper* m_bcHelper;

    /** @brief Print some helpful proc0cout information when setting up tasks **/
    void print_task_setup_info(
      std::string name, std::string type, std::string additional_info="NA" ){
      std::stringstream msg;
      if ( additional_info != "NA" ){
        msg << "     " << std::setw(6) << std::left << "Task: " << std::setw(20) << std::left << name << std::setw(6) << std::left << " Desc: " << std::setw(20) << std::left << type << " Additional Info: " << additional_info << std::endl;
        proc0cout << msg.str();
      } else {
        msg << "     " << std::setw(6) << std::left << "Task: " << std::setw(20) << std::left << name << std::setw(6) << std::left << " Desc: " << std::setw(20) << std::left << type << std::endl;
        proc0cout << msg.str();
      }
    }

  private:

    ArchesParticlesHelper* _part_helper;          ///< Particle Helper

  };
}
#endif
