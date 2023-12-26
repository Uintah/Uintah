#ifndef UT_TaskFactoryBase_h
#define UT_TaskFactoryBase_h

#include <Core/Grid/MaterialManager.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Task/AtomicTaskInterface.h>
#include <CCA/Components/Application/ApplicationCommon.h>
#include <CCA/Components/Arches/WBCHelper.h>
#include <Core/Util/DebugStream.h>
#include <string>
#include <iomanip>

namespace Uintah{

  class ArchesParticlesHelper;
  class ArchesFieldContainer;

  class TaskFactoryBase{

  public:

    TaskFactoryBase( const ApplicationCommon* arches );
    virtual ~TaskFactoryBase();

    typedef std::map< std::string, TaskInterface*>              TaskMap;
    typedef std::map< std::string, TaskInterface::TaskBuilder*> BuildMap;
    typedef std::map< std::string, AtomicTaskInterface*>              ATaskMap;
    typedef std::map< std::string, AtomicTaskInterface::AtomicTaskBuilder*> ABuildMap;
    typedef std::map<std::string, std::vector<std::string> >    TypeToTaskMap;

    //May need to overload for the builders.
    /** @brief Parse the input file and create builders for all tasks listed in the UPS file **/
    virtual void register_all_tasks( ProblemSpecP& db ) = 0;

    /** @brief Register tasks in a convenient container **/
    void register_task( std::string task_name,
                        TaskInterface::TaskBuilder* builder,
                        ProblemSpecP db = nullptr );

    /** @brief Register tasks in a convenient container **/
    void register_atomic_task( std::string task_name,
                               AtomicTaskInterface::AtomicTaskBuilder* builder,
                               ProblemSpecP db = nullptr );

    /** @brief Retrieve a subset (collection) of tasks given the subset name **/
    virtual std::vector<std::string> retrieve_task_subset(const std::string subset) = 0;

    /** @brief Retrieve a task by name **/
    TaskInterface* retrieve_task( const std::string task_name,
                                  const bool ignore_missing_task=false );

    /** @brief Retrieve an atomic task by name **/
    AtomicTaskInterface* retrieve_atomic_task( const std::string task_name );

    /** @brief Set the particle helper **/
    void set_particle_helper( ArchesParticlesHelper* part_helper ){ _part_helper = part_helper; }

    /** @brief Set the Material Manager **/
    void set_materialManager( MaterialManagerP materialManager ){ _materialManager = materialManager; }

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
    bool has_task( const std::string name ){

        std::vector<std::string>::const_iterator i_full_tasks   = std::find( _active_tasks.begin(), _active_tasks.end(), name );
        std::vector<std::string>::const_iterator i_atomic_tasks = std::find( _active_atomic_tasks.begin(), _active_atomic_tasks.end(), name );

        if ( i_full_tasks == _active_tasks.end() && i_atomic_tasks == _active_atomic_tasks.end() ){
          return false;
        }

        return true;
    }

    virtual void add_task( ProblemSpecP& db ){
      // The derived class must do this for each specific factory.
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
    virtual void schedule_initialization( const LevelP      & level,
                                                SchedulerP  & sched,
                                          const MaterialSet * matls,
                                                bool          doing_restart ){
      throw InvalidValue("Error: Task factory specific initialization for this factory is not implemented", __FILE__, __LINE__);
    }

    /** @brief Public interface for scheduling a single task through the factory

        @param task_name The Task name
        @param ignore_missing_task If TRUE and the task isn't found, then only provide a warning.

    **/
    void schedule_task(  const std::string task_name,
                         TaskInterface::TASK_TYPE type,
                         const LevelP& level,
                         SchedulerP& sched,
                         const MaterialSet* matls,
                         const int time_substep=0,
                         const bool reinitialize=false,
                         const bool ignore_missing_task=false);

    /** @brief Public interface for scheduling a set of tasks through the factory **/
    void schedule_task_group( const std::string group_name,
                              TaskInterface::TASK_TYPE type,
                              const bool pack_tasks,
                              const LevelP& level,
                              SchedulerP& sched,
                              const MaterialSet* matls,
                              const int time_substep=0,
                              const bool reinitialize=false );

    /** @brief Public interface for scheduling a set of tasks through the factory when the
               group of tasks was determined upstream. This is useful when one needs to
               enforce a specific task exe order. **/
    void schedule_task_group( const std::string task_group_name,
                              std::vector<std::string> task_names,
                              TaskInterface::TASK_TYPE type,
                              const bool pack_tasks,
                              const LevelP& level,
                              SchedulerP& sched,
                              const MaterialSet* matls,
                              const int time_substep=0,
                              const bool reinitialize=false );

    /** @brief Allow the factory to execute the boundary conditions for each task that it owns **/
    virtual void schedule_applyBCs( const LevelP& level,
                                    SchedulerP& sched,
                                    const MaterialSet * matls,
                                    const int time_substep ){
      throw InvalidValue("Error: Task factory specific boundary condition application for this factory is not implemented", __FILE__, __LINE__);
    }

    /** @brief Interface to the Uintah scheduler **/
    void factory_schedule_task( const LevelP& level,
                                SchedulerP& sched,
                                const MaterialSet* matls,
                                TaskInterface::TASK_TYPE type,
                                std::vector<TaskInterface*> arches_task,
                                const std::string task_group_name,
                                int time_substep,
                                const bool reinitialize,
                                const bool pack_tasks );

    /** @brief Task callback **/
    template <typename ExecSpace, typename MemSpace>
    void do_task ( const PatchSubset* patches,
                   const MaterialSubset* matls,
                   OnDemandDataWarehouse* old_dw,
                   OnDemandDataWarehouse* new_dw,
                   UintahParams& uintahParams,
                   ExecutionObject<ExecSpace, MemSpace>& execObj,
                   std::vector<ArchesFieldContainer::VariableInformation>  variable_registry,
                   std::vector<TaskInterface*> arches_task,
                   TaskInterface::TASK_TYPE type,
                   int time_substep,
                   const bool pack_tasks );

    /** @brief A container to hold variable information across tasks **/
    struct GhostHelper{
      int numTasksNewDW{0};
      int numTasksOldDW{0};
      std::vector<std::string> taskNamesOldDW{};
      std::vector<std::string> taskNamesNewDW{};
      int max_newdw_ghost{-99};
      int max_olddw_ghost{-99};
      int min_newdw_ghost{99};
      int min_olddw_ghost{99};
    };

    /** @brief Clean out the list of max ghost cells **/
    void clear_max_ghost_list(){
      m_variable_ghost_info.clear();
    }

    /** @brief Potentially insert a new variable to the max ghost list **/
    void insert_max_ghost(const ArchesFieldContainer::VariableInformation& var_info,
                          const std::string task_group_name ){
      //Store max ghost information per variable:
      bool in_new_dw = false;
      if ( var_info.dw == ArchesFieldContainer::NEWDW ){
        in_new_dw = true;
      }
      auto iter = m_variable_ghost_info.find(var_info.name);
      if ( iter == m_variable_ghost_info.end() ){

        //first record of this variable
        GhostHelper ghelp;
        if ( in_new_dw ){
          ghelp.taskNamesNewDW.push_back(task_group_name);
          ghelp.numTasksNewDW = 1;
          ghelp.max_newdw_ghost = var_info.nGhost;
          ghelp.min_newdw_ghost = var_info.nGhost;
        } else {
          ghelp.taskNamesOldDW.push_back(task_group_name);
          ghelp.numTasksOldDW = 1;
          ghelp.min_olddw_ghost = var_info.nGhost;
          ghelp.max_olddw_ghost = var_info.nGhost;
        }
        m_variable_ghost_info.insert(std::make_pair(var_info.name, ghelp));

      } else {

        //variable already in list, update it
        if ( in_new_dw ){
          iter->second.taskNamesNewDW.push_back(task_group_name);
          iter->second.numTasksNewDW += 1;
          if ( iter->second.numTasksNewDW == 1 ){
            //This is the first time this variable is encountered for
            // this DW so set max and min ghosts equal
            iter->second.max_newdw_ghost = var_info.nGhost;
            iter->second.min_newdw_ghost = var_info.nGhost;
          } else {
            if ( var_info.nGhost > iter->second.max_newdw_ghost ){
              iter->second.max_newdw_ghost = var_info.nGhost;
            }
            if ( var_info.nGhost < iter->second.min_newdw_ghost ){
              iter->second.min_newdw_ghost = var_info.nGhost;
            }
          }
        } else {
          iter->second.taskNamesOldDW.push_back(task_group_name);
          iter->second.numTasksOldDW += 1;
          if ( iter->second.numTasksOldDW == 1 ){
            //This is the first time this variable is encountered for
            // this DW so set max and min ghosts equal
            iter->second.max_olddw_ghost = var_info.nGhost;
            iter->second.min_olddw_ghost = var_info.nGhost;
          } else {
            if ( var_info.nGhost > iter->second.max_olddw_ghost ){
              iter->second.max_olddw_ghost = var_info.nGhost;
            }
            if ( var_info.nGhost < iter->second.min_olddw_ghost ){
              iter->second.min_olddw_ghost = var_info.nGhost;
            }
          }
        }
      }
    }

    /** @brief Print ghost cell requirements for all variables in this task **/
    /** @brief Get the ghost cell information **/
    std::map<std::string, GhostHelper>& get_max_ghost_info(){
      return m_variable_ghost_info;
    }
    void print_variable_max_ghost();

  protected:

    BuildMap  _builders;                           ///< Builder map
    ABuildMap  _atomic_builders;                    ///< Builders for atomic tasks
    std::vector<std::string> _active_tasks;        ///< Active tasks
    std::vector<std::string> _active_atomic_tasks; ///< Active atomic tasks
    TypeToTaskMap _type_to_tasks;                  ///< Collects all tasks of a common type
    MaterialManagerP _materialManager;                ///< Uintah MaterialManager
    std::string _all_tasks_str{"all_tasks"};                    ///< Common name across all factories indicating _active_tasks
    std::string _factory_name;                     ///< Name of the factory
    std::vector<std::string> m_task_init_order;    ///< Allows a factory to set an execution order for the tasks
    const ApplicationCommon* m_arches;             ///< Reference to the mother ship

    WBCHelper* m_bcHelper;

    /** @brief Print some helpful proc0cout information when setting up tasks **/
    void print_task_setup_info(
      std::string name, std::string type, std::string additional_info="NA" ){
      std::stringstream msg;
      if ( additional_info != "NA" ){
        msg << "     " << std::setw(6) << std::left << "Task: " << std::setw(20) << std::left << name << std::setw(6) << std::left << " Desc: " << std::setw(20) << std::left << type << " Additional Info: " << additional_info << std::endl;
        proc0cout << msg.str();
      }
      else {
        msg << "     " << std::setw(6) << std::left << "Task: " << std::setw(20) << std::left << name << std::setw(6) << std::left << " Desc: " << std::setw(20) << std::left << type << std::endl;
        proc0cout << msg.str();
      }
    }

  private:

    ArchesParticlesHelper* _part_helper;          ///< Particle Helper
    int m_matl_index;
    std::map<std::string, GhostHelper> m_variable_ghost_info;   ///< Stores ghost info for variables across all tasks in this factory

  };

}
#endif
