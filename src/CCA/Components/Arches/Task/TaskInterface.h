#ifndef Uintah_Component_Arches_TaskInterface_h
#define Uintah_Component_Arches_TaskInterface_h

#include <CCA/Components/Arches/Task/TaskVariableTools.h>
#include <CCA/Components/Arches/WBCHelper.h>
#include <CCA/Components/Arches/Task/FieldContainer.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/LevelP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidValue.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Task.h>
#include <string>
#include <vector>
#include <boost/foreach.hpp>

//==================================================================================================

/**
* @class  Task Interface for Arches
* @author Jeremy Thornock
* @date   2014
*
* @brief Serves as the interface to a standard Uintah task.
*
**/

//==================================================================================================

namespace Uintah{

  class Task;
  class VarLabel;
  class Level;
  class WBCHelper;
  class TaskInterface{

public:

    typedef ArchesFieldContainer AFC;

    enum TASK_TYPE { STANDARD_TASK, BC_TASK, INITIALIZE, TIMESTEP_INITIALIZE, TIMESTEP_EVAL, BC,
                     RESTART_INITIALIZE };

    static const std::string get_task_type_string( TASK_TYPE type ){
      if ( type == TIMESTEP_INITIALIZE ){
        return "Timestep Initialize";
      } else if ( type == INITIALIZE ){
        return "Initialize";
      } else if ( type == TIMESTEP_EVAL ){
        return "Timestep Evaluation";
      } else if ( type == BC ) {
        return "Boundary Condition Evalulation";
      } else if ( type == RESTART_INITIALIZE ){
        return "Restart Initialize";
      } else {
        std::cout << type << std::endl;
        //return "Unknown task type. Please fix."
        throw InvalidValue("Error: TaskType enum not valid.",__FILE__,__LINE__);
      }
    }

    typedef std::tuple<ParticleVariable<double>*, ParticleSubset*> ParticleTuple;

    typedef std::tuple<constParticleVariable<double>*, ParticleSubset*> ConstParticleTuple;

    /** @brief Default constructor **/
    TaskInterface( std::string task_name, int matl_index );

    /** @brief Default destructor **/
    virtual ~TaskInterface();

    /** @brief Print task name. **/
    void print_task_name(){
      std::cout << "Task: " << _task_name << std::endl;
    }

    /** @brief Get task name **/
    const std::string get_task_name(){ return _task_name; }

    /** @brief Input file interface **/
    virtual void problemSetup( ProblemSpecP& db ) = 0;

    /** @brief Create local labels for the task **/
    virtual void create_local_labels() = 0;

    /** @brief Initialization method **/
    virtual void register_initialize( std::vector<AFC::VariableInformation>& variable_registry,
                                      const bool pack_tasks ) = 0;

    /** @brief Schedules work done at the top of a timestep (which might be nothing) **/
    virtual void register_timestep_init( std::vector<AFC::VariableInformation>& variable_registry,
                                         const bool pack_tasks ) = 0;

    /** @brief Registers all variables with pertinent information for the
     *         uintah dw interface **/
    virtual void register_timestep_eval( std::vector<AFC::VariableInformation>& variable_registry,
                                         const int time_substep, const bool packed_tasks ) = 0;

    /** @brief Register all variables needed to compute boundary conditions **/
    virtual void register_compute_bcs( std::vector<AFC::VariableInformation>& variable_registry,
                                       const int time_substep, const bool packed_tasks ) = 0;

    /** @brief Register initialization work to be accomplished only on restart **/
    virtual void register_restart_initialize(
      std::vector<AFC::VariableInformation>& variable_registry, const bool packed_tasks ){}

    /** @brief Add this task to the Uintah task scheduler **/
    void schedule_task( const LevelP& level,
                        SchedulerP& sched,
                        const MaterialSet* matls,
                        TASK_TYPE task_type,
                        int time_substep );

    /** @brief The actual task interface function that references the
     *         derived class implementation **/
    void do_task( const ProcessorGroup* pc,
                  const PatchSubset* patches,
                  const MaterialSubset* matls,
                  DataWarehouse* old_dw,
                  DataWarehouse* new_dw,
                  std::vector<AFC::VariableInformation> variable_registry,
                  int time_substep );

    /** @brief Add this task to the Uintah task scheduler **/
    void schedule_init( const LevelP& level,
                        SchedulerP& sched,
                        const MaterialSet* matls,
                        const bool is_restart,
                        const bool reinitialize=false );

    /** @brief The actual task interface function that references the
     *         derived class implementation **/
    void do_init( const ProcessorGroup* pc,
                  const PatchSubset* patches,
                  const MaterialSubset* matls,
                  DataWarehouse* old_dw,
                  DataWarehouse* new_dw,
                  std::vector<AFC::VariableInformation> variable_registry );

    /** @brief The actual task interface function that references the
     *         derived class implementation **/
    void do_restart_init( const ProcessorGroup* pc,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw,
                          std::vector<AFC::VariableInformation> variable_registry );

    /** @brief Add this task to the Uintah task scheduler **/
    void schedule_timestep_init( const LevelP& level,
                                 SchedulerP& sched,
                                 const MaterialSet* matls );

    /** @brief The actual task interface function that references the
     *         derived class implementation **/
    void do_timestep_init( const ProcessorGroup* pc,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw,
                           std::vector<AFC::VariableInformation> variable_registry );

    /** @brief The actual task interface function that references the
     *         derived class implementation **/
    void do_bcs( const ProcessorGroup* pc,
                 const PatchSubset* patches,
                 const MaterialSubset* matls,
                 DataWarehouse* old_dw,
                 DataWarehouse* new_dw,
                 std::vector<AFC::VariableInformation> variable_registry,
                 int time_substep );

    /** @brief Builder class containing instructions on how to build the task **/
    class TaskBuilder {

      public:

        TaskBuilder(){};

        virtual ~TaskBuilder() {}

        virtual TaskInterface* build() = 0;

      protected:

    };

    void set_bcHelper( Uintah::WBCHelper* helper ){
      m_bcHelper = helper;
    }

    /** @brief The actual work done within the derived class **/
    virtual void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr ) = 0;

    /** @brief The actual work done within the derived class **/
    virtual void restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr ){}

    /** @brief Work done at the top of a timestep **/
    virtual void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr ) = 0;

    /** @brief The actual work done within the derived class **/
    virtual void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr ) = 0;

    /** @brief The actual work done within the derived class for computing the boundary conditions **/
    virtual void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr ) = 0;

protected:

    typedef std::map<std::string, GridVariableBase* > UintahVarMap;
    typedef std::map<std::string, constVariableBase<GridVariableBase>* > ConstUintahVarMap;

    WBCHelper* m_bcHelper;

    std::string                  _task_name;
    const int                    _matl_index;
    std::vector<const VarLabel*> _local_labels;

    /** @brief A helper struct for creating new varlabels as requested by the task **/
    template <typename T>
    struct RegisterNewVariableHelper{

      RegisterNewVariableHelper(){};

      void create_variable( const std::string name, std::vector<const VarLabel*>& local_labels ){
        const VarLabel* test = nullptr;
        test = VarLabel::find( name );

        if ( test == nullptr ){

          const VarLabel* label = VarLabel::create( name, T::getTypeDescription() );
          local_labels.push_back(label);

        } else {

          std::stringstream msg;
          msg << "Error: VarLabel already registered: " << name << " (name your task variable something else and try again)." << std::endl;
          throw InvalidValue(msg.str(), __FILE__, __LINE__);

        }
      }
    };

    /** @brief Register a local varlabel for this task **/
    template <typename T>
    void register_new_variable(const std::string name){

      RegisterNewVariableHelper<T>* helper = scinew RegisterNewVariableHelper<T>();
      helper->create_variable( name, _local_labels );
      delete helper;

    }

  };
}

#endif
