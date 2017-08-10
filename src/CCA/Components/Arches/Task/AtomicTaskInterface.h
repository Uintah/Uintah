#ifndef Uintah_Component_Arches_AtomicTaskInterface_h
#define Uintah_Component_Arches_AtomicTaskInterface_h

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

//===============================================================

/**
* @class  Atomic Task Interface for Arches
* @author Jeremy Thornock
* @date   2016
*
* @brief An atomic task (schedule + call back only). 
*
**/

//===============================================================

namespace Uintah{

  class Task;
  class VarLabel;
  class Level;
  class AtomicTaskInterface{

public:

    enum ATOMIC_TASK_TYPE { ATOMIC_STANDARD_TASK };

    typedef std::tuple<ParticleVariable<double>*, ParticleSubset*> ParticleTuple;

    typedef std::tuple<constParticleVariable<double>*, ParticleSubset*> ConstParticleTuple;

    /** @brief Default constructor **/
    AtomicTaskInterface( std::string task_name, int matl_index );

    /** @brief Default destructor **/
    virtual ~AtomicTaskInterface();

    /** @brief Print task name. **/
    void print_task_name(){
      std::cout << "Task: " << m_task_name << std::endl;
    }

    /** @brief Input file interface **/
    virtual void problemSetup( ProblemSpecP& db ) = 0;

    /** @brief Create local labels for the task **/
    virtual void create_local_labels() = 0;

    /** @brief Registers all variables with pertinent information for the
     *         uintah dw interface **/
    virtual void register_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
                                const int time_substep ) = 0;

    /** @brief Add this task to the Uintah task scheduler **/
    void schedule_task( const LevelP& level,
                        SchedulerP& sched,
                        const MaterialSet* matls,
                        ATOMIC_TASK_TYPE task_type,
                        int time_substep );

    /** @brief The actual task interface function that references the
     *         derived class implementation **/
    void do_task( const ProcessorGroup* pc,
                  const PatchSubset* patches,
                  const MaterialSubset* matls,
                  DataWarehouse* old_dw,
                  DataWarehouse* new_dw,
                  std::vector<ArchesFieldContainer::VariableInformation> variable_registry,
                  int time_substep );

    /** @brief Builder class containing instructions on how to build the task **/
    class AtomicTaskBuilder {

      public:

        AtomicTaskBuilder(){};

        virtual ~AtomicTaskBuilder() {}

        virtual AtomicTaskInterface* build() = 0;

      protected:

    };

    void set_bcHelper( Uintah::WBCHelper* helper ){
      m_bcHelper = helper;
    }

protected:

    typedef std::map<std::string, GridVariableBase* > UintahVarMap;
    typedef std::map<std::string, constVariableBase<GridVariableBase>* > ConstUintahVarMap;

    /** @brief The actual work done within the derived class **/
    virtual void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr ) = 0;

    std::string                  m_task_name;
    const int                    m_matl_index;
    std::vector<const VarLabel*> m_local_labels;
    WBCHelper* m_bcHelper;

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
      helper->create_variable( name, m_local_labels );
      delete helper;

    }
  };
}

#endif
