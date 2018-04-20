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
  class AtomicTaskInterface : public TaskInterface {

public:

    typedef std::tuple<ParticleVariable<double>*, ParticleSubset*> ParticleTuple;

    typedef std::tuple<constParticleVariable<double>*, ParticleSubset*> ConstParticleTuple;

    /** @brief Default constructor **/
    AtomicTaskInterface( std::string task_name, int matl_index );

    /** @brief Default destructor **/
    virtual ~AtomicTaskInterface();

    /** @brief Input file interface **/
    virtual void problemSetup( ProblemSpecP& db ) = 0;

    /** @brief Create local labels for the task **/
    virtual void create_local_labels() = 0;

    /** @brief Registers all variables with pertinent information for the
     *         uintah dw interface **/
    virtual void register_timestep_eval(
      std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
      const int time_substep, const bool packed_tasks ) = 0;

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

    //These method definitions are required per the TaskInterface def:
    // They are defined here to avoid forcing the user to define empty functions
    // and to keep the spirit of the Atomic Task (eval only)
    void register_initialize( std::vector<AFC::VariableInformation>& variable_registry,
                                      const bool pack_tasks ){}
    void register_compute_bcs( std::vector<AFC::VariableInformation>& variable_registry,
                                       const int time_substep, const bool packed_tasks ){}
    void register_restart_initialize(
      std::vector<AFC::VariableInformation>& variable_registry, const bool packed_tasks ){}
    void register_timestep_init( std::vector<AFC::VariableInformation>& variable_registry,
                                         const bool pack_tasks ){}
    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr ){}
    void restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr ){}
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr ){}
    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr ){}

protected:

    typedef std::map<std::string, GridVariableBase* > UintahVarMap;
    typedef std::map<std::string, constVariableBase<GridVariableBase>* > ConstUintahVarMap;

    /** @brief The actual work done within the derived class **/
    virtual void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr ) = 0;

    std::vector<const VarLabel*> m_local_labels;
    WBCHelper* m_bcHelper {nullptr};

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
