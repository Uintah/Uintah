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

namespace Uintah {

class Task;
class VarLabel;
class Level;
class WBCHelper;
class TaskInterface {

public:

  typedef ArchesFieldContainer AFC;

  enum TASK_TYPE { INITIALIZE           // initialize()
                 , TIMESTEP_INITIALIZE  // timestep_init()
                 , TIMESTEP_EVAL        // eval()
                 , BC                   // compute_bcs()
                 , RESTART_INITIALIZE   // restart_initialize()
                 , ATOMIC               // eval()
                 };

  static const std::string get_task_type_string( TASK_TYPE type )
  {
    if ( type == TIMESTEP_INITIALIZE ) {
      return "Time step Initialize";
    }
    else if ( type == INITIALIZE ) {
      return "INITIALIZE";
    }
    else if ( type == TIMESTEP_EVAL ) {
      return "TIMESTEP_EVAL";
    }
    else if ( type == BC ) {
      return "BC";
    }
    else if ( type == RESTART_INITIALIZE ) {
      return "RESTART_INITIALIZE";
    }
    else if ( type == ATOMIC ) {
      return "ATOMIC";
    }
    else {
      std::cout << type << std::endl;
      // Return "Unknown task type. Please fix."
      throw InvalidValue( "Error: TaskType enum not valid.",__FILE__,__LINE__ );
    }
  }

  typedef std::tuple<ParticleVariable<double>*, ParticleSubset*> ParticleTuple;

  typedef std::tuple<constParticleVariable<double>*, ParticleSubset*> ConstParticleTuple;

  /** @brief Default constructor **/
  TaskInterface( std::string task_name, int matl_index );

  /** @brief Default destructor **/
  virtual ~TaskInterface();

  /** @brief Print task name. **/
  void print_task_name(){ std::cout << "Task: " << m_task_name << std::endl; }

  /** @brief Get task name **/
  const std::string get_task_name(){ return m_task_name; }

  /** @brief Get task function **/
  const std::string get_task_function(){ return m_task_function; }

  /** @brief Input file interface **/
  virtual void problemSetup( ProblemSpecP& db ) = 0;

  /** @brief Create local labels for the task **/
  virtual void create_local_labels() = 0;

  /** @brief Initialization method **/
  virtual void register_initialize(       std::vector<AFC::VariableInformation> & variable_registry
                                  , const bool                                    pack_tasks
                                  ) = 0;

  /** @brief Schedules work done at the top of a timestep (which might be nothing) **/
  virtual void register_timestep_init(       std::vector<AFC::VariableInformation> & variable_registry
                                     , const bool                                    pack_tasks
                                     ) = 0;

  /** @brief Registers all variables with pertinent information for the
   *         uintah dw interface **/
  virtual void register_timestep_eval(       std::vector<AFC::VariableInformation> & variable_registry
                                     , const int                                     time_substep
                                     , const bool                                    packed_tasks
                                     ) = 0;

  /** @brief Register all variables needed to compute boundary conditions **/
  virtual void register_compute_bcs(       std::vector<AFC::VariableInformation> & variable_registry
                                   , const int                                     time_substep
                                   , const bool                                    packed_tasks
                                   ) = 0;

  /** @brief Register initialization work to be accomplished only on restart **/
  virtual void register_restart_initialize(       std::vector<AFC::VariableInformation> & variable_registry
                                          , const bool                                    packed_tasks
                                          ){}

  /** @brief Builder class containing instructions on how to build the task **/
  class TaskBuilder {

    public:

      TaskBuilder(){};

      virtual ~TaskBuilder() {}

      virtual TaskInterface* build() = 0;

    protected:

  }; // end class TaskBuilder

  void set_bcHelper( Uintah::WBCHelper* helper ){ m_bcHelper = helper; }

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

  std::string                  m_task_name{"undefined"};      ///< String identifier of the task
  std::string                  m_task_function{"undefined"};  ///< String identifier on the task function (what does the task do?)
  const int                    m_matl_index;                  ///< Uintah material index
  std::vector<const VarLabel*> m_local_labels;                ///< Labels held by the task

  /** @brief A helper struct for creating new varlabels as requested by the task **/
  template <typename T>
  struct RegisterNewVariableHelper
  {
    RegisterNewVariableHelper(const std::string task_name):m_task_name(task_name){};

    void create_variable( const std::string name, std::vector<const VarLabel*>& local_labels )
    {
      const VarLabel* test = nullptr;

      test = VarLabel::find(name);

      if ( test == nullptr ) {

        //std::cout << "[Task Interface]  Registering new variable: " << name << " in task: " << m_task_name << std::endl;
        const VarLabel* label = VarLabel::create( name, T::getTypeDescription() );
        local_labels.push_back(label);

      } else {

        std::stringstream msg;
        msg << "Error: Trying to register a variable, " << name << ", in Task " << m_task_name <<
        ", that was created elsewhere. " << std::endl;
        throw InvalidValue( msg.str(), __FILE__, __LINE__ );

      }
    }

    const std::string m_task_name;

  };

  /** @brief Register a local varlabel for this task **/
  template <typename T>
  void register_new_variable( const std::string name ){

    RegisterNewVariableHelper<T>* helper = scinew RegisterNewVariableHelper<T>(m_task_name);
    helper->create_variable( name, m_local_labels );
    delete helper;

  }

  /** @brief Strip the class name from the m_task_name **/
  /** This assumes that the format is: [CLASSNAME]*
      where * = wildcard name
      so this should return *
  **/
  std::string strip_class_name(){
    return m_task_name.substr(m_task_name.find("]")+1, m_task_name.size());
  }

}; // end class TaskInterface
} // end namespace Uintah

#endif
