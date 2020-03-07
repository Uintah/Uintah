#ifndef Uintah_Component_Arches_TaskInterface_h
#define Uintah_Component_Arches_TaskInterface_h

#include <CCA/Components/Arches/Task/TaskVariableTools.h>
#include <CCA/Components/Arches/WBCHelper.h>
#include <CCA/Components/Arches/Task/FieldContainer.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/LevelP.h>
#include <Core/Parallel/LoopExecution.hpp>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/InternalError.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Task.h>
#include <string>
#include <vector>
#include <map>
#include <typeinfo>
#include <typeindex>

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
class TaskInterface;

template <typename ExecSpace, typename MemSpace>
using archesFunctionPtr  = void (TaskInterface::*)( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr, ExecutionObject<ExecSpace, MemSpace>& execObj );

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

  //--------------------------------------------------------------------------------------------------
  // Begin Portability Support Members
  //--------------------------------------------------------------------------------------------------

  /** @brief Tells TaskFactoryBase which execution space this Arches task was assigned to.**/
  virtual TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers()   = 0;
  virtual TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers()   = 0;
  virtual TaskAssignedExecutionSpace loadTaskEvalFunctionPointers()         = 0;
  virtual TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers() = 0;
  virtual TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers()  = 0;

  // Hacky code warning.  Before portability was introduced, Arches had its own task execution system
  // that launched the correct method using polymorphism.
  // These Arches tasks are distinct from Uintah tasks, and are run *inside* Uintah tasks.
  // The difficulty with portability is that Uintah/Kokkos portability works best when templates flow
  // all the way down from task declaration to lambda/functor execution.
  // However, polymorphism and templates don't mix.
  // One way around it is having the derived class supply function pointers to the base class of all
  // possible template options.  Then when the "polymorphic" code is executed, it instead invokes the base
  // class version of the method, not the derived classes version, and the base class version searches
  // up the correct function pointer and executes it.
  // It also has a second helpful benefit with the compiler compiling all template options, for
  // Kokkos::Cuda, even if it's not desired, it may still compile and just won't generate compiler errors.
  // Yes, it's ugly.  But it's the best least ugly solution I could find.  -- Brad Peterson

  //--------------------------------------------------------------------------------------------------
  template <typename ExecSpace, typename MemSpace>
  void
  compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr, ExecutionObject<ExecSpace, MemSpace>& execObj )
  {
    archesFunctionPtr<UintahSpaces::CPU, UintahSpaces::HostSpace> function_ptr{nullptr};

    auto index = std::type_index(typeid(ExecSpace));
    auto handler = this->computeBCsFunctionPtrs.find(index);

    if ( handler != this->computeBCsFunctionPtrs.end() ) {
      function_ptr = handler->second;
    }
    else {
      throw InternalError( "Derived class version of Arches task compute_bcs() not found!", __FILE__, __LINE__ );
    }

    // Found the compute_bcs() function pointer associated with the execution space.  Run it.
    archesFunctionPtr<ExecSpace, MemSpace> handler_ptr = reinterpret_cast< archesFunctionPtr<ExecSpace, MemSpace> >(function_ptr);

    if ( handler_ptr ) {
      (this->*handler_ptr)( patch, tsk_info_mngr, execObj );
    }
  }

  //--------------------------------------------------------------------------------------------------
  template <typename ExecSpace, typename MemSpace>
  void
  addComputeBCsFunctionPtr( std::type_index ti, archesFunctionPtr<ExecSpace, MemSpace> ep )
  {
    computeBCsFunctionPtrs.emplace( ti, reinterpret_cast< archesFunctionPtr<UintahSpaces::CPU, UintahSpaces::HostSpace> >(ep) );
  }

  //--------------------------------------------------------------------------------------------------
  template <typename ExecSpace, typename MemSpace>
  void
  initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr, ExecutionObject<ExecSpace, MemSpace>& execObj )
  {
    archesFunctionPtr<UintahSpaces::CPU, UintahSpaces::HostSpace> function_ptr{nullptr};

    auto index = std::type_index(typeid(ExecSpace));
    auto handler = this->initializeFunctionPtrs.find(index);

    if ( handler != this->initializeFunctionPtrs.end() ) {
      function_ptr = handler->second;
    }
    else {
      std::string eSpace_name = typeid(ExecSpace).name();
      std::string mSpace_name = typeid(MemSpace).name();
      throw InternalError( "Derived class version of Arches task initialize() not found for"+m_task_name+" in: "+eSpace_name+" with: "+mSpace_name, __FILE__, __LINE__ );
    }

    // Found the initialize() function pointer associated with the execution space.  Run it.
    archesFunctionPtr<ExecSpace, MemSpace> handler_ptr = reinterpret_cast< archesFunctionPtr<ExecSpace, MemSpace> >(function_ptr);

    if ( handler_ptr ) {
      (this->*handler_ptr)( patch, tsk_info_mngr, execObj );
    }
  }

  //--------------------------------------------------------------------------------------------------
  template <typename ExecSpace, typename MemSpace>
  void
  addInitializeFunctionPtr( std::type_index ti, archesFunctionPtr<ExecSpace, MemSpace> ep )
  {
    initializeFunctionPtrs.emplace( ti, reinterpret_cast< archesFunctionPtr<UintahSpaces::CPU, UintahSpaces::HostSpace> >(ep) );
  }

  //--------------------------------------------------------------------------------------------------
  template <typename ExecSpace, typename MemSpace>
  void
  eval( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr, ExecutionObject<ExecSpace, MemSpace>& execObj )
  {
    archesFunctionPtr<UintahSpaces::CPU, UintahSpaces::HostSpace> function_ptr{nullptr};

    auto index = std::type_index(typeid(ExecSpace));
    auto handler = this->evalFunctionPtrs.find(index);

    if ( handler != this->evalFunctionPtrs.end() ) {
      function_ptr = handler->second;
    }
    else {
      throw InternalError( "Derived class version of Arches task eval() not found!", __FILE__, __LINE__ );
    }

    // Found the eval() function pointer associated with the execution space.  Run it.
    archesFunctionPtr<ExecSpace, MemSpace> handler_ptr = reinterpret_cast< archesFunctionPtr<ExecSpace, MemSpace> >(function_ptr);

    if ( handler_ptr ) {
      (this->*handler_ptr)( patch, tsk_info_mngr, execObj );
    }
  }

  //--------------------------------------------------------------------------------------------------
  template <typename ExecSpace, typename MemSpace>
  void
  addEvalFunctionPtr( std::type_index ti, archesFunctionPtr<ExecSpace, MemSpace> ep )
  {
    evalFunctionPtrs.emplace( ti, reinterpret_cast< archesFunctionPtr<UintahSpaces::CPU, UintahSpaces::HostSpace> >(ep) );
  }

  //--------------------------------------------------------------------------------------------------
  template <typename ExecSpace, typename MemSpace>
  void
  timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr, ExecutionObject<ExecSpace, MemSpace>& execObj )
  {
    archesFunctionPtr<UintahSpaces::CPU, UintahSpaces::HostSpace> function_ptr{nullptr};

    auto index = std::type_index(typeid(ExecSpace));
    auto handler = this->timestepInitFunctionPtrs.find(index);

    if ( handler != this->timestepInitFunctionPtrs.end() ) {
      function_ptr = handler->second;
    }
    else {
      throw InternalError( "Derived class version of Arches task timestep_init() not found!", __FILE__, __LINE__ );
    }

    // Found the timestep_init() function pointer associated with the execution space.  Run it.
    archesFunctionPtr<ExecSpace, MemSpace> handler_ptr = reinterpret_cast< archesFunctionPtr<ExecSpace, MemSpace> >(function_ptr);

    if ( handler_ptr ) {
      (this->*handler_ptr)( patch, tsk_info_mngr, execObj );
    }
  }

  //--------------------------------------------------------------------------------------------------
  template <typename ExecSpace, typename MemSpace>
  void
  addTimestepInitFunctionPtr( std::type_index ti, archesFunctionPtr<ExecSpace, MemSpace> ep )
  {
    timestepInitFunctionPtrs.emplace( ti, reinterpret_cast< archesFunctionPtr<UintahSpaces::CPU, UintahSpaces::HostSpace> >(ep) );
  }

  //--------------------------------------------------------------------------------------------------
  template <typename ExecSpace, typename MemSpace>
  void
  restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr, ExecutionObject<ExecSpace, MemSpace>& execObj )
  {
    archesFunctionPtr<UintahSpaces::CPU, UintahSpaces::HostSpace> function_ptr{nullptr};

    auto index = std::type_index(typeid(ExecSpace));
    auto handler = this->restartInitFunctionPtrs.find(index);

    if ( handler != this->restartInitFunctionPtrs.end() ) {
      function_ptr = handler->second;
    }
    else {
      throw InternalError( "Derived class version of Arches task restart_initialize() not found!", __FILE__, __LINE__ );
    }

    // Found the restart_initialize() function pointer associated with the execution space.  Run it.
    archesFunctionPtr<ExecSpace, MemSpace> handler_ptr = reinterpret_cast< archesFunctionPtr<ExecSpace, MemSpace> >(function_ptr);

    if ( handler_ptr ) {
      (this->*handler_ptr)( patch, tsk_info_mngr, execObj );
    }
  }

  //--------------------------------------------------------------------------------------------------
  template <typename ExecSpace, typename MemSpace>
  void
  addRestartInitFunctionPtr( std::type_index ti, archesFunctionPtr<ExecSpace, MemSpace> ep )
  {
    restartInitFunctionPtrs.emplace( ti, reinterpret_cast< archesFunctionPtr<UintahSpaces::CPU, UintahSpaces::HostSpace> >(ep) );
  }

private:

  std::map<std::type_index, archesFunctionPtr<UintahSpaces::CPU, UintahSpaces::HostSpace> > computeBCsFunctionPtrs;
  std::map<std::type_index, archesFunctionPtr<UintahSpaces::CPU, UintahSpaces::HostSpace> > initializeFunctionPtrs;
  std::map<std::type_index, archesFunctionPtr<UintahSpaces::CPU, UintahSpaces::HostSpace> > evalFunctionPtrs;
  std::map<std::type_index, archesFunctionPtr<UintahSpaces::CPU, UintahSpaces::HostSpace> > timestepInitFunctionPtrs;
  std::map<std::type_index, archesFunctionPtr<UintahSpaces::CPU, UintahSpaces::HostSpace> > restartInitFunctionPtrs;

  //--------------------------------------------------------------------------------------------------
  // End Portability Support Members
  //--------------------------------------------------------------------------------------------------

public:

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

      if ( name == "char_gas_reaction0_qn0" ) {
        printf( " Registering variable char_gas_reaction0_qn0 for task \n" );
      }

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

//--------------------------------------------------------------------------------------------------
// Template Specialization
template <unsigned int arches_mode>
struct ArchesSchedulingHelper
{
  template <typename ExecSpace, typename MemSpace>
  void
  addFunctionPtr( std::type_index ti, archesFunctionPtr<ExecSpace, MemSpace> ep, TaskInterface* ArchTI )
  {
    std::cout << "Invalid use of ArchesSchedulingHelper  \n";
  }
};

//--------------------------------------------------------------------------------------------------
template < >
struct ArchesSchedulingHelper<TaskInterface::INITIALIZE>
{
  template <typename ExecSpace, typename MemSpace>
  void
  addFunctionPtr( std::type_index ti, archesFunctionPtr<ExecSpace, MemSpace> ep, TaskInterface* ArchTI )
  {
    ArchTI->addInitializeFunctionPtr<ExecSpace, MemSpace>( ti, ep );
  }
};

//--------------------------------------------------------------------------------------------------
template < >
struct ArchesSchedulingHelper<TaskInterface::TIMESTEP_EVAL>
{
  template <typename ExecSpace, typename MemSpace>
  void
  addFunctionPtr( std::type_index ti, archesFunctionPtr<ExecSpace, MemSpace> ep, TaskInterface* ArchTI )
  {
    ArchTI->addEvalFunctionPtr<ExecSpace, MemSpace>( ti, ep );
  }
};

//--------------------------------------------------------------------------------------------------
template < >
struct ArchesSchedulingHelper<TaskInterface::BC>
{
  template <typename ExecSpace, typename MemSpace>
  void
  addFunctionPtr( std::type_index ti, archesFunctionPtr<ExecSpace, MemSpace> ep, TaskInterface* ArchTI )
  {
    ArchTI->addComputeBCsFunctionPtr<ExecSpace, MemSpace>( ti, ep );
  }
};

//--------------------------------------------------------------------------------------------------
template < >
struct ArchesSchedulingHelper<TaskInterface::RESTART_INITIALIZE>
{
  template <typename ExecSpace, typename MemSpace>
  void
  addFunctionPtr( std::type_index ti, archesFunctionPtr<ExecSpace, MemSpace> ep, TaskInterface* ArchTI )
  {
    ArchTI->addRestartInitFunctionPtr<ExecSpace, MemSpace>( ti, ep );
  }
};

//--------------------------------------------------------------------------------------------------
template < >
struct ArchesSchedulingHelper<TaskInterface::TIMESTEP_INITIALIZE>
{
  template <typename ExecSpace, typename MemSpace>
  void
  addFunctionPtr( std::type_index ti, archesFunctionPtr<ExecSpace, MemSpace> ep, TaskInterface* ArchTI )
  {
    ArchTI->addTimestepInitFunctionPtr<ExecSpace, MemSpace>( ti, ep );
  }
};

//--------------------------------------------------------------------------------------------------
// Three tag overloaded version of create_portable_arches_tasks()
template < unsigned int ArchesTaskType
         , typename ArchesTaskObject
         , typename ExecSpace1, typename MemSpace1
         , typename ExecSpace2, typename MemSpace2
         , typename ExecSpace3, typename MemSpace3 >
TaskAssignedExecutionSpace create_portable_arches_tasks( ArchesTaskObject * taskPtr
                                                       , void (ArchesTaskObject::*afp1)( const Patch                                  * patch
                                                                                       ,       ArchesTaskInfoManager                  * tsk_info_mngr
                                                                                       ,       ExecutionObject<ExecSpace1, MemSpace1> & execObj
                                                                                       )
                                                       , void (ArchesTaskObject::*afp2)( const Patch                                  * patch
                                                                                       ,       ArchesTaskInfoManager                  * tsk_info_mngr
                                                                                       ,       ExecutionObject<ExecSpace2, MemSpace2> & execObj
                                                                                       )
                                                       , void (ArchesTaskObject::*afp3)( const Patch                                  * patch
                                                                                       ,       ArchesTaskInfoManager                  * tsk_info_mngr
                                                                                       ,       ExecutionObject<ExecSpace3, MemSpace3> & execObj
                                                                                       )
                                                       )
{
  ArchesSchedulingHelper<ArchesTaskType> helpMe;
  TaskAssignedExecutionSpace assignedTag{};

  // Check for GPU tasks
  // GPU tasks take top priority
  if ( Uintah::Parallel::usingDevice() ) {
    if ( std::is_same<Kokkos::Cuda, ExecSpace1>::value || std::is_same<Kokkos::Cuda, ExecSpace2>::value || std::is_same<Kokkos::Cuda, ExecSpace3>::value ){
      if ( std::is_same<Kokkos::Cuda, ExecSpace1>::value ) {           /* Task supports Kokkos::Cuda builds */
        helpMe.addFunctionPtr( std::type_index(typeid(Kokkos::Cuda)), static_cast< archesFunctionPtr<ExecSpace1, MemSpace1> >(afp1), taskPtr );
      }
      else if ( std::is_same<Kokkos::Cuda, ExecSpace2>::value ) {      /* Task supports Kokkos::Cuda builds */
        helpMe.addFunctionPtr( std::type_index(typeid(Kokkos::Cuda)), static_cast< archesFunctionPtr<ExecSpace2, MemSpace2> >(afp2), taskPtr );
      }
      else if ( std::is_same<Kokkos::Cuda , ExecSpace3>::value ) {     /* Task supports Kokkos::Cuda builds */
        helpMe.addFunctionPtr( std::type_index(typeid(Kokkos::Cuda)), static_cast< archesFunctionPtr<ExecSpace3, MemSpace3> >(afp3), taskPtr );
      }
      assignedTag = KOKKOS_CUDA;
    }
  }

  // Check for CPU tasks if a GPU task did not get loaded
  if ( assignedTag == TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE ) {
    if ( std::is_same<Kokkos::OpenMP, ExecSpace1>::value || std::is_same<Kokkos::OpenMP, ExecSpace2>::value || std::is_same<Kokkos::OpenMP, ExecSpace3>::value ) {
      if ( std::is_same<Kokkos::OpenMP, ExecSpace1>::value ) {         /* Task supports Kokkos::OpenMP builds */
        helpMe.addFunctionPtr( std::type_index(typeid(Kokkos::OpenMP)), static_cast< archesFunctionPtr<ExecSpace1, MemSpace1> >(afp1), taskPtr );
      }
      else if ( std::is_same<Kokkos::OpenMP, ExecSpace2>::value ) {    /* Task supports Kokkos::OpenMP builds */
        helpMe.addFunctionPtr( std::type_index(typeid(Kokkos::OpenMP)), static_cast< archesFunctionPtr<ExecSpace2, MemSpace2> >(afp2), taskPtr );
      }
      else if ( std::is_same<Kokkos::OpenMP, ExecSpace3>::value ) {    /* Task supports Kokkos::OpenMP builds */
        helpMe.addFunctionPtr( std::type_index(typeid(Kokkos::OpenMP)), static_cast< archesFunctionPtr<ExecSpace3, MemSpace3> >(afp3), taskPtr );
      }
      assignedTag = KOKKOS_OPENMP;
    }
    else if ( std::is_same<UintahSpaces::CPU, ExecSpace1>::value || std::is_same<UintahSpaces::CPU, ExecSpace2>::value || std::is_same<UintahSpaces::CPU, ExecSpace3>::value ) {
      if ( std::is_same<UintahSpaces::CPU, ExecSpace1>::value) {       /* Task supports non-Kokkos builds */
        helpMe.addFunctionPtr( std::type_index(typeid(UintahSpaces::CPU)), static_cast< archesFunctionPtr<ExecSpace1, MemSpace1> >(afp1), taskPtr );
      }
      else if ( std::is_same<UintahSpaces::CPU, ExecSpace2>::value ) { /* Task supports non-Kokkos builds */
        helpMe.addFunctionPtr( std::type_index(typeid(UintahSpaces::CPU)), static_cast< archesFunctionPtr<ExecSpace2, MemSpace2> >(afp2), taskPtr );
      }
      else if ( std::is_same<UintahSpaces::CPU, ExecSpace3>::value ) { /* Task supports non-Kokkos builds */
        helpMe.addFunctionPtr( std::type_index(typeid(UintahSpaces::CPU)), static_cast< archesFunctionPtr<ExecSpace3, MemSpace3> >(afp3), taskPtr );
      }
      assignedTag = UINTAH_CPU;
    }
  }

  return assignedTag;
}

//--------------------------------------------------------------------------------------------------
// Two tag overloaded version of create_portable_arches_tasks()
template < unsigned int ArchesTaskType
         , typename ArchesTaskObject
         , typename ExecSpace1, typename MemSpace1
         , typename ExecSpace2, typename MemSpace2 >
TaskAssignedExecutionSpace create_portable_arches_tasks( ArchesTaskObject * taskPtr
                                                       , void (ArchesTaskObject::*afp1)( const Patch                                  * patch
                                                                                       ,       ArchesTaskInfoManager                  * tsk_info_mngr
                                                                                       ,       ExecutionObject<ExecSpace1, MemSpace1> & execObj
                                                                                       )
                                                       , void (ArchesTaskObject::*afp2)( const Patch                                  * patch
                                                                                       ,       ArchesTaskInfoManager                  * tsk_info_mngr
                                                                                       ,       ExecutionObject<ExecSpace2, MemSpace2> & execObj
                                                                                       )
                                                       )
{
  ArchesSchedulingHelper<ArchesTaskType> helpMe;
  TaskAssignedExecutionSpace assignedTag{};

  // Check for GPU tasks
  // GPU tasks take top priority
  if ( Uintah::Parallel::usingDevice() ) {
    if ( std::is_same<Kokkos::Cuda, ExecSpace1>::value || std::is_same<Kokkos::Cuda, ExecSpace2>::value ) {
      if ( std::is_same<Kokkos::Cuda, ExecSpace1>::value ) {           /* Task supports Kokkos::Cuda builds */
        helpMe.addFunctionPtr( std::type_index(typeid(Kokkos::Cuda)), static_cast< archesFunctionPtr<ExecSpace1, MemSpace1> >(afp1), taskPtr );
      }
      else if ( std::is_same<Kokkos::Cuda, ExecSpace2>::value ) {      /* Task supports Kokkos::Cuda builds */
        helpMe.addFunctionPtr( std::type_index(typeid(Kokkos::Cuda)), static_cast< archesFunctionPtr<ExecSpace2, MemSpace2> >(afp2), taskPtr );
      }
      assignedTag = KOKKOS_CUDA;
    }
  }

  // Check for CPU tasks if a GPU task did not get loaded
  if ( assignedTag == TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE ) {
    if ( std::is_same<Kokkos::OpenMP, ExecSpace1>::value || std::is_same<Kokkos::OpenMP, ExecSpace2>::value ) {
      if ( std::is_same<Kokkos::OpenMP, ExecSpace1>::value ) {         /* Task supports Kokkos::OpenMP builds */
        helpMe.addFunctionPtr( std::type_index(typeid(Kokkos::OpenMP)), static_cast< archesFunctionPtr<ExecSpace1, MemSpace1> >(afp1), taskPtr );
      }
      else if ( std::is_same<Kokkos::OpenMP, ExecSpace2>::value ) {    /* Task supports Kokkos::OpenMP builds */
        helpMe.addFunctionPtr( std::type_index(typeid(Kokkos::OpenMP)), static_cast< archesFunctionPtr<ExecSpace2, MemSpace2> >(afp2), taskPtr );
      }
      assignedTag = KOKKOS_OPENMP;
    }
    else if ( std::is_same<UintahSpaces::CPU, ExecSpace1>::value || std::is_same<UintahSpaces::CPU, ExecSpace2>::value ) {
      if ( std::is_same<UintahSpaces::CPU, ExecSpace1>::value ) {      /* Task supports non-Kokkos builds */
        helpMe.addFunctionPtr( std::type_index(typeid(UintahSpaces::CPU)), static_cast< archesFunctionPtr<ExecSpace1, MemSpace1> >(afp1), taskPtr );
      }
      else if ( std::is_same<UintahSpaces::CPU, ExecSpace2>::value ) { /* Task supports non-Kokkos builds */
        helpMe.addFunctionPtr( std::type_index(typeid(UintahSpaces::CPU)), static_cast< archesFunctionPtr<ExecSpace2, MemSpace2> >(afp2), taskPtr );
      }
      assignedTag = UINTAH_CPU;
    }
  }

  return assignedTag;
}

//--------------------------------------------------------------------------------------------------
// One tag overloaded version of create_portable_arches_tasks()
template < unsigned int ArchesTaskType
         , typename ArchesTaskObject
         , typename ExecSpace1, typename MemSpace1 >
TaskAssignedExecutionSpace create_portable_arches_tasks( ArchesTaskObject * taskPtr
                                                       , void (ArchesTaskObject::*afp1)( const Patch                                  * patch
                                                                                       ,       ArchesTaskInfoManager                  * tsk_info_mngr
                                                                                       ,       ExecutionObject<ExecSpace1, MemSpace1> & execObj
                                                                                       )
                                                       )
{
  ArchesSchedulingHelper<ArchesTaskType> helpMe;
  TaskAssignedExecutionSpace assignedTag{};

  // Check for GPU tasks
  // GPU tasks take top priority
  if ( Uintah::Parallel::usingDevice() ) {
    if ( std::is_same<Kokkos::Cuda, ExecSpace1>::value ) {           /* Task supports Kokkos::Cuda builds */
      helpMe.addFunctionPtr( std::type_index(typeid(Kokkos::Cuda)), static_cast< archesFunctionPtr<ExecSpace1, MemSpace1> >(afp1), taskPtr );
      assignedTag = KOKKOS_CUDA;
    }
  }

  // Check for CPU tasks if a GPU task did not get loaded
  if ( assignedTag == TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE ) {
    if ( std::is_same<Kokkos::OpenMP, ExecSpace1>::value ) {         /* Task supports Kokkos::OpenMP builds */
      helpMe.addFunctionPtr( std::type_index(typeid(Kokkos::OpenMP)), static_cast< archesFunctionPtr<ExecSpace1, MemSpace1> >(afp1), taskPtr );
      assignedTag = KOKKOS_OPENMP;
    }
    else if ( std::is_same<UintahSpaces::CPU, ExecSpace1>::value ) { /* Task supports non-Kokkos builds */
      helpMe.addFunctionPtr( std::type_index(typeid(UintahSpaces::CPU)), static_cast< archesFunctionPtr<ExecSpace1, MemSpace1> >(afp1), taskPtr );
      assignedTag = UINTAH_CPU;
    }
  }

  return assignedTag;
}

} // end namespace Uintah

#endif
