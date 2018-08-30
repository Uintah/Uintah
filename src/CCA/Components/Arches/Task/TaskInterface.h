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
  class TaskInterface;



  template <typename ES, typename MS>
  using archesFunctionPtr  = void (TaskInterface::*)( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr, ExecutionObject<ES, MS>& executionObject );

  class TaskInterface{

public:

    typedef ArchesFieldContainer AFC;

    enum TASK_TYPE { INITIALIZE           // initialize()
                   , TIMESTEP_INITIALIZE  // timestep_init()
                   , TIMESTEP_EVAL        // eval()
                   , BC                   // compute_bcs()
                   , RESTART_INITIALIZE   // restart_initialize()
                   , ATOMIC               // eval()
                   };

    static const std::string get_task_type_string( TASK_TYPE type ){

      if ( type == TIMESTEP_INITIALIZE ) {
        return "Timestep Initialize";
      }
      else if ( type == INITIALIZE ) {
        return "Initialize";
      }
      else if ( type == TIMESTEP_EVAL ) {
        return "Timestep Evaluation";
      }
      else if ( type == BC ) {
        return "Boundary Condition Evalulation";
      }
      else if ( type == RESTART_INITIALIZE ) {
        return "Restart Initialize";
      }
      else if ( type == ATOMIC ) {
        return "Atomic Task"; 
      }
      else {
        std::cout << type << std::endl;
        // Return "Unknown task type. Please fix."
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

    //------begin portability support members--------------------------------------------------

    /** @brief Tells TaskFactoryBase which execution space this Arches task was assigned to.**/
    virtual TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers() = 0;
    virtual TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers() = 0;
    virtual TaskAssignedExecutionSpace loadTaskEvalFunctionPointers() = 0;

    virtual TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers() = 0;
    virtual TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers() = 0;

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
    template<typename ExecutionSpace, typename MemorySpace>
    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr, ExecutionObject< ExecutionSpace, MemorySpace>& executionObject ) {

      archesFunctionPtr< UintahSpaces::CPU, UintahSpaces::HostSpace > function_ptr{nullptr};

      auto index = std::type_index(typeid(ExecutionSpace));
      auto handler = this->computeBCsFunctionPtrs.find(index);

      if ( handler != this->computeBCsFunctionPtrs.end() ) {
        function_ptr = handler->second;
      }
      else {
        throw InternalError("Derived class version of Arches task compute_bcs() not found!", __FILE__, __LINE__);
      }

      // Found the compute_bcs() function pointer associated with the execution space.  Run it.
      archesFunctionPtr<ExecutionSpace, MemorySpace> handler_ptr =
          reinterpret_cast< archesFunctionPtr< ExecutionSpace, MemorySpace > >(function_ptr);

      if ( handler_ptr ) {
        (this->*handler_ptr)( patch, tsk_info_mngr, executionObject );
      }
    }

    //--------------------------------------------------------------------------------------------------
    template<typename ExecutionSpace, typename MemorySpace>
    void addComputeBCsFunctionPtr( std::type_index ti, archesFunctionPtr<ExecutionSpace, MemorySpace> ep ) {
      computeBCsFunctionPtrs.emplace( ti, reinterpret_cast< archesFunctionPtr< UintahSpaces::CPU, UintahSpaces::HostSpace > >(ep) );
    }

    //--------------------------------------------------------------------------------------------------
    template<typename ExecutionSpace, typename MemorySpace>
    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr, ExecutionObject< ExecutionSpace, MemorySpace>& executionObject ) {

      archesFunctionPtr< UintahSpaces::CPU, UintahSpaces::HostSpace > function_ptr{nullptr};

      auto index = std::type_index(typeid(ExecutionSpace));
      auto handler = this->initializeFunctionPtrs.find(index);

      if ( handler != this->initializeFunctionPtrs.end() ) {
        function_ptr = handler->second;
      }
      else {
        throw InternalError("Derived class version of Arches task initialize() not found!", __FILE__, __LINE__);
      }

      // Found the initialize() function pointer associated with the execution space.  Run it.
      archesFunctionPtr<ExecutionSpace, MemorySpace> handler_ptr =
          reinterpret_cast< archesFunctionPtr< ExecutionSpace, MemorySpace > >(function_ptr);

      if ( handler_ptr ) {
        (this->*handler_ptr)( patch, tsk_info_mngr, executionObject );
      }
    }

    //--------------------------------------------------------------------------------------------------
    template<typename ExecutionSpace, typename MemorySpace>
    void addInitializeFunctionPtr( std::type_index ti, archesFunctionPtr<ExecutionSpace, MemorySpace> ep ) {
      initializeFunctionPtrs.emplace( ti, reinterpret_cast< archesFunctionPtr< UintahSpaces::CPU, UintahSpaces::HostSpace > >(ep) );
    }

    //--------------------------------------------------------------------------------------------------
    template<typename ExecutionSpace, typename MemorySpace>
    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr, ExecutionObject< ExecutionSpace, MemorySpace>& executionObject ) {

      archesFunctionPtr< UintahSpaces::CPU, UintahSpaces::HostSpace > function_ptr{nullptr};

      auto index = std::type_index(typeid(ExecutionSpace));
      auto handler = this->evalFunctionPtrs.find(index);

      if ( handler != this->evalFunctionPtrs.end() ) {
        function_ptr = handler->second;
      }
      else {
        throw InternalError("Derived class version of Arches task eval() not found!", __FILE__, __LINE__);
      }

      // Found the eval() function pointer associated with the execution space.  Run it.
      archesFunctionPtr<ExecutionSpace, MemorySpace> handler_ptr =
          reinterpret_cast< archesFunctionPtr< ExecutionSpace, MemorySpace > >(function_ptr);

      if ( handler_ptr ) {
        (this->*handler_ptr)( patch, tsk_info_mngr, executionObject );
      }
    }

    //--------------------------------------------------------------------------------------------------
    template<typename ExecutionSpace, typename MemorySpace  >
    void 
    addEvalFunctionPtr( std::type_index ti, archesFunctionPtr<ExecutionSpace, MemorySpace> ep ) {
      evalFunctionPtrs.emplace( ti, reinterpret_cast< archesFunctionPtr< UintahSpaces::CPU, UintahSpaces::HostSpace > >(ep) );
    }

    template<typename ExecutionSpace, typename MemorySpace>
    void restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr, ExecutionObject< ExecutionSpace, MemorySpace>& executionObject ) {

      archesFunctionPtr< UintahSpaces::CPU, UintahSpaces::HostSpace > function_ptr{nullptr};

      auto index = std::type_index(typeid(ExecutionSpace));
      auto handler = this->restartInitFunctionPtrs.find(index);

      if ( handler != this->restartInitFunctionPtrs.end() ) {
        function_ptr = handler->second;
      }
      else {
        throw InternalError("Derived class version of Arches task restart_initialize() not found!", __FILE__, __LINE__);
      }

      // Found the restart_initialize() function pointer associated with the execution space.  Run it.
      archesFunctionPtr<ExecutionSpace, MemorySpace> handler_ptr =
          reinterpret_cast< archesFunctionPtr< ExecutionSpace, MemorySpace > >(function_ptr);

      if ( handler_ptr ) {
        (this->*handler_ptr)( patch, tsk_info_mngr, executionObject );
      }
    }

    template<typename ExecutionSpace, typename MemorySpace  >
    void 
    addRestartInitFunctionPtr( std::type_index ti, archesFunctionPtr<ExecutionSpace, MemorySpace> ep ) {
      restartInitFunctionPtrs.emplace( ti, reinterpret_cast< archesFunctionPtr< UintahSpaces::CPU, UintahSpaces::HostSpace > >(ep) );
    }

    template<typename ExecutionSpace, typename MemorySpace>
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr, ExecutionObject< ExecutionSpace, MemorySpace>& executionObject ) {

      archesFunctionPtr< UintahSpaces::CPU, UintahSpaces::HostSpace > function_ptr{nullptr};

      auto index = std::type_index(typeid(ExecutionSpace));
      auto handler = this->timestepInitFunctionPtrs.find(index);

      if ( handler != this->timestepInitFunctionPtrs.end() ) {
        function_ptr = handler->second;
      }
      else {
        throw InternalError("Derived class version of Arches task timestep_init() not found!", __FILE__, __LINE__);
      }

      // Found the timestep_init() function pointer associated with the execution space.  Run it.
      archesFunctionPtr<ExecutionSpace, MemorySpace> handler_ptr =
          reinterpret_cast< archesFunctionPtr< ExecutionSpace, MemorySpace > >(function_ptr);

      if ( handler_ptr ) {
        (this->*handler_ptr)( patch, tsk_info_mngr, executionObject );
      }
    }


    template<typename ExecutionSpace, typename MemorySpace  >
    void 
    addTimestepInitFunctionPtr( std::type_index ti, archesFunctionPtr<ExecutionSpace, MemorySpace> ep ) {
      timestepInitFunctionPtrs.emplace( ti, reinterpret_cast< archesFunctionPtr< UintahSpaces::CPU, UintahSpaces::HostSpace > >(ep) );
    }



private:
    std::map<std::type_index, archesFunctionPtr< UintahSpaces::CPU, UintahSpaces::HostSpace > > computeBCsFunctionPtrs;
    std::map<std::type_index, archesFunctionPtr< UintahSpaces::CPU, UintahSpaces::HostSpace > > initializeFunctionPtrs;
    std::map<std::type_index, archesFunctionPtr< UintahSpaces::CPU, UintahSpaces::HostSpace > > evalFunctionPtrs;

    std::map<std::type_index, archesFunctionPtr< UintahSpaces::CPU, UintahSpaces::HostSpace > > timestepInitFunctionPtrs;
    std::map<std::type_index, archesFunctionPtr< UintahSpaces::CPU, UintahSpaces::HostSpace > > restartInitFunctionPtrs;

public:
    //------end portability support members----------------------------------------------------

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

        if ( name == "char_gas_reaction0_qn0" ) {
          printf(" Registering variable char_gas_reaction0_qn0 for task \n");
        }
        test = VarLabel::find( name );

        if ( test == nullptr ){

          const VarLabel* label = VarLabel::create( name, T::getTypeDescription() );
          local_labels.push_back(label);

        } else {

          std::stringstream msg;
          msg << "Error: VarLabel already registered (with Uintah): " << name << " (name your task variable something else and try again)." << std::endl;
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

  // Template Specialization
  template< unsigned int arches_mode>
  struct ArchesSchedulingHelper{
     template<typename ExecutionSpace, typename MemorySpace  >
     void 
     addFunctionPtr( std::type_index ti, archesFunctionPtr<ExecutionSpace, MemorySpace> ep, TaskInterface* ArchTI  ) {
        std::cout << "Invalid use of ArchesSchedulingHelper  \n";
     }
  };


  template< >
  struct ArchesSchedulingHelper<TaskInterface::INITIALIZE>{
     template<typename ExecutionSpace, typename MemorySpace  >
     void 
     addFunctionPtr( std::type_index ti, archesFunctionPtr<ExecutionSpace, MemorySpace> ep, TaskInterface* ArchTI ) {
       ArchTI->addInitializeFunctionPtr<ExecutionSpace,MemorySpace>(ti,ep);
     }
  };

  template< >
  struct ArchesSchedulingHelper<TaskInterface::TIMESTEP_EVAL>{
     template<typename ExecutionSpace, typename MemorySpace  >
     void 
     addFunctionPtr( std::type_index ti, archesFunctionPtr<ExecutionSpace, MemorySpace> ep, TaskInterface* ArchTI ) {
       ArchTI->addEvalFunctionPtr<ExecutionSpace,MemorySpace>(ti,ep);
     }
  };

  template< >
  struct ArchesSchedulingHelper<TaskInterface::BC>{
     template<typename ExecutionSpace, typename MemorySpace  >
     void 
     addFunctionPtr( std::type_index ti, archesFunctionPtr<ExecutionSpace, MemorySpace> ep, TaskInterface* ArchTI ) {
       ArchTI->addComputeBCsFunctionPtr<ExecutionSpace,MemorySpace>(ti,ep);
     }
  };

  template< >
  struct ArchesSchedulingHelper<TaskInterface::RESTART_INITIALIZE>{
     template<typename ExecutionSpace, typename MemorySpace  >
     void 
     addFunctionPtr( std::type_index ti, archesFunctionPtr<ExecutionSpace, MemorySpace> ep, TaskInterface* ArchTI ) {
       ArchTI->addRestartInitFunctionPtr<ExecutionSpace,MemorySpace>(ti,ep);
     }
  };

  template< >
  struct ArchesSchedulingHelper<TaskInterface::TIMESTEP_INITIALIZE>{
     template<typename ExecutionSpace, typename MemorySpace  >
     void 
     addFunctionPtr( std::type_index ti, archesFunctionPtr<ExecutionSpace, MemorySpace> ep, TaskInterface* ArchTI ) {
       ArchTI->addTimestepInitFunctionPtr<ExecutionSpace,MemorySpace>(ti,ep);
     }
  };


  template <unsigned int ArchesTaskType 
           , typename ArchesTaskObject
           , typename ES1, typename MS1
           , typename ES2, typename MS2
           , typename ES3, typename MS3
           >
  TaskAssignedExecutionSpace create_portable_arches_tasks( ArchesTaskObject* taskPtr
                                                         , void ( ArchesTaskObject::*afp1 )( const Patch                     * patch
                                                                                           ,       ArchesTaskInfoManager     * tsk_info_mngr
                                                                                           ,       ExecutionObject<ES1, MS1> & executionObject
                                                                                           )
                                                         , void ( ArchesTaskObject::*afp2 )( const Patch                     * patch
                                                                                           ,       ArchesTaskInfoManager     * tsk_info_mngr
                                                                                           ,       ExecutionObject<ES2, MS2> & executionObject
                                                                                           )
                                                         , void ( ArchesTaskObject::*afp3 )( const Patch                     * patch
                                                                                           ,       ArchesTaskInfoManager     * tsk_info_mngr
                                                                                           ,       ExecutionObject<ES3, MS3> & executionObject
                                                                                           )
                                                         )
  {
    ArchesSchedulingHelper<ArchesTaskType> helpMe;
    TaskAssignedExecutionSpace assignedTag{};

    // Check for CUDA tasks
    // GPU tasks take top priority
    if (Uintah::Parallel::usingDevice()) {

      if ( std::is_same< Kokkos::Cuda , ES1 >::value || std::is_same< Kokkos::Cuda , ES2 >::value || std::is_same< Kokkos::Cuda , ES3 >::value ) {

        if (std::is_same< Kokkos::Cuda , ES1 >::value) {          // Task supports Kokkos::Cuda builds
            helpMe.addFunctionPtr(std::type_index(typeid(Kokkos::Cuda)),static_cast< archesFunctionPtr<ES1, MS1> >(afp1), taskPtr);
        }
        else if (std::is_same< Kokkos::Cuda , ES2 >::value) {     // Task supports Kokkos::Cuda builds
            helpMe.addFunctionPtr(std::type_index(typeid(Kokkos::Cuda)),static_cast< archesFunctionPtr<ES2, MS2> >(afp2), taskPtr);
        }
        else if (std::is_same< Kokkos::Cuda , ES3 >::value) {     // Task supports Kokkos::Cuda builds
            helpMe.addFunctionPtr(std::type_index(typeid(Kokkos::Cuda)),static_cast< archesFunctionPtr<ES3, MS3> >(afp3), taskPtr);

        }
        assignedTag = KOKKOS_CUDA;
      }
    }

    if (assignedTag == TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE) {

      if ( std::is_same< Kokkos::OpenMP , ES1 >::value || std::is_same< Kokkos::OpenMP , ES2 >::value || std::is_same< Kokkos::OpenMP , ES3 >::value ) {

        if (std::is_same< Kokkos::OpenMP , ES1 >::value) {        // Task supports Kokkos::OpenMP builds
            helpMe.addFunctionPtr(std::type_index(typeid(Kokkos::OpenMP)),static_cast< archesFunctionPtr<ES1, MS1> >(afp1), taskPtr);
        }
        else if (std::is_same< Kokkos::OpenMP , ES2 >::value) {   // Task supports Kokkos::OpenMP builds
            helpMe.addFunctionPtr(std::type_index(typeid(Kokkos::OpenMP)),static_cast< archesFunctionPtr<ES2, MS2> >(afp2), taskPtr);
        }
        else if (std::is_same< Kokkos::OpenMP , ES3 >::value) {   // Task supports Kokkos::OpenMP builds
            helpMe.addFunctionPtr(std::type_index(typeid(Kokkos::OpenMP)),static_cast< archesFunctionPtr<ES3, MS3> >(afp3), taskPtr);
        }
        assignedTag = KOKKOS_OPENMP;
      }
      else if ( std::is_same< UintahSpaces::CPU , ES1 >::value || std::is_same< UintahSpaces::CPU , ES2 >::value || std::is_same< UintahSpaces::CPU , ES3 >::value ) {

        if (std::is_same< UintahSpaces::CPU , ES1 >::value) { // Task supports non-Kokkos builds
            helpMe.addFunctionPtr(std::type_index(typeid(UintahSpaces::CPU)),static_cast< archesFunctionPtr<ES1, MS1> >(afp1), taskPtr);
        }
        else if (std::is_same< UintahSpaces::CPU , ES2 >::value) { // Task supports non-Kokkos builds
            helpMe.addFunctionPtr(std::type_index(typeid(UintahSpaces::CPU)),static_cast< archesFunctionPtr<ES2, MS2> >(afp2), taskPtr);
        }
        else if (std::is_same< UintahSpaces::CPU , ES3 >::value) { // Task supports non-Kokkos builds
            helpMe.addFunctionPtr(std::type_index(typeid(UintahSpaces::CPU)),static_cast< archesFunctionPtr<ES3, MS3> >(afp3), taskPtr);
        }
        assignedTag = UINTAH_CPU;
      }
    }
    return assignedTag;
  }

  // The two tag overloaded version of create_portable_arches_tasks()
  template <unsigned int ArchesTaskType 
           , typename ArchesTaskObject
           , typename ES1, typename MS1
           , typename ES2, typename MS2
           >
  TaskAssignedExecutionSpace create_portable_arches_tasks( ArchesTaskObject * taskPtr
                                                         , void ( ArchesTaskObject::*afp1 )( const Patch                     * patch
                                                                                           ,       ArchesTaskInfoManager     * tsk_info_mngr
                                                                                           ,       ExecutionObject<ES1, MS1> & executionObject
                                                                                           )
                                                         , void ( ArchesTaskObject::*afp2 )( const Patch                     * patch
                                                                                           ,       ArchesTaskInfoManager     * tsk_info_mngr
                                                                                           ,       ExecutionObject<ES2, MS2> & executionObject
                                                                                           )
                                                         )
  {
    ArchesSchedulingHelper<ArchesTaskType> helpMe;
    TaskAssignedExecutionSpace assignedTag{};

    // Check for CUDA tasks
    // GPU tasks take top priority
    if (Uintah::Parallel::usingDevice()) {

      if ( std::is_same< Kokkos::Cuda , ES1 >::value || std::is_same< Kokkos::Cuda , ES2 >::value ) {

        if ( std::is_same< Kokkos::Cuda , ES1 >::value ) {        // Task supports Kokkos::Cuda builds
            helpMe.addFunctionPtr(std::type_index(typeid(Kokkos::Cuda)),static_cast< archesFunctionPtr<ES1, MS1> >(afp1), taskPtr);
          }
        }
        else if (std::is_same< Kokkos::Cuda , ES2 >::value) {     // Task supports Kokkos::Cuda builds
            helpMe.addFunctionPtr(std::type_index(typeid(Kokkos::Cuda)),static_cast< archesFunctionPtr<ES2, MS2> >(afp2), taskPtr);
        }
        assignedTag = KOKKOS_CUDA;
      }

    if (assignedTag == TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE) {

      if ( std::is_same< Kokkos::OpenMP , ES1 >::value || std::is_same< Kokkos::OpenMP , ES2 >::value) {

        if (std::is_same< Kokkos::OpenMP , ES1 >::value) {        // Task supports Kokkos::OpenMP builds
            helpMe.addFunctionPtr(std::type_index(typeid(Kokkos::OpenMP)),static_cast< archesFunctionPtr<ES1, MS1> >(afp1), taskPtr);
        }
        else if ( std::is_same< Kokkos::OpenMP , ES2 >::value ) {  // Task supports Kokkos::OpenMP builds
            helpMe.addFunctionPtr(std::type_index(typeid(Kokkos::OpenMP)),static_cast< archesFunctionPtr<ES2, MS2> >(afp2), taskPtr);
        }
        assignedTag = KOKKOS_OPENMP;
      }
      else if ( std::is_same< UintahSpaces::CPU , ES1 >::value || std::is_same< UintahSpaces::CPU , ES2 >::value ) {

        if ( std::is_same< UintahSpaces::CPU , ES1 >::value ) {   // Task supports non-Kokkos builds
            helpMe.addFunctionPtr(std::type_index(typeid(UintahSpaces::CPU)),static_cast< archesFunctionPtr<ES1, MS1> >(afp1), taskPtr);
        }
        else if ( std::is_same< UintahSpaces::CPU , ES2 >::value ) {  // Task supports non-Kokkos builds
            helpMe.addFunctionPtr(std::type_index(typeid(UintahSpaces::CPU)),static_cast< archesFunctionPtr<ES2, MS2> >(afp2), taskPtr);
        }
        assignedTag = UINTAH_CPU;
      }
    }
    return assignedTag;
  }

  // The one tag overloaded version of create_portable_arches_tasks()
  template <unsigned int ArchesTaskType 
           , typename ArchesTaskObject
           , typename ES1, typename MS1
           >
  TaskAssignedExecutionSpace create_portable_arches_tasks( ArchesTaskObject * taskPtr
                                                         , void ( ArchesTaskObject::*afp1 )( const Patch                     * patch
                                                                                           ,       ArchesTaskInfoManager     * tsk_info_mngr
                                                                                           ,       ExecutionObject<ES1, MS1> & executionObject
                                                                                           )
                              )
  {
    ArchesSchedulingHelper<ArchesTaskType> helpMe;
    TaskAssignedExecutionSpace assignedTag{};

    // Check for CUDA tasks
    // GPU tasks take top priority
    if ( Uintah::Parallel::usingDevice() ) {

      if ( std::is_same< Kokkos::Cuda , ES1 >::value) {         // Task supports Kokkos::Cuda builds
        helpMe.addFunctionPtr(std::type_index(typeid(Kokkos::Cuda)),static_cast< archesFunctionPtr<ES1, MS1> >(afp1), taskPtr);
        assignedTag = KOKKOS_CUDA;
      }
    }

    if ( assignedTag == TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE ) {

      if ( std::is_same< Kokkos::OpenMP , ES1 >::value ) {      // Task supports Kokkos::OpenMP builds
        helpMe.addFunctionPtr(std::type_index(typeid(Kokkos::OpenMP)),static_cast< archesFunctionPtr<ES1, MS1> >(afp1), taskPtr);
        assignedTag = KOKKOS_OPENMP;
      }
      else if ( std::is_same< UintahSpaces::CPU , ES1 >::value ) {  // Task supports non-Kokkos builds
        helpMe.addFunctionPtr(std::type_index(typeid(UintahSpaces::CPU)),static_cast< archesFunctionPtr<ES1, MS1> >(afp1), taskPtr);
        assignedTag = UINTAH_CPU;
      }
    }
    return assignedTag;
  }

}

#endif
