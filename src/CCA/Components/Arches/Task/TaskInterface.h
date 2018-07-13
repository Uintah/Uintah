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

  using evalFunctionPtr  = void (TaskInterface::*)( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr, ExecutionObject& executionObject );

// This is the older mechanism for trying to manage the boilerplate of Arches tasks.
// Leaving it in as it may be useful one day.
//#define SUPPORTED_UINTAH__TRYING_UINTAH_EXECUTION                 UintahSpaces::CPU
//#define SUPPORTED_UINTAH__TRYING_UINTAH_MEMORY                    UintahSpaces::HostSpace
//#define SUPPORTED_UINTAH__TRYING_UINTAH_TAG                       TaskAssignedExecutionSpace::UINTAH_CPU
//
////Note that we don't support both OPENMP and CPU in the same build.
//#if defined(UINTAH_ENABLE_KOKKOS) && defined(HAVE_CUDA)
//  #define SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_CUDA_EXECUTION     Kokkos::Cuda
//  #define SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_CUDA_MEMORY        Kokkos::CudaSpace
//  #define SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_CUDA_TAG           TaskAssignedExecutionSpace::KOKKOS_CUDA
//
//  #define SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_OPENMP_EXECUTION   Kokkos::OpenMP
//  #define SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_OPENMP_MEMORY      Kokkos::HostSpace
//  #define SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_OPENMP_TAG         TaskAssignedExecutionSpace::KOKKOS_OPENMP
//
//  #define SUPPORTED_UINTAH_OPENMP__TRYING_OPENMP_EXECUTION        Kokkos::OpenMP
//  #define SUPPORTED_UINTAH_OPENMP__TRYING_OPENMP_MEMORY           Kokkos::HostSpace
//  #define SUPPORTED_UINTAH_OPENMP__TRYING_OPENMP_TAG              TaskAssignedExecutionSpace::KOKKOS_OPENMP
//#elif defined(UINTAH_ENABLE_KOKKOS) && !defined(HAVE_CUDA)
//  #define SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_CUDA_EXECUTION     Kokkos::OpenMP
//  #define SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_CUDA_MEMORY        Kokkos::HostSpace
//  #define SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_CUDA_TAG           TaskAssignedExecutionSpace::KOKKOS_OPENMP
//
//  #define SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_OPENMP_EXECUTION   Kokkos::OpenMP
//  #define SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_OPENMP_MEMORY      Kokkos::HostSpace
//  #define SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_OPENMP_TAG         TaskAssignedExecutionSpace::KOKKOS_OPENMP
//
//  #define SUPPORTED_UINTAH_OPENMP__TRYING_OPENMP_EXECUTION        Kokkos::OpenMP
//  #define SUPPORTED_UINTAH_OPENMP__TRYING_OPENMP_MEMORY           Kokkos::HostSpace
//  #define SUPPORTED_UINTAH_OPENMP__TRYING_OPENMP_TAG              TaskAssignedExecutionSpace::KOKKOS_OPENMP
//#elif !defined(UINTAH_ENABLE_KOKKOS)
//  #define SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_CUDA_EXECUTION     UintahSpaces::CPU
//  #define SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_CUDA_MEMORY        UintahSpaces::HostSpace
//  #define SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_CUDA_TAG           TaskAssignedExecutionSpace::UINTAH_CPU
//
//  #define SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_OPENMP_EXECUTION   UintahSpaces::CPU
//  #define SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_OPENMP_MEMORY      UintahSpaces::HostSpace
//  #define SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_OPENMP_TAG         TaskAssignedExecutionSpace::UINTAH_CPU
//
//  #define SUPPORTED_UINTAH_OPENMP__TRYING_OPENMP_EXECUTION        UintahSpaces::CPU
//  #define SUPPORTED_UINTAH_OPENMP__TRYING_OPENMP_MEMORY           UintahSpaces::HostSpace
//  #define SUPPORTED_UINTAH_OPENMP__TRYING_OPENMP_TAG              TaskAssignedExecutionSpace::UINTAH_CPU
//#endif
//  // When uncommenting, put back in the \ characters needed for macros
//#define LOAD_ARCHES_UINTAH_OPENMP_CUDA(ASSIGNED_TAG, FUNCTION_CODE_NAME) {
//  if (Uintah::Parallel::usingDevice()) {
//    this->addEvalFunctionPtr(std::type_index(
//                             typeid(SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_CUDA_EXECUTION)),
//      static_cast<evalFunctionPtr>(&FUNCTION_CODE_NAME<
//                                   SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_CUDA_EXECUTION,
//                                   SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_CUDA_MEMORY>));
//    ASSIGNED_TAG = SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_CUDA_TAG;
//  }
//  if (ASSIGNED_TAG == TaskAssignedExecutionSpace::NONE_SPACE) {
//    this->addEvalFunctionPtr(std::type_index(
//                             typeid(SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_OPENMP_EXECUTION)),
//      static_cast<evalFunctionPtr>(&FUNCTION_CODE_NAME<
//                                   SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_OPENMP_EXECUTION,
//                                   SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_OPENMP_MEMORY>));
//    ASSIGNED_TAG = SUPPORTED_UINTAH_OPENMP_CUDA__TRYING_OPENMP_TAG;
//  }
//}
//
////User specified that CUDA is not an option.
////In this mode, we don't allow the regular CPU version to compile.
//#define LOAD_ARCHES_UINTAH_OPENMP(ASSIGNED_TAG, FUNCTION_CODE_NAME) {
//  this->addEvalFunctionPtr(std::type_index(
//                           typeid(SUPPORTED_UINTAH_OPENMP__TRYING_OPENMP_EXECUTION)),
//    static_cast<evalFunctionPtr>(&FUNCTION_CODE_NAME<
//                                 SUPPORTED_UINTAH_OPENMP__TRYING_OPENMP_EXECUTION,
//                                 SUPPORTED_UINTAH_OPENMP__TRYING_OPENMP_MEMORY>));
//  ASSIGNED_TAG = SUPPORTED_UINTAH_OPENMP__TRYING_OPENMP_TAG;
//}
//
////User specified that CUDA or OpenMP is not an option.
////In this most only the CPU version can compile
//#define LOAD_ARCHES_UINTAH(ASSIGNED_TAG, FUNCTION_CODE_NAME) {
//    this->addEvalFunctionPtr(std::type_index(
//                             typeid(SUPPORTED_UINTAH__TRYING_UINTAH_EXECUTION)),
//      static_cast<evalFunctionPtr>(&FUNCTION_CODE_NAME<
//                                   SUPPORTED_UINTAH__TRYING_UINTAH_EXECUTION,
//                                   SUPPORTED_UINTAH__TRYING_UINTAH_MEMORY>));
//    ASSIGNED_TAG = SUPPORTED_UINTAH__TRYING_UINTAH_TAG;
//}



  //See Core/Parallel/LoopExecution.h for the purpose behind these macros, as that file uses a similar pattern
#define LOAD_ARCHES_EVAL_TASK_3TAGS(TAG1, TAG2, TAG3, ASSIGNED_TAG, FUNCTION_CODE_NAME) {           \
                                                                                                    \
  if (Uintah::Parallel::usingDevice()) {                                                            \
    if        (strcmp(STRVX(ORIGINAL_KOKKOS_CUDA_TAG), STRVX(TAG1)) == 0) {                         \
      this->addEvalFunctionPtr(std::type_index(typeid(Kokkos::Cuda)),                               \
            static_cast<evalFunctionPtr>(&FUNCTION_CODE_NAME<TAG1>));                               \
      ASSIGNED_TAG = KOKKOS_CUDA;                                                                   \
    } else if (strcmp(STRVX(ORIGINAL_KOKKOS_CUDA_TAG), STRVX(TAG2)) == 0) {                         \
      this->addEvalFunctionPtr(std::type_index(typeid(Kokkos::Cuda)),                               \
            static_cast<evalFunctionPtr>(&FUNCTION_CODE_NAME<TAG2>));                               \
      ASSIGNED_TAG = KOKKOS_CUDA;                                                                   \
    } else if (strcmp(STRVX(ORIGINAL_KOKKOS_CUDA_TAG), STRVX(TAG3)) == 0) {                         \
      this->addEvalFunctionPtr(std::type_index(typeid(Kokkos::Cuda)),                               \
            static_cast<evalFunctionPtr>(&FUNCTION_CODE_NAME<TAG3>));                               \
      ASSIGNED_TAG = KOKKOS_CUDA;                                                                   \
    }                                                                                               \
  }                                                                                                 \
                                                                                                    \
  if (ASSIGNED_TAG == TaskAssignedExecutionSpace::NONE_SPACE) {                                     \
    if        (strcmp(STRVX(ORIGINAL_KOKKOS_OPENMP_TAG), STRVX(TAG1)) == 0) {                       \
        this->addEvalFunctionPtr(std::type_index(typeid(Kokkos::OpenMP)),                           \
              static_cast<evalFunctionPtr>(&FUNCTION_CODE_NAME<TAG1>));                             \
        ASSIGNED_TAG = KOKKOS_OPENMP;                                                               \
    } else if (strcmp(STRVX(ORIGINAL_KOKKOS_OPENMP_TAG), STRVX(TAG2)) == 0) {                       \
        this->addEvalFunctionPtr(std::type_index(typeid(Kokkos::OpenMP)),                           \
              static_cast<evalFunctionPtr>(&FUNCTION_CODE_NAME<TAG2>));                             \
        ASSIGNED_TAG = KOKKOS_OPENMP;                                                               \
    } else if (strcmp(STRVX(ORIGINAL_KOKKOS_OPENMP_TAG), STRVX(TAG3)) == 0) {                       \
        this->addEvalFunctionPtr(std::type_index(typeid(Kokkos::OpenMP)),                           \
              static_cast<evalFunctionPtr>(&FUNCTION_CODE_NAME<TAG3>));                             \
        ASSIGNED_TAG = KOKKOS_OPENMP;                                                               \
    } else if (strcmp(STRVX(ORIGINAL_UINTAH_CPU_TAG), STRVX(TAG1)) == 0) {                          \
        this->addEvalFunctionPtr(std::type_index(typeid(UintahSpaces::CPU)),                        \
              static_cast<evalFunctionPtr>(&FUNCTION_CODE_NAME<TAG1>));                             \
        ASSIGNED_TAG = UINTAH_CPU;                                                                  \
    } else if (strcmp(STRVX(ORIGINAL_UINTAH_CPU_TAG), STRVX(TAG2)) == 0) {                          \
        this->addEvalFunctionPtr(std::type_index(typeid(UintahSpaces::CPU)),                        \
              static_cast<evalFunctionPtr>(&FUNCTION_CODE_NAME<TAG2>));                             \
        ASSIGNED_TAG = UINTAH_CPU;                                                                  \
    } else if (strcmp(STRVX(ORIGINAL_UINTAH_CPU_TAG), STRVX(TAG3)) == 0) {                          \
        this->addEvalFunctionPtr(std::type_index(typeid(UintahSpaces::CPU)),                        \
              static_cast<evalFunctionPtr>(&FUNCTION_CODE_NAME<TAG3>));                             \
        ASSIGNED_TAG = UINTAH_CPU;                                                                  \
    }                                                                                               \
  }                                                                                                 \
}

//If only 2 execution space tags are specified
#define LOAD_ARCHES_EVAL_TASK_2TAGS(TAG1, TAG2, ASSIGNED_TAG, FUNCTION_CODE_NAME) {                 \
                                                                                                    \
  if (Uintah::Parallel::usingDevice()) {                                                            \
    if        (strcmp(STRVX(ORIGINAL_KOKKOS_CUDA_TAG), STRVX(TAG1)) == 0) {                         \
      this->addEvalFunctionPtr(std::type_index(typeid(Kokkos::Cuda)),                               \
            static_cast<evalFunctionPtr>(&FUNCTION_CODE_NAME<TAG1>));                               \
      ASSIGNED_TAG = KOKKOS_CUDA;                                                                   \
    } else if (strcmp(STRVX(ORIGINAL_KOKKOS_CUDA_TAG), STRVX(TAG2)) == 0) {                         \
      this->addEvalFunctionPtr(std::type_index(typeid(Kokkos::Cuda)),                               \
            static_cast<evalFunctionPtr>(&FUNCTION_CODE_NAME<TAG2>));                               \
      ASSIGNED_TAG = KOKKOS_CUDA;                                                                   \
    }                                                                                               \
  }                                                                                                 \
                                                                                                    \
  if (ASSIGNED_TAG == TaskAssignedExecutionSpace::NONE_SPACE) {                                     \
    if        (strcmp(STRVX(ORIGINAL_KOKKOS_OPENMP_TAG), STRVX(TAG1)) == 0) {                       \
        this->addEvalFunctionPtr(std::type_index(typeid(Kokkos::OpenMP)),                           \
              static_cast<evalFunctionPtr>(&FUNCTION_CODE_NAME<TAG1>));                             \
        ASSIGNED_TAG = KOKKOS_OPENMP;                                                               \
    } else if (strcmp(STRVX(ORIGINAL_KOKKOS_OPENMP_TAG), STRVX(TAG2)) == 0) {                       \
        this->addEvalFunctionPtr(std::type_index(typeid(Kokkos::OpenMP)),                           \
              static_cast<evalFunctionPtr>(&FUNCTION_CODE_NAME<TAG2>));                             \
        ASSIGNED_TAG = KOKKOS_OPENMP;                                                               \
    } else if (strcmp(STRVX(ORIGINAL_UINTAH_CPU_TAG), STRVX(TAG1)) == 0) {                          \
        this->addEvalFunctionPtr(std::type_index(typeid(UintahSpaces::CPU)),                        \
              static_cast<evalFunctionPtr>(&FUNCTION_CODE_NAME<TAG1>));                             \
        ASSIGNED_TAG = UINTAH_CPU;                                                                  \
    } else if (strcmp(STRVX(ORIGINAL_UINTAH_CPU_TAG), STRVX(TAG2)) == 0) {                          \
        this->addEvalFunctionPtr(std::type_index(typeid(UintahSpaces::CPU)),                        \
              static_cast<evalFunctionPtr>(&FUNCTION_CODE_NAME<TAG2>));                             \
        ASSIGNED_TAG = UINTAH_CPU;                                                                  \
    }                                                                                               \
  }                                                                                                 \
}


//If only 1 execution space tag is specified
#define LOAD_ARCHES_EVAL_TASK_1TAG(TAG1, ASSIGNED_TAG, FUNCTION_CODE_NAME) {                        \
                                                                                                    \
  if (Uintah::Parallel::usingDevice()) {                                                            \
    if        (strcmp(STRVX(ORIGINAL_KOKKOS_CUDA_TAG), STRVX(TAG1)) == 0) {                         \
      this->addEvalFunctionPtr(std::type_index(typeid(Kokkos::Cuda)),                               \
            static_cast<evalFunctionPtr>(&FUNCTION_CODE_NAME<TAG1>));                               \
      ASSIGNED_TAG = KOKKOS_CUDA;                                                                   \
    }                                                                                               \
  }                                                                                                 \
                                                                                                    \
  if (ASSIGNED_TAG == TaskAssignedExecutionSpace::NONE_SPACE) {                                     \
    if        (strcmp(STRVX(ORIGINAL_KOKKOS_OPENMP_TAG), STRVX(TAG1)) == 0) {                       \
        this->addEvalFunctionPtr(std::type_index(typeid(Kokkos::OpenMP)),                           \
              static_cast<evalFunctionPtr>(&FUNCTION_CODE_NAME<TAG1>));                             \
        ASSIGNED_TAG = KOKKOS_OPENMP;                                                               \
    } else if (strcmp(STRVX(ORIGINAL_UINTAH_CPU_TAG), STRVX(TAG1)) == 0) {                          \
       this->addEvalFunctionPtr(std::type_index(typeid(UintahSpaces::CPU)),                         \
              static_cast<evalFunctionPtr>(&FUNCTION_CODE_NAME<TAG1>));                             \
        ASSIGNED_TAG = UINTAH_CPU;                                                                  \
    }                                                                                               \
  }                                                                                                 \
}

  class TaskInterface{

public:

    typedef ArchesFieldContainer AFC;

    enum TASK_TYPE { INITIALIZE, TIMESTEP_INITIALIZE, TIMESTEP_EVAL, BC,
                     RESTART_INITIALIZE, ATOMIC };

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
      } else if ( type == ATOMIC ){ 
        return "Atomic Task"; 
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

    //------begin portability support members--------------------------------------------------

    /** @brief Tells TaskFactoryBase which execution space this Arches task was assigned to.**/
    virtual TaskAssignedExecutionSpace loadTaskEvalFunctionPointers() = 0;

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
    template<typename ExecutionSpace, typename MemorySpace>
    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr, ExecutionObject& executionObject ) {
      evalFunctionPtr handler_ptr{nullptr};
      auto index = std::type_index(typeid(ExecutionSpace));
      auto handler = this->evalFunctionPtrs.find(index);
      if(handler != this->evalFunctionPtrs.end()) {
        handler_ptr = handler->second;
      } else {
        throw InternalError("Derived class version of Arches task eval() not found!", __FILE__, __LINE__);
      }

      // Found the eval() function pointer associated with the execution space.  Run it.
      if (handler_ptr) {
        (this->*handler_ptr)( patch, tsk_info_mngr, executionObject );
      }

    }
protected:

    void addEvalFunctionPtr( std::type_index ti, evalFunctionPtr ep ) {
      evalFunctionPtrs.emplace( ti, ep );
    }
private:
    std::map<std::type_index, evalFunctionPtr> evalFunctionPtrs;

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

    /** @brief The actual work done within the derived class **/
    virtual void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr , ExecutionObject& executionObject ) = 0;

    /** @brief The actual work done within the derived class **/
    virtual void restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr ){}

    /** @brief Work done at the top of a timestep **/
    virtual void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr ) = 0;

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
}

#endif
