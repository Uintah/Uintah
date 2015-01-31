#ifndef Uintah_Component_Arches_TaskInterface_h
#define Uintah_Component_Arches_TaskInterface_h

#include <CCA/Components/Wasatch/Operators/UpwindInterpolant.h>
#include <CCA/Components/Wasatch/Operators/FluxLimiterInterpolant.h>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>

#include <spatialops/structured/FVStaggered.h>
#include <spatialops/structured/MemoryWindow.h>
#include <spatialops/particles/ParticleFieldTypes.h>
#include <spatialops/particles/ParticleOperators.h>
#include <CCA/Components/Arches/Task/FieldContainer.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/LevelP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidValue.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Task.h>
#include <string>
#include <vector>
#include <boost/foreach.hpp>

//===============================================================

/** 
* @class  Task Interface for Arches
* @author Jeremy Thornock
* @date   2014
* 
* @brief Serves as the interface to a standard uintah task. 
* 
**/ 

//===============================================================

namespace Uintah{ 

  class Task; 
  class VarLabel; 
  class Level;  

  class TaskInterface{ 




public: 

    enum VAR_DEPEND { COMPUTES, MODIFIES, REQUIRES, LOCAL_COMPUTES };
    enum WHICH_DW { OLDDW, NEWDW, LATEST };
    enum VAR_TYPE { CC_INT, CC_DOUBLE, CC_VEC, FACEX, FACEY, FACEZ, SUM, MAX, MIN, PARTICLE };

    template <typename FieldT>
    struct VarTypeHelper{
      VarTypeHelper(){ 
        throw InvalidValue("Arches Task Error: Not able to deduce a type.", __FILE__, __LINE__); 
      }
      ~VarTypeHelper(){}
    };


    /** @brief The variable registry information **/ 
    struct VariableInformation { 

      std::string name;
      VAR_TYPE    type; 
      VAR_DEPEND  depend; 
      WHICH_DW    dw;
      int         nGhost;
      bool        dw_inquire; 
      const VarLabel* label; 
      Task::WhichDW uintah_task_dw; 
      Ghost::GhostType ghost_type; 
      bool        local;  
    
    };

    /** @brief Default constructor **/ 
    TaskInterface( std::string take_name, int matl_index ); 

    /** @brief Default destructor **/ 
    virtual ~TaskInterface();

    /** @brief Print task name. **/ 
    void print_task_name(){ 
      std::cout << "Task: " << _task_name << std::endl; 
    }

    /** @brief Input file interface **/ 
    virtual void problemSetup( ProblemSpecP& db ) = 0; 

    /** @brief Create local labels for the task **/ 
    virtual void create_local_labels() = 0; 

    /** @brief Initialization method **/ 
    virtual void register_initialize( std::vector<VariableInformation>& variable_registry ) = 0; 

    /** @brief Schedules work done at the top of a timestep (which might be nothing) **/ 
    virtual void register_timestep_init( std::vector<VariableInformation>& ) = 0; 

    /** @brief Registers all variables with pertinent information for the 
     *         uintah dw interface **/ 
    virtual void register_timestep_eval( std::vector<VariableInformation>& variable_registry, 
                                         const int time_substep ) = 0; 

    /** @brief Register all variables needed to compute boundary conditions **/
    virtual void register_compute_bcs( std::vector<VariableInformation>& variable_registry, const int time_substep ) = 0; 

    /** @brief Matches labels to variables in the registry **/ 
    void resolve_labels( std::vector<VariableInformation>& variable_registry ); 


    /** @brief Add this task to the Uintah task scheduler **/ 
    void schedule_task( const LevelP& level, 
                        SchedulerP& sched, 
                        const MaterialSet* matls,
                        int time_substep );

    /** @brief The actual task interface function that references the 
     *         derived class implementation **/ 
    void do_task( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw, 
                  std::vector<VariableInformation> variable_registry, 
                  int time_substep );

    /** @brief Add this task to the Uintah task scheduler **/ 
    void schedule_init( const LevelP& level, 
                        SchedulerP& sched, 
                        const MaterialSet* matls );

    /** @brief The actual task interface function that references the 
     *         derived class implementation **/ 
    void do_init( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw, 
                  std::vector<VariableInformation> variable_registry );

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
                           std::vector<VariableInformation> variable_registry );

    /** @brief Add this task to the Uintah task scheduler **/ 
    void schedule_bcs( const LevelP& level, 
                        SchedulerP& sched, 
                        const MaterialSet* matls,
                        int time_substep );

    /** @brief The actual task interface function that references the 
     *         derived class implementation **/ 
    void do_bcs( const ProcessorGroup* pc, 
                 const PatchSubset* patches, 
                 const MaterialSubset* matls, 
                 DataWarehouse* old_dw, 
                 DataWarehouse* new_dw, 
                 std::vector<VariableInformation> variable_registry, 
                 int time_substep );

    /** @brief Builder class containing instructions on how to build the task **/ 
    class TaskBuilder { 

      public: 

        TaskBuilder(){}; 

        virtual ~TaskBuilder() {}

        virtual TaskInterface* build() = 0; 

      protected: 

    }; 



protected: 

    /** @brief Inteface to register_variable_work -- this function is overloaded. **/ 
    void register_variable( std::string name, 
                            VAR_TYPE type, 
                            VAR_DEPEND dep, 
                            int nGhost, 
                            WHICH_DW dw, 
                            std::vector<VariableInformation>& var_reg, 
                            const int time_substep );

    /** @brief Inteface to register_variable_work -- this function is overloaded. **/ 
    void register_variable( std::string name, 
                            VAR_TYPE type, 
                            VAR_DEPEND dep, 
                            int nGhost, 
                            WHICH_DW dw, 
                            std::vector<VariableInformation>& var_reg );

    /** @brief Inteface to register_variable_work -- this function is overloaded. 
     *         This version assumes NewDW and zero ghosts. **/ 
    void register_variable( std::string name, 
                            VAR_TYPE type, 
                            VAR_DEPEND dep, 
                            std::vector<VariableInformation>& var_reg );

    /** @brief Inteface to register_variable_work -- this function is overloaded. 
     *         This version assumes NewDW and zero ghosts. **/ 
    void register_variable( std::string name, 
                            VAR_TYPE type, 
                            VAR_DEPEND dep, 
                            std::vector<VariableInformation>& var_reg, 
                            const int timesubstep );

    /** @brief Builds a struct for each variable containing all pertinent uintah
     * DW information **/ 
    void register_variable_work( std::string name, 
                                 VAR_TYPE type, 
                                 VAR_DEPEND dep, 
                                 int nGhost, 
                                 WHICH_DW dw, 
                                 std::vector<VariableInformation>& var_reg, 
                                 const int time_substep );

    /** @brief A container to hold a small amount of other information to 
     *         pass into the task exe. **/ 
    struct SchedToTaskInfo{ 
      int time_substep; 
      double dt; 
    };

    /** @brief Return the enum type of a spatial ops type **/ 
   

    /** @brief Task grid variable storage **/ 
    template <typename T>
    struct VarContainer{ 
      T* variable; 
      constVariableBase<T*> const_variable; 
    };

    typedef std::map<std::string, GridVariableBase* > UintahVarMap; 
    typedef std::map<std::string, constVariableBase<GridVariableBase>* > ConstUintahVarMap; 

    /** @brief A class for managing the retrieval of uintah/so fields during task exe **/ 
    class ArchesTaskInfoManager{

      public: 

        enum MAPCHECK {CHECK_FIELD,CONST_FIELD,NONCONST_FIELD};

        ArchesTaskInfoManager( std::vector<VariableInformation>& var_reg, const Patch* patch, SchedToTaskInfo& info ):
                        _var_reg(var_reg), _patch(patch), _tsk_info(info){

        }; 

        ~ArchesTaskInfoManager(){

        }; 

        /** @brief return the time substep **/ 
        inline int get_time_substep(){ return _tsk_info.time_substep; }; 

        /** @brief return the dt **/ 
        inline double get_dt(){ return _tsk_info.dt; }; 

        /** @brief return the variable registry **/ 
        inline std::vector<VariableInformation>& get_variable_reg(){ return _var_reg; }

        /** @brief Set the references to the variable maps in the Field Collector for easier 
         * management of the fields when trying to retrieve from the DW **/ 
        void set_field_container(ArchesFieldContainer* field_container){

          _field_container = field_container; 
          
        }

        //====================================================================================
        // GRID VARIABLE ACCESS
        //====================================================================================

        /** @brief Return a CONST UINTAH field **/ 
        template <typename T>
        T* get_uintah_const_field( const std::string name ){ 
          return _field_container->get_const_field<T>(name); 
        } 

        /** @brief Return a UINTAH field **/ 
        template <typename T>
        T* get_uintah_field( const std::string name ){ 
          return _field_container->get_field<T>(name); 
        }

        /** @brief Return a SPATIAL field **/ 
        template <typename T>
        SpatialOps::SpatFldPtr<T> get_so_field( const std::string name ){ 
          return _field_container->get_so_field<T>(name); 
        }

        /** @brief Return a CONST SPATIAL field **/ 
        template <typename T>
        SpatialOps::SpatFldPtr<T> get_const_so_field( const std::string name ){ 
          return _field_container->get_const_so_field<T>(name); 
        }

        /** @brief Return a SPATIAL OPS PARTICLE FIELD **/ 
        SpatialOps::SpatFldPtr<ParticleField> get_particle_field( const std::string name ){ 
          return _field_container->get_so_particle_field(name); 
        }

        /** @brief Return a CONST SPATIAL OPS PARTICLE FIELD **/ 
        SpatialOps::SpatFldPtr<ParticleField> get_const_particle_field( const std::string name ){ 
          return _field_container->get_const_so_particle_field(name); 
        }

      private: 

        ArchesFieldContainer* _field_container; 

        std::vector<VariableInformation> _var_reg; 
        const Patch* _patch; 
        SchedToTaskInfo& _tsk_info; 

    }; //End ArchesTaskInfoManager

    /** @brief Resolves the DW fields with the dependency **/ 
    void resolve_fields( DataWarehouse* old_dw, 
                         DataWarehouse* new_dw, 
                         const Patch* patch, 
                         ArchesFieldContainer* field_container, 
                         ArchesTaskInfoManager* f_collector, 
                         const bool doing_initialization );

    /** @brief The actual work done within the derived class **/ 
    virtual void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr, 
                             SpatialOps::OperatorDatabase& opr ) = 0; 

    /** @brief Work done at the top of a timestep **/ 
    virtual void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr, 
                                SpatialOps::OperatorDatabase& opr ) = 0; 

    /** @brief The actual work done within the derived class **/ 
    virtual void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr, 
                       SpatialOps::OperatorDatabase& opr ) = 0; 

    /** @brief The actual work done within the derived class for computing the boundary conditions **/ 
    virtual void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info_mngr, 
                              SpatialOps::OperatorDatabase& opr ) = 0; 

    std::string _task_name; 
    const int _matl_index; 
    std::vector<const VarLabel*> _local_labels;

    /** @brief Get the Uintah typeDescription for a TaskType **/ 
    inline const TypeDescription* get_TD(VAR_TYPE type){ 

      if ( type == CC_DOUBLE ){ 
        return CCVariable<double>::getTypeDescription(); 
      } else if ( type == CC_INT ){ 
        return CCVariable<int>::getTypeDescription(); 
      } else if ( type == CC_VEC ){ 
        return CCVariable<Vector>::getTypeDescription(); 
      } else if ( type == FACEX ){ 
        return SFCXVariable<double>::getTypeDescription(); 
      } else if ( type == FACEY ){ 
        return SFCYVariable<double>::getTypeDescription(); 
      } else if ( type == FACEZ ){ 
        return SFCZVariable<double>::getTypeDescription(); 
      } else if ( type == PARTICLE ){ 
        return ParticleVariable<double>::getTypeDescription();
      } else { 
        throw InvalidValue("Error: Variable type not recognized.",__FILE__,__LINE__); 
      }

    }

    /** @brief Register a local varlabel for this task **/ 
    void register_new_variable(const std::string name, VAR_TYPE type ){ 

      const VarLabel* test = NULL; 
      test = VarLabel::find( name );

      if ( test == NULL ){

        const VarLabel* label = VarLabel::create( name, get_TD(type) );
        _local_labels.push_back(label); 

      } else { 

        std::stringstream msg; 
        msg << "Error: Varlabel already registered: " << name << " (name your task variable something else and try again)." << std::endl;
        throw InvalidValue(msg.str(), __FILE__, __LINE__); 

      }
    }


private: 

    /** @brief Performs all DW get*,allocateAndPut, etc.. for all variables for this 
     *         task. **/
    template<class T>
    void resolve_field_modifycompute( DataWarehouse* old_dw, DataWarehouse* new_dw, T* field, 
                                      VariableInformation& info, const Patch* patch, const int time_substep );

    /** @brief Performs all DW get*,allocateAndPut, etc.. for all variables for this 
     *         task. **/
    template<class T>
    void resolve_field_requires( DataWarehouse* old_dw, DataWarehouse* new_dw, 
                                 T* field, VariableInformation& info, 
                                 const Patch* patch, const int time_substep );


  
  };

  template <>
  struct
  TaskInterface::VarTypeHelper<SpatialOps::SVolField>{ 
    VarTypeHelper(){ _var_type = TaskInterface::CC_DOUBLE; }
    TaskInterface::VAR_TYPE get_vartype(){ return _var_type; }
    ~VarTypeHelper(){}

    private: 
    TaskInterface::VAR_TYPE _var_type; 
  };
  template <>
  struct
  TaskInterface::VarTypeHelper<CCVariable<double> >{ 
    VarTypeHelper(){ _var_type = TaskInterface::CC_DOUBLE; }
    TaskInterface::VAR_TYPE get_vartype(){ return _var_type; }
    ~VarTypeHelper(){}

    private: 
    TaskInterface::VAR_TYPE _var_type; 
  };
  template <>
  struct
  TaskInterface::VarTypeHelper<SpatialOps::XVolField >{ 
    VarTypeHelper(){ _var_type = TaskInterface::FACEX; }
    TaskInterface::VAR_TYPE get_vartype(){ return _var_type; }
    ~VarTypeHelper(){}

    private: 
    TaskInterface::VAR_TYPE _var_type; 
  };
  template <>
  struct
  TaskInterface::VarTypeHelper<SpatialOps::YVolField >{ 
    VarTypeHelper(){ _var_type = TaskInterface::FACEY; }
    TaskInterface::VAR_TYPE get_vartype(){ return _var_type; }
    ~VarTypeHelper(){}

    private: 
    TaskInterface::VAR_TYPE _var_type; 
  };
  template <>
  struct
  TaskInterface::VarTypeHelper<SpatialOps::ZVolField >{ 
    VarTypeHelper(){ _var_type = TaskInterface::FACEZ; }
    TaskInterface::VAR_TYPE get_vartype(){ return _var_type; }
    ~VarTypeHelper(){}

    private: 
    TaskInterface::VAR_TYPE _var_type; 
  };
  template <>
  struct
  TaskInterface::VarTypeHelper<ParticleField>{ 
    VarTypeHelper(){ _var_type = TaskInterface::PARTICLE; }
    TaskInterface::VAR_TYPE get_vartype(){ return _var_type; }
    ~VarTypeHelper(){}

    private: 
    TaskInterface::VAR_TYPE _var_type; 
  };

}

#endif 
