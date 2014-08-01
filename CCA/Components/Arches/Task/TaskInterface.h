#ifndef Uintah_Component_Arches_TaskInterface_h
#define Uintah_Component_Arches_TaskInterface_h

#include <spatialops/structured/FVStaggered.h>
#include <spatialops/structured/MemoryWindow.h>

#include <CCA/Components/Arches/Operators/Operators.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
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
    enum VAR_TYPE { CC_INT, CC_DOUBLE, CC_VEC, FACEX, FACEY, FACEZ, SUM, MAX, MIN };

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

    /** @brief Get the type for templated tasks **/ 
    template <class T> 
    void set_type(){

      if ( typeid(T) == typeid(CCVariable<double>) ){
        _mytype = CC_DOUBLE; 
      } else if ( typeid(T) == typeid(SFCXVariable<double>)){ 
        _mytype = FACEX; 
      } else if ( typeid(T) == typeid(SFCYVariable<double>)){ 
        _mytype = FACEY; 
      } else if ( typeid(T) == typeid(SFCZVariable<double>)){ 
        _mytype = FACEZ; 
      } else if ( typeid(T) == typeid(SpatialOps::SVolField)){ 
        _mytype = CC_DOUBLE; 
      } else if ( typeid(T) == typeid(SpatialOps::SSurfXField)){ 
        _mytype = FACEX; 
      } else if ( typeid(T) == typeid(SpatialOps::SSurfYField)){ 
        _mytype = FACEY; 
      } else if ( typeid(T) == typeid(SpatialOps::SSurfZField)){ 
        _mytype = FACEZ; 
      }

    }


    /** @brief Input file interface **/ 
    virtual void problemSetup( ProblemSpecP& db ) = 0; 

    /** @brief Initialization method **/ 
    virtual void register_initialize( std::vector<VariableInformation>& variable_registry ) = 0; 

    /** @brief Schedules work done at the top of a timestep (which might be nothing) **/ 
    virtual void register_timestep_init( std::vector<VariableInformation>& ) = 0; 

    /** @brief Registers all variables with pertinent information for the 
     *         uintah dw interface **/ 
    virtual void register_timestep_eval( std::vector<VariableInformation>& variable_registry, 
                                         const int time_substep ) = 0; 

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
   

    /** @brief Task grid variable storage **/ 
    template <typename T>
    struct VarContainer{ 
      T* variable; 
      constVariableBase<T*> const_variable; 
    };

    typedef std::map<std::string, GridVariableBase* > UintahVarMap; 
    typedef std::map<std::string, constVariableBase<GridVariableBase>* > ConstUintahVarMap; 

    /** @brief A class for managing the retrieval of uintah/so fields during task exe **/ 
    class FieldCollector{

      public: 

        enum MAPCHECK {CHECK_FIELD,CONST_FIELD,NONCONST_FIELD};

        FieldCollector( std::vector<VariableInformation>& var_reg, const Patch* patch, SchedToTaskInfo& info):
                        _var_reg(var_reg), _patch(patch), _tsk_info(info){

          _variable_map.clear();
          _const_variable_map.clear(); 
        
        }; 

        ~FieldCollector(){
        
          //clean up 
          //for ( UintahVarMap::iterator i = _variable_map.begin(); i != _variable_map.end(); i++ ){
          //  delete i->second; 
          //}
          //for ( ConstUintahVarMap::iterator i = _const_variable_map.begin(); i != _const_variable_map.end(); i++ ){
          //  delete i->second; 
          //}

          for ( std::vector<SpatialOps::SVolField*>::iterator i = _cc_fields.begin(); i != _cc_fields.end(); i++){ 
            delete *i; 
          }
          for ( std::vector<SpatialOps::SSurfXField*>::iterator i = _fx_fields.begin(); i != _fx_fields.end(); i++){ 
            delete *i; 
          }
          for ( std::vector<SpatialOps::SSurfYField*>::iterator i = _fy_fields.begin(); i != _fy_fields.end(); i++){ 
            delete *i; 
          }
          for ( std::vector<SpatialOps::SSurfZField*>::iterator i = _fz_fields.begin(); i != _fz_fields.end(); i++){ 
            delete *i; 
          }

        }; 

        /** @brief return the time substep **/ 
        int get_time_substep(){ return _tsk_info.time_substep; }; 

        /** @brief return the dt **/ 
        double get_dt(){ return _tsk_info.dt; }; 

        /** @brief return the variable registry **/ 
        std::vector<VariableInformation>& get_variable_reg(){ return _var_reg; }

        /** @brief Set the references to the variable maps in the Field Collector for easier 
         * management of the fields when trying to retrieve from the DW **/ 
        void set_var_maps(UintahVarMap& var_map, ConstUintahVarMap const_var_map){
          _variable_map = var_map; 
          _const_variable_map = const_var_map;
        }

        //Shamelessly stolen from Wasatch: 
        /**
         * \fn void get_bc_logicals( const Uintah::Patch* const, SpatialOps::IntVec&, SpatialOps::IntVec& );
         * \brief Given the patch, populate information about whether a physical
         *        boundary exists on each side of the patch.
         * \param patch   - the patch of interest
         * \param bcMinus - assigned to 0 if no BC present on (-) faces, 1 if present
         * \param bcPlus  - assigned to 0 if no BC present on (+) faces, 1 if present
         */
        static void get_bc_logicals( const Uintah::Patch* const patch,
                              SpatialOps::IntVec& bcMinus,
                              SpatialOps::IntVec& bcPlus )
        {
          for( int i=0; i<3; ++i ){
            bcMinus[i] = 1;
            bcPlus [i] = 1;
          }
          std::vector<Uintah::Patch::FaceType> faces;
          patch->getNeighborFaces(faces);
          for( std::vector<Uintah::Patch::FaceType>::const_iterator i=faces.begin(); i!=faces.end(); ++i ){
            SCIRun::IntVector dir = patch->getFaceDirection(*i);
            for( int j=0; j<3; ++j ){
              if( dir[j] == -1 ) bcMinus[j]=0;
              if( dir[j] ==  1 ) bcPlus [j]=0;
            }
          }
        }

        /** @brief wrap a uintah field as a spatialops field. **/ 
        template< typename FieldT, typename UFT >
        static inline FieldT* wrap_uintah_field_as_spatialops( UFT& uintahVar,
                                                        const Uintah::Patch* const patch,
                                                        const int nGhost )
    //                                                    const SpatialOps::MemoryType mtype=SpatialOps::LOCAL_RAM,
    //                                                    const unsigned short int deviceIndex=0,
    //                                                    double* uintahDeviceVar = NULL )
        {
    
          namespace SS = SpatialOps;
    
          using SCIRun::IntVector;
    
          const SCIRun::IntVector lowIx       = uintahVar->getLowIndex();
          const SCIRun::IntVector highIx      = uintahVar->getHighIndex();
          const SCIRun::IntVector fieldSize   = uintahVar->getWindow()->getData()->size();
          const SCIRun::IntVector fieldOffset = uintahVar->getWindow()->getOffset();
          const SCIRun::IntVector fieldExtent = highIx - lowIx;
    
          const SS::IntVec   size( fieldSize[0],   fieldSize[1],   fieldSize[2]   );
          const SS::IntVec extent( fieldExtent[0], fieldExtent[1], fieldExtent[2] );
          const SS::IntVec offset( lowIx[0]-fieldOffset[0], lowIx[1]-fieldOffset[1], lowIx[2]-fieldOffset[2] );
    
          SS::IntVec bcMinus, bcPlus;
          get_bc_logicals( patch, bcMinus, bcPlus );
    
          const unsigned short int deviceIndex = CPU_INDEX; 
    
          double* fieldValues_ = NULL;
    //      if( mtype == SpatialOps::EXTERNAL_CUDA_GPU ){
    //#       ifdef HAVE_CUDA
    //        fieldValues_ = const_cast<double*>( uintahDeviceVar );
    //#       endif
    //      }
    //      else{
            fieldValues_ = const_cast<typename FieldT::value_type*>( uintahVar->getPointer() );
    //      }
    
          return new FieldT( SS::MemoryWindow( size, offset, extent ),
                             SS::BoundaryCellInfo::build<FieldT>(bcPlus),
                             nGhost,
                             fieldValues_,
                             SS::ExternalStorage,
                             deviceIndex );
        }

        /** @brief Add a (non-const) variable to the list **/ 
        void add_variable(std::string name, GridVariableBase* var){

          _variable_map.insert(std::make_pair(name, var)); 
        
        }

        /** @brief Add a constant variable to the list **/ 
        void add_constant_variable(std::string name, constVariableBase<GridVariableBase>* var){

          _const_variable_map.insert(std::make_pair(name, var)); 
        
        }

        /** @brief Struct for template function specialization to return the SO field **/ 
        template<class ST, class M>
        struct SO{ 
          ST* get_so_grid_var( std::string name, M& var_map, const Patch* patch, const int nGhost ){
            throw InvalidValue("Arches Task Error: (SPATIAL OPS) Cannot resolve grid variable "+name, __FILE__, __LINE__); 
          };
        };

        template<class M>
        struct SO<SpatialOps::SVolField,M>{
          SpatialOps::SVolField* get_so_grid_var( std::string name, M& var_map, const Patch* patch, const int nGhost ){
            std::map<std::string, constVariableBase<GridVariableBase>* > test_const;
            bool do_const = false; 
            if (typeid(var_map) == typeid(test_const)){ 
              do_const = true; 
            }
            if ( var_map.find(name) != var_map.end() ) {

              if ( do_const ){ 
                constCCVariable<double>* var = dynamic_cast<constCCVariable<double>* >(var_map.find(name)->second); 
                return wrap_uintah_field_as_spatialops<SpatialOps::SVolField>( var, patch, nGhost );
              } else { 
                CCVariable<double>* var = dynamic_cast<CCVariable<double>* >(var_map.find(name)->second); 
                return wrap_uintah_field_as_spatialops<SpatialOps::SVolField>( var, patch, nGhost );
              }

            }
            std::ostringstream msg; 
            msg << " Arches Task Error: Cannot resolve grid variable: "<< name << "\n" << "(try checking var_map)" << std::endl;
            throw InvalidValue(msg.str(), __FILE__, __LINE__); 
          };
        };
        template<class M>
        struct SO<SpatialOps::SSurfXField,M>{
          SpatialOps::SSurfXField* get_so_grid_var( std::string name, M& var_map, const Patch* patch, const int nGhost ){
            std::map<std::string, constVariableBase<GridVariableBase>* > test_const;
            bool do_const = false; 
            if (typeid(var_map) == typeid(test_const)){ 
              do_const = true; 
            }
            if ( var_map.find(name) != var_map.end() ) {

              if ( do_const ){ 
                constSFCXVariable<double>* var = dynamic_cast<constSFCXVariable<double>* >(var_map.find(name)->second); 
                return wrap_uintah_field_as_spatialops<SpatialOps::SSurfXField>( var, patch, nGhost );
              } else { 
                SFCXVariable<double>* var = dynamic_cast<SFCXVariable<double>* >(var_map.find(name)->second); 
                return wrap_uintah_field_as_spatialops<SpatialOps::SSurfXField>( var, patch, nGhost );
              }

            }

            throw InvalidValue("Arches Task Error: (SPATIAL OPS) Cannot resolve grid variable "+name, __FILE__, __LINE__); 

          }
        };
        template<class M>
        struct SO<SpatialOps::SSurfYField,M>{
          SpatialOps::SSurfYField* get_so_grid_var( std::string name, M& var_map, const Patch* patch, const int nGhost ){
            std::map<std::string, constVariableBase<GridVariableBase>* > test_const;
            bool do_const = false; 
            if (typeid(var_map) == typeid(test_const)){ 
              do_const = true; 
            }
            if ( var_map.find(name) != var_map.end() ) {

              if ( do_const ){ 
                constSFCYVariable<double>* var = dynamic_cast<constSFCYVariable<double>* >(var_map.find(name)->second); 
                return wrap_uintah_field_as_spatialops<SpatialOps::SSurfYField>( var, patch, nGhost );
              } else { 
                SFCYVariable<double>* var = dynamic_cast<SFCYVariable<double>* >(var_map.find(name)->second); 
                return wrap_uintah_field_as_spatialops<SpatialOps::SSurfYField>( var, patch, nGhost );
              }

            }

            throw InvalidValue("Arches Task Error: (SPATIAL OPS) Cannot resolve grid variable "+name, __FILE__, __LINE__); 

          }
        };
        template<class M>
        struct SO<SpatialOps::SSurfZField,M>{
          SpatialOps::SSurfZField* get_so_grid_var( std::string name, M& var_map, const Patch* patch, const int nGhost ){
            std::map<std::string, constVariableBase<GridVariableBase>* > test_const;
            bool do_const = false; 
            if (typeid(var_map) == typeid(test_const)){ 
              do_const = true; 
            }
            if ( var_map.find(name) != var_map.end() ) {

              if ( do_const ){ 
                constSFCZVariable<double>* var = dynamic_cast<constSFCZVariable<double>* >(var_map.find(name)->second); 
                return wrap_uintah_field_as_spatialops<SpatialOps::SSurfZField>( var, patch, nGhost );
              } else { 
                SFCZVariable<double>* var = dynamic_cast<SFCZVariable<double>* >(var_map.find(name)->second); 
                return wrap_uintah_field_as_spatialops<SpatialOps::SSurfZField>( var, patch, nGhost );
              }

            }

            throw InvalidValue("Arches Task Error: (SPATIAL OPS) Cannot resolve grid variable "+name, __FILE__, __LINE__); 

          }
        };


        /** @brief Get an SpatialOps field to work with **/ 
        template<class ST, class M>
        ST* retrieve_so_field(const std::string name, M& var_map, const Patch* patch, const int nGhost ){ 
          SO<ST,M> func; 
          ST* field = func.get_so_grid_var(name, var_map, patch, nGhost ); 
          return field; 
        }

        /** @brief Get a Uintah field to work with **/ 
        template<class T, class M>
        inline T* get_uintah_grid_var(const std::string name, M& var_map ){

          if ( var_map.find(name) != var_map.end() ) return dynamic_cast<T* >(var_map.find(name)->second); 

          throw InvalidValue("Arches Task Error: (UINTAH) Cannot resolve grid variable "+name, __FILE__, __LINE__); 
        }

        //====================================================================================
        // GRID VARIABLE ACCESS
        //====================================================================================
        //SO FIELD
        /** @brief The interface to the actual task for retrieving variables for spatialops types.**/
        template<class ST>
        ST* get_so_field( const std::string name, 
                          const WHICH_DW which_dw ){ 

          //search through the registry for the variable: 
          //Note: In its most basic operation, this assumes that the variable 
          //name will be present ONLY ONCE in the variable registry. 
          //One can use the which_dw to force a matching if the variable 
          //is being requested from two different DW's
          VariableInformation* var_info=0; 
          BOOST_FOREACH( VariableInformation &ivar, _var_reg ){ 
            if ( ivar.name == name && ivar.dw == which_dw ){ 

              var_info = &ivar; 
              break; 

            }
          }

          if (var_info == 0){ 
            throw InvalidValue("Arches Task Error: (UINTAH) Cannot find information on variable: "+name, __FILE__, __LINE__); 
          }

          //utilize the appropriate map: 
          //Note we passing the patch here because we might want to 
          //interface directly with the Wasatch wrap function later. 
          int nGhost = var_info->nGhost; 
          ST* field; 

          if ( var_info->depend == REQUIRES ){ 
            //const map (requires)
            field = retrieve_so_field<ST>(name, _const_variable_map, _patch, nGhost ); 
          } else { 
            //non-const map (computes, modifies)
            field = retrieve_so_field<ST>(name, _variable_map, _patch, nGhost ); 
          }

          //clunky: but we need to track the variables to destroy them later --
          //how bad is the dynamic casting?
          if ( var_info->type == CC_DOUBLE ){ 
            _cc_fields.push_back(dynamic_cast<SpatialOps::SVolField*>(field)); 
          } else if ( var_info->type == FACEX ){ 
            _fx_fields.push_back(dynamic_cast<SpatialOps::SSurfXField*>(field)); 
          } else if ( var_info->type == FACEY ){ 
            _fy_fields.push_back(dynamic_cast<SpatialOps::SSurfYField*>(field)); 
          } else if ( var_info->type == FACEZ ){ 
            _fz_fields.push_back(dynamic_cast<SpatialOps::SSurfZField*>(field)); 
          }

          return field; 

        }

        //====================================================================================
        // GRID VARIABLE ACCESS
        //====================================================================================
        //UINTAH
        /** @brief The interface to the actual task for retrieving variables for uintah types **/ 
        template<class ST>
        ST* get_uintah_field( const std::string name, const WHICH_DW which_dw ){

          //search through the registry for the variable: 
          //Note: In its most basic operation, this assumes that the variable 
          //name will be present ONLY ONCE in the variable registry. 
          //One can use the which_dw to force a matching if the variable 
          //is being requested from two different DW's
          VariableInformation* var_info=0; 
          BOOST_FOREACH( VariableInformation &ivar, _var_reg ){ 
            if ( ivar.name == name && ivar.dw == which_dw ){ 

              var_info = &ivar; 
              break; 

            }
          }

          if (var_info == 0){ 
            throw InvalidValue("Arches Task Error: (UINTAH) Cannot find information on variable: "+name, __FILE__, __LINE__); 
          }

          //utilize the appropriate map: 
          ST* field; 
          if ( var_info->depend == REQUIRES ){ 
            //const map (requires)
            field = get_uintah_grid_var<ST>(name, _const_variable_map );
            //field = new_get_uintah_grid_var<ST>(name, CONST_FIELD ); 
          } else { 
            //non-const map (computes, modifies)
            field = get_uintah_grid_var<ST>(name, _variable_map );
            //field = new_get_uintah_grid_var<ST>(name, NONCONST_FIELD ); 
          }

          //uintah fields are automagically stored in the variable_maps.
          //they get destroyed later when FieldCollector gets destroyed so 
          //no need to do any more tracking. 

          return field; 

        }


      private: 

        UintahVarMap _variable_map;
        ConstUintahVarMap _const_variable_map; 

        std::vector<VariableInformation> _var_reg; 
        const Patch* _patch; 
        SchedToTaskInfo& _tsk_info; 

        std::vector<void *> _all_so_fields; 
      
        //lists of spatialops variables to be destroyed: 
        std::vector<SpatialOps::SVolField*> _cc_fields;
        std::vector<SpatialOps::SSurfXField*> _fx_fields;
        std::vector<SpatialOps::SSurfYField*> _fy_fields;
        std::vector<SpatialOps::SSurfZField*> _fz_fields;

    }; //End FieldCollector

    /** @brief Resolves the DW fields with the dependency **/ 
    void resolve_fields( DataWarehouse* old_dw, 
                         DataWarehouse* new_dw, 
                         const Patch* patch, 
                         UintahVarMap& var_map, 
                         ConstUintahVarMap& const_var_map, 
                         FieldCollector* f_collector );

    /** @brief The actual work done within the derived class **/ 
    virtual void initialize( const Patch* patch, FieldCollector* field_collector, 
                             SpatialOps::OperatorDatabase& opr ) = 0; 

    /** @brief Work done at the top of a timestep **/ 
    virtual void timestep_init( const Patch* patch, FieldCollector* field_collector, 
                                SpatialOps::OperatorDatabase& opr ) = 0; 

    /** @brief The actual work done within the derived class **/ 
    virtual void eval( const Patch* patch, FieldCollector* field_collector, 
                       SpatialOps::OperatorDatabase& opr ) = 0; 

    std::string _task_name; 
    const int _matl_index; 
    VAR_TYPE _mytype;
    std::vector<const VarLabel*> _local_labels;

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
}

#endif 
