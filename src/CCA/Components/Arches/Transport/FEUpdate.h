#ifndef Uintah_Component_Arches_FEUpdate_h
#define Uintah_Component_Arches_FEUpdate_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <spatialops/structured/FVStaggered.h>

namespace Uintah{ 

  template <typename T>
  class FEUpdate : public TaskInterface { 

public: 

    FEUpdate<T>( std::string task_name, int matl_index, std::vector<std::string> eqn_names, bool divide_out_density ); 
    ~FEUpdate<T>(); 

    /** @brief Input file interface **/ 
    void problemSetup( ProblemSpecP& db ); 

    void create_local_labels(){}

    /** @brief Build instruction for this class **/ 
    class Builder : public TaskInterface::TaskBuilder { 

      public: 

      Builder( std::string task_name, int matl_index, std::vector<std::string> eqn_names, bool divide_out_density ) : 
        _task_name(task_name), _matl_index(matl_index), _eqn_names(eqn_names), _div_density(divide_out_density){}
      ~Builder(){}

      FEUpdate* build()
      { return scinew FEUpdate<T>( _task_name, _matl_index, _eqn_names, _div_density ); }

      private: 

      std::string _task_name; 
      int _matl_index; 
      std::vector<std::string> _eqn_names; 
      bool _div_density; 

    };

protected: 

    void register_initialize( std::vector<VariableInformation>& variable_registry );

    void register_timestep_init( std::vector<VariableInformation>& variable_registry ){}

    void register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ); 

    void register_compute_bcs( std::vector<VariableInformation>& variable_registry, const int time_substep ){}; 

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                      SpatialOps::OperatorDatabase& opr ){}; 

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                     SpatialOps::OperatorDatabase& opr );
    
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                        SpatialOps::OperatorDatabase& opr ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
               SpatialOps::OperatorDatabase& opr ); 

private:

    std::vector<std::string> _eqn_names; 
    VAR_TYPE _mytype; 

    bool _div_density; 

  
  };

  //Function definitions: 

  template <typename T>
  FEUpdate<T>::FEUpdate( std::string task_name, int matl_index, std::vector<std::string> eqn_names, bool divide_out_density ) : 
  TaskInterface( task_name, matl_index ){

    VarTypeHelper<T> helper; 
    _mytype = helper.get_vartype(); 

    _eqn_names = eqn_names; 
    _div_density = divide_out_density; 
  
  }

  template <typename T>
  FEUpdate<T>::~FEUpdate()
  {
  }

  template <typename T>
  void FEUpdate<T>::problemSetup( ProblemSpecP& db ){ 

    _do_ts_init_task = false;
    _do_init_task    = false;
    _do_bcs_task     = false;

  }


  template <typename T>
  void FEUpdate<T>::register_initialize( std::vector<VariableInformation>& variable_registry ){ 
  }
  
  //This is the work for the task.  First, get the variables. Second, do the work! 
  template <typename T> 
  void FEUpdate<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                                SpatialOps::OperatorDatabase& opr ){ 
  }


  template <typename T> 
  void FEUpdate<T>::register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ){
   
    //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
    //register_variable( "templated_variable", _mytype, COMPUTES, 0, NEWDW, variable_registry, time_substep ); 
    typedef std::vector<std::string> SV;
    for ( SV::iterator i = _eqn_names.begin(); i != _eqn_names.end(); i++){ 
      register_variable( *i, _mytype, MODIFIES, 0, NEWDW, variable_registry, time_substep ); 
      std::string rhs_name = *i + "_RHS"; 
      register_variable( rhs_name, _mytype, REQUIRES, 0, NEWDW, variable_registry, time_substep ); 
    }
    register_variable( "density", CC_DOUBLE, REQUIRES, 0, LATEST, variable_registry, time_substep ); 
  
  }

  template <typename T>
  void FEUpdate<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                          SpatialOps::OperatorDatabase& opr ){ 

    using namespace SpatialOps;
    using SpatialOps::operator *; 
    typedef SpatialOps::SVolField   SVolF;
    typedef SpatialOps::SpatFldPtr<SVolF> SVolFP; 
    typedef SpatialOps::SpatFldPtr<T> STFP; 
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolF, T>::type I_svol_to_T;

    SVolFP const rho = tsk_info->get_const_so_field<SVolF>( "density" ); 

    const I_svol_to_T* const interp_rho = opr.retrieve_operator<I_svol_to_T>(); 

    typedef std::vector<std::string> SV;

    for ( SV::iterator i = _eqn_names.begin(); i != _eqn_names.end(); i++){ 

      STFP phi = tsk_info->get_so_field<T>( *i );
      STFP rhs = tsk_info->get_const_so_field<T>( *i+"_RHS" ); 

      //update: 
      if ( _div_density ){ 
        //dividing out the density
        *phi <<= *rhs / ((*interp_rho)(*rho)); 
      } else { 
        //solving for rho*phi
        *phi <<= *rhs; 
      }

    }
  }
}
#endif 
