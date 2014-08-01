#ifndef Uintah_Component_Arches_FEUpdate_h
#define Uintah_Component_Arches_FEUpdate_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <spatialops/structured/FVStaggered.h>

namespace Uintah{ 

  template <typename T>
  class FEUpdate : public TaskInterface { 

public: 

    FEUpdate<T>( std::string task_name, int matl_index, std::vector<std::string> eqn_names ); 
    ~FEUpdate<T>(); 

    /** @brief Input file interface **/ 
    void problemSetup( ProblemSpecP& db ); 

    /** @brief Build instruction for this class **/ 
    class Builder : public TaskInterface::TaskBuilder { 

      public: 

      Builder( std::string task_name, int matl_index, std::vector<std::string> eqn_names ) : 
        _task_name(task_name), _matl_index(matl_index), _eqn_names(eqn_names){}
      ~Builder(){}

      FEUpdate* build()
      { return scinew FEUpdate<T>( _task_name, _matl_index, _eqn_names ); }

      private: 

      std::string _task_name; 
      int _matl_index; 
      std::vector<std::string> _eqn_names; 

    };

protected: 

    void register_initialize( std::vector<VariableInformation>& variable_registry );

    void register_timestep_init( std::vector<VariableInformation>& variable_registry ){}

    void register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ); 

    void initialize( const Patch* patch, FieldCollector* field_collector, 
                     SpatialOps::OperatorDatabase& opr );
    
    void timestep_init( const Patch* patch, FieldCollector* field_collector, 
                        SpatialOps::OperatorDatabase& opr ){}

    void eval( const Patch* patch, FieldCollector* field_collector, 
               SpatialOps::OperatorDatabase& opr ); 

private:

    std::vector<std::string> _eqn_names; 

  
  };

  //Function definitions: 

  template <typename T>
  FEUpdate<T>::FEUpdate( std::string task_name, int matl_index, std::vector<std::string> eqn_names ) : 
  TaskInterface( task_name, matl_index ){

    // This needs to be done to set the variable type 
    // for this function. All templated tasks should do this. 
    set_task_type<T>(); 

    _eqn_names = eqn_names; 
  
  }

  template <typename T>
  FEUpdate<T>::~FEUpdate()
  {
  }

  template <typename T>
  void FEUpdate<T>::problemSetup( ProblemSpecP& db ){ 
  }


  template <typename T>
  void FEUpdate<T>::register_initialize( std::vector<VariableInformation>& variable_registry ){ 
  }
  
  //This is the work for the task.  First, get the variables. Second, do the work! 
  template <typename T> 
  void FEUpdate<T>::initialize( const Patch* patch, FieldCollector* field_collector, 
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
  void FEUpdate<T>::eval( const Patch* patch, FieldCollector* field_collector, 
                          SpatialOps::OperatorDatabase& opr ){ 

    using namespace SpatialOps;
    using SpatialOps::operator *; 
    typedef SpatialOps::SVolField   SVol;

    const SVol* const rho = field_collector->get_so_field<SVol>( "density", LATEST ); 
    typedef std::vector<std::string> SV;
    for ( SV::iterator i = _eqn_names.begin(); i != _eqn_names.end(); i++){ 

      T* const phi = field_collector->get_so_field<T>( *i, NEWDW );
      T* const rhs = field_collector->get_so_field<T>( *i+"_RHS", NEWDW ); 

      //update: 
      *phi <<= *rhs / *rho; 

    }
  }
}
#endif 
