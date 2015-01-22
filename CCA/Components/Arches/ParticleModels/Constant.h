#ifndef Uintah_Component_Arches_Constant_h
#define Uintah_Component_Arches_Constant_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <spatialops/structured/FVStaggered.h>
#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif
//-------------------------------------------------------

/**
 * @class    Constant
 * @author   Alex Abboud
 * @date     October 2014
 *
 * @brief    This class sets a constant source term for particles
 *
 * @details  A constant source term for easier debugging
 *
 */

//-------------------------------------------------------

namespace Uintah{
  
  //T is the dependent variable type
  template <typename T>
  class Constant : public TaskInterface {
    
  public:
    
    Constant<T>( std::string task_name, int matl_index, const std::string var_name, const int N );
    ~Constant<T>();
    
    void problemSetup( ProblemSpecP& db );
    
    void create_local_labels();
    
    class Builder : public TaskInterface::TaskBuilder {
      
    public:
      
      Builder( std::string task_name, int matl_index, std::string base_var_name, const int N ) :
      _task_name(task_name), _matl_index(matl_index), _base_var_name(base_var_name), _N(N){}
      ~Builder(){}
      
      Constant* build()
      { return scinew Constant<T>( _task_name, _matl_index, _base_var_name, _N ); }
      
    private:
      
      std::string _task_name;
      int _matl_index;
      std::string _base_var_name;
      const int _N;
      
    };
    
  protected:
    
    void register_initialize( std::vector<VariableInformation>& variable_registry );
    
    void register_timestep_init( std::vector<VariableInformation>& variable_registry );
    
    void register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep );
    
    void register_compute_bcs( std::vector<VariableInformation>& variable_registry, const int time_substep ){};
    
    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                     SpatialOps::OperatorDatabase& opr ){};
    
    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                    SpatialOps::OperatorDatabase& opr );
    
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                       SpatialOps::OperatorDatabase& opr );
    
    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
              SpatialOps::OperatorDatabase& opr );
    
  private:
    
    const std::string _base_var_name;
    VAR_TYPE _D_type;
    
    const int _N;                      //<<< The number of "environments"
    std::vector<double> _const;        //<<< constant source value/environment
    
    const std::string get_name(const int i, const std::string base_name){
      std::stringstream out;
      std::string env;
      out << i;
      env = out.str();
      return base_name + "_" + env;
    }
    
  };
  
  //Function definitions:
  
  template <typename T>
  void Constant<T>::create_local_labels(){
    for ( int i = 0; i < _N; i++ ){
      const std::string name = get_name(i, _base_var_name);
      register_new_variable(name, _D_type);
    }
  }
  
  template <typename T>
  Constant<T>::Constant( std::string task_name, int matl_index,
                         const std::string base_var_name, const int N ) :
  TaskInterface( task_name, matl_index ), _base_var_name(base_var_name), _N(N){

    VarTypeHelper<T> dhelper; 
    _D_type = dhelper.get_vartype(); 

  }
  
  template <typename T>
  Constant<T>::~Constant()
  {}
  
  template <typename T>
  void Constant<T>::problemSetup( ProblemSpecP& db ){
    
    db->require("constant",_const);

  }
  
  //======INITIALIZATION:
  template <typename T>
  void Constant<T>::register_initialize( std::vector<VariableInformation>& variable_registry ){
    
    for ( int i = 0; i < _N; i++ ){

      const std::string name = get_name(i, _base_var_name);
      register_variable( name, _D_type, COMPUTES, 0, NEWDW, variable_registry );
      
    }
  }
  
  template <typename T>
  void Constant<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                    SpatialOps::OperatorDatabase& opr ){
    
    using namespace SpatialOps;
    using SpatialOps::operator *;
    typedef SpatialOps::SpatFldPtr<T> Tptr;
    
    for ( int i = 0; i < _N; i++ ){
      const std::string name = get_name(i, _base_var_name);
      Tptr model_value = tsk_info->get_so_field<T>(name);
      
      *model_value <<= _const[i];

    }
  }
  
  //======TIME STEP INITIALIZATION:
  template <typename T>
  void Constant<T>::register_timestep_init( std::vector<VariableInformation>& variable_registry ){
    for ( int i = 0; i < _N; i++ ){

      //dependent variables(s) or model values
      const std::string name = get_name(i, _base_var_name);
      register_variable( name, _D_type, COMPUTES, variable_registry );

    }
  }
  
  template <typename T>
  void Constant<T>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                       SpatialOps::OperatorDatabase& opr ){
    using namespace SpatialOps;
    using SpatialOps::operator *;
    typedef SpatialOps::SpatFldPtr<T> Tptr;

    for ( int i = 0; i < _N; i++ ){
      
      const std::string name = get_name(i, _base_var_name);
      Tptr model_value = tsk_info->get_so_field<T>(name);

      *model_value <<= _const[i];
      
    }
    
  }
  
  //======TIME STEP EVALUATION:
  template <typename T>
  void Constant<T>::register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ){
    
  }
  
  template <typename T>
  void Constant<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                              SpatialOps::OperatorDatabase& opr ) {
  }
}
#endif
