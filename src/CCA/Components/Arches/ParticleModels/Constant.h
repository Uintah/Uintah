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
 * @brief    This class sets a constnat source term for particles
 *
 * @details  A constant source term for easier debugging
 *
 */

//-------------------------------------------------------

namespace Uintah{
  
  //IT is the independent variable type
  //DT is the dependent variable type
  template <typename IT, typename DT>
  class Constant : public TaskInterface {
    
  public:
    
    Constant<IT, DT>( std::string task_name, int matl_index, const std::string var_name, const int N );
    ~Constant<IT, DT>();
    
    void problemSetup( ProblemSpecP& db );
    
    void create_local_labels();
    
    class Builder : public TaskInterface::TaskBuilder {
      
    public:
      
      Builder( std::string task_name, int matl_index, std::string base_var_name, const int N ) :
      _task_name(task_name), _matl_index(matl_index), _base_var_name(base_var_name), _N(N){}
      ~Builder(){}
      
      Constant* build()
      { return scinew Constant<IT, DT>( _task_name, _matl_index, _base_var_name, _N ); }
      
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
    VAR_TYPE _I_type;
    
    const int _N;                 //<<< The number of "environments"
    double _const;        //cosntant source value
    
    const std::string get_name(const int i, const std::string base_name){
      std::stringstream out;
      std::string env;
      out << i;
      env = out.str();
      return base_name + "_" + env;
    }
    
  };
  
  //Function definitions:
  
  template <typename IT, typename DT>
  void Constant<IT,DT>::create_local_labels(){
    for ( int i = 0; i < _N; i++ ){
      const std::string name = get_name(i, _base_var_name);
      register_new_variable(name, _D_type);
    }
  }
  
  template <typename IT, typename DT>
  Constant<IT, DT>::Constant( std::string task_name, int matl_index,
                               const std::string base_var_name, const int N ) :
  _base_var_name(base_var_name), TaskInterface( task_name, matl_index ), _N(N){
  }
  
  template <typename IT, typename DT>
  Constant<IT, DT>::~Constant()
  {}
  
  template <typename IT, typename DT>
  void Constant<IT, DT>::problemSetup( ProblemSpecP& db ){
    
    _do_ts_init_task = false;
    _do_bcs_task = false;
    
    //This sets the type of the independent and dependent variable types as needed by the variable
    //registration step.
    DT* d_test;
    IT* i_test;
    
    set_type(d_test, _D_type);
    set_type(i_test, _I_type);
    
    db->require("constant",_const);
  }
  
  //======INITIALIZATION:
  template <typename IT, typename DT>
  void Constant<IT, DT>::register_initialize( std::vector<VariableInformation>& variable_registry ){
    
    for ( int i = 0; i < _N; i++ ){
      const std::string name = get_name(i, _base_var_name);
      std::cout << "Source label " << name << std::endl;
      register_variable( name, _D_type, COMPUTES, 0, NEWDW, variable_registry );
      
    }
  }
  
  template <typename IT, typename DT>
  void Constant<IT,DT>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                    SpatialOps::OperatorDatabase& opr ){
    
    using namespace SpatialOps;
    using SpatialOps::operator *;
    typedef SpatialOps::SpatFldPtr<DT> DTptr;
    
    for ( int i = 0; i < _N; i++ ){
      const std::string name = get_name(i, _base_var_name);
      DTptr model_value = tsk_info->get_so_field<DT>(name);
      
      *model_value <<= 0.0;
    }
  }
  
  //======TIME STEP INITIALIZATION:
  template <typename IT, typename DT>
  void Constant<IT, DT>::register_timestep_init( std::vector<VariableInformation>& variable_registry ){
  }
  
  template <typename IT, typename DT>
  void Constant<IT,DT>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                       SpatialOps::OperatorDatabase& opr ){
  }
  
  //======TIME STEP EVALUATION:
  template <typename IT, typename DT>
  void Constant<IT, DT>::register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ){
    
    for ( int i = 0; i < _N; i++ ){
      //dependent variables(s) or model values
      const std::string name = get_name(i, _base_var_name);
      register_variable( name, _D_type, COMPUTES, 0, NEWDW, variable_registry, time_substep );
    }
  }
  
  template <typename IT, typename DT>
  void Constant<IT,DT>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                              SpatialOps::OperatorDatabase& opr ) {
    using namespace SpatialOps;
    using SpatialOps::operator *;
    typedef SpatialOps::SpatFldPtr<DT> DTptr;
    typedef SpatialOps::SpatFldPtr<IT> ITptr;
    typedef typename OperatorTypeBuilder< SpatialOps::Interpolant, IT, DT >::type InterpT;
    const InterpT* const interp = opr.retrieve_operator<InterpT>();

    for ( int i = 0; i < _N; i++ ){
      
      const std::string name = get_name(i, _base_var_name);
      DTptr model_value = tsk_info->get_so_field<DT>(name);

      *model_value <<= _const;
      
    }
    
  }
}
#endif
