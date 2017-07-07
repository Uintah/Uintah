#ifndef Uintah_Component_Arches_Constant_h
#define Uintah_Component_Arches_Constant_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
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

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){}

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

  private:

    const std::string _base_var_name;

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
      register_new_variable<T>(name);
    }
  }

  template <typename T>
  Constant<T>::Constant( std::string task_name, int matl_index,
                         const std::string base_var_name, const int N ) :
  TaskInterface( task_name, matl_index ), _base_var_name(base_var_name), _N(N){
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
  void Constant<T>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

    for ( int ei = 0; ei < _N; ei++ ){

      const std::string name = get_name(ei, _base_var_name);
      register_variable( name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );

    }
  }

  template <typename T>
  void Constant<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    for ( int ei = 0; ei < _N; ei++ ){

      const std::string name = get_name(ei, _base_var_name);
      T& model_value = tsk_info->get_uintah_field_add<T>(name);

      Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
      Uintah::parallel_for( range, [&](int i, int j, int k){
        model_value(i,j,k) = _const[ei];
      });
    }
  }

  //======TIME STEP INITIALIZATION:
  template <typename T>
  void Constant<T>::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){
    for ( int ei = 0; ei < _N; ei++ ){

      //dependent variables(s) or model values
      const std::string name = get_name(ei, _base_var_name);
      register_variable( name, ArchesFieldContainer::COMPUTES, variable_registry );

    }
  }

  template <typename T>
  void Constant<T>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    for ( int ei = 0; ei < _N; ei++ ){

      const std::string name = get_name(ei, _base_var_name);
      T& model_value = tsk_info->get_uintah_field_add<T>(name);

      Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
      Uintah::parallel_for( range, [&](int i, int j, int k){
        model_value(i,j,k) = _const[ei];
      });
    }
  }

}
#endif
