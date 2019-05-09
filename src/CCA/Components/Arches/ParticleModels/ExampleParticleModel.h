#ifndef Uintah_Component_Arches_ExampleParticleModel_h
#define Uintah_Component_Arches_ExampleParticleModel_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>

namespace Uintah{

  //IT is the independent variable type
  //DT is the dependent variable type
  template <typename IT, typename DT>
  class ExampleParticleModel : public TaskInterface {

public:

    ExampleParticleModel<IT, DT>( std::string task_name, int matl_index, const std::string var_name, const int N );
    ~ExampleParticleModel<IT, DT>();

    void problemSetup( ProblemSpecP& db );



    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index, std::string base_var_name, const int N ) :
        m_task_name(task_name), m_matl_index(matl_index), _base_var_name(base_var_name), _N(N){}
      ~Builder(){}

      ExampleParticleModel* build()
      { return scinew ExampleParticleModel<IT, DT>( m_task_name, m_matl_index, _base_var_name, _N ); }

      private:

      std::string m_task_name;
      int m_matl_index;
      std::string _base_var_name;
      const int _N;

    };

protected:

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();

private:

    const std::string _base_var_name;
    std::string _temperature_var_name;
    std::string _conc_var_name;

    const int _N;                 //<<< The number of "environments"

    double _A;
    double _ER;
    double _m;

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
  void ExampleParticleModel<IT,DT>::create_local_labels(){

    for ( int i = 0; i < _N; i++ ){

      const std::string name = get_name(i, _base_var_name);
      register_new_variable<IT>( name );

    }

  }


  template <typename IT, typename DT>
  ExampleParticleModel<IT, DT>::ExampleParticleModel( std::string task_name, int matl_index,
                                                      const std::string base_var_name, const int N ) :
  TaskInterface( task_name, matl_index ), _base_var_name(base_var_name), _N(N){
  }

  template <typename IT, typename DT>
  ExampleParticleModel<IT, DT>::~ExampleParticleModel()
  {}

  template <typename IT, typename DT>
  void ExampleParticleModel<IT, DT>::problemSetup( ProblemSpecP& db ){

    db->require("A",_A);
    db->require("ER",_ER);
    db->require("m",_m);
    db->findBlock("temperature")->getAttribute("label",_temperature_var_name);
    db->findBlock("concentration")->getAttribute("label",_conc_var_name);

  }

  //======INITIALIZATION:
  template <typename IT, typename DT>
  void ExampleParticleModel<IT, DT>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

    for ( int i = 0; i < _N; i++ ){

      const std::string name = get_name(i, _base_var_name);
      register_variable( name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );

    }

  }

  template <typename IT, typename DT>
  void ExampleParticleModel<IT,DT>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    for ( int ienv = 0; ienv < _N; ienv++ ){

      const std::string name = get_name(ienv, _base_var_name);
      DT& model_value = *(tsk_info->get_uintah_field<DT>(name));

      Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
      Uintah::parallel_for( range, [&](int i, int j, int k){
        model_value(i,j,k) = 0.0;
      });

    }
  }

  //======TIME STEP INITIALIZATION:
  template <typename IT, typename DT>
  void ExampleParticleModel<IT, DT>::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){
  }

  template <typename IT, typename DT>
  void ExampleParticleModel<IT,DT>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

  //======TIME STEP EVALUATION:
  template <typename IT, typename DT>
  void ExampleParticleModel<IT, DT>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){


    for ( int ienv = 0; ienv < _N; ienv++ ){

      //dependent variables(s) or model values
      const std::string name = get_name(ienv, _base_var_name);
      register_variable( name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

    }

    register_variable( _temperature_var_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    register_variable( _conc_var_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

  }

  template <typename IT, typename DT>
  void ExampleParticleModel<IT,DT>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    typedef typename ArchesCore::VariableHelper<IT>::ConstType CIT;

    for ( int ienv = 0; ienv < _N; ienv++ ){

      const std::string name = get_name(ienv, _base_var_name);
      DT& model_value = *(tsk_info->get_uintah_field<DT>(name));

      CIT& temperature = *(tsk_info->get_const_uintah_field<CIT>(_temperature_var_name));
      CIT& conc = *(tsk_info->get_const_uintah_field<CIT>(_conc_var_name));

      Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
      Uintah::parallel_for( range, [&](int i, int j, int k){

        //compute a rate term
        model_value(i,j,k) = _A * std::exp( _ER * temperature(i,j,k)) * std::pow( conc(i,j,k), _m );

      });

    }
  }
}
#endif
