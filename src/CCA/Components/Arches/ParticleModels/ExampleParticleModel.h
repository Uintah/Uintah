#ifndef Uintah_Component_Arches_ExampleParticleModel_h
#define Uintah_Component_Arches_ExampleParticleModel_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <spatialops/structured/FVStaggered.h>

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
        _task_name(task_name), _matl_index(matl_index), _base_var_name(base_var_name), _N(N){}
      ~Builder(){}

      ExampleParticleModel* build()
      { return new ExampleParticleModel<IT, DT>( _task_name, _matl_index, _base_var_name, _N ); }

      private:

      std::string _task_name;
      int _matl_index;
      std::string _base_var_name;
      const int _N;

    };

protected:

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry );

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry );

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep );

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                      SpatialOps::OperatorDatabase& opr ){};

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                     SpatialOps::OperatorDatabase& opr );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                        SpatialOps::OperatorDatabase& opr );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
               SpatialOps::OperatorDatabase& opr );

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
  void ExampleParticleModel<IT, DT>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

    for ( int i = 0; i < _N; i++ ){

      const std::string name = get_name(i, _base_var_name);
      register_variable( name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );

    }

  }

  template <typename IT, typename DT>
  void ExampleParticleModel<IT,DT>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
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
  void ExampleParticleModel<IT, DT>::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
  }

  template <typename IT, typename DT>
  void ExampleParticleModel<IT,DT>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                        SpatialOps::OperatorDatabase& opr ){

  }

  //======TIME STEP EVALUATION:
  template <typename IT, typename DT>
  void ExampleParticleModel<IT, DT>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){


    for ( int i = 0; i < _N; i++ ){

      //dependent variables(s) or model values
      const std::string name = get_name(i, _base_var_name);
      register_variable( name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

      //independent variable
      //const std::string temperature_name = get_name( i, _temperature_var_name );
      //register_variable( temperature_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );

      //const std::string conc_name = get_name( i, _conc_var_name );
      //register_variable( conc_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );

    }

    //temp: remove this after testing:
    //and uncomment the statements above.
    register_variable( _temperature_var_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( _conc_var_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    //end temp:

  }

  template <typename IT, typename DT>
  void ExampleParticleModel<IT,DT>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                        SpatialOps::OperatorDatabase& opr ){

    using namespace SpatialOps;
    using SpatialOps::operator *;
    typedef SpatialOps::SpatFldPtr<DT> DTptr;
    typedef SpatialOps::SpatFldPtr<IT> ITptr;
    typedef typename OperatorTypeBuilder< SpatialOps::Interpolant, IT, DT >::type InterpT;
    const InterpT* const interp = opr.retrieve_operator<InterpT>();

    for ( int i = 0; i < _N; i++ ){

      const std::string name = get_name(i, _base_var_name);
      DTptr model_value = tsk_info->get_so_field<DT>(name);

      //temp: remove this after testing:
      ITptr temperature = tsk_info->get_const_so_field<IT>(_temperature_var_name);
      ITptr conc = tsk_info->get_const_so_field<IT>(_conc_var_name);
      //end temp:

      //const std::string temperature_name = get_name( i, _temperature_var_name );
      //ITptr temperature = tsk_info->get_const_so_field<IT>(temperature_name);

      //const std::string conc_name = get_name( i, _conc_var_name );
      //ITptr conc = tsk_info->get_const_so_field<IT>(conc_name);

      //compute a rate term
      *model_value <<= _A * exp( _ER * (*interp)(*temperature)) * pow( (*interp)(*conc), _m );

    }
  }
}
#endif
