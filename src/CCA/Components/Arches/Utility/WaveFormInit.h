#ifndef Uintah_Component_Arches_WaveFormInit_h
#define Uintah_Component_Arches_WaveFormInit_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/DiscretizationTools.h>

namespace Uintah{

  template <typename T>
  class WaveFormInit : public TaskInterface {

public:

    enum WAVE_TYPE { SINE, SQUARE };

    WaveFormInit<T>( std::string task_name, int matl_index, const std::string var_name );
    ~WaveFormInit<T>();

    void problemSetup( ProblemSpecP& db );

    //Build instructions for this (WaveFormInit) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index, std::string var_name ) :
        _task_name(task_name), _matl_index(matl_index), _var_name(var_name){}
      ~Builder(){}

      WaveFormInit* build()
      { return scinew WaveFormInit<T>( _task_name, _matl_index, _var_name ); }

      private:

      std::string _task_name;
      int _matl_index;
      std::string _var_name;

    };

protected:

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry );

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){}

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){}

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                      SpatialOps::OperatorDatabase& opr ){};

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                     SpatialOps::OperatorDatabase& opr );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                        SpatialOps::OperatorDatabase& opr ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
               SpatialOps::OperatorDatabase& opr ){}

    void create_local_labels(){};

private:

    const std::string _var_name;
    std::string _ind_var_name;
    std::string _ind_var_name_2;
    WAVE_TYPE _wtype;
    double _amp;
    double _two_pi;
    double _f1;
    double _A;
    double _offset;
    double _min_sq;
    double _max_sq;

    typedef typename VariableHelper<T>::ConstType CT;

  };

  //Function definitions:

  template <typename T>
  WaveFormInit<T>::WaveFormInit( std::string task_name, int matl_index, const std::string var_name ) :
  TaskInterface( task_name, matl_index ), _var_name(var_name){

    _two_pi = 2.0*acos(-1.0);

  }

  template <typename T>
  WaveFormInit<T>::~WaveFormInit()
  {}

  template <typename T>
  void WaveFormInit<T>::problemSetup( ProblemSpecP& db ){

    std::string wave_type;
    db->findBlock("wave")->getAttribute("type",wave_type);
    db->findBlock("wave")->findBlock("independent_variable")->getAttribute("label",_ind_var_name);

    if ( wave_type == "sine"){

      ProblemSpecP db_sine = db->findBlock("wave")->findBlock("sine");

      _wtype = SINE;
      db_sine->getAttribute("A",_A);
      db_sine->getAttribute("f",_f1);
      db_sine->getAttribute("offset",_offset);

    } else if ( wave_type == "square"){

      ProblemSpecP db_square= db->findBlock("wave")->findBlock("square");

      _wtype = SQUARE;
      db_square->getAttribute("f",_f1);
      db_square->getAttribute("min",_min_sq );
      db_square->getAttribute("max",_max_sq );
      db_square->getAttribute("offset",_offset);

    } else {

      throw InvalidValue("Error: Wave type not recognized.",__FILE__,__LINE__);

    }
  }

  template <typename T>
  void WaveFormInit<T>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

    register_variable( _ind_var_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW,  variable_registry );
    register_variable( _var_name,     ArchesFieldContainer::MODIFIES, variable_registry );

  }

  template <typename T>
  void WaveFormInit<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                        SpatialOps::OperatorDatabase& opr ){

    T& dep_field = *(tsk_info->get_uintah_field<T>( _var_name ));
    CT& ind_field = *(tsk_info->get_const_uintah_field<CT>( _ind_var_name ));
    Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex());

    switch (_wtype){
      case SINE:

        Uintah::parallel_for(range, [&dep_field, &ind_field, this](int i, int j, int k){
          dep_field(i,j,k) = _A * sin( _two_pi * _f1 * ind_field(i,j,k) ) + _offset;
        });

        break;
      case SQUARE:

        Uintah::parallel_for(range, [&dep_field, &ind_field, this](int i, int j, int k){
          dep_field(i,j,k) = sin( _two_pi * _f1 * ind_field(i,j,k)) + _offset;
          dep_field(i,j,k) = (dep_field(i,j,k) <= 0.0) ? _min_sq : _max_sq ;
         });

        break;
      default:
        break;

    }
  }
}
#endif
