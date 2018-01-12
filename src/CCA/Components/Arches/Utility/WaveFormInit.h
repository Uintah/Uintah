#ifndef Uintah_Component_Arches_WaveFormInit_h
#define Uintah_Component_Arches_WaveFormInit_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>

namespace Uintah{

  template <typename T, typename CT>
  class WaveFormInit : public TaskInterface {

public:

    enum WAVE_TYPE { SINE, SQUARE };

    WaveFormInit<T,CT>( std::string task_name, int matl_index, const std::string var_name );
    ~WaveFormInit<T,CT>();

    void problemSetup( ProblemSpecP& db );

    //Build instructions for this (WaveFormInit) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index, std::string var_name ) :
        _task_name(task_name), _matl_index(matl_index), _var_name(var_name){}
      ~Builder(){}

      WaveFormInit* build()
      { return scinew WaveFormInit<T,CT>( _task_name, _matl_index, _var_name ); }

      private:

      std::string _task_name;
      int _matl_index;
      std::string _var_name;

    };

protected:

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){}

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){}

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

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

  };

  //Function definitions ---------------------------------------------------------------------------

  template <typename T, typename CT>
  WaveFormInit<T,CT>::WaveFormInit( std::string task_name, int matl_index, const std::string var_name ) :
  TaskInterface( task_name, matl_index ), _var_name(var_name){

    _two_pi = 2.0*acos(-1.0);

  }

  template <typename T, typename CT>
  WaveFormInit<T, CT>::~WaveFormInit()
  {}

  template <typename T, typename CT>
  void WaveFormInit<T, CT>::problemSetup( ProblemSpecP& db ){

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

  template <typename T, typename CT>
  void WaveFormInit<T, CT>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

    register_variable( _ind_var_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW,  variable_registry );
    register_variable( _var_name,     ArchesFieldContainer::MODIFIES, variable_registry );

  }

  template <typename T, typename CT>
  void WaveFormInit<T, CT>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    T& dep_field = *(tsk_info->get_uintah_field<T>( _var_name ));
    CT& ind_field = *(tsk_info->get_const_uintah_field<CT>( _ind_var_name ));
    Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex());

    ArchesCore::oneDInterp<T,CT> interpolator;
    double weight = interpolator.get_central_weight();
    int dir = interpolator.dir;

    switch (_wtype){
      case SINE:

        if ( dir > 2 ){
          Uintah::parallel_for(range, [&](int i, int j, int k){
            dep_field(i,j,k) = _A * sin( _two_pi * _f1 * ind_field(i,j,k) ) + _offset;
          });
        } else {
          Uintah::parallel_for(range, [&](int i, int j, int k){
            STENCIL3_1D(dir);
            dep_field(IJK_) = _A * sin( _two_pi * _f1 * weight * (ind_field(IJK_)+ind_field(IJK_M_) ) ) + _offset;
          });
        }

        break;
      case SQUARE:

        if ( dir > 2 ){
          Uintah::parallel_for(range, [&](int i, int j, int k){
            dep_field(i,j,k) = sin( _two_pi * _f1 * ind_field(i,j,k)) + _offset;
            dep_field(i,j,k) = (dep_field(i,j,k) <= 0.0) ? _min_sq : _max_sq ;
          });
        } else {
          Uintah::parallel_for(range, [&](int i, int j, int k){
            STENCIL3_1D(dir);
            dep_field(IJK_) = sin( _two_pi * _f1 * weight * (ind_field(IJK_)+ind_field(IJK_M_)) )+ _offset;
            dep_field(IJK_) = (dep_field(IJK_) <= 0.0) ? _min_sq : _max_sq ;
          });
        }

        break;
      default:
        break;

    }
  }
}
#endif
