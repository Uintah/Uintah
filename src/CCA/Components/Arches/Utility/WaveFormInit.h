#ifndef Uintah_Component_Arches_WaveFormInit_h
#define Uintah_Component_Arches_WaveFormInit_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <spatialops/structured/FVStaggered.h>

namespace Uintah{ 

  template <typename T>
  class WaveFormInit : public TaskInterface { 

public: 

    enum WAVE_TYPE { SINE, SQUARE, SINECOS };

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
      std::string _var_name; 
      int _matl_index; 


    };

protected: 

    void register_initialize( std::vector<VariableInformation>& variable_registry );

    void register_timestep_init( std::vector<VariableInformation>& variable_registry ){} 

    void register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ){} 

    void initialize( const Patch* patch, FieldCollector* field_collector, 
                     SpatialOps::OperatorDatabase& opr );
    
    void timestep_init( const Patch* patch, FieldCollector* field_collector, 
                        SpatialOps::OperatorDatabase& opr ){}

    void eval( const Patch* patch, FieldCollector* field_collector, 
               SpatialOps::OperatorDatabase& opr ){}

private:

    const std::string _var_name; 
    std::string _dir; 
    WAVE_TYPE _wtype; 
    double _amp; 
    double _two_pi; 
    double _f1, _f2; 
    double _A, _B; 
    double _offset; 
    double _min_sq; 
    double _max_sq; 
  
  };

  //Function definitions: 

  template <typename T>
  WaveFormInit<T>::WaveFormInit( std::string task_name, int matl_index, const std::string var_name ) : 
  _var_name(var_name), TaskInterface( task_name, matl_index ){

    // This needs to be done to set the variable type 
    // for this function. All templated tasks should do this. 
    set_task_type<T>(); 

    _two_pi = 2.0*acos(-1.0);
  
  }

  template <typename T>
  WaveFormInit<T>::~WaveFormInit()
  {}

  template <typename T>
  void WaveFormInit<T>::problemSetup( ProblemSpecP& db ){ 

    std::string wave_type; 
    db->findBlock("wave")->getAttribute("type",wave_type); 
    db->findBlock("wave")->findBlock("direction")->getAttribute("value",_dir); 
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

    } else if ( wave_type == "sinecos"){ 
    
      ProblemSpecP db_sinecos= db->findBlock("wave")->findBlock("sinecos"); 

      _wtype = SINECOS; 
      db_sinecos->getAttribute("A",_A); 
      db_sinecos->getAttribute("B",_B); 
      db_sinecos->getAttribute("f1",_f1 ); 
      db_sinecos->getAttribute("f2",_f2 ); 
      db_sinecos->getAttribute("offset",_offset); 

    } else { 

      throw InvalidValue("Error: Wave type not recognized.",__FILE__,__LINE__);

    }

  }

  template <typename T>
  void WaveFormInit<T>::register_initialize( std::vector<VariableInformation>& variable_registry ){ 

    //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
    register_variable( "gridX",             CC_DOUBLE, REQUIRES,       0, NEWDW,  variable_registry ); 
    register_variable( "gridY",             CC_DOUBLE, REQUIRES,       0, NEWDW,  variable_registry ); 
    register_variable( "gridZ",             CC_DOUBLE, REQUIRES,       0, NEWDW,  variable_registry ); 
    register_variable( _var_name,           _mytype,   MODIFIES,       0, NEWDW,  variable_registry );
  
  }
  
  //This is the work for the task.  First, get the variables. Second, do the work! 
  template <typename T> 
  void WaveFormInit<T>::initialize( const Patch* patch, FieldCollector* field_collector, 
                                    SpatialOps::OperatorDatabase& opr ){ 

    using namespace SpatialOps;
    using SpatialOps::operator *; 
    typedef SpatialOps::SVolField   SVolF;
    typedef typename OperatorTypeBuilder< SpatialOps::Interpolant, SVolF, T >::type InterpT;

    const InterpT* const interp = opr.retrieve_operator<InterpT>();

    T* const field = field_collector->get_so_field<T>( _var_name, NEWDW ); 

    const SVolF* const x = field_collector->get_so_field<SVolF>( "gridX", NEWDW ); 
    const SVolF* const y = field_collector->get_so_field<SVolF>( "gridY", NEWDW ); 
    const SVolF* const z = field_collector->get_so_field<SVolF>( "gridZ", NEWDW ); 
    IntVector c = IntVector(1,1,1); 

    switch (_wtype){ 
      case SINE:
        
        if ( _dir == "x"){
          *field <<= _A*sin( _two_pi * _f1 * (*interp)( *x )) + _offset; 
        } else if ( _dir == "y" ){ 
          *field <<= _A*sin( _two_pi * _f1 * (*interp)( *y )) + _offset; 
        } else { 
          *field <<= _A*sin( _two_pi * _f1 * (*interp)( *z )) + _offset; 
        }

        break; 
      case SQUARE: 

        if ( _dir == "x"){
          *field <<= sin( _two_pi * _f1 * (*interp)( *x )) + _offset; 
        } else if ( _dir == "y" ){ 
          *field <<= sin( _two_pi * _f1 * (*interp)( *y )) + _offset; 
        } else { 
          *field <<= sin( _two_pi * _f1 * (*interp)( *z )) + _offset; 
        }

        *field <<= cond( *field < 0.0, _min_sq )
                       ( *field > 0.0, _max_sq )
                       ( 0.0 ); 

        break; 

      case SINECOS:

        if ( _dir == "x"){ 
          *field <<= _A*sin(_two_pi * _f1 * (*interp)(*x)) + _B*cos(_two_pi * _f2 * (*interp)(*x)); 
        } else if ( _dir == "y" ){
          *field <<= _A*sin(_two_pi * _f1 * (*interp)(*y)) + _B*cos(_two_pi * _f2 * (*interp)(*y)); 
        } else {
          *field <<= _A*sin(_two_pi * _f1 * (*interp)(*z)) + _B*cos(_two_pi * _f2 * (*interp)(*z)); 
        }

        break;

      default:
        break;

    } 

  }

}
#endif 
