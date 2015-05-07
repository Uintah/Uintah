#ifndef Uintah_Component_Arches_CoalTemperatureNebo_h
#define Uintah_Component_Arches_CoalTemperatureNebo_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{ 

  class Operators; 
  class CoalTemperatureNebo : public TaskInterface { 

public: 

    CoalTemperatureNebo( std::string task_name, int matl_index ); 
    ~CoalTemperatureNebo(); 

    void problemSetup( ProblemSpecP& db ); 

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

    void create_local_labels(); 

    const std::string get_env_name( const int i, const std::string base_name ){ 
      std::stringstream out;
      std::string env;
      out << i;
      env = out.str();
      return base_name + "_" + env;
    }
               

    //Build instructions for this (CoalTemperatureNebo) class. 
    class Builder : public TaskInterface::TaskBuilder { 

      public: 

      Builder( std::string task_name, int matl_index ) : _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      CoalTemperatureNebo* build()
      { return scinew CoalTemperatureNebo( _task_name, _matl_index ); }

      private: 

      std::string _task_name; 
      int _matl_index; 

    };

private: 

    int _Nenv; 
    double _value; 
    double _rhop_o;
    double _pi; 
    double _initial_temperature; 
    double _Ha0; 
    double _Hc0; 
    double _Hh0; 
    double _Rgas; 
    double _RdC; 
    double _RdMW; 
    double _MW_avg; 

    std::vector<double> _init_ash;
    std::vector<double> _init_rawcoal;
    std::vector<double> _init_char;
    std::vector<double> _sizes;
    std::vector<double> _denom; 

    std::string _rawcoal_base_name; 
    std::string _char_base_name; 
    std::string _enthalpy_base_name; 
    std::string _dTdt_base_name; 

    struct CoalAnalysis{ 
      double C;
      double H; 
      double O; 
      double N; 
      double S; 
      double CHAR; 
      double ASH; 
      double H2O; 
    };




  
  };
}
#endif 
