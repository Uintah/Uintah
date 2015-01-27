#ifndef Uintah_Component_Arches_TotNumDensity_h
#define Uintah_Component_Arches_TotNumDensity_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{ 

  class Operators; 
  class TotNumDensity : public TaskInterface { 

public: 

    TotNumDensity( std::string task_name, int matl_index ); 
    ~TotNumDensity(); 

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
               

    //Build instructions for this (TotNumDensity) class. 
    class Builder : public TaskInterface::TaskBuilder { 

      public: 

      Builder( std::string task_name, int matl_index ) : _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      TotNumDensity* build()
      { return scinew TotNumDensity( _task_name, _matl_index ); }

      private: 

      std::string _task_name; 
      int _matl_index; 

    };

private: 

    int _Nenv; 
    double _value; 
    double _rhop_o;
    double _pi; 

    std::vector<double> _init_ash;
    std::vector<double> _init_rawcoal;
    std::vector<double> _init_char;
    std::vector<double> _sizes;
    std::vector<double> _denom; 

    std::string _rawcoal_base_name; 
    std::string _char_base_name; 

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
