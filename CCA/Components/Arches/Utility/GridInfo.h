#ifndef Uintah_Component_Arches_GridInfo_h
#define Uintah_Component_Arches_GridInfo_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{ 

  class Operators; 
  class GridInfo : public TaskInterface { 

public: 

    GridInfo( std::string task_name, int matl_index ); 
    ~GridInfo(); 

    void problemSetup( ProblemSpecP& db ); 

    void register_initialize( std::vector<VariableInformation>& variable_registry );

    void register_timestep_init( std::vector<VariableInformation>& variable_registry ); 

    void register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ); 

    void initialize( const Patch* patch, FieldCollector* field_collector, 
                     SpatialOps::OperatorDatabase& opr );
    
    void timestep_init( const Patch* patch, FieldCollector* field_collector, 
                        SpatialOps::OperatorDatabase& opr );

    void eval( const Patch* patch, FieldCollector* field_collector, 
               SpatialOps::OperatorDatabase& opr );
               

    //Build instructions for this (GridInfo) class. 
    class Builder : public TaskInterface::TaskBuilder { 

      public: 

      Builder( std::string task_name, int matl_index ) : _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      GridInfo* build()
      { return scinew GridInfo( _task_name, _matl_index ); }

      private: 

      std::string _task_name; 
      int _matl_index; 

    };

private: 

    double _value; 
  
  };
}
#endif 
