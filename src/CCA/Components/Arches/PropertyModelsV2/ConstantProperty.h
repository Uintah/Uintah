#ifndef Uintah_Component_Arches_ConstantProperty_h
#define Uintah_Component_Arches_ConstantProperty_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/DiscretizationTools.h>

namespace Uintah{

  template <typename T>
  class ConstantProperty : public TaskInterface {

public:

    typedef std::vector<ArchesFieldContainer::VariableInformation> VIVec;

    ConstantProperty<T>( std::string task_name, int matl_index );
    ~ConstantProperty<T>();

    void problemSetup( ProblemSpecP& db );

    void create_local_labels(){
      register_new_variable<T>( _task_name );
    }

    void register_initialize( VIVec& variable_registry );

    void register_timestep_init( VIVec& variable_registry );

    void register_restart_initialize( VIVec& variable_registry ){};

    void register_timestep_eval( VIVec& variable_registry, const int time_substep ){};

    void register_compute_bcs( VIVec& variable_registry, const int time_substep ){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                      SpatialOps::OperatorDatabase& opr ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                     SpatialOps::OperatorDatabase& opr );

    void restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                             SpatialOps::OperatorDatabase& opr ){};

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                        SpatialOps::OperatorDatabase& opr );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
               SpatialOps::OperatorDatabase& opr ){};

    //Build instructions for this (ConstantProperty) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index )
        : _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      ConstantProperty* build()
      { return scinew ConstantProperty( _task_name, _matl_index ); }

      private:

      std::string _task_name;
      int _matl_index;

    };

private:

    typedef std::vector<ArchesFieldContainer::VariableInformation> AVarInfo;

    double m_constant;

  };

  //---------------------------------------------------------------------------------------
  //Function definitions:
  template <typename T>
  ConstantProperty<T>::ConstantProperty( std::string task_name, int matl_index ) :
  TaskInterface( task_name, matl_index ){
  }

  template <typename T>
  ConstantProperty<T>::~ConstantProperty(){}

  template <typename T>
  void ConstantProperty<T>::problemSetup( ProblemSpecP& db ){

    db->require("value", m_constant );

  }

  template <typename T>
  void ConstantProperty<T>::register_initialize( AVarInfo& variable_registry ){
    register_variable( _task_name, ArchesFieldContainer::COMPUTES, variable_registry );
  }

  template <typename T>
  void ConstantProperty<T>::initialize( const Patch*, ArchesTaskInfoManager* tsk_info,
                                SpatialOps::OperatorDatabase& opr ){

    T& property = *(tsk_info->get_uintah_field<T>( _task_name ));

    property.initialize(m_constant);

  }

  template <typename T>
  void ConstantProperty<T>::register_timestep_init( AVarInfo& variable_registry ){
    register_variable( _task_name, ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( _task_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );
  }

  template <typename T>
  void ConstantProperty<T>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                           SpatialOps::OperatorDatabase& opr ){

    typedef typename VariableHelper<T>::ConstType CT;
    T& property = *(tsk_info->get_uintah_field<T>( _task_name ));
    CT& old_property = *(tsk_info->get_const_uintah_field<CT>( _task_name ));

    property.copyData(old_property);

  }
}
#endif
