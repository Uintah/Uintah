#ifndef Uintah_Component_Arches_TaylorGreen3D_h
#define Uintah_Component_Arches_TaylorGreen3D_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>

namespace Uintah{

  template <typename T>
  class TaylorGreen3D : public TaskInterface {

public:

    TaylorGreen3D<T>( std::string task_name, int matl_index, const std::string var_name );
    ~TaylorGreen3D<T>();

    TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

    TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

    TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

    TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

    TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();

    void problemSetup( ProblemSpecP& db );

    //Build instructions for this (TaylorGreen3D) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index, std::string var_name ) :
        m_task_name(task_name), m_matl_index(matl_index), m_var_name(var_name){}
      ~Builder(){}

      TaylorGreen3D* build()
      { return scinew TaylorGreen3D<T>( m_task_name, m_matl_index, m_var_name ); }

      private:

      std::string m_task_name;
      int m_matl_index;
      std::string m_var_name;

    };

protected:

    typedef ArchesFieldContainer AFC;

    void register_initialize( std::vector<AFC::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_init( std::vector<AFC::VariableInformation>& variable_registry , const bool packed_tasks){}

    void register_timestep_eval( std::vector<AFC::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){}

    void register_compute_bcs( std::vector<AFC::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){};

    template <typename ExecSpace, typename MemSpace>
    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    template <typename ExecSpace, typename MemSpace>
    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    template <typename ExecSpace, typename MemSpace>
    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    void create_local_labels(){};

private:

    const std::string m_var_name;
    double m_c;
    double m_two_pi;

    std::string m_x_name;
    std::string m_y_name;
    std::string m_z_name;
    std::string m_which_vel;

  };

  //------------------------------------------------------------------------------------------------
  template <typename T>
  TaylorGreen3D<T>::TaylorGreen3D( std::string task_name, int matl_index, const std::string var_name ) :
  TaskInterface( task_name, matl_index ), m_var_name(var_name){

    m_two_pi = 2.0*acos(-1.0);

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  TaylorGreen3D<T>::~TaylorGreen3D()
  {}

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace TaylorGreen3D<T>::loadTaskComputeBCsFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace TaylorGreen3D<T>::loadTaskInitializeFunctionPointers()
  {
    return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                       , &TaylorGreen3D<T>::initialize<UINTAH_CPU_TAG>               // Task supports non-Kokkos builds
                                       //, &TaylorGreen3D<T>::initialize<KOKKOS_OPENMP_TAG>          // Task supports Kokkos::OpenMP builds
                                       //, &TaylorGreen3D<T>::initialize<KOKKOS_DEFAULT_HOST_TAG>    // Task supports Kokkos::DefaultHostExecutionSpace builds
                                       //, &TaylorGreen3D<T>::initialize<KOKKOS_DEFAULT_DEVICE_TAG>  // Task supports Kokkos::DefaultExecutionSpace builds
                                       //, &TaylorGreen3D<T>::initialize<KOKKOS_DEFAULT_DEVICE_TAG>            // Task supports Kokkos builds
                                       );
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace TaylorGreen3D<T>::loadTaskEvalFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace TaylorGreen3D<T>::loadTaskTimestepInitFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace TaylorGreen3D<T>::loadTaskRestartInitFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void TaylorGreen3D<T>::problemSetup( ProblemSpecP& db ){

    db->getWithDefault( "c", m_c, 2.0);
    db->require("which_vel", m_which_vel);
    ProblemSpecP db_coord = db->findBlock("coordinates");
    if ( db_coord ){
      db_coord->getAttribute("x", m_x_name);
      db_coord->getAttribute("y", m_y_name);
      db_coord->getAttribute("z", m_z_name);
    } else {
      throw InvalidValue("Error: must have coordinates specified for almgren MMS init condition", __FILE__, __LINE__);
    }

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void TaylorGreen3D<T>::register_initialize(
    std::vector<AFC::VariableInformation>& variable_registry,
    const bool packed_tasks ){

    register_variable( m_var_name,     AFC::MODIFIES, variable_registry );
    register_variable( m_x_name, AFC::REQUIRES, 0, AFC::NEWDW, variable_registry, m_task_name );
    register_variable( m_y_name, AFC::REQUIRES, 0, AFC::NEWDW, variable_registry, m_task_name );
    register_variable( m_z_name, AFC::REQUIRES, 0, AFC::NEWDW, variable_registry, m_task_name );

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  template <typename ExecSpace, typename MemSpace>
  void TaylorGreen3D<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

    T& var = tsk_info->get_field<T>( m_var_name );
    constCCVariable<double>& x = tsk_info->get_field<constCCVariable<double> >(m_x_name);
    constCCVariable<double>& y = tsk_info->get_field<constCCVariable<double> >(m_y_name);
    constCCVariable<double>& z = tsk_info->get_field<constCCVariable<double> >(m_z_name);

    Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex());

    if ( m_which_vel == "u" ){
      Uintah::parallel_for( range, [&](int i, int j, int k){
        var(i,j,k) = 2.0/sqrt(3.0) * sin( m_c + m_two_pi/3.0 )
                     * sin( m_two_pi * x(i,j,k) )
                     * cos( m_two_pi * y(i,j,k) )
                     * cos( m_two_pi * z(i,j,k) );
      });
    } else if ( m_which_vel == "v" ){
      Uintah::parallel_for( range, [&](int i, int j, int k){
        var(i,j,k) = 2.0/sqrt(3.0) * sin( m_c - m_two_pi/3.0 )
                     * sin( m_two_pi * y(i,j,k) )
                     * cos( m_two_pi * x(i,j,k) )
                     * cos( m_two_pi * z(i,j,k) );
      });
    } else {
      Uintah::parallel_for( range, [&](int i, int j, int k){
        var(i,j,k) = 2.0/sqrt(3.0) * sin(m_c )
                     * sin( m_two_pi * z(i,j,k) )
                     * cos( m_two_pi * x(i,j,k) )
                     * cos( m_two_pi * y(i,j,k) );
      });
    }


  }
}
#endif
