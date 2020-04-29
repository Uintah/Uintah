#ifndef Uintah_Component_Arches_ShunnMMS_h
#define Uintah_Component_Arches_ShunnMMS_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>

namespace Uintah{

  template <typename T>
  class ShunnMMS : public TaskInterface {

public:

    ShunnMMS<T>( std::string task_name, int matl_index, const std::string var_name );
    ~ShunnMMS<T>();

    TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

    TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

    TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

    TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

    TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();

    void problemSetup( ProblemSpecP& db );

    //Build instructions for this (ShunnMMS) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index, std::string var_name ) :
        m_task_name(task_name), m_matl_index(matl_index), m_var_name(var_name){}
      ~Builder(){}

      ShunnMMS* build()
      { return scinew ShunnMMS<T>( m_task_name, m_matl_index, m_var_name ); }

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

    const double m_pi = acos(-1.0);
    double m_k2 ;
    double m_k1 ;
    double m_w0 ;
    double m_rho0;
    double m_rho1;

    const std::string m_var_name;

    std::string m_x_name;
    std::string m_which_vel;
    std::string m_density_name;
    bool m_use_density;

  };

  //------------------------------------------------------------------------------------------------
  template <typename T>
  ShunnMMS<T>::ShunnMMS( std::string task_name, int matl_index, const std::string var_name ) :
  TaskInterface( task_name, matl_index ), m_var_name(var_name){


  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  ShunnMMS<T>::~ShunnMMS()
  {}

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace ShunnMMS<T>::loadTaskComputeBCsFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace ShunnMMS<T>::loadTaskInitializeFunctionPointers()
  {
    return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                       , &ShunnMMS<T>::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                       //, &ShunnMMS<T>::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                       //, &ShunnMMS<T>::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                       );
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace ShunnMMS<T>::loadTaskEvalFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace ShunnMMS<T>::loadTaskTimestepInitFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace ShunnMMS<T>::loadTaskRestartInitFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void ShunnMMS<T>::problemSetup( ProblemSpecP& db ){

    db->getWithDefault( "k1", m_k1, 4.0);
    db->getWithDefault( "k2", m_k2, 2.0);
    db->getWithDefault( "w0", m_w0, 50.0);
    db->getWithDefault( "rho0", m_rho0, 20.0);
    db->getWithDefault( "rho1", m_rho1, 1.0);

    db->require("which_vel", m_which_vel);

    m_use_density = false;
    if (db->findBlock("which_density")) {
      db->findBlock("which_density")->getAttribute("label", m_density_name);
      m_use_density = true;
    }

    ProblemSpecP db_coord = db->findBlock("grid");
    if ( db_coord ){
      db_coord->getAttribute("x", m_x_name);
    } else {
      throw InvalidValue("Error: must have coordinates specified for almgren MMS init condition", __FILE__, __LINE__);
    }

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void ShunnMMS<T>::register_initialize(
    std::vector<AFC::VariableInformation>& variable_registry,
    const bool packed_tasks ){

    register_variable( m_var_name,     AFC::MODIFIES, variable_registry );
    register_variable( m_x_name, AFC::REQUIRES, 0, AFC::NEWDW, variable_registry, m_task_name );

    if (m_use_density){
      register_variable( m_density_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, m_task_name );
    }
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  template <typename ExecSpace, typename MemSpace>
  void ShunnMMS<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

    T& f_mms = tsk_info->get_field<T>(m_var_name);
    constCCVariable<double>& x = tsk_info->get_field<constCCVariable<double> >(m_x_name);

    Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
    const double time_d = 0.0;
    const double k12 = m_k1-m_k2;
    //const double k21 = m_k2-m_k1;
    const double z1 = std::exp(-m_k1 * time_d);
    const double r01 = m_rho0 - m_rho1;

    if ( m_which_vel == "u" ){
    // for velocity
      constCCVariable<double>& density = tsk_info->get_field<constCCVariable<double> >(m_density_name);
      Uintah::parallel_for( range, [&](int i, int j, int k){
        //const double z2 = std::cosh (m_w0 * std::exp (-m_k2 * time_d) * x(i,j,k)); // x at face
        //const double phi_f = (z1-z2)/(z1 * (1.0 - m_rho0/m_rho1)-z2);
        const double u1  = std::exp(m_w0*std::exp(-m_k2*time_d)*x(i,j,k));
//        const double rho = 1.0/(phi_f/m_rho1 + (1.0- phi_f )/m_rho0);
        //I use density that was computed with same mms but at cc, becuase I want to get same x-mom as option rho_u
        const double rho = 0.5*(density(i-1,j,k) + density(i,j,k)); // only works for x. Fix me !!!

        f_mms(i,j,k) = (2.0*m_k2*x(i,j,k)*r01*std::exp(-m_k1*time_d)*u1/(u1*u1 + 1.0) +
          r01*k12*std::exp(-k12*time_d)/m_w0*(2.0*std::atan(u1)-m_pi/2.0))/rho;

    });
    } else if ( m_which_vel == "rho_u" ){
    // for velocity
      Uintah::parallel_for( range, [&](int i, int j, int k){
        const double u1  = std::exp(m_w0*std::exp(-m_k2*time_d)*x(i,j,k));
        //const double rho = 1.0/(phi_f/m_rho1 + (1.0- phi_f )/m_rho0);

        f_mms(i,j,k) = 2.0*m_k2*x(i,j,k)*r01*std::exp(-m_k1*time_d)*u1/(u1*u1 + 1.0) +
          r01*k12*std::exp(-k12*time_d)/m_w0*(2.0*std::atan(u1)-m_pi/2.0);

    });

    } else {
    // for scalar
      Uintah::parallel_for( range, [&](int i, int j, int k){
        const double z2 = std::cosh(m_w0 * std::exp (-m_k2 * time_d) * x(i,j,k)); // x is cc value
        f_mms(i,j,k) = (z1-z2)/(z1 * (1.0 - m_rho0/m_rho1)-z2);
    });
    }
  }
}
#endif
