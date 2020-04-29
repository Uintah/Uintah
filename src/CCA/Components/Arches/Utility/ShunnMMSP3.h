#ifndef Uintah_Component_Arches_ShunnMMSP3_h
#define Uintah_Component_Arches_ShunnMMSP3_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>

namespace Uintah{

  template <typename T>
  class ShunnMMSP3 : public TaskInterface {

public:

    ShunnMMSP3<T>( std::string task_name, int matl_index, const std::string var_name );
    ~ShunnMMSP3<T>();

    TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

    TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

    TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

    TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

    TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();

    void problemSetup( ProblemSpecP& db );

    //Build instructions for this (ShunnMMSP3) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index, std::string var_name ) :
        m_task_name(task_name), m_matl_index(matl_index), m_var_name(var_name){}
      ~Builder(){}

      ShunnMMSP3* build()
      { return scinew ShunnMMSP3<T>( m_task_name, m_matl_index, m_var_name ); }

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
    double m_k;
    double m_w0 ;
    double m_rho0;
    double m_rho1;
    double m_uf;
    double m_vf;

    std::vector<int> m_ijk_off;
    const std::string m_var_name;

    std::string m_x_name;
    std::string m_y_name;
    std::string m_which_vel;
    std::string m_density_name;
    int m_dir;
    bool m_use_density;

  };

  //------------------------------------------------------------------------------------------------
  template <typename T>
  ShunnMMSP3<T>::ShunnMMSP3( std::string task_name, int matl_index, const std::string var_name ) :
  TaskInterface( task_name, matl_index ), m_var_name(var_name){

  ArchesCore::VariableHelper<T> helper;
  m_ijk_off.push_back(0);
  m_ijk_off.push_back(0);
  m_ijk_off.push_back(0);


  if ( helper.dir == ArchesCore::XDIR ||
       helper.dir == ArchesCore::YDIR ||
       helper.dir == ArchesCore::ZDIR ){
       m_dir = helper.dir;
       m_ijk_off[0] = helper.ioff;
       m_ijk_off[1] = helper.joff;
       m_ijk_off[2] = helper.koff;
  }

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  ShunnMMSP3<T>::~ShunnMMSP3()
  {}

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace ShunnMMSP3<T>::loadTaskComputeBCsFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace ShunnMMSP3<T>::loadTaskInitializeFunctionPointers()
  {
    return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                       , &ShunnMMSP3<T>::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                       //, &ShunnMMSP3<T>::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                       //, &ShunnMMSP3<T>::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                       );
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace ShunnMMSP3<T>::loadTaskEvalFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace ShunnMMSP3<T>::loadTaskTimestepInitFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace ShunnMMSP3<T>::loadTaskRestartInitFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void ShunnMMSP3<T>::problemSetup( ProblemSpecP& db ){

    // Going to grab density from the cold flow properties list.
    // Note that the original Shunn paper mapped the fuel density (f=1) to rho1 and the air density (f=0)
    // to rho0. This is opposite of what Arches traditionally has done. So, in this file, we stick to
    // the Shunn notation but remap the density names for convienence.

    //NOTE: We are going to assume that the property the code is looking for is called "density"
    //      (as specified by the user)
    ProblemSpecP db_prop = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("StateProperties");
    bool found_coldflow_density = false;

    for ( ProblemSpecP db_p = db_prop->findBlock("model");
          db_p.get_rep() != nullptr;
          db_p = db_p->findNextBlock("model")){

      std::string label;
      std::string type;

      db_p->getAttribute("label", label);
      db_p->getAttribute("type", type);

      if ( type == "coldflow" ){

        for ( ProblemSpecP db_cf = db_p->findBlock("property");
              db_cf.get_rep() != nullptr;
              db_cf = db_cf->findNextBlock("property") ){

          std::string label;
          double value0;
          double value1;

          db_cf->getAttribute("label", label);

          if ( label == "density" ){
            db_cf->getAttribute("stream_0", value0);
            db_cf->getAttribute("stream_1", value1);

            found_coldflow_density = true;

            //NOTICE: We are inverting the mapping here. See note above.
            m_rho0 = value1;
            m_rho1 = value0;

          }
        }
      }
    }

    if ( !found_coldflow_density ){
      throw InvalidValue("Error: Cold flow property specification wasnt found which is needed to use the ShunnP3 initial condition.", __FILE__, __LINE__);
    }

    db->getWithDefault( "k", m_k, 2.0);
    db->getWithDefault( "w0", m_w0, 2.0);
    db->getWithDefault( "uf", m_uf, 0.0);
    db->getWithDefault( "vf", m_vf, 0.0);

    db->require("which_vel", m_which_vel);
    m_use_density = false;
    if (db->findBlock("which_density")) {
      db->findBlock("which_density")->getAttribute("label", m_density_name);
      m_use_density = true;
    }
    ProblemSpecP db_coord = db->findBlock("coordinates");
    if ( db_coord ){
      db_coord->getAttribute("x", m_x_name);
      db_coord->getAttribute("y", m_y_name);
    } else {
      throw InvalidValue("Error: must have coordinates specified for almgren MMS init condition", __FILE__, __LINE__);
    }

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void ShunnMMSP3<T>::register_initialize(
    std::vector<AFC::VariableInformation>& variable_registry,
    const bool packed_tasks ){

    register_variable( m_var_name,     AFC::MODIFIES, variable_registry );
    register_variable( m_x_name, AFC::REQUIRES, 0, AFC::NEWDW, variable_registry, m_task_name );
    register_variable( m_y_name, AFC::REQUIRES, 0, AFC::NEWDW, variable_registry, m_task_name );
    if (m_use_density){
      register_variable( m_density_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, m_task_name );
    }

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  template <typename ExecSpace, typename MemSpace>
  void ShunnMMSP3<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

    T& f_mms = tsk_info->get_field<T>(m_var_name);
    constCCVariable<double>& x = tsk_info->get_field<constCCVariable<double> >(m_x_name);
    constCCVariable<double>& y = tsk_info->get_field<constCCVariable<double> >(m_y_name);

    Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
    const double time_d = 0.0;

    if ( m_which_vel == "u" ){
      constCCVariable<double>& rho = tsk_info->get_field<constCCVariable<double> >(m_density_name);
    // for velocity
      const int ioff = m_ijk_off[0];
      const int joff = m_ijk_off[1];
      const int koff = m_ijk_off[2];
      Uintah::parallel_for( range, [&, ioff, joff, koff](int i, int j, int k){
        //const double phi_f = (1.0 + sin(m_k*m_pi*(x(i,j,k)-m_uf*time_d))*
        //                sin(m_k*m_pi*(y(i,j,k)-m_vf*time_d))*cos(m_w0*m_pi*time_d))/(1.0 +
        //                m_rho0/m_rho1+(1.0-m_rho0/m_rho1)*sin(m_k*m_pi*(x(i,j,k)-m_uf*time_d))*
        //                sin(m_k*m_pi*(y(i,j,k)-m_vf*time_d))*cos(m_w0*m_pi*time_d));
        //const double rho_d = 1.0/(phi_f/m_rho1 + (1.0- phi_f )/m_rho0);
        const double rho_inter = 0.5 * (rho(i,j,k)+rho(i-ioff,j-joff,k-koff));

        //const double f_resolution = (rho_d/rho_inter); // Only for testing liner interpolation of density
        const double f_resolution = 1.0;
        f_mms(i,j,k) = m_uf*f_resolution  - m_w0/m_k/4.0*cos(m_k*m_pi*(x(i,j,k)-m_uf*time_d))*sin(m_k*m_pi*(y(i,j,k)-m_vf*time_d))*sin(m_w0*m_pi*time_d)*(m_rho1-m_rho0)/rho_inter;
      });
    } else if ( m_which_vel == "rho_u" ){
    // for velocity
      Uintah::parallel_for( range, [&](int i, int j, int k){
        const double phi_f = (1.0 + sin(m_k*m_pi*(x(i,j,k)-m_uf*time_d))*
                        sin(m_k*m_pi*(y(i,j,k)-m_vf*time_d))*cos(m_w0*m_pi*time_d))/(1.0 +
                        m_rho0/m_rho1+(1.0-m_rho0/m_rho1)*sin(m_k*m_pi*(x(i,j,k)-m_uf*time_d))*
                        sin(m_k*m_pi*(y(i,j,k)-m_vf*time_d))*cos(m_w0*m_pi*time_d));

        const double rho = 1.0/(phi_f/m_rho1 + (1.0- phi_f )/m_rho0);
        f_mms(i,j,k) = m_uf*rho -m_w0/m_k/4.0*cos(m_k*m_pi*(x(i,j,k)-m_uf*time_d))*sin(m_k*m_pi*(y(i,j,k)-m_vf*time_d))*sin(m_w0*m_pi*time_d)*(m_rho1-m_rho0);
    });

    } else {
    // for scalar
      Uintah::parallel_for( range, [&](int i, int j, int k){
        f_mms(i,j,k) = (1.0 + sin(m_k*m_pi*(x(i,j,k)-m_uf*time_d))*
                        sin(m_k*m_pi*(y(i,j,k)-m_vf*time_d))*cos(m_w0*m_pi*time_d))/(1.0 +
                        m_rho0/m_rho1+(1.0-m_rho0/m_rho1)*sin(m_k*m_pi*(x(i,j,k)-m_uf*time_d))*
                        sin(m_k*m_pi*(y(i,j,k)-m_vf*time_d))*cos(m_w0*m_pi*time_d));
    });
    }
  }
}
#endif
