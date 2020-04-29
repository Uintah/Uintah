#ifndef Uintah_Component_Arches_MMS_Shunn_h
#define Uintah_Component_Arches_MMS_Shunn_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <Core/Grid/MaterialManager.h>

namespace Uintah{

  template <typename T>
  class MMS_Shunn : public TaskInterface {

public:

    MMS_Shunn<T>( std::string task_name, int matl_index, MaterialManagerP materialManager  );
    ~MMS_Shunn<T>();

    TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

    TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

    TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

    TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

    TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();

    void problemSetup( ProblemSpecP& db );

    //Build instructions for this (MMS_Shunn) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index, MaterialManagerP materialManager ) :
        m_task_name(task_name), m_matl_index(matl_index), _materialManager(materialManager){}
      ~Builder(){}

      MMS_Shunn* build()
      { return scinew MMS_Shunn<T>( m_task_name, m_matl_index, _materialManager  ); }

      private:

      std::string m_task_name;
      int m_matl_index;

      MaterialManagerP _materialManager;
    };

 protected:

    typedef ArchesFieldContainer::VariableInformation VarInfo;

    void register_initialize( std::vector<VarInfo>& variable_registry, const bool pack_tasks);

    void register_timestep_init( std::vector<VarInfo>& variable_registry, const bool pack_tasks);

    void register_timestep_eval( std::vector<VarInfo>& variable_registry, const int time_substep,
                                 const bool pack_tasks);

    void register_compute_bcs( std::vector<VarInfo>& variable_registry, const int time_substep,
                               const bool pack_tasks){}

    template <typename ExecSpace, typename MemSpace>
    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    template <typename ExecSpace, typename MemSpace>
    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    void create_local_labels();

private:

    const double m_pi = acos(-1.0);
    double m_k2 ;
    double m_k1 ;
    double m_w0 ;
    double m_rho0; 
    double m_rho1;
    double m_D;

    std::string m_x_face_name;
    std::string m_x_name;

    std::string m_MMS_label;
    std::string m_rho_u_label;
    std::string m_MMS_scalar_label;
    std::string m_MMS_source_label;
    std::string m_MMS_source_scalar_label;
    std::string m_MMS_rho_scalar_label;
    std::string m_MMS_rho_label;
    std::string m_MMS_rho_face_label;
    std::string m_MMS_drhodt_label;
    
    //std::string m_MMS_source_diff_label;
    //std::string m_MMS_source_t_label;

    MaterialManagerP _materialManager;

    void compute_source( const Patch* patch, ArchesTaskInfoManager* tsk_info );

  };

//--------------------------------------------------------------------------------------------------
template <typename T>
MMS_Shunn<T>::MMS_Shunn( std::string task_name, int matl_index, MaterialManagerP materialManager ) :
TaskInterface( task_name, matl_index ) , _materialManager(materialManager){

}

//--------------------------------------------------------------------------------------------------
template <typename T>
MMS_Shunn<T>::~MMS_Shunn(){
}

//--------------------------------------------------------------------------------------------------
template <typename T>
TaskAssignedExecutionSpace MMS_Shunn<T>::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
template <typename T>
TaskAssignedExecutionSpace MMS_Shunn<T>::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &MMS_Shunn<T>::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &MMS_Shunn<T>::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &MMS_Shunn<T>::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
template <typename T>
TaskAssignedExecutionSpace MMS_Shunn<T>::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &MMS_Shunn<T>::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &MMS_Shunn<T>::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &MMS_Shunn<T>::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
template <typename T>
TaskAssignedExecutionSpace MMS_Shunn<T>::loadTaskTimestepInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
template <typename T>
TaskAssignedExecutionSpace MMS_Shunn<T>::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_Shunn<T>::problemSetup( ProblemSpecP& db ){

  db->getWithDefault( "k1", m_k1, 4.0);
  db->getWithDefault( "k2", m_k2, 2.0);
  db->getWithDefault( "w0", m_w0, 50.0);
  db->getWithDefault( "rho0", m_rho0, 20.0);
  db->getWithDefault( "rho1", m_rho1, 1.0);
  db->getWithDefault( "D", m_D, 0.03);
    
    
  ProblemSpecP db_coord = db->findBlock("grid");
  if ( db_coord ){
    db_coord->getAttribute("x", m_x_name);
    db_coord->getAttribute("x_face", m_x_face_name);
  } else {
    throw InvalidValue("Error: must have coordinates specified for shunn MMS init condition",
      __FILE__, __LINE__);
  }

  m_MMS_label             = m_task_name;
  m_rho_u_label           = m_task_name + "_rho_u";
  m_MMS_scalar_label      = m_task_name+"_scalar";
  m_MMS_source_label      = m_task_name + "_source";
  m_MMS_source_scalar_label = m_task_name + "_source_scalar";
  
  m_MMS_rho_scalar_label   = m_task_name+"_rho_scalar";
  m_MMS_rho_label          = m_task_name+"_rho";
  m_MMS_rho_face_label          = m_task_name+"_rho_face";
  m_MMS_drhodt_label       = m_task_name+"_drhodt";
  

  //m_MMS_source_diff_label = m_task_name + "_source_diff";
  //m_MMS_source_t_label    = m_task_name + "_source_time";

}

//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_Shunn<T>::create_local_labels(){

  register_new_variable< T >( m_MMS_label);
  register_new_variable< T >( m_rho_u_label);
  register_new_variable< CCVariable<double> >( m_MMS_scalar_label);
  register_new_variable< CCVariable<double> >( m_MMS_source_scalar_label);
  register_new_variable< CCVariable<double> >( m_MMS_drhodt_label);
  register_new_variable< T >( m_MMS_source_label);
 // register_new_variable< T >( m_MMS_source_diff_label);
  //register_new_variable< T >( m_MMS_source_t_label);
  register_new_variable< CCVariable<double> >( m_MMS_rho_scalar_label);
  register_new_variable< CCVariable<double> >( m_MMS_rho_label);
  register_new_variable< T >( m_MMS_rho_face_label);

}

//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_Shunn<T>::register_initialize( std::vector<VarInfo>&
                                      variable_registry , const bool pack_tasks){

  register_variable( m_MMS_label,               ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_rho_u_label,             ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_MMS_scalar_label,        ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_MMS_source_label,        ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_MMS_source_scalar_label, ArchesFieldContainer::COMPUTES, variable_registry );
  //register_variable( m_MMS_source_diff_label, ArchesFieldContainer::COMPUTES, variable_registry );
  //register_variable( m_MMS_source_t_label,    ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_MMS_rho_scalar_label,    ArchesFieldContainer::COMPUTES ,  variable_registry);
  register_variable( m_MMS_rho_label,           ArchesFieldContainer::COMPUTES ,  variable_registry );
  register_variable( m_MMS_rho_face_label,      ArchesFieldContainer::COMPUTES ,  variable_registry);
  register_variable( m_MMS_drhodt_label,        ArchesFieldContainer::COMPUTES ,  variable_registry );

  register_variable( m_x_face_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry );
  register_variable( m_x_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry );
  //register_variable( m_y_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry );

}

//--------------------------------------------------------------------------------------------------
template <typename T>
template <typename ExecSpace, typename MemSpace>
void MMS_Shunn<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  compute_source(patch, tsk_info);

}
//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_Shunn<T>::register_timestep_init( std::vector<VarInfo>&
                                          variable_registry , const bool pack_tasks){
}

//--------------------------------------------------------------------------------------------------
template <typename T>
template <typename ExecSpace, typename MemSpace>
void MMS_Shunn<T>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

}

//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_Shunn<T>::register_timestep_eval( std::vector<VarInfo>&
                                          variable_registry, const int time_substep , const bool pack_tasks){

  register_variable( m_MMS_label,               ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  register_variable( m_rho_u_label,             ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  register_variable( m_MMS_scalar_label,        ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  register_variable( m_MMS_source_label,        ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  register_variable( m_MMS_source_scalar_label, ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  register_variable( m_MMS_rho_scalar_label,    ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  register_variable( m_MMS_rho_label,           ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  register_variable( m_MMS_rho_face_label,      ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  register_variable( m_MMS_drhodt_label,        ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );

 // register_variable( m_MMS_source_diff_label, ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
 // register_variable( m_MMS_source_t_label,    ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );

  register_variable( m_x_face_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  register_variable( m_x_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

  //register_variable( m_y_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

}

//--------------------------------------------------------------------------------------------------
template <typename T>
template <typename ExecSpace, typename MemSpace>
void MMS_Shunn<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  compute_source( patch, tsk_info );

}

template <typename T>
void MMS_Shunn<T>::compute_source( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  T& u_mms = tsk_info->get_field<T>(m_MMS_label);
  T& rho_u_mms = tsk_info->get_field<T>(m_rho_u_label);
  CCVariable<double>& phi_mms = tsk_info->get_field<CCVariable<double> >(m_MMS_scalar_label);
  CCVariable<double>& rho_phi_mms = tsk_info->get_field<CCVariable<double> >(m_MMS_rho_scalar_label);
  CCVariable<double>& rho_mms = tsk_info->get_field<CCVariable<double> >(m_MMS_rho_label);
  T& rho_face_mms = tsk_info->get_field<T>(m_MMS_rho_face_label);

  CCVariable<double>& phi_source_mms = tsk_info->get_field<CCVariable<double> >(m_MMS_source_scalar_label);
  CCVariable<double>& drhodt_mms = tsk_info->get_field<CCVariable<double> >(m_MMS_drhodt_label);

  
  //T& s_mms = tsk_info->get_field<T>(m_MMS_source_label);// convection source term u
  //T& s_diff_mms = tsk_info->get_field<T>(m_MMS_source_diff_label);// diffusion source term u
  //T& s_t_mms = tsk_info->get_field<T>(m_MMS_source_t_label);// time source term u
  

  constCCVariable<double>& x_f = tsk_info->get_field<constCCVariable<double> >(m_x_face_name);
  constCCVariable<double>& x = tsk_info->get_field<constCCVariable<double> >(m_x_name);
  
  //constCCVariable<double>& y = tsk_info->get_field<constCCVariable<double> >(m_y_name);

  double time_d      = tsk_info->get_time(); //_materialManager->getElapsedSimTime();
  int   time_substep = tsk_info->get_time_substep();
  double factor      = tsk_info->get_ssp_time_factor(time_substep);
  double dt          = tsk_info->get_dt();
  time_d = time_d + factor*dt;

  const double k12 = m_k1-m_k2;
  //const double k21 = m_k2-m_k1;
  const double z1 = std::exp(-m_k1 * time_d);
  const double r01 = m_rho0 - m_rho1; 
 
    
  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );

    Uintah::parallel_for( range, [&](int i, int j, int k){
      const double z2 = std::cosh (m_w0 * std::exp (-m_k2 * time_d) * x_f(i,j,k)); // x at face
      const double phi_f = (z1-z2)/(z1 * (1.0 - m_rho0/m_rho1)-z2);
      const double u1  = std::exp(m_w0*std::exp(-m_k2*time_d)*x_f(i,j,k));
      const double rho = 1.0/(phi_f/m_rho1 + (1.0- phi_f )/m_rho0); 
      
      u_mms(i,j,k) = (2.0*m_k2*x_f(i,j,k)*r01*std::exp(-m_k1*time_d)*u1/(u1*u1 + 1.0) + 
        r01*k12*std::exp(-k12*time_d)/m_w0*(2.0*std::atan(u1)-m_pi/2.0))/rho; 
      rho_u_mms(i,j,k) = u_mms(i,j,k)* rho;
      rho_face_mms(i,j,k) = rho; // density at faces  
      
    });
          
    Uintah::parallel_for( range, [&](int i, int j, int k){
      const double z2 = std::cosh(m_w0 * std::exp (-m_k2 * time_d) * x(i,j,k)); // x is cc value
      phi_mms(i,j,k) = (z1-z2)/(z1 * (1.0 - m_rho0/m_rho1)-z2);
      rho_mms(i,j,k) = 1.0/(phi_mms(i,j,k)/m_rho1 + (1.0- phi_mms(i,j,k) )/m_rho0); 
      rho_phi_mms(i,j,k) = phi_mms(i,j,k)*rho_mms(i,j,k);

    });


    
    double s0 = std::exp(-m_k1*time_d);
    double s1 = std::exp(-m_k2*time_d);
    const double s5 = s0/s1;

    Uintah::parallel_for( range, [&](int i, int j, int k){
    
      const double s2 = 1.0/std::cosh(m_w0*s1*x(i,j,k));
      const double s3 = std::tanh(m_w0*s1*x(i,j,k));
      const double s4 = std::exp(m_w0*s1*x(i,j,k));
      const double s6 = s0*s2;
      const double s7 = m_rho1 + r01*s6;
      const double s8 = 2.0*std::atan(s4)-1.0/2.0*m_pi;
      const double s9 = s6*s3*m_w0;
      const double s10 =-s9*s1*m_rho1 + s9*s1*m_rho0;
      const double s11 = 1.0-s3*s3;
      const double s12 = s4*s4 + 1.0;
      const double s13 = -1.0 + s6;
      const double s14 = -m_k1*s6 + s9*m_k2*s1*x(i,j,k);
      const double s15 = s6 * s3 *s3 * m_w0 *m_w0 * s1 *s1;
      const double s16 = 2.0 * m_k2 * r01 * s0 * s4/s12;
      const double s17 = s6 * s11 * m_w0 *m_w0 * s1 *s1;
      const double s18 = s16 * x(i,j,k) + r01 * k12 * s5/m_w0 * s8;
      const double s19 = s9 * s1 * s7-s13 * s10;
      const double s20 = 2.0 * s4 * s1 * (s16 * x(i,j,k) * s4 * m_w0-r01 * k12 * s5)/s12;
      phi_source_mms(i,j,k) = -s14*m_rho1-(s16 + s16 * x(i,j,k) * m_w0 * s1-s20)*
        s13 * m_rho1/s7 + s18 * m_rho1 * s19/(s7 *s7) - m_D * m_rho1 * (-s15 * s7*s7+ s17 * s7 *s7 +
        2.0 * s9 * s1 * s10 * s7-2.0 * s13 * s10 *s10-s13 * r01 * s7 * s17 + s13 * r01 * s7 * s15)/(s7*s7*s7);
    });
    // Because I need drhodt at t+dt
    s0 = std::exp(-m_k1*(time_d+dt));
    s1 = std::exp(-m_k2*(time_d+dt));

    Uintah::parallel_for( range, [&](int i, int j, int k){
      const double s2 = std::sinh(m_w0*x(i,j,k)*s1);
      const double s3 = std::cosh(m_w0*x(i,j,k)*s1);
      const double s4 = s3 + s0*(m_rho0/m_rho1 - 1.0);
      const double s6 = s0*(m_rho0/m_rho1 - 1.0);
      const double s5 = ((s3 - s0)/(s3 + s6) - 1.0)/m_rho0-(s3 - s0)/(m_rho1*(s3 + s6));
      const double s7 = m_k2*m_w0*x(i,j,k)*s1*s2;

      drhodt_mms(i,j,k) = -((m_k1*s0 - s7)/(m_rho1*(s3 + s6)) - ((m_k1*s0 - s7)/(s3 + s6) +
        ((m_k1*s6 + s7)*(s3 - s0))/(s4*s4))/m_rho0 +
        ((m_k1*s6 + s7)*(s3 - s0))/(m_rho1*s4*s4))/(s5*s5);

    });
 }


}

#endif
