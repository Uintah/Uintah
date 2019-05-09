#ifndef Uintah_Component_Arches_MMS_mom_h
#define Uintah_Component_Arches_MMS_mom_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

  template <typename T>
  class MMS_mom : public TaskInterface {

public:

    MMS_mom<T>( std::string task_name, int matl_index, MaterialManagerP materialManager  );
    ~MMS_mom<T>();

    void problemSetup( ProblemSpecP& db );

    //Build instructions for this (MMS_mom) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index, MaterialManagerP materialManager ) :
        m_task_name(task_name), m_matl_index(matl_index), _materialManager(materialManager){}
      ~Builder(){}

      MMS_mom* build()
      { return scinew MMS_mom<T>( m_task_name, m_matl_index, _materialManager  ); }

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

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();

private:

    double m_amp;
    double m_freq;
    double m_two_pi = 2*acos(-1.0);

    std::string m_x_name;
    std::string m_y_name;
    std::string m_which_vel;

    std::string m_MMS_label;
    std::string m_MMS_source_label;
    std::string m_MMS_source_diff_label;
    std::string m_MMS_source_t_label;

    MaterialManagerP _materialManager;

    void compute_source( const Patch* patch, ArchesTaskInfoManager* tsk_info );

  };

//--------------------------------------------------------------------------------------------------
template <typename T>
MMS_mom<T>::MMS_mom( std::string task_name, int matl_index, MaterialManagerP materialManager ) :
TaskInterface( task_name, matl_index ) , _materialManager(materialManager){

}

//--------------------------------------------------------------------------------------------------
template <typename T>
MMS_mom<T>::~MMS_mom(){
}

//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_mom<T>::problemSetup( ProblemSpecP& db ){
    std::string wave_type;

    db->getWithDefault( "amplitude", m_amp, 1.0);
    db->getWithDefault( "frequency", m_freq, 1.0);
    db->require("which_vel", m_which_vel);
    ProblemSpecP db_coord = db->findBlock("coordinates");
    if ( db_coord ){
      db_coord->getAttribute("x", m_x_name);
      db_coord->getAttribute("y", m_y_name);
    } else {
      throw InvalidValue("Error: must have coordinates specified for almgren MMS init condition",
        __FILE__, __LINE__);
    }

  m_MMS_label             = m_task_name;
  m_MMS_source_label      = m_task_name + "_source";
  m_MMS_source_diff_label = m_task_name + "_source_diff";
  m_MMS_source_t_label    = m_task_name + "_source_time";

}

//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_mom<T>::create_local_labels(){

  register_new_variable< T >( m_MMS_label);
  register_new_variable< T >( m_MMS_source_label);
  register_new_variable< T >( m_MMS_source_diff_label);
  register_new_variable< T >( m_MMS_source_t_label);

}

//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_mom<T>::register_initialize( std::vector<VarInfo>&
                                      variable_registry , const bool pack_tasks){

  register_variable( m_MMS_label,             ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_MMS_source_label,      ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_MMS_source_diff_label, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_MMS_source_t_label,    ArchesFieldContainer::COMPUTES, variable_registry );

  register_variable( m_x_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry );
  register_variable( m_y_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry );

}

//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_mom<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  compute_source(patch, tsk_info);

}
//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_mom<T>::register_timestep_init( std::vector<VarInfo>&
                                          variable_registry , const bool pack_tasks){
}

//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_mom<T>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

}

//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_mom<T>::register_timestep_eval( std::vector<VarInfo>&
                                          variable_registry, const int time_substep , const bool pack_tasks){

  register_variable( m_MMS_label,             ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  register_variable( m_MMS_source_label,      ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  register_variable( m_MMS_source_diff_label, ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  register_variable( m_MMS_source_t_label,    ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );

  register_variable( m_x_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  register_variable( m_y_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

}

//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_mom<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  compute_source( patch, tsk_info );

}

template <typename T>
void MMS_mom<T>::compute_source( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  T& f_mms = *(tsk_info->get_uintah_field<T>(m_MMS_label));
  T& s_mms = *(tsk_info->get_uintah_field<T>(m_MMS_source_label));
  T& s_diff_mms = *(tsk_info->get_uintah_field<T>(m_MMS_source_diff_label));
  // T& s_t_mms = *(tsk_info->get_uintah_field<T>(m_MMS_source_t_label));

  constCCVariable<double>& x = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_x_name);
  constCCVariable<double>& y = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_y_name);

//  double time_d      = _materialManager->getElapsedTime();
//  int   time_substep = tsk_info->get_time_substep();
//  double factor      = tsk_info->get_ssp_time_factor(time_substep);
//  double dt          = tsk_info->get_dt();
//  time_d = time_d + factor*dt;

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );

   if ( m_which_vel == "u" ){

     Uintah::parallel_for( range, [&](int i, int j, int k){

       f_mms(i,j,k) = 1.0  - m_amp * cos( m_two_pi * x(i,j,k) )
                                 * sin( m_two_pi * y(i,j,k) );

       s_mms(i,j,k) = - m_amp*m_two_pi*cos(m_two_pi*x(i,j,k))*cos(m_two_pi*y(i,j,k))
                         *(m_amp*cos(m_two_pi*y(i,j,k))*sin(m_two_pi*x(i,j,k)) + 1.0)
                         - m_amp*m_two_pi*sin(m_two_pi*x(i,j,k))*sin(m_two_pi*y(i,j,k))
                         *(m_amp*cos(m_two_pi*x(i,j,k))*sin(m_two_pi*y(i,j,k)) - 1.0); // convection

       s_diff_mms(i,j,k) = -2.0*m_amp*m_two_pi*m_two_pi*cos(m_two_pi*x(i,j,k))*sin(m_two_pi*y(i,j,k));


     });
  }  else if (m_which_vel == "p") {
     const double u_x = 1.0;
     const double u_y = 1.0;
     Uintah::parallel_for( range, [&](int i, int j, int k){

       f_mms(i,j,k) = 1.0  - m_amp * cos( m_two_pi * x(i,j,k) )
                                  * sin( m_two_pi * y(i,j,k) );

       s_mms(i,j,k) =  m_amp*m_two_pi*u_x*sin(m_two_pi*x(i,j,k))*sin(m_two_pi*y(i,j,k))
                        - m_amp*m_two_pi*u_y*cos(m_two_pi*x(i,j,k))*cos(m_two_pi*y(i,j,k));

       s_diff_mms(i,j,k) = -2.0*m_amp*m_two_pi*m_two_pi*cos(m_two_pi*x(i,j,k))*sin(m_two_pi*y(i,j,k));

     });

  } else {

    Uintah::parallel_for( range, [&](int i, int j, int k){

      f_mms(i,j,k) = 1.0  + m_amp * sin( m_two_pi * x(i,j,k) )
                                * cos( m_two_pi * y(i,j,k) );

      s_mms(i,j,k) =  - m_amp*m_two_pi*cos(m_two_pi*x(i,j,k))*cos(m_two_pi*y(i,j,k))
                       *(m_amp*cos(m_two_pi*x(i,j,k))*sin(m_two_pi*y(i,j,k)) - 1.0)
                        - m_amp*m_two_pi*sin(m_two_pi*x(i,j,k))*sin(m_two_pi*y(i,j,k))
                        *(m_amp*cos(m_two_pi*y(i,j,k))*sin(m_two_pi*x(i,j,k)) + 1.0); // convection

      s_diff_mms(i,j,k)  = 2.0*m_amp*m_two_pi*m_two_pi*cos(m_two_pi*y(i,j,k))*sin(m_two_pi*x(i,j,k));

    });

  }
}


}

#endif
