#ifndef Uintah_Component_Arches_MMS_mom_csmag_h
#define Uintah_Component_Arches_MMS_mom_csmag_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

  template <typename T>
  class MMS_mom_csmag : public TaskInterface {

public:

    MMS_mom_csmag<T>( std::string task_name, int matl_index, SimulationStateP shared_state  );
    ~MMS_mom_csmag<T>();

    void problemSetup( ProblemSpecP& db );

    //Build instructions for this (MMS_mom_csmag) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index, SimulationStateP shared_state ) :
        _task_name(task_name), _matl_index(matl_index), _shared_state(shared_state){}
      ~Builder(){}

      MMS_mom_csmag* build()
      { return scinew MMS_mom_csmag<T>( _task_name, _matl_index, _shared_state  ); }

      private:

      std::string _task_name;
      int _matl_index;

      SimulationStateP _shared_state;
    };

 protected:

    typedef ArchesFieldContainer::VariableInformation VarInfo;

    void register_initialize( std::vector<VarInfo>& variable_registry , const bool pack_tasks);

    void register_timestep_init( std::vector<VarInfo>& variable_registry , const bool pack_tasks);

    void register_timestep_eval( std::vector<VarInfo>& variable_registry, const int time_substep , const bool pack_tasks);

    void register_compute_bcs( std::vector<VarInfo>& variable_registry, const int time_substep , const bool pack_tasks){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();

private:

    double m_amp;
    double m_freq;
    double m_two_pi = 2*acos(-1.0);
    double m_molecular_visc;
    double Cs;

    std::string m_x_name;
    std::string m_y_name;
    std::string m_which_vel;

    std::string m_MMS_label;
    std::string m_MMS_source_label;
    std::string m_MMS_source_diff_label;
    std::string m_MMS_source_t_label;

    SimulationStateP _shared_state;

    /** @brief Helper function to reduce code - Called in initialize and eval **/
    void compute_source( const Patch* patch, ArchesTaskInfoManager* tsk_info );

  };
  //------------------------------------------------------------------------------------------------

template <typename T>
MMS_mom_csmag<T>::MMS_mom_csmag( std::string task_name, int matl_index,
  SimulationStateP shared_state ) :
TaskInterface( task_name, matl_index ) , _shared_state(shared_state)
{}

//--------------------------------------------------------------------------------------------------
template <typename T>
MMS_mom_csmag<T>::~MMS_mom_csmag()
{}

//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_mom_csmag<T>::problemSetup( ProblemSpecP& db ){
    std::string wave_type;

  db->getWithDefault( "amplitude", m_amp, 1.0);
  db->getWithDefault( "frequency", m_freq, 1.0);
  db->require("which_vel", m_which_vel);
  ProblemSpecP db_coord = db->findBlock("coordinates");
  if ( db_coord ){
    db_coord->getAttribute("x", m_x_name);
    db_coord->getAttribute("y", m_y_name);
  } else {
    throw InvalidValue(
      "Error: must have coordinates specified for almgren MMS init condition",
      __FILE__, __LINE__);
  }

  m_MMS_label             = _task_name;
  m_MMS_source_label      = _task_name + "_source";
  m_MMS_source_diff_label = _task_name + "_source_diff";
  m_MMS_source_t_label    = _task_name + "_source_time";

  const ProblemSpecP params_root = db->getRootNode();

  if (params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("TurbulenceModels")) {
    params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("TurbulenceModels")->
      findBlock("model")->findBlock("Smagorinsky_constant")->getAttribute("Cs",Cs);
  } else {
    throw InvalidValue(
      "ERROR: No turbulence model for verification  problemSetup(): Missing <Cs> section in input file!"
      ,__FILE__,__LINE__);
  }

  if (params_root->findBlock("PhysicalConstants")) {
    params_root->findBlock("PhysicalConstants")->require("viscosity", m_molecular_visc);
    if( m_molecular_visc == 0 ) {
      throw InvalidValue(
        "ERROR: Constant Smagorinsky: problemSetup(): Zero viscosity specified in <PhysicalConstants> section of input file.",
        __FILE__,__LINE__);
    }
  } else {
    throw InvalidValue(
      "ERROR: Constant Smagorinsky: problemSetup(): Missing <PhysicalConstants> section in input file!",
      __FILE__,__LINE__);
  }
}

  //------------------------------------------------------------------------------------------------
template <typename T>
void MMS_mom_csmag<T>::create_local_labels(){

  register_new_variable< T >( m_MMS_label);
  register_new_variable< T >( m_MMS_source_label);
  register_new_variable< T >( m_MMS_source_diff_label);
  register_new_variable< T >( m_MMS_source_t_label);

}

//--------------------------------------------------------------------------------------------------

template <typename T>
void MMS_mom_csmag<T>::register_initialize( std::vector<VarInfo>&
                                            variable_registry , const bool pack_tasks){

  register_variable( m_MMS_label,             AFC::COMPUTES, variable_registry );
  register_variable( m_MMS_source_label,      AFC::COMPUTES, variable_registry );
  register_variable( m_MMS_source_diff_label, AFC::COMPUTES, variable_registry );
  register_variable( m_MMS_source_t_label,    AFC::COMPUTES, variable_registry );

  register_variable( m_x_name, AFC::REQUIRES, 0, AFC::NEWDW, variable_registry );
  register_variable( m_y_name, AFC::REQUIRES, 0, AFC::NEWDW, variable_registry );

}

//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_mom_csmag<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  compute_source( patch, tsk_info );

}

//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_mom_csmag<T>::register_timestep_init( std::vector<VarInfo>&
                                               variable_registry , const bool pack_tasks){

  register_variable( m_MMS_label,        AFC::COMPUTES, variable_registry );
  register_variable( m_MMS_source_label, AFC::COMPUTES, variable_registry );
  register_variable( m_MMS_source_diff_label, AFC::COMPUTES, variable_registry );
  register_variable( m_MMS_source_t_label, AFC::COMPUTES, variable_registry );
}

//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_mom_csmag<T>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  T& f_mms = *(tsk_info->get_uintah_field<T>(m_MMS_label));
  T& s_mms = *(tsk_info->get_uintah_field<T>(m_MMS_source_label));
  T& s_diff_mms = *(tsk_info->get_uintah_field<T>(m_MMS_source_diff_label));
  T& s_t_mms = *(tsk_info->get_uintah_field<T>(m_MMS_source_t_label));
}

//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_mom_csmag<T>::register_timestep_eval( std::vector<VarInfo>&
                                               variable_registry, const int time_substep ,
                                               const bool pack_tasks){

  register_variable( m_MMS_label,             AFC::MODIFIES ,  variable_registry, time_substep );
  register_variable( m_MMS_source_label,      AFC::MODIFIES ,  variable_registry, time_substep );
  register_variable( m_MMS_source_diff_label, AFC::MODIFIES ,  variable_registry, time_substep );
  register_variable( m_MMS_source_t_label,    AFC::MODIFIES ,  variable_registry, time_substep );

  register_variable( m_x_name, AFC::REQUIRES, 0, AFC::NEWDW, variable_registry, time_substep );
  register_variable( m_y_name, AFC::REQUIRES, 0, AFC::NEWDW, variable_registry, time_substep );

}

//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_mom_csmag<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  compute_source( patch, tsk_info );

}

//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_mom_csmag<T>::compute_source( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  T& f_mms = *(tsk_info->get_uintah_field<T>(m_MMS_label));
  T& s_mms = *(tsk_info->get_uintah_field<T>(m_MMS_source_label));
  T& s_diff_mms = *(tsk_info->get_uintah_field<T>(m_MMS_source_diff_label));
  T& s_t_mms = *(tsk_info->get_uintah_field<T>(m_MMS_source_t_label));

  constCCVariable<double>& x = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_x_name);
  constCCVariable<double>& y = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_y_name);

  const Vector Dx = patch->dCell();
  const double delta = pow(Dx.x()*Dx.y()*Dx.z(),1./3.);
  double rho =1.0 ; // it is constant

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );

  if ( m_which_vel == "u" ){

    Uintah::parallel_for( range, [&](int i, int j, int k){

      f_mms(i,j,k) = 1.0  - m_amp * cos( m_two_pi * x(i,j,k) )
                                * sin( m_two_pi * y(i,j,k) );

      s_mms(i,j,k) = - m_amp*m_two_pi*cos(m_two_pi*x(i,j,k))*cos(m_two_pi*y(i,j,k))
                        *(m_amp*cos(m_two_pi*y(i,j,k))*sin(m_two_pi*x(i,j,k)) + 1.0)
                        - m_amp*m_two_pi*sin(m_two_pi*x(i,j,k))*sin(m_two_pi*y(i,j,k))
                        *(m_amp*cos(m_two_pi*x(i,j,k))*sin(m_two_pi*y(i,j,k)) - 1.0); // convection

       s_diff_mms(i,j,k) = (4.0*pow(Cs*delta,2.0)*pow(m_amp,3.0)*pow(m_two_pi,4.0)*rho*cos(m_two_pi*x(i,j,k))*pow(cos(m_two_pi*y(i,j,k))*sin(m_two_pi*x(i,j,k)),2.0)*sin(m_two_pi*y(i,j,k)))/
                          (2.0*m_amp*m_two_pi*sin(m_two_pi*x(i,j,k))*sin(m_two_pi*y(i,j,k)))
                           - (4.0*pow(Cs*delta,2.0)*pow(m_amp,3.0)*pow(m_two_pi,4.0)*rho*cos(m_two_pi*x(i,j,k))*pow(sin(m_two_pi*x(i,j,k)),2.0)*pow(sin(m_two_pi*y(i,j,k)),3.0))/
                          (2.0*m_amp*m_two_pi*sin(m_two_pi*x(i,j,k))*sin(m_two_pi*y(i,j,k)))
                           - 2.0*m_amp*pow(m_two_pi,2.0)*cos(m_two_pi*x(i,j,k))*sin(m_two_pi*y(i,j,k))
                          *(m_molecular_visc + pow(Cs*delta,2.0)*rho*(2.0*m_amp*m_two_pi*sin(m_two_pi*x(i,j,k))*sin(m_two_pi*y(i,j,k))));

//      s_diff_mms(i,j,k) = -2.0*m_amp*m_two_pi*m_two_pi*cos(m_two_pi*x(i,j,k))*sin(m_two_pi*y(i,j,k));


    });

  } else { // then v

    Uintah::parallel_for( range, [&](int i, int j, int k){

      f_mms(i,j,k) = 1.0  + m_amp * sin( m_two_pi * x(i,j,k) )
                                * cos( m_two_pi * y(i,j,k) );

      s_mms(i,j,k) =  - m_amp*m_two_pi*cos(m_two_pi*x(i,j,k))*cos(m_two_pi*y(i,j,k))
                       *(m_amp*cos(m_two_pi*x(i,j,k))*sin(m_two_pi*y(i,j,k)) - 1.0)
                        - m_amp*m_two_pi*sin(m_two_pi*x(i,j,k))*sin(m_two_pi*y(i,j,k))
                        *(m_amp*cos(m_two_pi*y(i,j,k))*sin(m_two_pi*x(i,j,k)) + 1.0); // convection

//      s_diff_mms(i,j,k)  = 2.0*m_amp*m_two_pi*m_two_pi*cos(m_two_pi*y(i,j,k))*sin(m_two_pi*x(i,j,k));

      s_diff_mms(i,j,k)  = 2.0*m_amp*pow(m_two_pi,2.0)*cos(m_two_pi*y(i,j,k))*sin(m_two_pi*x(i,j,k))
                           *(m_molecular_visc + pow(Cs*delta,2.0)*rho*(2.0*m_amp*m_two_pi*sin(m_two_pi*x(i,j,k))*sin(m_two_pi*y(i,j,k))))
                           + (4.0*pow(Cs*delta,2.0)*pow(m_amp,3.0)*pow(m_two_pi,4.0)*rho*cos(m_two_pi*y(i,j,k))*pow(sin(m_two_pi*x(i,j,k)),3.0)*pow(sin(m_two_pi*y(i,j,k)),2.0))
                           /(2.0*m_amp*m_two_pi*sin(m_two_pi*x(i,j,k))*sin(m_two_pi*y(i,j,k)))
                           - (4.0*pow(Cs*delta,2.0)*pow(m_amp,3.0)*pow(m_two_pi,4.0)*rho*pow(cos(m_two_pi*x(i,j,k)),2.0)*cos(m_two_pi*y(i,j,k))*sin(m_two_pi*x(i,j,k))*pow(sin(m_two_pi*y(i,j,k)),2.0))
                            /(2.0*m_amp*m_two_pi*sin(m_two_pi*x(i,j,k))*sin(m_two_pi*y(i,j,k)));
    });
  }
}

}

#endif
