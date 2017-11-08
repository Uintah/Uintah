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

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void create_local_labels(){};

private:

    const double m_pi = acos(-1.0);
    double m_k;
    double m_w0 ;
    double m_rho0; 
    double m_rho1;
    double m_uf;
    double m_vf;

    const std::string m_var_name;

    std::string m_x_name;
    std::string m_y_name;
    std::string m_which_vel;

  };

  //------------------------------------------------------------------------------------------------
  template <typename T>
  ShunnMMSP3<T>::ShunnMMSP3( std::string task_name, int matl_index, const std::string var_name ) :
  TaskInterface( task_name, matl_index ), m_var_name(var_name){


  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  ShunnMMSP3<T>::~ShunnMMSP3()
  {}

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void ShunnMMSP3<T>::problemSetup( ProblemSpecP& db ){

    db->getWithDefault( "k", m_k, 2.0);
    db->getWithDefault( "w0", m_w0, 2.0);
    db->getWithDefault( "rho0", m_rho0, 20.0);
    db->getWithDefault( "rho1", m_rho1, 1.0);
    db->getWithDefault( "uf", m_uf, 0.0);
    db->getWithDefault( "vf", m_vf, 0.0);

    db->require("which_vel", m_which_vel);
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
    register_variable( m_x_name, AFC::REQUIRES, 0, AFC::NEWDW, variable_registry, _task_name );
    register_variable( m_y_name, AFC::REQUIRES, 0, AFC::NEWDW, variable_registry, _task_name );

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void ShunnMMSP3<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){
    
    T& f_mms = *(tsk_info->get_uintah_field<T>(m_var_name));
    constCCVariable<double>& x = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_x_name);
    constCCVariable<double>& y = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_y_name);

    Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
    const double time_d = 0.0;

    if ( m_which_vel == "u" ){
    // for velocity
      Uintah::parallel_for( range, [&](int i, int j, int k){
        const double phi_f = (1.0 + sin(m_k*m_pi*(x(i,j,k)-m_uf*time_d))*
                        sin(m_k*m_pi*(y(i,j,k)-m_vf*time_d))*cos(m_w0*m_pi*time_d))/(1.0 + 
                        m_rho0/m_rho1+(1.0-m_rho0/m_rho1)*sin(m_k*m_pi*(x(i,j,k)-m_uf*time_d))*
                        sin(m_k*m_pi*(y(i,j,k)-m_vf*time_d))*cos(m_w0*m_pi*time_d));
        const double rho = 1.0/(phi_f/m_rho1 + (1.0- phi_f )/m_rho0); 
        
        f_mms(i,j,k) = m_uf -m_w0/m_k/4.0*cos(m_k*m_pi*(x(i,j,k)-m_uf*time_d))*sin(m_k*m_pi*(y(i,j,k)-m_vf*time_d))*sin(m_w0*m_pi*time_d)*(m_rho1-m_rho0)/rho;
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
