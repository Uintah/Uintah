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

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

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

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void ShunnMMS<T>::problemSetup( ProblemSpecP& db ){

    db->getWithDefault( "k1", m_k1, 4.0);
    db->getWithDefault( "k2", m_k2, 2.0);
    db->getWithDefault( "w0", m_w0, 50.0);
    db->getWithDefault( "rho0", m_rho0, 20.0);
    db->getWithDefault( "rho1", m_rho1, 1.0);

    db->require("which_vel", m_which_vel);
    ProblemSpecP db_coord = db->findBlock("coordinates");
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
    register_variable( m_x_name, AFC::REQUIRES, 0, AFC::NEWDW, variable_registry, _task_name );

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void ShunnMMS<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    T& f_mms = *(tsk_info->get_uintah_field<T>(m_var_name));
    constCCVariable<double>& x = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_x_name);

    Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
    const double time_d = 0.0;
    const double k12 = m_k1-m_k2;
    //const double k21 = m_k2-m_k1;
    const double z1 = std::exp(-m_k1 * time_d);
    const double r01 = m_rho0 - m_rho1;

    if ( m_which_vel == "u" ){
    // for velocity
      Uintah::parallel_for( range, [&](int i, int j, int k){
        const double z2 = std::cosh (m_w0 * std::exp (-m_k2 * time_d) * x(i,j,k)); // x at face
        const double phi_f = (z1-z2)/(z1 * (1.0 - m_rho0/m_rho1)-z2);
        const double u1  = std::exp(m_w0*std::exp(-m_k2*time_d)*x(i,j,k));
        const double rho = 1.0/(phi_f/m_rho1 + (1.0- phi_f )/m_rho0);

        f_mms(i,j,k) = (2.0*m_k2*x(i,j,k)*r01*std::exp(-m_k1*time_d)*u1/(u1*u1 + 1.0) +
          r01*k12*std::exp(-k12*time_d)/m_w0*(2.0*std::atan(u1)-m_pi/2.0))/rho;

    });
    } else {
    // for scalar
      Uintah::parallel_for( range, [&](int i, int j, int k){
        const double z2 = std::cosh(m_w0 * std::exp (-m_k2 * time_d) * x(i,j,k)); // x is cc value
        const double phi = (z1-z2)/(z1 * (1.0 - m_rho0/m_rho1)-z2);
        //const double rho = 1.0/(phi/m_rho1 + (1.0- phi )/m_rho0);
        f_mms(i,j,k) = phi;
    });
    }
  }
}
#endif
