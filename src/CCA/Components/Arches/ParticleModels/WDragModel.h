#ifndef Uintah_Component_Arches_WDragModel_h
#define Uintah_Component_Arches_WDragModel_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>

namespace Uintah{

  //IT is the independent variable type
  template <typename T>
  class WDragModel : public TaskInterface {

public:

    WDragModel<T>( std::string task_name, int matl_index, int Nenv );
    ~WDragModel<T>();

    void problemSetup( ProblemSpecP& db );



    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index, int Nenv ) :
        _task_name(task_name), _matl_index(matl_index), _Nenv(Nenv){}
      ~Builder(){}

      WDragModel* build()
      { return scinew WDragModel<T>( _task_name, _matl_index, _Nenv ); }

      private:

      std::string _task_name;
      int _matl_index;
      int _Nenv;

    };

protected:

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();

private:
    typedef typename ArchesCore::VariableHelper<T>::XFaceType FXT;
    typedef typename ArchesCore::VariableHelper<T>::YFaceType FYT;
    typedef typename ArchesCore::VariableHelper<T>::ZFaceType FZT;
    typedef typename ArchesCore::VariableHelper<T>::ConstType CT;

    int _Nenv; 
    std::string m_model_name;
    std::string m_gasSource_name;
    std::string m_density_gas_name;
    std::string m_cc_u_vel_name;
    std::string m_cc_v_vel_name;
    std::string m_cc_w_vel_name;
    std::string m_volFraction_name;
    std::string m_up_name;
    std::string m_vp_name;
    std::string m_wp_name;
    std::string m_length_name;
    std::string m_particle_density_name;
    std::string m_w_qn_name;
    std::string m_w_name;
    std::string m_vel_dir_name;
    std::string m_pvel_dir_name;
    std::string m_wvelp_name;
    //std:: string ;
    double m_kvisc;
    double m_scaling_constant;
    double m_gravity;
  };

  //Function definitions:

  template <typename T>
  void WDragModel<T>::create_local_labels(){

    register_new_variable<T>( m_model_name );
    register_new_variable<T>( m_gasSource_name );

  }


  template <typename T>
  WDragModel<T>::WDragModel( std::string task_name, int matl_index,
                                                      int Nenv ) :
  TaskInterface( task_name, matl_index ), _Nenv(Nenv){
  }

  template <typename T>
  WDragModel<T>::~WDragModel()
  {}

  template <typename T>
  void WDragModel<T>::problemSetup( ProblemSpecP& db ){

    m_model_name     = _task_name; 
    m_gasSource_name = _task_name + "_gasSource";

    m_density_gas_name = ArchesCore::parse_ups_for_role( ArchesCore::DENSITY,   db, "density" );
    m_cc_u_vel_name = ArchesCore::parse_ups_for_role( ArchesCore::CCUVELOCITY, db, "CCUVelocity" );
    m_cc_v_vel_name = ArchesCore::parse_ups_for_role( ArchesCore::CCVVELOCITY, db, "CCVVelocity" );
    m_cc_w_vel_name = ArchesCore::parse_ups_for_role( ArchesCore::CCWVELOCITY, db, "CCWVelocity" );
    m_volFraction_name = "volFraction";

    // check for particle velocity
    std::string up_root = ArchesCore::parse_for_particle_role_to_label( db, ArchesCore::P_XVEL );
    std::string vp_root = ArchesCore::parse_for_particle_role_to_label( db, ArchesCore::P_YVEL );
    std::string wp_root = ArchesCore::parse_for_particle_role_to_label( db, ArchesCore::P_ZVEL );
    std::string density_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_DENSITY);
    std::string length_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_SIZE);
    
    m_up_name = ArchesCore::append_env( up_root, _Nenv);
    m_vp_name = ArchesCore::append_env( vp_root, _Nenv);
    m_wp_name = ArchesCore::append_env( wp_root, _Nenv);

    m_length_name  = ArchesCore::append_env( length_root, _Nenv );
    m_particle_density_name  = ArchesCore::append_env( density_root, _Nenv );
    m_w_qn_name              = ArchesCore::append_qn_env("w", _Nenv ); // w_qn
    m_w_name                 = ArchesCore::append_env("w", _Nenv ); // w
    
    // check for gravity
    const ProblemSpecP params_root = db->getRootNode();
    std::vector<double> gravity;
    if (params_root->findBlock("PhysicalConstants")) {
      ProblemSpecP db_phys = params_root->findBlock("PhysicalConstants");
      db_phys->require("gravity", gravity);
      db_phys->require("viscosity",m_kvisc);
    } else {
      throw InvalidValue("Error: Missing <PhysicalConstants> section in input file required for drag model.",__FILE__,__LINE__);
    }

    std::string coord;
    db->require("direction",coord);
    
    if ( coord == "x" || coord == "X" ){

      m_scaling_constant = ArchesCore::get_scaling_constant(db,up_root,_Nenv );
      m_vel_dir_name     = m_cc_u_vel_name;
      m_pvel_dir_name    = m_up_name;
      m_wvelp_name       = ArchesCore::append_qn_env(up_root, _Nenv ) ; 
      m_gravity          = gravity[0]; 

    } else if ( coord == "y" || coord == "Y" ){

      m_scaling_constant = ArchesCore::get_scaling_constant(db,vp_root,_Nenv );
      m_vel_dir_name     = m_cc_v_vel_name;
      m_pvel_dir_name    = m_vp_name;
      m_wvelp_name       = ArchesCore::append_qn_env(vp_root, _Nenv ) ; 
      m_gravity          = gravity[1]; 

    } else {

      m_scaling_constant = ArchesCore::get_scaling_constant(db,wp_root,_Nenv );
      m_vel_dir_name     = m_cc_w_vel_name;
      m_pvel_dir_name    = m_wp_name;
      m_wvelp_name       = ArchesCore::append_qn_env(wp_root, _Nenv ) ; 
      m_gravity          = gravity[2]; 

    }


  }

  //======INITIALIZATION:
  template <typename T>
  void WDragModel<T>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

  register_variable( m_model_name, ArchesFieldContainer::COMPUTES, variable_registry, _task_name  );
  register_variable( m_gasSource_name, ArchesFieldContainer::COMPUTES, variable_registry, _task_name  );

  }

  template <typename T>
  void WDragModel<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  T& model      = tsk_info->get_uintah_field_add<T>(m_model_name);
  T& gas_source = tsk_info->get_uintah_field_add<T>(m_gasSource_name);

  model.initialize( 0.0 );
  gas_source.initialize( 0.0 );

  }

  //======TIME STEP INITIALIZATION:
  template <typename T>
  void WDragModel<T>::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){
  }

  template <typename T>
  void WDragModel<T>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

  //======TIME STEP EVALUATION:
  template <typename T>
  void WDragModel<T>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){

  register_variable( m_model_name, ArchesFieldContainer::COMPUTES, variable_registry, _task_name  );
  register_variable( m_gasSource_name, ArchesFieldContainer::COMPUTES, variable_registry, _task_name  );

  register_variable( m_cc_u_vel_name,    ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  register_variable( m_cc_v_vel_name,    ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  register_variable( m_cc_w_vel_name,    ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  register_variable( m_density_gas_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  register_variable( m_volFraction_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );

  register_variable( m_w_qn_name,            ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  register_variable( m_w_name,               ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  register_variable( m_w_qn_name + "_RHS",   ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

  register_variable( m_up_name,               ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  register_variable( m_vp_name,               ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  register_variable( m_wp_name,               ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  register_variable( m_particle_density_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  register_variable( m_length_name,           ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  register_variable( m_wvelp_name,            ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  register_variable( m_wvelp_name + "_RHS",   ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

  }

  template <typename T>
  void WDragModel<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  const double dt = tsk_info->get_dt();
  Vector Dx = patch->dCell();
  const double vol = Dx.x()* Dx.y()* Dx.z();

  T& model      = tsk_info->get_uintah_field_add<T>(m_model_name);
  T& gas_source = tsk_info->get_uintah_field_add<T>(m_gasSource_name);

  // gas variables
  CT& CCuVel = tsk_info->get_const_uintah_field_add<CT>( m_cc_u_vel_name );
  CT& CCvVel = tsk_info->get_const_uintah_field_add<CT>( m_cc_v_vel_name );
  CT& CCwVel = tsk_info->get_const_uintah_field_add<CT>( m_cc_w_vel_name );
  CT& den    = tsk_info->get_const_uintah_field_add<CT>( m_density_gas_name );
  CT& volFraction  = tsk_info->get_const_uintah_field_add<CT>( m_volFraction_name );

  
  CT& up    = tsk_info->get_const_uintah_field_add< CT >( m_up_name );
  CT& vp    = tsk_info->get_const_uintah_field_add< CT >( m_vp_name );
  CT& wp    = tsk_info->get_const_uintah_field_add< CT >( m_wp_name );
  CT& rho_p = tsk_info->get_const_uintah_field_add< CT >( m_particle_density_name );
  CT& l_p   = tsk_info->get_const_uintah_field_add< CT >( m_length_name );

  // DQMOM valiables 
  CT& w_qn         = tsk_info->get_const_uintah_field_add<CT>(m_w_qn_name);
  CT& weight       = tsk_info->get_const_uintah_field_add<CT>(m_w_name);
  CT& RHS_weight   = tsk_info->get_const_uintah_field_add<CT>(m_w_qn_name + "_RHS");
  CT& Vel          = tsk_info->get_const_uintah_field_add<CT>(m_vel_dir_name );
  CT& pVel         = tsk_info->get_const_uintah_field_add<CT>(m_pvel_dir_name );
  CT& weight_p_vel = tsk_info->get_const_uintah_field_add<CT>(m_wvelp_name);// 
  CT& RHS_source   = tsk_info->get_const_uintah_field_add<CT>(m_wvelp_name + "_RHS");

  T& lambdaBirth = tsk_info->get_uintah_field_add<T>("lambdaBirth",0);
  lambdaBirth.initialize( 0.0 );

  Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){

    if (volFraction(i,j,k) > 0.0) {

      const double denph    = den(i,j,k);
      const double rho_pph  = rho_p(i,j,k);
      const double l_pph    = l_p(i,j,k);

      const  double relative_velocity = std::sqrt( ( CCuVel(i,j,k) - up(i,j,k) ) * ( CCuVel(i,j,k) - up(i,j,k) ) +
                                                   ( CCvVel(i,j,k) - vp(i,j,k) ) * ( CCvVel(i,j,k) - vp(i,j,k) ) +
                                                   ( CCwVel(i,j,k) - wp(i,j,k) ) * ( CCwVel(i,j,k) - wp(i,j,k) )   ); // [m/s]

      const double Re  = relative_velocity * l_pph / ( m_kvisc / denph );

      const double fDrag    = Re <994 ? 1.0 + 0.15*std::pow(Re, 0.687) : 0.0183*Re;

      const double t_p = ( rho_pph * l_pph * l_pph )/( 18.0 * m_kvisc );
      const double tau=t_p/fDrag;

      if (tau > dt ){ 
        model(i,j,k)      = w_qn(i,j,k) * ( fDrag / t_p * (Vel(i,j,k)-pVel(i,j,k)) + m_gravity) / m_scaling_constant;
        gas_source(i,j,k) = -weight(i,j,k) * rho_pph / 6.0 * M_PI * fDrag / t_p * ( Vel(i,j,k)-pVel(i,j,k) ) * std::pow(l_pph,3.0);
      } else {  // rate clip, if we aren't resolving timescale
        const double updated_weight = std::max(w_qn(i,j,k) + dt / vol * ( RHS_weight(i,j,k) ) , 1e-15);
        model(i,j,k) = 1. / m_scaling_constant * ( updated_weight * Vel(i,j,k) - weight_p_vel(i,j,k) ) / dt - ( RHS_source(i,j,k) / vol + lambdaBirth(i,j,k));
      } // end timescale if

    }

  });
  

  }
}
#endif
