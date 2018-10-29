#ifndef Uintah_Component_Arches_DragModel_h
#define Uintah_Component_Arches_DragModel_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>

//-------------------------------------------------------

/**
 * @class    Drag mdel
 * @author   Alex Abboud
 * @date     August 2014
 *
 * @brief    This class calculates a generalized drag term for particle flow that is independent of coal properties.
 *
 * @details  The class calculates the drag term for the particle phase only for each of the quadrature nodes of the system.
 *           When using lagrangian particles N = 1. This generalization should allow for the same code to be utilized for
 *           any particle method - DQMOM, CQMOM, or Lagrangian.
 *
 *           \f$ du_p/dt = f_{drag} / \tau_p (u_{g,i} - u_{p,i} ) \f$ with
 *           \f$ \tau_p = \frac{\rho_p d_p^2}{18 \mu_g} \f$
 *           note: usign 994 instead of 1000 limits incontinuity
 *           \f$ f_{drag} = 1+0.15 Re_p^{0.687} \f$ for \f$ Re_p < 994\f$
 *           \f$ f_{drag} = 0.0183 Re_p \f$ for \f$Re_p > 994\f$  with
 *           \f$ Re_p \frac{ \rho_g d_p | u_p - u_g |}{\mu_g}
 *           the corrseponding gas source term is then
 *           \f$ du_g/dt = - du_p/dt * \rho_p / \rho_g * w
 */

//-------------------------------------------------------

namespace Uintah{

  //IT is the independent variable type
  //DT is the dependent variable type
  template <typename IT, typename DT>
  class DragModel : public TaskInterface {

  public:

    DragModel<IT, DT>( std::string task_name, int matl_index, const std::string var_name, const int N );
    ~DragModel<IT, DT>();

    void problemSetup( ProblemSpecP& db );

    void create_local_labels();

    class Builder : public TaskInterface::TaskBuilder {

    public:

      Builder( std::string task_name, int matl_index, std::string base_var_name, const int N ) :
      m_task_name(task_name), m_matl_index(matl_index), _base_var_name(base_var_name), _N(N){}
      ~Builder(){}

      DragModel* build()
      { return scinew DragModel<IT, DT>( m_task_name, m_matl_index, _base_var_name, _N ); }

    private:

      std::string m_task_name;
      int m_matl_index;
      std::string _base_var_name;
      std::string _base_gas_var_name;
      const int _N;

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

  private:

    enum DRAG_DIRECTION {XDIR,YDIR,ZDIR};

    DRAG_DIRECTION m_drag_direction;

    const std::string _base_var_name;
    std::string _base_gas_var_name;
    std::string _base_diameter_name;
    std::string _base_density_name;
    std::string _base_u_velocity_name;
    std::string _base_v_velocity_name;
    std::string _base_w_velocity_name;
    std::string _gas_u_velocity_name;
    std::string _gas_v_velocity_name;
    std::string _gas_w_velocity_name;
    std::string _gas_density_name;
    std::string _direction;

    const int _N;                 //<<< The number of "environments"
    double m_pi;

    double _visc;

    const std::string get_name(const int i, const std::string base_name){
      std::stringstream out;
      std::string env;
      out << i;
      env = out.str();
      return base_name + "_" + env;
    }

  };

  //Function definitions:

  template <typename IT, typename DT>
  DragModel<IT, DT>::DragModel( std::string task_name, int matl_index,
                                const std::string base_var_name, const int N ) :
  TaskInterface( task_name, matl_index ), _base_var_name(base_var_name), _N(N){
    m_pi = std::acos(-1.0);
  }

  template <typename IT, typename DT>
  DragModel<IT, DT>::~DragModel()
  {}

  template <typename IT, typename DT>
  void DragModel<IT, DT>::problemSetup( ProblemSpecP& db ){
    proc0cout << "WARNING: ParticleModels DragModel needs to be made consistent with DQMOM models and use correct DW, use model at your own risk."
      << "\n" << "\n" << "\n" << "\n" << "\n" << "\n" << "\n" << "\n" << "\n" << "\n"<< std::endl;
    _base_u_velocity_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_XVEL);
    _base_v_velocity_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_YVEL);
    _base_w_velocity_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_ZVEL);
    _base_diameter_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_SIZE);
    _base_density_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_DENSITY);

    db->require("direction",_direction);

    if ( _direction == "x" || _direction == "X" ){
      m_drag_direction = XDIR;
    } else if ( _direction == "y" || _direction == "Y" ){
      m_drag_direction = YDIR;
    } else if ( _direction == "z" || _direction == "Z" ){
      m_drag_direction = ZDIR;
    } else {
      throw InvalidValue("Error: direction in drag model not recognized.", __FILE__, __LINE__ );
    }

    _gas_u_velocity_name = "CCUVelocity";
    _gas_v_velocity_name = "CCVVelocity";
    _gas_w_velocity_name = "CCWVelocity";
    _gas_density_name = "densityCP";

    const ProblemSpecP params_doot = db->getRootNode();
    ProblemSpecP db_phys = params_doot->findBlock("PhysicalConstants");
    db_phys->require("viscosity", _visc);

    _base_gas_var_name = "gas_" + _base_var_name;
  }

  template <typename IT, typename DT>
  void DragModel<IT, DT>::create_local_labels(){
    for ( int i = 0; i < _N; i++ ){

      const std::string name = get_name(i, _base_var_name);
      register_new_variable<DT>( name );

      const std::string gas_name = get_name(i, _base_gas_var_name);
      register_new_variable<DT>( gas_name );

    }
  }

  //------------------------------------------------------------------------------------------------
  template <typename IT, typename DT>
  void DragModel<IT, DT>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

    for ( int i = 0; i < _N; i++ ){
      const std::string name = get_name(i, _base_var_name);
      register_variable( name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );

      const std::string gas_name = get_name(i, _base_gas_var_name);
      register_variable( gas_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );
    }
  }

  template <typename IT, typename DT>
  void DragModel<IT,DT>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    for ( int ienv = 0; ienv < _N; ienv++ ){
      const std::string name = get_name(ienv, _base_var_name);
      DT& model_value = *(tsk_info->get_uintah_field<DT>(name));
      const std::string gas_name = get_name(ienv, _base_gas_var_name);
      DT& gas_model_value = *(tsk_info->get_uintah_field<DT>(gas_name));

      Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
      Uintah::parallel_for( range, [&](int i, int j, int k){
        model_value(i,j,k) = 0.0;
        gas_model_value(i,j,k) = 0.0;
      });

    }
  }

  //------------------------------------------------------------------------------------------------
  template <typename IT, typename DT>
  void DragModel<IT, DT>::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

    for ( int i = 0; i < _N; i++ ){
      const std::string name = get_name(i, _base_var_name);
      register_variable( name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );

      const std::string gas_name = get_name(i, _base_gas_var_name);
      register_variable( gas_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );
    }
  }

  template <typename IT, typename DT>
  void DragModel<IT,DT>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    for ( int ienv = 0; ienv < _N; ienv++ ){
      const std::string name = get_name(ienv, _base_var_name);
      DT& model_value = *(tsk_info->get_uintah_field<DT>(name));
      const std::string gas_name = get_name(ienv, _base_gas_var_name);
      DT& gas_model_value = *(tsk_info->get_uintah_field<DT>(gas_name));

      Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
      Uintah::parallel_for( range, [&](int i, int j, int k){
        model_value(i,j,k) = 0.0;
        gas_model_value(i,j,k) = 0.0;
      });

    }
  }

  //======TIME STEP EVALUATION:
  template <typename IT, typename DT>
  void DragModel<IT, DT>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){

    for ( int i = 0; i < _N; i++ ){
      //dependent variables(s) or model values
      const std::string name = get_name(i, _base_var_name);
      register_variable( name, ArchesFieldContainer::MODIFIES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

      const std::string gas_name = get_name(i, _base_gas_var_name);
      register_variable( gas_name, ArchesFieldContainer::MODIFIES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

      //independent variables
      const std::string diameter_name = get_name( i, _base_diameter_name );
      register_variable( diameter_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

      const std::string density_name = get_name( i, _base_density_name );
      register_variable( density_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

      const std::string weight_name = get_name( i, "w" );
      register_variable( weight_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

      const std::string u_velocity_name = get_name( i, _base_u_velocity_name );
      register_variable( u_velocity_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

      const std::string v_velocity_name = get_name( i, _base_v_velocity_name );
      register_variable( v_velocity_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

      const std::string w_velocity_name = get_name( i, _base_w_velocity_name );
      register_variable( w_velocity_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    }

    register_variable( _gas_u_velocity_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( _gas_v_velocity_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( _gas_w_velocity_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( _gas_density_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  }

  template <typename IT, typename DT>
  void DragModel<IT,DT>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    IT& rhoG = *(tsk_info->get_const_uintah_field<IT>(_gas_density_name));
    IT& velU = *(tsk_info->get_const_uintah_field<IT>(_gas_u_velocity_name));
    IT& velV = *(tsk_info->get_const_uintah_field<IT>(_gas_v_velocity_name));
    IT& velW = *(tsk_info->get_const_uintah_field<IT>(_gas_w_velocity_name));

    for ( int ienv = 0; ienv < _N; ienv++ ){

      const std::string name = get_name(ienv, _base_var_name);
      const std::string gas_name = get_name(ienv, _base_gas_var_name);
      const std::string u_vel_name = get_name( ienv, _base_u_velocity_name );
      const std::string v_vel_name = get_name( ienv, _base_v_velocity_name );
      const std::string w_vel_name = get_name( ienv, _base_w_velocity_name );
      const std::string density_name = get_name( ienv, _base_density_name );
      const std::string diameter_name = get_name( ienv, _base_diameter_name );
      const std::string w_name = get_name( ienv, "w" );

      DT& model_value     = *(tsk_info->get_uintah_field<DT>(name));
      DT& gas_model_value = *(tsk_info->get_uintah_field<DT>(gas_name));
      IT& partVelU        = *(tsk_info->get_const_uintah_field<IT>(u_vel_name));
      IT& partVelV        = *(tsk_info->get_const_uintah_field<IT>(v_vel_name));
      IT& partVelW        = *(tsk_info->get_const_uintah_field<IT>(w_vel_name));
      IT& density         = *(tsk_info->get_const_uintah_field<IT>(density_name));
      IT& diameter        = *(tsk_info->get_const_uintah_field<IT>(diameter_name));
      IT& weight          = *(tsk_info->get_const_uintah_field<IT>(w_name));

      Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
      Uintah::parallel_for( range, [&](int i, int j, int k){

        const double gasVelMag = std::sqrt( velU(i,j,k) * velU(i,j,k) +
                                            velV(i,j,k) * velV(i,j,k) +
                                            velW(i,j,k) * velW(i,j,k));

        const double partVelMag = std::sqrt( partVelU(i,j,k) * partVelU(i,j,k) +
                                             partVelV(i,j,k) * partVelV(i,j,k) +
                                             partVelW(i,j,k) * partVelW(i,j,k));

        const double tauP = ( density(i,j,k) * diameter(i,j,k) * diameter(i,j,k) ) / (18.0 * _visc);

        const double Re   = std::abs( gasVelMag - partVelMag ) * diameter(i,j,k) * rhoG(i,j,k) / _visc;

        const double fDrag = (Re < 994.0) ? 1.0 + 0.15 * std::pow( Re, 0.687) :  0.0183 * Re;
        //an alternative drag law in case its needed later
        //*fDrag <<= 1.0 + 0.15 * pow( (*Re), 0.687 ) + 0.0175* (*Re) / ( 1.0 + 4.25e4 * pow( (*Re), -1.16) ); //valid over all Re

        double pred_model_value = 0.0;

        if ( _direction=="x" ) {
          pred_model_value = tauP != 0.0 ? (fDrag) / (tauP) * ( velU(i,j,k) - partVelU(i,j,k) ) : 0.0;
        } else if ( _direction=="y" ) {
          pred_model_value = tauP != 0.0 ? (fDrag) / (tauP) * ( velV(i,j,k) - partVelV(i,j,k) ) : 0.0;
        } else if ( _direction=="z" ) {
          pred_model_value = tauP != 0.0 ? (fDrag) / (tauP) * ( velW(i,j,k) - partVelW(i,j,k) ) : 0.0;
        }

        model_value(i,j,k) = pred_model_value;

        double pred_gas_value = -1.* pred_model_value * weight(i,j,k) * density(i,j,k) / rhoG(i,j,k) * m_pi/6.0 * diameter(i,j,k) * diameter(i,j,k) * diameter(i,j,k);

        pred_gas_value = (diameter(i,j,k) == 0.0) ? 0.0 : pred_gas_value;

        gas_model_value(i,j,k) = pred_gas_value;

      });
    }
  }
}
#endif
