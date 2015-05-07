#ifndef Uintah_Component_Arches_DragModel_h
#define Uintah_Component_Arches_DragModel_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <spatialops/structured/FVStaggered.h>

#ifndef PI
#define PI 3.141592653589793
#endif

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
      _task_name(task_name), _matl_index(matl_index), _base_var_name(base_var_name), _N(N){}
      ~Builder(){}
      
      DragModel* build()
      { return scinew DragModel<IT, DT>( _task_name, _matl_index, _base_var_name, _N ); }
      
    private:
      
      std::string _task_name;
      int _matl_index;
      std::string _base_var_name;
      std::string _base_gas_var_name;
      const int _N;
      
    };
    
  protected:
    
    void register_initialize( std::vector<VariableInformation>& variable_registry );
    
    void register_timestep_init( std::vector<VariableInformation>& variable_registry );
    
    void register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep );
    
    void register_compute_bcs( std::vector<VariableInformation>& variable_registry, const int time_substep ){}; 

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                      SpatialOps::OperatorDatabase& opr ){}; 

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                     SpatialOps::OperatorDatabase& opr );
    
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                        SpatialOps::OperatorDatabase& opr );
    
    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
               SpatialOps::OperatorDatabase& opr );
    
  private:
    
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
    VAR_TYPE _D_type;
    VAR_TYPE _I_type;
    
    const int _N;                 //<<< The number of "environments"
    
    double _visc;
    double _rho;
    double _d;
    
    bool constDensity;
    bool constDiameter;
    
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

    VarTypeHelper<DT> dhelper; 
    _D_type = dhelper.get_vartype(); 

    VarTypeHelper<IT> ihelper; 
    _I_type = ihelper.get_vartype(); 

  }
  
  template <typename IT, typename DT>
  DragModel<IT, DT>::~DragModel()
  {}
  
  template <typename IT, typename DT>
  void DragModel<IT, DT>::problemSetup( ProblemSpecP& db ){

    db->getWithDefault("u_velocity_label",_base_u_velocity_name,"none");
    db->getWithDefault("v_velocity_label",_base_v_velocity_name,"none");
    db->getWithDefault("w_velocity_label",_base_w_velocity_name,"none");
    
    std::string tempType;
    db->findBlock("particle_density")->getAttribute("type",tempType);
    if ( tempType == "constant" ) {
      db->findBlock("particle_density")->require("constant",_rho);
      constDensity = true;
    } else {
      db->findBlock("particle_density")->require("density_label",_base_density_name);
      constDensity = false;
    }
    
    db->findBlock("diameter")->getAttribute("type",tempType);
    if ( tempType == "constant" ) {
      db->findBlock("diameter")->require("constant",_d);
      constDiameter = true;
    } else {
      db->findBlock("diameter")->require("diameter_label",_base_diameter_name);
      constDiameter = false;
    }
    
    db->require("direction",_direction);

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
      register_new_variable( name, _D_type ); 
      
      const std::string gas_name = get_name(i, _base_gas_var_name);
      register_new_variable( gas_name, _D_type );
      
    }
  }
  
  //======INITIALIZATION:
  template <typename IT, typename DT>
  void DragModel<IT, DT>::register_initialize( std::vector<VariableInformation>& variable_registry ){

    for ( int i = 0; i < _N; i++ ){
      const std::string name = get_name(i, _base_var_name);
      register_variable( name, _D_type, COMPUTES, 0, NEWDW, variable_registry );
      
      const std::string gas_name = get_name(i, _base_gas_var_name);
      register_variable( gas_name, _D_type, COMPUTES, 0, NEWDW, variable_registry );
    }
  }
  
  template <typename IT, typename DT>
  void DragModel<IT,DT>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                     SpatialOps::OperatorDatabase& opr ){
    
    using namespace SpatialOps;
    using SpatialOps::operator *;
    typedef SpatialOps::SpatFldPtr<DT> DTptr;

    for ( int i = 0; i < _N; i++ ){
      const std::string name = get_name(i, _base_var_name);
      DTptr model_value = tsk_info->get_so_field<DT>(name);
      *model_value <<= 0.0;
      
      const std::string gas_name = get_name(i, _base_gas_var_name);
      DTptr gas_model_value = tsk_info->get_so_field<DT>(gas_name);
      *gas_model_value <<= 0.0;
    }
  }
  
  //======TIME STEP INITIALIZATION:
  template <typename IT, typename DT>
  void DragModel<IT, DT>::register_timestep_init( std::vector<VariableInformation>& variable_registry ){
  }
  
  template <typename IT, typename DT>
  void DragModel<IT,DT>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                        SpatialOps::OperatorDatabase& opr ){
  }
  
  //======TIME STEP EVALUATION:
  template <typename IT, typename DT>
  void DragModel<IT, DT>::register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ){
    
    for ( int i = 0; i < _N; i++ ){
      //dependent variables(s) or model values
      const std::string name = get_name(i, _base_var_name);
      register_variable( name, _D_type, COMPUTES, 0, NEWDW, variable_registry, time_substep );
      
      const std::string gas_name = get_name(i, _base_gas_var_name);
      register_variable( gas_name, _D_type, COMPUTES, 0, NEWDW, variable_registry, time_substep );
      
      //independent variables
      if ( !constDiameter ) {
        const std::string diameter_name = get_name( i, _base_diameter_name );
        register_variable( diameter_name, _I_type, REQUIRES, 0, LATEST, variable_registry, time_substep );
      }
  
      if ( !constDensity ) {
        const std::string density_name = get_name( i, _base_density_name );
        register_variable( density_name, _I_type, REQUIRES, 0, LATEST, variable_registry, time_substep );
      }
      
      const std::string weight_name = get_name( i, "w" );
      register_variable( weight_name, _I_type, REQUIRES, 0, LATEST, variable_registry, time_substep );
      
      const std::string velocity_name = get_name( i, _base_u_velocity_name );
      register_variable( velocity_name, _I_type, REQUIRES, 0, LATEST, variable_registry, time_substep );
      //using if statements on v/w particle velocities to allow testing in 1&2D
      if ( _base_v_velocity_name != "none" ) {
        const std::string velocity_name = get_name( i, _base_v_velocity_name );
        register_variable( velocity_name, _I_type, REQUIRES, 0, LATEST, variable_registry, time_substep );
      }
      if (_base_w_velocity_name != "none" ) {
        const std::string velocity_name = get_name( i, _base_w_velocity_name );
        register_variable( velocity_name, _I_type, REQUIRES, 0, LATEST, variable_registry, time_substep );
      }
    }
    
    register_variable( _gas_u_velocity_name, _I_type, REQUIRES, 0, LATEST, variable_registry, time_substep );
    register_variable( _gas_v_velocity_name, _I_type, REQUIRES, 0, LATEST, variable_registry, time_substep );
    register_variable( _gas_w_velocity_name, _I_type, REQUIRES, 0, LATEST, variable_registry, time_substep );
    register_variable( _gas_density_name, _I_type, REQUIRES, 0, LATEST, variable_registry, time_substep );
  }
  
  template <typename IT, typename DT>
  void DragModel<IT,DT>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                               SpatialOps::OperatorDatabase& opr ) {
    using namespace SpatialOps;
    using SpatialOps::operator *;
    typedef SpatialOps::SpatFldPtr<DT> DTptr;
    typedef SpatialOps::SpatFldPtr<IT> ITptr;
    typedef typename OperatorTypeBuilder< SpatialOps::Interpolant, IT, DT >::type InterpT;
    const InterpT* const interp = opr.retrieve_operator<InterpT>();

    ITptr rhoG = tsk_info->get_const_so_field<IT>(_gas_density_name);
    SpatialOps::SpatFldPtr<IT> gasVelMag = SpatialFieldStore::get<IT>( *rhoG );
    ITptr velU = tsk_info->get_const_so_field<IT>(_gas_u_velocity_name);
    ITptr velV = tsk_info->get_const_so_field<IT>(_gas_v_velocity_name);
    ITptr velW = tsk_info->get_const_so_field<IT>(_gas_w_velocity_name);
    
    //interpolate velocities in case particle is lagrangian
    *gasVelMag <<= (*interp)(*velU) * (*interp)(*velU);
    *gasVelMag <<= *gasVelMag + (*interp)(*velV) * (*interp)(*velV);
    *gasVelMag <<= *gasVelMag + (*interp)(*velW) * (*interp)(*velW);
    *gasVelMag <<= sqrt( *gasVelMag );
    
    for ( int i = 0; i < _N; i++ ){
      
      const std::string name = get_name(i, _base_var_name);
      DTptr model_value = tsk_info->get_so_field<DT>(name);
      
      const std::string gas_name = get_name(i, _base_gas_var_name);
      DTptr gas_model_value = tsk_info->get_so_field<DT>(gas_name);

      SpatialOps::SpatFldPtr<DT> tauP = SpatialFieldStore::get<DT>( *model_value );
      SpatialOps::SpatFldPtr<DT> Re = SpatialFieldStore::get<DT>( *model_value );
      SpatialOps::SpatFldPtr<DT> fDrag = SpatialFieldStore::get<DT>( *model_value );
      
      SpatialOps::SpatFldPtr<DT> partVelMag = SpatialFieldStore::get<DT>( *model_value );
      const std::string u_vel_name = get_name( i, _base_u_velocity_name );
      ITptr partVelU = tsk_info->get_const_so_field<IT>(u_vel_name);
      ITptr partVelV;
      ITptr partVelW;
      ITptr density;
      ITptr diameter;
      
      *partVelMag <<= *partVelU * *partVelU;
      if ( _base_v_velocity_name != "none") {
        const std::string v_vel_name = get_name( i, _base_v_velocity_name );
        partVelV = tsk_info->get_const_so_field<IT>(v_vel_name);
        *partVelMag <<= *partVelMag + *partVelV * *partVelV;
      }
      if ( _base_w_velocity_name != "none" ) {
        const std::string w_vel_name = get_name( i, _base_w_velocity_name );
        partVelW = tsk_info->get_const_so_field<IT>(w_vel_name);
        *partVelMag <<= *partVelMag + *partVelW * *partVelW;
      }
      *partVelMag <<= sqrt( *partVelMag );

      if (!constDensity && !constDiameter) {
        const std::string density_name = get_name( i, _base_density_name );
        density = tsk_info->get_const_so_field<IT>(density_name);
        const std::string diameter_name = get_name( i, _base_diameter_name );
        diameter = tsk_info->get_const_so_field<IT>(diameter_name);
        *tauP <<= (*density) * (*diameter) * (*diameter) / (18.0 * _visc);
        *Re <<= abs( *gasVelMag - *partVelMag ) * (*diameter) * (*interp)(*rhoG) / _visc ;
      } else if ( !constDensity && constDiameter ) {
        const std::string density_name = get_name( i, _base_density_name );
        density = tsk_info->get_const_so_field<IT>(density_name);
        *tauP <<= (*density) * _d * _d / (18.0 * _visc);
        *Re <<= abs( *gasVelMag - *partVelMag ) * _d * (*interp)(*rhoG) / _visc;
      } else if ( constDensity && !constDiameter ) {
        const std::string diameter_name = get_name( i, _base_diameter_name );
        diameter = tsk_info->get_const_so_field<IT>(diameter_name);
        *tauP <<= _rho * (*diameter) * (*diameter) / (18.0 * _visc);
        *Re <<= abs( *gasVelMag - *partVelMag ) * (*diameter) * (*interp)(*rhoG) / _visc;
      } else {
        *tauP <<= _rho * _d * _d / (18.0 * _visc);
        *Re <<= abs( *gasVelMag - *partVelMag ) * _d * (*interp)(*rhoG) / _visc;
      }
      
      *fDrag <<= cond( *Re < 994.0, 1.0 + 0.15 * pow( (*Re), 0.687) )
                     ( 0.0183* (*Re) );
      //an alternative drag law in case its needed later
      //*fDrag <<= 1.0 + 0.15 * pow( (*Re), 0.687 ) + 0.0175* (*Re) / ( 1.0 + 4.25e4 * pow( (*Re), -1.16) ); //valid over all Re
      
      //compute a rate term
      if ( _direction=="x" ) {
        *model_value <<= cond( *tauP != 0.0, (*fDrag) / (*tauP) * ( *velU - *partVelU ) )
                             (0.0);
      } else if ( _direction=="y" ) {
        *model_value <<= cond( *tauP != 0.0, (*fDrag) / (*tauP) * ( *velV - *partVelV ) )
                             (0.0);
      } else if ( _direction=="z" ) {
        *model_value <<= cond( *tauP != 0.0, (*fDrag) / (*tauP) * ( *velW - *partVelW ) )
                             (0.0);
      }
      
      const std::string w_name = get_name( i, "w" );
      ITptr weight = tsk_info->get_const_so_field<IT>(w_name);
      
      if (!constDensity && !constDiameter) {
        *gas_model_value <<= cond( *diameter!=0.0, - *model_value * *weight * *density / (*interp)(*rhoG) * PI/6.0 * (*diameter) * (*diameter) * (*diameter) )
                                 (0.0);
      } else if (!constDensity && constDiameter ) {
        *gas_model_value <<= - *model_value * *weight * *density / (*interp)(*rhoG) * PI/6.0 * _d * _d * _d;
      } else if (constDensity && !constDiameter ) {
        *gas_model_value <<= cond( *diameter!=0.0, - *model_value * *weight * _rho / (*interp)(*rhoG) * PI/6.0 * (*diameter) * (*diameter) * (*diameter) )
                                 (0.0);
      } else {
        *gas_model_value <<= - *model_value * *weight * _rho / (*interp)(*rhoG) * PI/6.0 * _d * _d * _d;
      }
      
    }
  }
}
#endif 
