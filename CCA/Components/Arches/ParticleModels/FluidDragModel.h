#ifndef Uintah_Component_Arches_FluidDragModel_h
#define Uintah_Component_Arches_FluidDragModel_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <spatialops/structured/FVStaggered.h>
#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif
//-------------------------------------------------------

/**
 * @class    Fluid Drag Model
 * @author   Alex Abboud
 * @date     August 2014
 *
 * @brief    This class should be coupled with the Drag Model term for the gas phase of the system
 *
 * @details  The class calculates the drag term for the gas phase only for each of the quadrature nodes of the system.
 *           The schedulign of this class needs to occur slightly differently to use as a gas source instead of particle one.
 *           When using lagrangian particles N = 1. This generalization should allow for the same code to be utilized for
 *           any particle method - DQMOM, CQMOM, or Lagrangian.
 *
 */

//-------------------------------------------------------

namespace Uintah{
  
  //IT is the independent variable type
  //DT is the dependent variable type
  template <typename IT, typename DT>
  class FluidDragModel : public TaskInterface {
    
  public:
    
    FluidDragModel<IT, DT>( std::string task_name, int matl_index, const std::string var_name, const int N );
    ~FluidDragModel<IT, DT>();
    
    void problemSetup( ProblemSpecP& db );
    
    void create_local_labels();
    
    class Builder : public TaskInterface::TaskBuilder {
      
    public:
      
      Builder( std::string task_name, int matl_index, std::string base_var_name, const int N ) :
      _task_name(task_name), _matl_index(matl_index), _base_var_name(base_var_name), _N(N){}
      ~Builder(){}
      
      FluidDragModel* build()
      { return scinew FluidDragModel<IT, DT>( _task_name, _matl_index, _base_var_name, _N ); }
      
    private:
      
      std::string _task_name;
      int _matl_index;
      std::string _base_var_name;
      const int _N;
      
    };
    
  protected:
    
    void register_initialize( std::vector<VariableInformation>& variable_registry );
    
    void register_timestep_init( std::vector<VariableInformation>& variable_registry );
    
    void register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep );
    
    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                    SpatialOps::OperatorDatabase& opr );
    
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                       SpatialOps::OperatorDatabase& opr );
    
    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
              SpatialOps::OperatorDatabase& opr );
    
  private:
    
    const std::string _base_var_name;
    std::string _base_radius_name;
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
    
    double _kVisc;
    double _rho;
    double _r;
    
    bool constDensity;
    bool constRadius;
    
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
  void FluidDragModel<IT,DT>::create_local_labels(){
    for ( int i = 0; i < _N; i++ ){
      const std::string name = get_name(i, _base_var_name);
      register_new_variable(name, _D_type);
    }
  }
  
  template <typename IT, typename DT>
  FluidDragModel<IT, DT>::FluidDragModel( std::string task_name, int matl_index,
                               const std::string base_var_name, const int N ) :
  _base_var_name(base_var_name), TaskInterface( task_name, matl_index ), _N(N){
  }
  
  template <typename IT, typename DT>
  FluidDragModel<IT, DT>::~FluidDragModel()
  {}
  
  template <typename IT, typename DT>
  void FluidDragModel<IT, DT>::problemSetup( ProblemSpecP& db ){
    //This sets the type of the independent and dependent variable types as needed by the variable
    //registration step.
    DT* d_test;
    IT* i_test;
    
    set_type(d_test, _D_type);
    set_type(i_test, _I_type);
    
    db->require("kinematic_viscosity",_kVisc);
    
    db->getWithDefault("u_velocity_label",_base_u_velocity_name,"none");
    db->getWithDefault("v_velocity_label",_base_v_velocity_name,"none");
    db->getWithDefault("w_velocity_label",_base_w_velocity_name,"none");
    
    std::string tempType;
    db->findBlock("particle_density")->getAttribute("type",tempType);
    if ( tempType == "constant" ) {
      db->findBlock("particle_density")->require("constant",_rho);
      constDensity = true;
    } else {
      db->require("density_label",_base_density_name);
      constDensity = false;
    }
    
    db->findBlock("radius")->getAttribute("type",tempType);
    if ( tempType == "constant" ) {
      db->findBlock("radius")->require("constant",_r);
      constRadius = true;
    } else {
      db->require("radius_label",_base_radius_name);
      constRadius = false;
    }
    
    db->require("direction",_direction);
    
    _gas_u_velocity_name = "CCUVelocity";
    _gas_v_velocity_name = "CCVVelocity";
    _gas_w_velocity_name = "CCWVelocity";
    _gas_density_name = "densityCP";
  }
  
  //======INITIALIZATION:
  template <typename IT, typename DT>
  void FluidDragModel<IT, DT>::register_initialize( std::vector<VariableInformation>& variable_registry ){
    
    for ( int i = 0; i < _N; i++ ){
      const std::string name = get_name(i, _base_var_name);
      std::cout << "Source label " << name << std::endl;
      register_variable( name, _D_type, LOCAL_COMPUTES, 0, NEWDW, variable_registry );
      
    }
  }
  
  template <typename IT, typename DT>
  void FluidDragModel<IT,DT>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                    SpatialOps::OperatorDatabase& opr ){
    
    using namespace SpatialOps;
    using SpatialOps::operator *;
    typedef SpatialOps::SpatFldPtr<DT> DTptr;
    
    for ( int i = 0; i < _N; i++ ){
      const std::string name = get_name(i, _base_var_name);
      DTptr model_value = tsk_info->get_so_field<DT>(name);
      
      *model_value <<= 0.0;
    }
  }
  
  //======TIME STEP INITIALIZATION:
  template <typename IT, typename DT>
  void FluidDragModel<IT, DT>::register_timestep_init( std::vector<VariableInformation>& variable_registry ){
  }
  
  template <typename IT, typename DT>
  void FluidDragModel<IT,DT>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                       SpatialOps::OperatorDatabase& opr ){
  }
  
  //======TIME STEP EVALUATION:
  template <typename IT, typename DT>
  void FluidDragModel<IT, DT>::register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ){
    
    for ( int i = 0; i < _N; i++ ){
      //dependent variables(s) or model values
      const std::string name = get_name(i, _base_var_name);
      register_variable( name, _D_type, COMPUTES, 0, NEWDW, variable_registry, time_substep );
      
      //independent variables
      if ( !constRadius ) {
        const std::string radius_name = get_name( i, _base_radius_name );
        register_variable( radius_name, _I_type, REQUIRES, 0, LATEST, variable_registry, time_substep );
      }
      
      if ( !constDensity ) {
        const std::string density_name = get_name( i, _base_density_name );
        register_variable( density_name, _I_type, REQUIRES, 0, LATEST, variable_registry, time_substep );
      }
      
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
  void FluidDragModel<IT,DT>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
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
      
      SpatialOps::SpatFldPtr<DT> tauP = SpatialFieldStore::get<DT>( *model_value );
      SpatialOps::SpatFldPtr<DT> Re = SpatialFieldStore::get<DT>( *model_value );
      SpatialOps::SpatFldPtr<DT> psi = SpatialFieldStore::get<DT>( *model_value );
      
      SpatialOps::SpatFldPtr<DT> partVelMag = SpatialFieldStore::get<DT>( *model_value );
      const std::string u_vel_name = get_name( i, _base_u_velocity_name );
      ITptr partVelU = tsk_info->get_const_so_field<IT>(u_vel_name);
      ITptr partVelV;
      ITptr partVelW;
      
      *partVelMag <<= *partVelU * *partVelU;
      if ( _base_v_velocity_name != "none") {
        const std::string v_vel_name = get_name( i, _base_v_velocity_name );
        partVelV = tsk_info->get_const_so_field<IT>(v_vel_name);
        *partVelMag <<= *partVelMag + *partVelV * *partVelV;
      }
      if ( _base_w_velocity_name != "none" ) {
        const std::string w_vel_name = get_name( i, _base_w_velocity_name );
        ITptr partVelW = tsk_info->get_const_so_field<IT>(w_vel_name);
        *partVelMag <<= *partVelMag + *partVelW * *partVelW;
      }
      *partVelMag <<= sqrt( *partVelMag );
      
      
      ITptr density;
      ITptr radius;
      
      
      if (!constDensity && !constRadius) {
        const std::string density_name = get_name( i, _base_density_name );
        density = tsk_info->get_const_so_field<IT>(density_name);
        const std::string radius_name = get_name( i, _base_radius_name );
        radius = tsk_info->get_const_so_field<IT>(radius_name);
        *tauP <<= (*density) * (*radius) * (*radius) / (18.0 * _kVisc);
        *Re <<= abs( *gasVelMag - *partVelMag ) * (*radius) / ( _kVisc / (*interp)(*rhoG) );
      } else if ( !constDensity && constRadius ) {
        const std::string density_name = get_name( i, _base_density_name );
        density = tsk_info->get_const_so_field<IT>(density_name);
        *tauP <<= (*density) * _r * _r / (18.0 * _kVisc);
        *Re <<= abs( *gasVelMag - *partVelMag ) * _r / ( _kVisc / (*interp)(*rhoG) );
      } else if ( constDensity && !constRadius ) {
        const std::string radius_name = get_name( i, _base_radius_name );
        radius = tsk_info->get_const_so_field<IT>(radius_name);
        *tauP <<= _rho * (*radius) * (*radius) / (18.0 * _kVisc);
        *Re <<= abs( *gasVelMag - *partVelMag ) * (*radius) / ( _kVisc / (*interp)(*rhoG) );
      } else {
        *tauP <<= _rho * _r * _r / (18.0 * _kVisc);
        *Re <<= abs( *gasVelMag - *partVelMag ) * _r / ( _kVisc / (*interp)(*rhoG) );
      }
      
      *psi <<= cond( *Re < 1.0, 1.0 )
                   ( *Re > 1000.0, 0.183* (*Re) )
                   ( 1.0 + 0.15 * pow( (*Re), 0.687) );
      
      //ADD FOR GOR GET OVER THE WEIGHTS
      const std::string w_name = get_name( i, "w" );
      ITptr weight = tsk_info->get_const_so_field<IT>(w_name);
      
      //compute a rate term
      if ( !constDensity && !constRadius ) {
        if ( _direction=="x" ) {
          *model_value <<= -(*weight) * (*density) * PI * (*psi) / 6.0 / (*tauP) * ( *velU - *partVelU ) * (*radius) * (*radius) * (*radius);
        } else if ( _direction=="y" ) {
          *model_value <<= -(*weight) * (*density) * PI * (*psi) / 6.0 / (*tauP) * ( *velV - *partVelV ) * (*radius) * (*radius) * (*radius);
        } else if ( _direction=="z" ) {
          *model_value <<= -(*weight) * (*density) * PI * (*psi) / 6.0 / (*tauP) * ( *velW - *partVelW ) * (*radius) * (*radius) * (*radius);
        }
      } else if ( !constDensity && constRadius ) {
        if ( _direction=="x" ) {
          *model_value <<= -(*weight) * (*density) * PI * (*psi) / 6.0 / (*tauP) * ( *velU - *partVelU ) * _r * _r * _r;
        } else if ( _direction=="y" ) {
          *model_value <<= -(*weight) * (*density) * PI * (*psi) / 6.0 / (*tauP) * ( *velV - *partVelV ) * _r * _r * _r;
        } else if ( _direction=="z" ) {
          *model_value <<= -(*weight) * (*density) * PI * (*psi) / 6.0 / (*tauP) * ( *velW - *partVelW ) * _r * _r * _r;
        }
      } else if ( constDensity && !constRadius ) {
        if ( _direction=="x" ) {
          *model_value <<= -(*weight) * (_rho) * PI * (*psi) / 6.0 / (*tauP) * ( *velU - *partVelU ) * (*radius) * (*radius) * (*radius);
        } else if ( _direction=="y" ) {
          *model_value <<= -(*weight) * (_rho) * PI * (*psi) / 6.0 / (*tauP) * ( *velV - *partVelV ) * (*radius) * (*radius) * (*radius);
        } else if ( _direction=="z" ) {
          *model_value <<= -(*weight) * (_rho) * PI * (*psi) / 6.0 / (*tauP) * ( *velW - *partVelW ) * (*radius) * (*radius) * (*radius);
        }
      } else {
        if ( _direction=="x" ) {
          *model_value <<= -(*weight) * (_rho) * PI * (*psi) / 6.0 / (*tauP) * ( *velU - *partVelU ) * _r * _r * _r;
        } else if ( _direction=="y" ) {
          *model_value <<= -(*weight) * (_rho) * PI * (*psi) / 6.0 / (*tauP) * ( *velV - *partVelV ) * _r * _r * _r;
        } else if ( _direction=="z" ) {
          *model_value <<= -(*weight) * (_rho) * PI * (*psi) / 6.0 / (*tauP) * ( *velW - *partVelW ) * _r * _r * _r;
        }
      }
      
    }
  }
}
#endif
