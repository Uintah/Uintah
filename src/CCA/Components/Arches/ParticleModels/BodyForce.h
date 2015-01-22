#ifndef Uintah_Component_Arches_BodyForce_h
#define Uintah_Component_Arches_BodyForce_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <spatialops/structured/FVStaggered.h>
#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif
//-------------------------------------------------------

/**
 * @class    Body Force
 * @author   Alex Abboud
 * @date     August 2014
 *
 * @brief    This class sets up the body force on particles due to gravity
 *
 * @details  The class calculates the body force for particles as the simple \f$ du_i/dt = g_i (\rho_p-\rho_g)/\rho_p \f$
 *           When using lagrangian particles N = 1. This generalization should allow for the same code to be utilized for
 *           any particle method - DQMOM, CQMOM, or Lagrangian.
 *
 */

//-------------------------------------------------------

namespace Uintah{
  
  //IT is the independent variable type
  //DT is the dependent variable type
  template <typename IT, typename DT>
  class BodyForce : public TaskInterface {
    
  public:
    
    BodyForce<IT, DT>( std::string task_name, int matl_index, const std::string var_name, const int N );
    ~BodyForce<IT, DT>();
    
    void problemSetup( ProblemSpecP& db );
    
    void create_local_labels();
    
    class Builder : public TaskInterface::TaskBuilder {
      
    public:
      
      Builder( std::string task_name, int matl_index, std::string base_var_name, const int N ) :
      _task_name(task_name), _matl_index(matl_index), _base_var_name(base_var_name), _N(N){}
      ~Builder(){}
      
      BodyForce* build()
      { return scinew BodyForce<IT, DT>( _task_name, _matl_index, _base_var_name, _N ); }
      
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
    std::string _base_density_name;
    std::string _gas_density_name;
    std::string _direction;
    VAR_TYPE _D_type;
    VAR_TYPE _I_type;
    
    const int _N;                 //<<< The number of "environments"
    double _g;
    double _rho;
    bool constDensity;
    Vector gravity;
    
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
  void BodyForce<IT,DT>::create_local_labels(){
    for ( int i = 0; i < _N; i++ ){
      const std::string name = get_name(i, _base_var_name);
      register_new_variable(name, _D_type);
    }
  }
  
  template <typename IT, typename DT>
  BodyForce<IT, DT>::BodyForce( std::string task_name, int matl_index,
                                const std::string base_var_name, const int N ) :
  TaskInterface( task_name, matl_index ), _base_var_name(base_var_name), _N(N){

    VarTypeHelper<DT> dhelper; 
    _D_type = dhelper.get_vartype(); 

    VarTypeHelper<IT> ihelper; 
    _I_type = ihelper.get_vartype(); 

  }
  
  template <typename IT, typename DT>
  BodyForce<IT, DT>::~BodyForce()
  {}
  
  template <typename IT, typename DT>
  void BodyForce<IT, DT>::problemSetup( ProblemSpecP& db ){

    std::string tempType;
    db->findBlock("particle_density")->getAttribute("type",tempType);
    if ( tempType == "constant" ) {
      db->findBlock("particle_density")->require("constant",_rho);
      constDensity = true;
    } else {
      db->require("density_label",_base_density_name);
      constDensity = false;
    }

    db->require("direction",_direction);
    _gas_density_name = "densityCP";
    const ProblemSpecP params_root = db->getRootNode();
    if (params_root->findBlock("PhysicalConstants")) {
      ProblemSpecP db_phys = params_root->findBlock("PhysicalConstants");
      db_phys->require("gravity", gravity);
    }
    
    if ( _direction == "x" ) {
      _g = gravity.x();
    } else if ( _direction == "y" ) {
      _g = gravity.y();
    } else if ( _direction == "z" ) {
      _g = gravity.z();
    }
  }
  
  //======INITIALIZATION:
  template <typename IT, typename DT>
  void BodyForce<IT, DT>::register_initialize( std::vector<VariableInformation>& variable_registry ){
    
    for ( int i = 0; i < _N; i++ ){
      const std::string name = get_name(i, _base_var_name);
      register_variable( name, _D_type, COMPUTES, 0, NEWDW, variable_registry );
      
    }
  }
  
  template <typename IT, typename DT>
  void BodyForce<IT,DT>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
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
  void BodyForce<IT, DT>::register_timestep_init( std::vector<VariableInformation>& variable_registry ){
  }
  
  template <typename IT, typename DT>
  void BodyForce<IT,DT>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                            SpatialOps::OperatorDatabase& opr ){
  }
  
  //======TIME STEP EVALUATION:
  template <typename IT, typename DT>
  void BodyForce<IT, DT>::register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ){
    
    for ( int i = 0; i < _N; i++ ){
      //dependent variables(s) or model values
      const std::string name = get_name(i, _base_var_name);
      register_variable( name, _D_type, COMPUTES, 0, NEWDW, variable_registry, time_substep );
      
      //independent variables
      if ( !constDensity ) {
        const std::string density_name = get_name( i, _base_density_name );
        register_variable( density_name, _I_type, REQUIRES, 0, LATEST, variable_registry, time_substep );
      }
    }

    register_variable( _gas_density_name, _I_type, REQUIRES, 0, LATEST, variable_registry, time_substep );
  }
  
  template <typename IT, typename DT>
  void BodyForce<IT,DT>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                   SpatialOps::OperatorDatabase& opr ) {
    using namespace SpatialOps;
    using SpatialOps::operator *;
    typedef SpatialOps::SpatFldPtr<DT> DTptr;
    typedef SpatialOps::SpatFldPtr<IT> ITptr;
    typedef typename OperatorTypeBuilder< SpatialOps::Interpolant, IT, DT >::type InterpT;
    const InterpT* const interp = opr.retrieve_operator<InterpT>();
    
    ITptr rhoG = tsk_info->get_const_so_field<IT>(_gas_density_name);
    
    for ( int i = 0; i < _N; i++ ){
      
      const std::string name = get_name(i, _base_var_name);
      DTptr model_value = tsk_info->get_so_field<DT>(name);
      ITptr rhoP;
    
      
      if (!constDensity ) {
        const std::string density_name = get_name( i, _base_density_name );
        rhoP = tsk_info->get_const_so_field<IT>(density_name);
      }
      
      //compute a rate term
      if ( !constDensity  ) {
        *model_value <<= _g * ( *rhoP  - (*interp)(*rhoG) ) / (*rhoP);
      } else {
        *model_value <<= _g * ( _rho - (*interp)(*rhoG) ) / _rho;
      }

    }
  }
}
#endif
