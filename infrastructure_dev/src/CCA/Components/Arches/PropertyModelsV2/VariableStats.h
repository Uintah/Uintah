#ifndef Uintah_Component_Arches_VariableStats_h
#define Uintah_Component_Arches_VariableStats_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <spatialops/structured/FVStaggered.h>


namespace Uintah{ 

  class Operators; 
  template< typename T>
  class VariableStats : public TaskInterface { 

public: 

    typedef SpatialOps::SVolField SVolF;
    typedef SpatialOps::SpatFldPtr<SVolF> SVolFP; 
    typedef SpatialOps::SpatFldPtr<T> TP; 
    typedef SpatialOps::XVolField XVolF; 
    typedef SpatialOps::YVolField YVolF; 
    typedef SpatialOps::ZVolField ZVolF; 
    typedef SpatialOps::SpatFldPtr<XVolF> XVolFP; 
    typedef SpatialOps::SpatFldPtr<YVolF> YVolFP; 
    typedef SpatialOps::SpatFldPtr<ZVolF> ZVolFP; 
    //interpolants
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::XVolField, SVolF >::type IX;
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::YVolField, SVolF >::type IY;
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::ZVolField, SVolF >::type IZ;
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, T, SVolF >::type ITP; //Current type (T) to the p-cell center. 

    typedef std::vector<VariableInformation> VIVec; 

    VariableStats<T>( std::string task_name, int matl_index, SimulationStateP& shared_state ); 
    ~VariableStats(); 

    void problemSetup( ProblemSpecP& db ); 

    void register_initialize( VIVec& variable_registry );

    void register_timestep_init( VIVec& variable_registry );

    void register_restart_initialize( VIVec& variable_registry );

    void register_timestep_eval( VIVec& variable_registry, const int time_substep ); 

    void register_compute_bcs( VIVec& variable_registry, const int time_substep ){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                      SpatialOps::OperatorDatabase& opr ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                     SpatialOps::OperatorDatabase& opr );

    void restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                             SpatialOps::OperatorDatabase& opr );
    
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                        SpatialOps::OperatorDatabase& opr );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
               SpatialOps::OperatorDatabase& opr );

    void create_local_labels(); 


    //Build instructions for this (VariableStats) class. 
    class Builder : public TaskInterface::TaskBuilder { 

      public: 

      Builder( std::string task_name, int matl_index, SimulationStateP& shared_state ) 
        : _task_name(task_name), _matl_index(matl_index), _shared_state(shared_state){}
      ~Builder(){}

      VariableStats* build()
      { return new VariableStats<T>( _task_name, _matl_index, _shared_state ); }

      private: 

      std::string _task_name; 
      int _matl_index; 
      SimulationStateP _shared_state; 

    };

private: 

    std::vector<const VarLabel*> _ave_sum_labels; 
    std::vector<const VarLabel*> _ave_flux_sum_labels; 
    std::vector<const VarLabel*> _sqr_sum_labels; 

    //single variables
    std::vector<std::string> _ave_sum_names; 
    std::vector<std::string> _base_var_names; 
    std::vector<std::string> _new_variables; 
    std::vector<std::string> _sqr_variable_names; 

    std::string _rho_name;

    //fluxes
    bool _no_flux; 
    struct FluxInfo{ 
      std::string phi; 
      bool do_phi;
    };

    std::vector<std::string> _ave_x_flux_sum_names; 
    std::vector<std::string> _ave_y_flux_sum_names; 
    std::vector<std::string> _ave_z_flux_sum_names; 

    std::vector<std::string> _x_flux_sqr_sum_names; 
    std::vector<std::string> _y_flux_sqr_sum_names; 
    std::vector<std::string> _z_flux_sqr_sum_names; 
    std::vector<FluxInfo>    _flux_sum_info; 

    SimulationStateP _shared_state; 

    VAR_TYPE _V_type; 


  }; //end class header

  //---------------- CLASS METHODS ------------------

  using namespace SpatialOps;
  using SpatialOps::operator *; 
  
  template <typename T>
  VariableStats<T>::VariableStats( std::string task_name, int matl_index, SimulationStateP& shared_state ) : 
  TaskInterface( task_name, matl_index ) { 
  
    _shared_state = shared_state; 

    VarTypeHelper<T> dhelper; 
    _V_type = dhelper.get_vartype(); 
  
  }
  
  template <typename T>
  VariableStats<T>::~VariableStats(){ 
  
    std::vector<const VarLabel*>::iterator i = _ave_sum_labels.begin(); 
    for (;i!=_ave_sum_labels.end();i++){ 
  
      VarLabel::destroy(*i); 
  
    }

    i = _sqr_sum_labels.begin(); 
    for (;i!=_sqr_sum_labels.end();i++){ 
  
      VarLabel::destroy(*i); 
  
    }
  
    i = _ave_flux_sum_labels.begin(); 
    for (;i!=_ave_flux_sum_labels.end();i++){ 
  
      VarLabel::destroy(*i); 
  
    }
  
  }
  
  template <typename T>
  void VariableStats<T>::problemSetup( ProblemSpecP& db ){ 
  
    for ( ProblemSpecP var_db = db->findBlock("single_variable"); var_db != 0; 
          var_db = var_db->findNextBlock("single_variable") ){ 
  
      std::string var_name; 
      var_db->getAttribute("label", var_name);
  
      std::string var_ave_name = var_name + "_running_sum"; 
      std::string sqr_name = var_name + "_squared_sum";
  
      const VarLabel* label_sum = VarLabel::create( var_ave_name, CCVariable<double>::getTypeDescription() );
      const VarLabel* sqr_label = VarLabel::create( sqr_name    , CCVariable<double>::getTypeDescription() );
  
      _ave_sum_labels.push_back(label_sum); 
      _sqr_sum_labels.push_back(sqr_label); 
  
      _ave_sum_names.push_back(var_ave_name); 
      _sqr_variable_names.push_back(sqr_name); 
  
      _base_var_names.push_back(var_name); 
  
      if (var_db->findBlock("new")){ 
        _new_variables.push_back(var_ave_name); 
      }
  
    }
  
    bool do_fluxes = false; 
  
    for ( ProblemSpecP var_db = db->findBlock("flux_variable"); var_db != 0; 
          var_db = var_db->findNextBlock("flux_variable") ){
  
      do_fluxes = true; 
  
      std::string phi_name; 
      std::string flux_name; 
  
      var_db->getAttribute("label",flux_name);
  
      std::string x_var_name = flux_name + "_running_sum_x"; 
      std::string y_var_name = flux_name + "_running_sum_y"; 
      std::string z_var_name = flux_name + "_running_sum_z"; 

      std::string x_var_sqr_name = flux_name + "_squared_sum_x"; 
      std::string y_var_sqr_name = flux_name + "_squared_sum_y"; 
      std::string z_var_sqr_name = flux_name + "_squared_sum_z"; 
  
      const VarLabel* x_flux_label_sum = VarLabel::create( x_var_name, CCVariable<double>::getTypeDescription() );
      const VarLabel* y_flux_label_sum = VarLabel::create( y_var_name, CCVariable<double>::getTypeDescription() );
      const VarLabel* z_flux_label_sum = VarLabel::create( z_var_name, CCVariable<double>::getTypeDescription() );

      const VarLabel* x_flux_label_sqr_sum = VarLabel::create( x_var_sqr_name, CCVariable<double>::getTypeDescription() );
      const VarLabel* y_flux_label_sqr_sum = VarLabel::create( y_var_sqr_name, CCVariable<double>::getTypeDescription() );
      const VarLabel* z_flux_label_sqr_sum = VarLabel::create( z_var_sqr_name, CCVariable<double>::getTypeDescription() );
  
      _ave_flux_sum_labels.push_back(x_flux_label_sum); 
      _ave_flux_sum_labels.push_back(y_flux_label_sum); 
      _ave_flux_sum_labels.push_back(z_flux_label_sum); 

      _sqr_sum_labels.push_back(x_flux_label_sqr_sum); 
      _sqr_sum_labels.push_back(y_flux_label_sqr_sum); 
      _sqr_sum_labels.push_back(z_flux_label_sqr_sum); 
  
      _ave_x_flux_sum_names.push_back(x_var_name); 
      _ave_y_flux_sum_names.push_back(y_var_name); 
      _ave_z_flux_sum_names.push_back(z_var_name); 

      _x_flux_sqr_sum_names.push_back(x_var_sqr_name); 
      _y_flux_sqr_sum_names.push_back(y_var_sqr_name); 
      _z_flux_sqr_sum_names.push_back(z_var_sqr_name); 
  
      //get required information: 
      phi_name = "NA";
      var_db->getAttribute("phi",phi_name); 
  
      FluxInfo fi; 
      fi.phi = phi_name; 
  
      if ( phi_name == "NA" ){ 
        fi.do_phi = false; 
      } else { 
        fi.do_phi = true;
      }
  
      _flux_sum_info.push_back(fi); 
  
      if (var_db->findBlock("new")){ 
        _new_variables.push_back(x_var_name); 
        _new_variables.push_back(y_var_name); 
        _new_variables.push_back(z_var_name); 
      }
  
    }
  
    if ( do_fluxes ){ 
      if ( db->findBlock("density")){ 
        db->findBlock("density")->getAttribute("label", _rho_name);
        _no_flux = false; 
      } else { 
        _no_flux = true; 
        throw ProblemSetupException("Error: For time_ave property; must specify a density label for fluxes.",__FILE__,__LINE__);
      }
    } else { 
      _no_flux = true; 
    }
  
    for ( ProblemSpecP var_db = db->findBlock("new_single_variable"); var_db != 0; 
          var_db = var_db->findNextBlock("new_single_variable") ){
  
      std::string name; 
      var_db->getAttribute("label", name); 
      std::string final_name = name + "_running_sum"; 
      _new_variables.push_back( final_name );
      
    }
  
    for ( ProblemSpecP var_db = db->findBlock("new_flux_variable"); var_db != 0; 
          var_db = var_db->findNextBlock("new_flux_variable") ){
  
      std::string name; 
      var_db->getAttribute("label", name); 
      std::string final_name = name + "_running_sum_x"; 
      _new_variables.push_back( final_name );
      final_name = name + "_running_sum_y"; 
      _new_variables.push_back( final_name );
      final_name = name + "_running_sum_z"; 
      _new_variables.push_back( final_name );

      final_name = name + "_squared_sum_x";
      _new_variables.push_back( final_name ); 
      final_name = name + "_squared_sum_y";
      _new_variables.push_back( final_name ); 
      final_name = name + "_squared_sum_z";
      _new_variables.push_back( final_name ); 
      
    }
  
  }
  
  template <typename T>
  void VariableStats<T>::create_local_labels(){ 
  }
  
  //
  //------------------------------------------------
  //-------------- INITIALIZATION ------------------
  //------------------------------------------------
  //
  
  template <typename T>
  void VariableStats<T>::register_initialize( VIVec& variable_registry ){ 
  
    
    std::vector<std::string>::iterator i = _ave_sum_names.begin(); 
    for (;i!=_ave_sum_names.end();i++){ 
  
      register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 
  
    }

    i = _sqr_variable_names.begin(); 
    for (;i!=_sqr_variable_names.end();i++){

      register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 

    }
  
    i = _ave_x_flux_sum_names.begin(); 
    for (;i!=_ave_x_flux_sum_names.end();i++){ 
  
      register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 
  
    }
    i = _ave_y_flux_sum_names.begin(); 
    for (;i!=_ave_y_flux_sum_names.end();i++){ 
  
      register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 
  
    }
    i = _ave_z_flux_sum_names.begin(); 
    for (;i!=_ave_z_flux_sum_names.end();i++){ 
  
      register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 
  
    }
    i = _x_flux_sqr_sum_names.begin(); 
    for (;i!=_x_flux_sqr_sum_names.end();i++){ 
  
      register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 
  
    }
    i = _y_flux_sqr_sum_names.begin(); 
    for (;i!=_y_flux_sqr_sum_names.end();i++){ 
  
      register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 
  
    }
    i = _z_flux_sqr_sum_names.begin(); 
    for (;i!=_z_flux_sqr_sum_names.end();i++){ 
  
      register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 
  
    }
  
  }
  
  template <typename T>
  void VariableStats<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                        SpatialOps::OperatorDatabase& opr ){ 
  
    using namespace SpatialOps;
    using SpatialOps::operator *; 
  
    std::vector<std::string>::iterator i = _ave_sum_names.begin(); 
    for (;i!=_ave_sum_names.end();i++){ 
      SVolFP var = tsk_info->get_so_field<SVolF>( *i ); 
      *var <<= 0.0;
    }

    i = _sqr_variable_names.begin(); 
    for (;i!=_sqr_variable_names.end();i++){
      SVolFP var = tsk_info->get_so_field<SVolF>( *i ); 
      *var <<= 0.0;
    }
  
    i = _ave_x_flux_sum_names.begin(); 
    for (;i!=_ave_x_flux_sum_names.end();i++){ 
      SVolFP var = tsk_info->get_so_field<SVolF>( *i ); 
      *var <<= 0.0;
    }
  
    i = _ave_y_flux_sum_names.begin(); 
    for (;i!=_ave_y_flux_sum_names.end();i++){ 
      SVolFP var = tsk_info->get_so_field<SVolF>( *i ); 
      *var <<= 0.0;
    }
  
    i = _ave_z_flux_sum_names.begin(); 
    for (;i!=_ave_z_flux_sum_names.end();i++){ 
      SVolFP var = tsk_info->get_so_field<SVolF>( *i ); 
      *var <<= 0.0;
    }
    i = _x_flux_sqr_sum_names.begin(); 
    for (;i!=_x_flux_sqr_sum_names.end();i++){ 
      SVolFP var = tsk_info->get_so_field<SVolF>( *i ); 
      *var <<= 0.0;
    }
    i = _y_flux_sqr_sum_names.begin(); 
    for (;i!=_y_flux_sqr_sum_names.end();i++){ 
      SVolFP var = tsk_info->get_so_field<SVolF>( *i ); 
      *var <<= 0.0;
    }
    i = _z_flux_sqr_sum_names.begin(); 
    for (;i!=_z_flux_sqr_sum_names.end();i++){ 
      SVolFP var = tsk_info->get_so_field<SVolF>( *i ); 
      *var <<= 0.0;
    }
  
  }
  
  template <typename T>
  void VariableStats<T>::register_restart_initialize( VIVec& variable_registry ){ 
  
    typedef std::vector<std::string> StrVec; 
  
    for ( StrVec::iterator i = _new_variables.begin(); i != _new_variables.end(); i++ ){
      register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 
    }
    
  }
  
  template <typename T>
  void VariableStats<T>::restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                               SpatialOps::OperatorDatabase& opr ){ 
  
    using namespace SpatialOps;
    using SpatialOps::operator *; 
    typedef std::vector<std::string> StrVec; 
  
    for ( StrVec::iterator i = _new_variables.begin(); i != _new_variables.end(); i++ ){

      SVolFP variable = tsk_info->get_so_field<SVolF>(*i); 
  
      *variable <<= 0.0; 
  
    }
  }
  
  //
  //------------------------------------------------
  //------------- TIMESTEP INIT --------------------
  //------------------------------------------------
  //
  template <typename T>
  void VariableStats<T>::register_timestep_init( VIVec& variable_registry ){ 
  
    std::vector<std::string>::iterator i = _ave_sum_names.begin(); 
    for (;i!=_ave_sum_names.end();i++){ 
  
      register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 
  
    }

    i = _sqr_variable_names.begin(); 
    for (;i!=_sqr_variable_names.end();i++){
  
      register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 
  
    }
  
    i = _ave_x_flux_sum_names.begin(); 
    for (;i!=_ave_x_flux_sum_names.end();i++){ 
  
      register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 
  
    }
  
    i = _ave_y_flux_sum_names.begin(); 
    for (;i!=_ave_y_flux_sum_names.end();i++){ 
  
      register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 
  
    }
  
    i = _ave_z_flux_sum_names.begin(); 
    for (;i!=_ave_z_flux_sum_names.end();i++){ 
  
      register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 
  
    }

    i = _x_flux_sqr_sum_names.begin(); 
    for (;i!=_x_flux_sqr_sum_names.end();i++){ 
  
      register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 
  
    }

    i = _y_flux_sqr_sum_names.begin(); 
    for (;i!=_y_flux_sqr_sum_names.end();i++){ 
  
      register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 
  
    }

    i = _z_flux_sqr_sum_names.begin(); 
    for (;i!=_z_flux_sqr_sum_names.end();i++){ 
  
      register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 
  
    }

  }
  
  template <typename T>
  void VariableStats<T>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                          SpatialOps::OperatorDatabase& opr ){ 
  
    using namespace SpatialOps;
    using SpatialOps::operator *; 
  
    std::vector<std::string>::iterator i = _ave_sum_names.begin(); 
    for (;i!=_ave_sum_names.end();i++){ 
  
      SVolFP var = tsk_info->get_so_field<SVolF>( *i ); 
      *var <<= 0.0;
  
    }

    i = _sqr_variable_names.begin(); 
    for (;i!=_sqr_variable_names.end();i++){

      SVolFP var = tsk_info->get_so_field<SVolF>( *i ); 
      *var <<= 0.0;

    }

    i = _ave_x_flux_sum_names.begin(); 
    for (;i!=_ave_x_flux_sum_names.end();i++){

      SVolFP var = tsk_info->get_so_field<SVolF>( *i ); 
      *var <<= 0.0;

    }
  
    i = _ave_y_flux_sum_names.begin(); 
    for (;i!=_ave_y_flux_sum_names.end();i++){

      SVolFP var = tsk_info->get_so_field<SVolF>( *i ); 
      *var <<= 0.0;

    }

    i = _ave_z_flux_sum_names.begin(); 
    for (;i!=_ave_z_flux_sum_names.end();i++){

      SVolFP var = tsk_info->get_so_field<SVolF>( *i ); 
      *var <<= 0.0;

    }

    i = _x_flux_sqr_sum_names.begin(); 
    for (;i!=_x_flux_sqr_sum_names.end();i++){

      SVolFP var = tsk_info->get_so_field<SVolF>( *i ); 
      *var <<= 0.0;

    }

    i = _y_flux_sqr_sum_names.begin(); 
    for (;i!=_y_flux_sqr_sum_names.end();i++){

      SVolFP var = tsk_info->get_so_field<SVolF>( *i ); 
      *var <<= 0.0;

    }

    i = _z_flux_sqr_sum_names.begin(); 
    for (;i!=_z_flux_sqr_sum_names.end();i++){

      SVolFP var = tsk_info->get_so_field<SVolF>( *i ); 
      *var <<= 0.0;

    }
  
  }
  
  //
  //------------------------------------------------
  //------------- TIMESTEP WORK --------------------
  //------------------------------------------------
  //
  
  template <typename T>
  void VariableStats<T>::register_timestep_eval( VIVec& variable_registry, const int time_substep ){ 

    int nGhost; 

    if ( _V_type == CC_DOUBLE ){ 
      nGhost = 0;
    } else { 
      nGhost = 1; 
    }
  
    std::vector<std::string>::iterator i = _ave_sum_names.begin(); 
    for (;i!=_ave_sum_names.end();i++){ 
  
      register_variable( *i, CC_DOUBLE, MODIFIES, variable_registry ); 
      register_variable( *i, CC_DOUBLE, REQUIRES, 0, OLDDW, variable_registry ); 
  
    }

    i = _sqr_variable_names.begin(); 
    for (;i!=_sqr_variable_names.end();i++){ 
  
      register_variable( *i, CC_DOUBLE, MODIFIES, variable_registry ); 
      register_variable( *i, CC_DOUBLE, REQUIRES, 0, OLDDW, variable_registry ); 
  
    }
  
    i = _base_var_names.begin(); 
    for (;i!=_base_var_names.end();i++){ 
  
      register_variable( *i, _V_type, REQUIRES, nGhost, NEWDW, variable_registry ); 
  
    }
  
    if ( !_no_flux ){ 
      i = _ave_x_flux_sum_names.begin(); 
      for (;i!=_ave_x_flux_sum_names.end();i++){ 
  
        register_variable( *i, CC_DOUBLE, MODIFIES, variable_registry ); 
        register_variable( *i, CC_DOUBLE, REQUIRES, 0, OLDDW, variable_registry ); 
  
      }
  
      i = _ave_y_flux_sum_names.begin(); 
      for (;i!=_ave_y_flux_sum_names.end();i++){ 
  
        register_variable( *i, CC_DOUBLE, MODIFIES, variable_registry ); 
        register_variable( *i, CC_DOUBLE, REQUIRES, 0, OLDDW, variable_registry ); 
  
      }
  
      i = _ave_z_flux_sum_names.begin(); 
      for (;i!=_ave_z_flux_sum_names.end();i++){ 
  
        register_variable( *i, CC_DOUBLE, MODIFIES, variable_registry ); 
        register_variable( *i, CC_DOUBLE, REQUIRES, 0, OLDDW, variable_registry ); 
  
      }

      i = _x_flux_sqr_sum_names.begin(); 
      for (;i!=_x_flux_sqr_sum_names.end();i++){ 
  
        register_variable( *i, CC_DOUBLE, MODIFIES, variable_registry ); 
        register_variable( *i, CC_DOUBLE, REQUIRES, 0, OLDDW, variable_registry ); 
  
      }

      i = _y_flux_sqr_sum_names.begin(); 
      for (;i!=_y_flux_sqr_sum_names.end();i++){ 
  
        register_variable( *i, CC_DOUBLE, MODIFIES, variable_registry ); 
        register_variable( *i, CC_DOUBLE, REQUIRES, 0, OLDDW, variable_registry ); 
  
      }

      i = _z_flux_sqr_sum_names.begin(); 
      for (;i!=_z_flux_sqr_sum_names.end();i++){ 
  
        register_variable( *i, CC_DOUBLE, MODIFIES, variable_registry ); 
        register_variable( *i, CC_DOUBLE, REQUIRES, 0, OLDDW, variable_registry ); 
  
      }
  
      register_variable( "uVelocitySPBC" , FACEX     , REQUIRES , 1 , NEWDW , variable_registry );
      register_variable( "vVelocitySPBC" , FACEY     , REQUIRES , 1 , NEWDW , variable_registry );
      register_variable( "wVelocitySPBC" , FACEZ     , REQUIRES , 1 , NEWDW , variable_registry );
      register_variable( _rho_name        , CC_DOUBLE , REQUIRES , 0 , NEWDW , variable_registry );
  
      typename std::vector<FluxInfo>::iterator ii = _flux_sum_info.begin(); 
      for (;ii!=_flux_sum_info.end();ii++){ 
  
        if ( (*ii).do_phi )
          register_variable( (*ii).phi , _V_type, REQUIRES , nGhost , NEWDW , variable_registry );
  
      }
    }
  
  }
  
  template <typename T>
  void VariableStats<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                  SpatialOps::OperatorDatabase& opr ){ 
  
    using SpatialOps::operator *; 
    
    const double dt = tsk_info->get_dt(); 
    std::vector<std::string>::iterator i = _ave_sum_names.begin(); 
  
    int N = _ave_sum_names.size(); 
  
  
    //----------NEBO----------------
    // (Uintah implementation pasted below)
    //NOTE: For single variables, we will leave them in situ with regards to 
    //       their respective variable type (ie, T)
    //
    //      For fluxes, we will move them always to the cell center until 
    //       we need to do something else. 
    //
  
    //Single variables
    for ( int i = 0; i < N; i++ ){ 

      const ITP* const ip = opr.retrieve_operator<ITP>();
  
      SVolFP sum               = tsk_info->get_so_field<SVolF>( _ave_sum_names[i] );
      TP const var             = tsk_info->get_const_so_field<T>( _base_var_names[i] );
      SVolFP const old_sum     = tsk_info->get_const_so_field<SVolF>( _ave_sum_names[i] );
      SVolFP sqr_sum           = tsk_info->get_so_field<SVolF>( _sqr_variable_names[i] );
      SVolFP const old_sqr_sum = tsk_info->get_const_so_field<SVolF>( _sqr_variable_names[i] );
  
      *sum <<= *old_sum + dt * (*ip)(*var); 
      *sqr_sum <<= *old_sqr_sum + dt * (*ip)(*var)*(*ip)(*var); 
  
    }
  
    if ( !_no_flux ){ 
  
      //Fluxes 
      // NOTE: I WAS TRYING TO CREATE FACE FLUXES BUT WAS GETTING COMPILATION ERRORS
      //       WHEN TRYING TO CREATE THE INTERPOLANT (SVOL->XVOL) 
      //       But going to the cell centers requires one less interpolation. 
      XVolFP const u = tsk_info->get_const_so_field<XVolF>( "uVelocitySPBC" );
      YVolFP const v = tsk_info->get_const_so_field<YVolF>( "vVelocitySPBC" );
      ZVolFP const w = tsk_info->get_const_so_field<ZVolF>( "wVelocitySPBC" );
      SVolFP const rho = tsk_info->get_const_so_field<SVolF>( _rho_name );
  
      const IX* const ix = opr.retrieve_operator<IX>();
      const IY* const iy = opr.retrieve_operator<IY>();
      const IZ* const iz = opr.retrieve_operator<IZ>();
      const ITP* const ip = opr.retrieve_operator<ITP>();
  
      //X FLUX
      N = _ave_x_flux_sum_names.size(); 
      for ( int i = 0; i < N; i++ ){ 
  
        SVolFP sum            = tsk_info->get_so_field<SVolF>( _ave_x_flux_sum_names[i] );
        SVolFP const old_sum  = tsk_info->get_const_so_field<SVolF>( _ave_x_flux_sum_names[i] );

        SVolFP sqr_sum         = tsk_info->get_so_field<SVolF>( _x_flux_sqr_sum_names[i] ); 
        SVolFP const old_sqr_sum = tsk_info->get_const_so_field<SVolF>( _x_flux_sqr_sum_names[i] ); 

        SpatialOps::SpatFldPtr<SVolF> flux = SpatialFieldStore::get<SVolF>(*rho); 
        *flux <<= 0.0;
  
        if ( _flux_sum_info[i].do_phi ){ 
  
          TP const phi = tsk_info->get_const_so_field<T>( _flux_sum_info[i].phi ); 
         
          *flux <<= ( (*ix)(*u) * *rho * (*ip)(*phi) ); 
          *sum <<= *old_sum + dt * ( *flux ); 

          *sqr_sum <<= *old_sqr_sum + dt * ( *flux * *flux ); 
  
        } else { 
  
          *flux <<= ( *rho * (*ix)(*u) ); 

          *sum <<= *old_sum + dt * ( *flux ); 

          *sqr_sum <<= *old_sqr_sum + dt * ( *flux * *flux ); 
          
        }
  
      }
  
      //Y FLUX
      N = _ave_y_flux_sum_names.size(); 
      for ( int i = 0; i < N; i++ ){ 
  
        SVolFP sum            = tsk_info->get_so_field<SVolF>( _ave_y_flux_sum_names[i] );
        SVolFP const old_sum  = tsk_info->get_const_so_field<SVolF>( _ave_y_flux_sum_names[i] );

        SVolFP sqr_sum         = tsk_info->get_so_field<SVolF>( _y_flux_sqr_sum_names[i] ); 
        SVolFP const old_sqr_sum = tsk_info->get_const_so_field<SVolF>( _y_flux_sqr_sum_names[i] ); 

        SpatialOps::SpatFldPtr<SVolF> flux = SpatialFieldStore::get<SVolF>(*rho); 
        *flux <<= 0.0;
  
        if ( _flux_sum_info[i].do_phi ){ 
  
          TP const phi = tsk_info->get_const_so_field<T>( _flux_sum_info[i].phi ); 
         
          *flux <<= ( (*iy)(*v) * *rho * (*ip)(*phi) ); 
          *sum <<= *old_sum + dt * ( *flux ); 

          *sqr_sum <<= *old_sqr_sum + dt * ( *flux * *flux ); 
  
  
        } else { 
  
          *flux <<= ( *rho * (*iy)(*v) ); 

          *sum <<= *old_sum + dt * ( *flux ); 

          *sqr_sum <<= *old_sqr_sum + dt * ( *flux * *flux ); 
          
          
        }
  
      }
  
      //Z FLUX
      N = _ave_z_flux_sum_names.size(); 
      for ( int i = 0; i < N; i++ ){ 
  
        SVolFP sum            = tsk_info->get_so_field<SVolF>( _ave_z_flux_sum_names[i] );
        SVolFP const old_sum  = tsk_info->get_const_so_field<SVolF>( _ave_z_flux_sum_names[i] );

        SVolFP sqr_sum         = tsk_info->get_so_field<SVolF>( _z_flux_sqr_sum_names[i] ); 
        SVolFP const old_sqr_sum = tsk_info->get_const_so_field<SVolF>( _z_flux_sqr_sum_names[i] ); 

        SpatialOps::SpatFldPtr<SVolF> flux = SpatialFieldStore::get<SVolF>(*rho); 
        *flux <<= 0.0;
  
        if ( _flux_sum_info[i].do_phi ){ 
  
          TP const phi = tsk_info->get_const_so_field<T>( _flux_sum_info[i].phi ); 
         
          *flux <<= ( (*iz)(*w) * *rho * (*ip)(*phi) ); 
          *sum <<= *old_sum + dt * ( *flux ); 

          *sqr_sum <<= *old_sqr_sum + dt * ( *flux * *flux ); 
  
        } else { 
  
          *flux <<= ( *rho * (*iz)(*w) ); 

          *sum <<= *old_sum + dt * ( *flux ); 

          *sqr_sum <<= *old_sqr_sum + dt * ( *flux * *flux ); 
          
        }
      }
    }
  }
}
#endif 
    ////Uintah implementation
    ////Single Variables
    //for ( int i = 0; i < N; i++ ){ 
  
      //CCVariable<double>* sump          = tsk_info->get_uintah_field<CCVariable<double> >(_ave_sum_names[i]);
      //constCCVariable<double>* varp     = tsk_info->get_const_uintah_field<constCCVariable<double> >(_base_var_names[i]);
      //constCCVariable<double>* old_sump = tsk_info->get_const_uintah_field<constCCVariable<double> >(_ave_sum_names[i]);
  
      //CCVariable<double>& sum          = *sump;
      //constCCVariable<double>& var     = *varp;
      //constCCVariable<double>& old_sum = *old_sump;
  
      //sum.initialize(0.0);
  
      //for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
  
        //IntVector c = *iter; 
        
        //sum[c] = old_sum[c] + dt * var[c]; 
  
      //}
    //}
  
    ////Fluxes
    //if ( !_no_flux ){ 
      //constCCVariable<double>* rhop = tsk_info->get_const_uintah_field<constCCVariable<double> >(_rho_name); 
      //constSFCXVariable<double>* up = tsk_info->get_const_uintah_field<constSFCXVariable<double> >("uVelocitySPBC"); 
      //constSFCYVariable<double>* vp = tsk_info->get_const_uintah_field<constSFCYVariable<double> >("vVelocitySPBC"); 
      //constSFCZVariable<double>* wp = tsk_info->get_const_uintah_field<constSFCZVariable<double> >("wVelocitySPBC"); 
  
      //constCCVariable<double>& rho = *rhop; 
      //constSFCXVariable<double>& u = *up; 
      //constSFCYVariable<double>& v = *vp; 
      //constSFCZVariable<double>& w = *wp; 
  
      //N = _ave_x_flux_sum_names.size(); 
  
      //for ( int i = 0; i < N; i++ ){ 
  
        //CCVariable<double>* sump_x          = tsk_info->get_uintah_field<CCVariable<double> >(_ave_x_flux_sum_names[i]);
        //constCCVariable<double>* old_sump_x = tsk_info->get_const_uintah_field<constCCVariable<double> >(_ave_x_flux_sum_names[i]);
        //CCVariable<double>* sump_y          = tsk_info->get_uintah_field<CCVariable<double> >(_ave_y_flux_sum_names[i]);
        //constCCVariable<double>* old_sump_y = tsk_info->get_const_uintah_field<constCCVariable<double> >(_ave_y_flux_sum_names[i]);
        //CCVariable<double>* sump_z          = tsk_info->get_uintah_field<CCVariable<double> >(_ave_z_flux_sum_names[i]);
        //constCCVariable<double>* old_sump_z = tsk_info->get_const_uintah_field<constCCVariable<double> >(_ave_z_flux_sum_names[i]);
        //constCCVariable<double>* phip;
  
        //if ( _flux_sum_info[i].do_phi)
          //phip = tsk_info->get_const_uintah_field<constCCVariable<double> >(_flux_sum_info[i].phi); 
  
        //CCVariable<double>& sum_x          = *sump_x;
        //constCCVariable<double>& old_sum_x = *old_sump_x;
        //CCVariable<double>& sum_y          = *sump_y;
        //constCCVariable<double>& old_sum_y = *old_sump_y;
        //CCVariable<double>& sum_z          = *sump_z;
        //constCCVariable<double>& old_sum_z = *old_sump_z;
  
        //sum_x.initialize(12.0);
        //sum_y.initialize(0.0);
        //sum_z.initialize(0.0);
  
        //for (CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++) { 
  
          //IntVector c = *iter; 
  
          //if ( _flux_sum_info[i].do_phi ){ 
  
            //sum_x[c] = old_sum_x[c] + dt * rho[c] * ( u[c] + u[c+IntVector(1,0,0)] )/2.0 * (*phip)[c]; 
            //sum_y[c] = old_sum_y[c] + dt * rho[c] * ( v[c] + v[c+IntVector(0,1,0)] )/2.0 * (*phip)[c]; 
            //sum_z[c] = old_sum_z[c] + dt * rho[c] * ( w[c] + w[c+IntVector(0,0,1)] )/2.0 * (*phip)[c]; 
  
          //} else { 
  
            //sum_x[c] = old_sum_x[c] + dt * rho[c] * ( u[c] + u[c+IntVector(1,0,0)] )/2.0 ; 
            //sum_y[c] = old_sum_y[c] + dt * rho[c] * ( v[c] + v[c+IntVector(0,1,0)] )/2.0 ; 
            //sum_z[c] = old_sum_z[c] + dt * rho[c] * ( w[c] + w[c+IntVector(0,0,1)] )/2.0 ; 
  
          //}
  
        //}
      //}
    //}
  
