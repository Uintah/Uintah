#include <CCA/Components/Arches/PropertyModelsV2/TimeAve.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <spatialops/structured/FVStaggered.h>

using namespace Uintah;

using namespace SpatialOps;
using SpatialOps::operator *; 
typedef SpatialOps::SVolField SVolF;
typedef SpatialOps::SpatFldPtr<SVolF> SVolFP; 
typedef SpatialOps::XVolField XVolF; 
typedef SpatialOps::YVolField YVolF; 
typedef SpatialOps::ZVolField ZVolF; 
typedef SpatialOps::SpatFldPtr<XVolF> XVolFP; 
typedef SpatialOps::SpatFldPtr<YVolF> YVolFP; 
typedef SpatialOps::SpatFldPtr<ZVolF> ZVolFP; 
//interpolants
typedef OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::XVolField, SpatialOps::SVolField >::type IX;
typedef OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::YVolField, SpatialOps::SVolField >::type IY;
typedef OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::ZVolField, SpatialOps::SVolField >::type IZ;

TimeAve::TimeAve( std::string task_name, int matl_index, SimulationStateP& shared_state ) : 
TaskInterface( task_name, matl_index ) { 

  _shared_state = shared_state; 

}

TimeAve::~TimeAve(){ 

  std::vector<const VarLabel*>::iterator i = ave_sum_labels.begin(); 
  for (;i!=ave_sum_labels.end();i++){ 

    VarLabel::destroy(*i); 

  }

  i = ave_flux_sum_labels.begin(); 
  for (;i!=ave_flux_sum_labels.end();i++){ 

    VarLabel::destroy(*i); 

  }

}

void 
TimeAve::problemSetup( ProblemSpecP& db ){ 

  for ( ProblemSpecP var_db = db->findBlock("single_variable"); var_db != 0; 
        var_db = var_db->findNextBlock("single_variable") ){ 

    std::string var_name; 
    var_db->getAttribute("label", var_name);

    std::string var_ave_name = var_name + "_running_sum"; 

    const VarLabel* label_sum = VarLabel::create( var_ave_name     , CCVariable<double>::getTypeDescription() );

    ave_sum_labels.push_back(label_sum); 

    ave_sum_names.push_back(var_ave_name); 

    base_var_names.push_back(var_name); 

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

    const VarLabel* x_flux_label_sum = VarLabel::create( x_var_name , CCVariable<double>::getTypeDescription() );
    const VarLabel* y_flux_label_sum = VarLabel::create( y_var_name , CCVariable<double>::getTypeDescription() );
    const VarLabel* z_flux_label_sum = VarLabel::create( z_var_name , CCVariable<double>::getTypeDescription() );

    ave_flux_sum_labels.push_back(x_flux_label_sum); 
    ave_flux_sum_labels.push_back(y_flux_label_sum); 
    ave_flux_sum_labels.push_back(z_flux_label_sum); 

    ave_x_flux_sum_names.push_back(x_var_name); 
    ave_y_flux_sum_names.push_back(y_var_name); 
    ave_z_flux_sum_names.push_back(z_var_name); 

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

    flux_sum_info.push_back(fi); 

    if (var_db->findBlock("new")){ 
      _new_variables.push_back(x_var_name); 
      _new_variables.push_back(y_var_name); 
      _new_variables.push_back(z_var_name); 
    }

  }

  if ( do_fluxes ){ 
    if ( db->findBlock("density")){ 
      db->findBlock("density")->getAttribute("label", rho_name);
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
    
  }

}

void 
TimeAve::create_local_labels(){ 
}

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void 
TimeAve::register_initialize( VIVec& variable_registry ){ 

  
  std::vector<std::string>::iterator i = ave_sum_names.begin(); 
  for (;i!=ave_sum_names.end();i++){ 

    register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 

  }

  i = ave_x_flux_sum_names.begin(); 
  for (;i!=ave_x_flux_sum_names.end();i++){ 

    register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 

  }
  i = ave_y_flux_sum_names.begin(); 
  for (;i!=ave_y_flux_sum_names.end();i++){ 

    register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 

  }
  i = ave_z_flux_sum_names.begin(); 
  for (;i!=ave_z_flux_sum_names.end();i++){ 

    register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 

  }

}

void 
TimeAve::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                      SpatialOps::OperatorDatabase& opr ){ 

  using namespace SpatialOps;
  using SpatialOps::operator *; 
  typedef SpatialOps::SVolField   SVolF;
  typedef SpatialOps::SpatFldPtr<SVolF> SVolFP; 

  std::vector<std::string>::iterator i = ave_sum_names.begin(); 
  for (;i!=ave_sum_names.end();i++){ 
    SVolFP var = tsk_info->get_so_field<SVolF>( *i ); 
    *var <<= 0.0;
  }

  i = ave_x_flux_sum_names.begin(); 
  for (;i!=ave_x_flux_sum_names.end();i++){ 
    SVolFP var = tsk_info->get_so_field<SVolF>( *i ); 
    *var <<= 0.0;
  }

  i = ave_y_flux_sum_names.begin(); 
  for (;i!=ave_y_flux_sum_names.end();i++){ 
    SVolFP var = tsk_info->get_so_field<SVolF>( *i ); 
    *var <<= 0.0;
  }

  i = ave_z_flux_sum_names.begin(); 
  for (;i!=ave_z_flux_sum_names.end();i++){ 
    SVolFP var = tsk_info->get_so_field<SVolF>( *i ); 
    *var <<= 0.0;
  }

}

void 
TimeAve::register_restart_initialize( VIVec& variable_registry ){ 

  typedef std::vector<std::string> StrVec; 

  for ( StrVec::iterator i = _new_variables.begin(); i != _new_variables.end(); i++ ){
    register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 
  }
  
}

void 
TimeAve::restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                             SpatialOps::OperatorDatabase& opr ){ 

  using namespace SpatialOps;
  using SpatialOps::operator *; 
  typedef SpatialOps::SVolField   SVolF;
  typedef SpatialOps::SpatFldPtr<SVolF> SVolFP; 
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
void 
TimeAve::register_timestep_init( VIVec& variable_registry ){ 

  std::vector<std::string>::iterator i = ave_sum_names.begin(); 
  for (;i!=ave_sum_names.end();i++){ 

    register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 

  }

  i = ave_x_flux_sum_names.begin(); 
  for (;i!=ave_x_flux_sum_names.end();i++){ 

    register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 

  }

  i = ave_y_flux_sum_names.begin(); 
  for (;i!=ave_y_flux_sum_names.end();i++){ 

    register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 

  }

  i = ave_z_flux_sum_names.begin(); 
  for (;i!=ave_z_flux_sum_names.end();i++){ 

    register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry ); 

  }
}

void 
TimeAve::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                        SpatialOps::OperatorDatabase& opr ){ 

  using namespace SpatialOps;
  using SpatialOps::operator *; 

  std::vector<std::string>::iterator i = ave_sum_names.begin(); 
  for (;i!=ave_sum_names.end();i++){ 

    SVolFP var = tsk_info->get_so_field<SVolF>( *i ); 

  }

  i = ave_x_flux_sum_names.begin(); 
  for (;i!=ave_x_flux_sum_names.end();i++){ 

    SVolFP var = tsk_info->get_so_field<SVolF>( *i ); 

  }

  i = ave_y_flux_sum_names.begin(); 
  for (;i!=ave_y_flux_sum_names.end();i++){ 

    SVolFP var = tsk_info->get_so_field<SVolF>( *i ); 

  }

  i = ave_z_flux_sum_names.begin(); 
  for (;i!=ave_z_flux_sum_names.end();i++){ 

    SVolFP var = tsk_info->get_so_field<SVolF>( *i ); 

  }


}

//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

void 
TimeAve::register_timestep_eval( VIVec& variable_registry, const int time_substep ){ 

  std::vector<std::string>::iterator i = ave_sum_names.begin(); 
  for (;i!=ave_sum_names.end();i++){ 

    register_variable( *i, CC_DOUBLE, MODIFIES, variable_registry ); 
    register_variable( *i, CC_DOUBLE, REQUIRES, 0, OLDDW, variable_registry ); 

  }

  i = base_var_names.begin(); 
  for (;i!=base_var_names.end();i++){ 

    register_variable( *i, CC_DOUBLE, REQUIRES, 0, NEWDW, variable_registry ); 

  }

  if ( !_no_flux ){ 
    i = ave_x_flux_sum_names.begin(); 
    for (;i!=ave_x_flux_sum_names.end();i++){ 

      register_variable( *i, CC_DOUBLE, MODIFIES, variable_registry ); 
      register_variable( *i, CC_DOUBLE, REQUIRES, 0, OLDDW, variable_registry ); 

    }

    i = ave_y_flux_sum_names.begin(); 
    for (;i!=ave_y_flux_sum_names.end();i++){ 

      register_variable( *i, CC_DOUBLE, MODIFIES, variable_registry ); 
      register_variable( *i, CC_DOUBLE, REQUIRES, 0, OLDDW, variable_registry ); 

    }

    i = ave_z_flux_sum_names.begin(); 
    for (;i!=ave_z_flux_sum_names.end();i++){ 

      register_variable( *i, CC_DOUBLE, MODIFIES, variable_registry ); 
      register_variable( *i, CC_DOUBLE, REQUIRES, 0, OLDDW, variable_registry ); 

    }

    register_variable( "uVelocitySPBC"   , FACEX , REQUIRES , 1 , NEWDW , variable_registry );
    register_variable( "vVelocitySPBC"   , FACEY , REQUIRES , 1 , NEWDW , variable_registry );
    register_variable( "wVelocitySPBC"   , FACEZ , REQUIRES , 1 , NEWDW , variable_registry );
    register_variable( rho_name, CC_DOUBLE , REQUIRES , 1 , NEWDW , variable_registry );
    std::vector<FluxInfo>::iterator ii = flux_sum_info.begin(); 
    for (;ii!=flux_sum_info.end();ii++){ 

      if ( (*ii).do_phi )
        register_variable( (*ii).phi , CC_DOUBLE , REQUIRES , 0 , NEWDW , variable_registry );

    }
  }

}

void 
TimeAve::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                SpatialOps::OperatorDatabase& opr ){ 

  using SpatialOps::operator *; 
  
  const double dt = tsk_info->get_dt(); 
  std::vector<std::string>::iterator i = ave_sum_names.begin(); 

  int N = ave_sum_names.size(); 

  //Uintah implementation
  //Single Variables
  for ( int i = 0; i < N; i++ ){ 

    CCVariable<double>* sump = tsk_info->get_uintah_field<CCVariable<double> >(ave_sum_names[i]); 
    constCCVariable<double>* varp = tsk_info->get_const_uintah_field<constCCVariable<double> >(base_var_names[i]); 
    constCCVariable<double>* old_sump = tsk_info->get_const_uintah_field<constCCVariable<double> >(ave_sum_names[i]); 

    CCVariable<double>& sum = *sump; 
    constCCVariable<double>& var = *varp;
    constCCVariable<double>& old_sum = *old_sump;

    sum.initialize(0.0);

    for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 

      IntVector c = *iter; 
      
      sum[c] = old_sum[c] + dt * var[c]; 

    }
  }

  //Fluxes
  if ( !_no_flux ){ 
    constCCVariable<double>* rhop = tsk_info->get_const_uintah_field<constCCVariable<double> >(rho_name); 
    constSFCXVariable<double>* up = tsk_info->get_const_uintah_field<constSFCXVariable<double> >("uVelocitySPBC"); 
    constSFCYVariable<double>* vp = tsk_info->get_const_uintah_field<constSFCYVariable<double> >("vVelocitySPBC"); 
    constSFCZVariable<double>* wp = tsk_info->get_const_uintah_field<constSFCZVariable<double> >("wVelocitySPBC"); 

    constCCVariable<double>& rho = *rhop; 
    constSFCXVariable<double>& u = *up; 
    constSFCYVariable<double>& v = *vp; 
    constSFCZVariable<double>& w = *wp; 

    N = ave_x_flux_sum_names.size(); 

    for ( int i = 0; i < N; i++ ){ 

      CCVariable<double>* sump_x = tsk_info->get_uintah_field<CCVariable<double> >(ave_x_flux_sum_names[i]); 
      constCCVariable<double>* old_sump_x = tsk_info->get_const_uintah_field<constCCVariable<double> >(ave_x_flux_sum_names[i]); 
      CCVariable<double>* sump_y = tsk_info->get_uintah_field<CCVariable<double> >(ave_y_flux_sum_names[i]); 
      constCCVariable<double>* old_sump_y = tsk_info->get_const_uintah_field<constCCVariable<double> >(ave_y_flux_sum_names[i]); 
      CCVariable<double>* sump_z = tsk_info->get_uintah_field<CCVariable<double> >(ave_z_flux_sum_names[i]); 
      constCCVariable<double>* old_sump_z = tsk_info->get_const_uintah_field<constCCVariable<double> >(ave_z_flux_sum_names[i]); 
      constCCVariable<double>* phip;

      if ( flux_sum_info[i].do_phi)
        phip = tsk_info->get_const_uintah_field<constCCVariable<double> >(flux_sum_info[i].phi); 

      CCVariable<double>& sum_x = *sump_x; 
      constCCVariable<double>& old_sum_x = *old_sump_x;
      CCVariable<double>& sum_y = *sump_y; 
      constCCVariable<double>& old_sum_y = *old_sump_y;
      CCVariable<double>& sum_z = *sump_z; 
      constCCVariable<double>& old_sum_z = *old_sump_z;

      sum_x.initialize(12.0);
      sum_y.initialize(0.0);
      sum_z.initialize(0.0);

      for (CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++) { 

        IntVector c = *iter; 

        if ( flux_sum_info[i].do_phi ){ 

          sum_x[c] = old_sum_x[c] + dt * rho[c] * ( u[c] + u[c+IntVector(1,0,0)] )/2.0 * (*phip)[c]; 
          sum_y[c] = old_sum_y[c] + dt * rho[c] * ( v[c] + v[c+IntVector(0,1,0)] )/2.0 * (*phip)[c]; 
          sum_z[c] = old_sum_z[c] + dt * rho[c] * ( w[c] + w[c+IntVector(0,0,1)] )/2.0 * (*phip)[c]; 

        } else { 

          sum_x[c] = old_sum_x[c] + dt * rho[c] * ( u[c] + u[c+IntVector(1,0,0)] )/2.0 ; 
          sum_y[c] = old_sum_y[c] + dt * rho[c] * ( v[c] + v[c+IntVector(0,1,0)] )/2.0 ; 
          sum_z[c] = old_sum_z[c] + dt * rho[c] * ( w[c] + w[c+IntVector(0,0,1)] )/2.0 ; 

        }

      }
    }
  }


  ////----------NEBO----------------

  ////Single variables
  //for ( int i = 0; i < N; i++ ){ 

    //SVolFP sum = tsk_info->get_so_field<SVolF>( ave_sum_names[i] ); 
    //SVolFP const var = tsk_info->get_const_so_field<SVolF>( base_var_names[i] ); 
    //SVolFP const old_sum = tsk_info->get_const_so_field<SVolF>( ave_sum_names[i] ); 

    //*sum <<= *old_sum + dt * *var; 

  //}

  //if ( !_no_flux ){ 

    ////Fluxes 
    //// NOTE: I WAS TRYING TO CREATE FACE FLUXES BUT WAS GETTING COMPILATION ERRORS
    ////       WHEN TRYING TO CREATE THE INTERPOLANT (SVOL->XVOL) 
    ////       But going to the cell centers requires one less interpolation. 
    //XVolFP const u = tsk_info->get_const_so_field<XVolF>( "uVelocitySPBC" );
    //YVolFP const v = tsk_info->get_const_so_field<YVolF>( "vVelocitySPBC" );
    //ZVolFP const w = tsk_info->get_const_so_field<ZVolF>( "wVelocitySPBC" );
    //SVolFP const rho = tsk_info->get_const_so_field<SVolF>( rho_name );

    //const IX* const ix = opr.retrieve_operator<IX>();
    //const IY* const iy = opr.retrieve_operator<IY>();
    //const IZ* const iz = opr.retrieve_operator<IZ>();

    ////X FLUX
    //N = ave_x_flux_sum_names.size(); 
    //for ( int i = 0; i < N; i++ ){ 

      //SVolFP sum            = tsk_info->get_so_field<SVolF>( ave_x_flux_sum_names[i] );
      //SVolFP const old_sum  = tsk_info->get_const_so_field<SVolF>( ave_x_flux_sum_names[i] );

      //if ( flux_sum_info[i].do_phi ){ 

        //SVolFP const phi = tsk_info->get_const_so_field<SVolF>( flux_sum_info[i].phi ); 
        //*sum <<= *old_sum + dt * ( (*ix)(*u) * *rho * *phi ); 

      //} else { 

        ///sum <<= *old_sum + dt * ( *rho * (*ix)(*u) ); 
        
      //}

    //}

    ////Y FLUX
    //N = ave_y_flux_sum_names.size(); 
    //for ( int i = 0; i < N; i++ ){ 

      //SVolFP sum            = tsk_info->get_so_field<SVolF>( ave_y_flux_sum_names[i] );
      //SVolFP const old_sum  = tsk_info->get_const_so_field<SVolF>( ave_y_flux_sum_names[i] );

      //if ( flux_sum_info[i].do_phi ){ 

        //SVolFP const phi = tsk_info->get_const_so_field<SVolF>( flux_sum_info[i].phi ); 
        //*sum <<= *old_sum + dt * ( (*iy)(*v) * *rho * *phi ); 

      //} else { 

        //*sum <<= *old_sum + dt * ( *rho * (*iy)(*v) ); 
        
      //}

    //}

    ////Z FLUX
    //N = ave_z_flux_sum_names.size(); 
    //for ( int i = 0; i < N; i++ ){ 

      //SVolFP sum            = tsk_info->get_so_field<SVolF>( ave_z_flux_sum_names[i] );
      //SVolFP const old_sum  = tsk_info->get_const_so_field<SVolF>( ave_z_flux_sum_names[i] );

      //if ( flux_sum_info[i].do_phi ){ 

        //SVolFP const phi = tsk_info->get_const_so_field<SVolF>( flux_sum_info[i].phi ); 
        //*sum <<= *old_sum + dt * ( (*iz)(*w) * *rho * *phi ); 

      //} else { 

        //*sum <<= *old_sum + dt * ( *rho * (*iz)(*w) ); 
        
      //}

    //}
  //}

}
