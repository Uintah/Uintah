#include <CCA/Components/Arches/PropertyModelsV2/VariableStats.h>

typedef SpatialOps::SVolField SVolF;
typedef SpatialOps::SpatFldPtr<SVolF> SVolFP;
typedef SpatialOps::XVolField XVolF;
typedef SpatialOps::YVolField YVolF;
typedef SpatialOps::ZVolField ZVolF;
typedef SpatialOps::SpatFldPtr<XVolF> XVolFP;
typedef SpatialOps::SpatFldPtr<YVolF> YVolFP;
typedef SpatialOps::SpatFldPtr<ZVolF> ZVolFP;

//interpolants
typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant,
  SVolF, XVolF >::type SVtoXV;
typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant,
  SVolF, YVolF >::type SVtoYV;
typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant,
  SVolF, ZVolF >::type SVtoZV;

using namespace Uintah;

//--------------------------------------------------------------------------------------------------
VariableStats::VariableStats( std::string task_name,
                              int matl_index,
                              SimulationStateP& shared_state ) :
TaskInterface( task_name, matl_index )
{
  _shared_state = shared_state;
}

//--------------------------------------------------------------------------------------------------
VariableStats::~VariableStats(){
}

//--------------------------------------------------------------------------------------------------
void VariableStats::problemSetup( ProblemSpecP& db ){

  for ( ProblemSpecP var_db = db->findBlock("single_variable"); var_db != 0;
        var_db = var_db->findNextBlock("single_variable") ){

    std::string var_name;
    var_db->getAttribute("label", var_name);

    std::string var_ave_name = var_name + "_running_sum";
    std::string sqr_name = var_name + "_squared_sum";

    _ave_sum_names.push_back(var_ave_name);
    _sqr_variable_names.push_back(sqr_name);

    _base_var_names.push_back(var_name);

    if (var_db->findBlock("new")){
      _new_variables.push_back(var_ave_name);
      _new_variables.push_back(sqr_name);
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
      _new_variables.push_back(x_var_sqr_name);
      _new_variables.push_back(y_var_sqr_name);
      _new_variables.push_back(z_var_sqr_name);
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

//--------------------------------------------------------------------------------------------------
void VariableStats::create_local_labels(){

  std::vector<std::string>::iterator i = _ave_sum_names.begin();
  for (; i!= _ave_sum_names.end(); i++ ){
    register_new_variable( *i, CC_DOUBLE );
  }

  i = _sqr_variable_names.begin();
  for (; i!= _sqr_variable_names.end(); i++ ){
    register_new_variable( *i, CC_DOUBLE );
  }

  i = _ave_x_flux_sum_names.begin();
  for (; i!= _ave_x_flux_sum_names.end(); i++ ){
    register_new_variable( *i, FACEX );
  }

  i = _ave_y_flux_sum_names.begin();
  for (; i!= _ave_y_flux_sum_names.end(); i++ ){
    register_new_variable( *i, FACEY );
  }

  i = _ave_z_flux_sum_names.begin();
  for (; i!= _ave_z_flux_sum_names.end(); i++ ){
    register_new_variable( *i, FACEZ );
  }

  i = _x_flux_sqr_sum_names.begin();
  for (; i!= _x_flux_sqr_sum_names.end(); i++ ){
    register_new_variable( *i, FACEX );
  }

  i = _y_flux_sqr_sum_names.begin();
  for (; i!= _y_flux_sqr_sum_names.end(); i++ ){
    register_new_variable( *i, FACEY );
  }

  i = _z_flux_sqr_sum_names.begin();
  for (; i!= _z_flux_sqr_sum_names.end(); i++ ){
    register_new_variable( *i, FACEZ );
  }

}

//--------------------------------------------------------------------------------------------------
void VariableStats::register_initialize( VIVec& variable_registry ){

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

//--------------------------------------------------------------------------------------------------
void VariableStats::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
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

//--------------------------------------------------------------------------------------------------
void VariableStats::register_restart_initialize( VIVec& variable_registry ){

  typedef std::vector<std::string> StrVec;

  for ( StrVec::iterator i = _new_variables.begin(); i != _new_variables.end(); i++ ){
    register_variable( *i, CC_DOUBLE, COMPUTES, variable_registry );
  }
}

//--------------------------------------------------------------------------------------------------
void VariableStats::restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                        SpatialOps::OperatorDatabase& opr ){

  using namespace SpatialOps;
  using SpatialOps::operator *;
  typedef std::vector<std::string> StrVec;

  for ( StrVec::iterator i = _new_variables.begin(); i != _new_variables.end(); i++ ){

    SVolFP variable = tsk_info->get_so_field<SVolF>(*i);

    *variable <<= 0.0;

  }
}

//--------------------------------------------------------------------------------------------------
void VariableStats::register_timestep_init( VIVec& variable_registry ){

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

    register_variable( *i, FACEX, COMPUTES, variable_registry );

  }

  i = _ave_y_flux_sum_names.begin();
  for (;i!=_ave_y_flux_sum_names.end();i++){

    register_variable( *i, FACEY, COMPUTES, variable_registry );

  }

  i = _ave_z_flux_sum_names.begin();
  for (;i!=_ave_z_flux_sum_names.end();i++){

    register_variable( *i, FACEZ, COMPUTES, variable_registry );

  }

  i = _x_flux_sqr_sum_names.begin();
  for (;i!=_x_flux_sqr_sum_names.end();i++){

    register_variable( *i, FACEX, COMPUTES, variable_registry );

  }

  i = _y_flux_sqr_sum_names.begin();
  for (;i!=_y_flux_sqr_sum_names.end();i++){

    register_variable( *i, FACEY, COMPUTES, variable_registry );

  }

  i = _z_flux_sqr_sum_names.begin();
  for (;i!=_z_flux_sqr_sum_names.end();i++){

    register_variable( *i, FACEZ, COMPUTES, variable_registry );

  }
}

//--------------------------------------------------------------------------------------------------
void VariableStats::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
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

    XVolFP var = tsk_info->get_so_field<XVolF>( *i );
    *var <<= 0.0;

  }

  i = _ave_y_flux_sum_names.begin();
  for (;i!=_ave_y_flux_sum_names.end();i++){

    YVolFP var = tsk_info->get_so_field<YVolF>( *i );
    *var <<= 0.0;

  }

  i = _ave_z_flux_sum_names.begin();
  for (;i!=_ave_z_flux_sum_names.end();i++){

    ZVolFP var = tsk_info->get_so_field<ZVolF>( *i );
    *var <<= 0.0;

  }

  i = _x_flux_sqr_sum_names.begin();
  for (;i!=_x_flux_sqr_sum_names.end();i++){

    XVolFP var = tsk_info->get_so_field<XVolF>( *i );
    *var <<= 0.0;

  }

  i = _y_flux_sqr_sum_names.begin();
  for (;i!=_y_flux_sqr_sum_names.end();i++){

    YVolFP var = tsk_info->get_so_field<YVolF>( *i );
    *var <<= 0.0;

  }

  i = _z_flux_sqr_sum_names.begin();
  for (;i!=_z_flux_sqr_sum_names.end();i++){

    ZVolFP var = tsk_info->get_so_field<ZVolF>( *i );
    *var <<= 0.0;

  }
}

//--------------------------------------------------------------------------------------------------
void VariableStats::register_timestep_eval( VIVec& variable_registry, const int time_substep ){

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

    register_variable( *i, CC_DOUBLE, REQUIRES, 0, NEWDW, variable_registry );

  }

  if ( !_no_flux ){
    i = _ave_x_flux_sum_names.begin();
    for (;i!=_ave_x_flux_sum_names.end();i++){

      register_variable( *i, FACEX, MODIFIES, variable_registry );
      register_variable( *i, FACEX, REQUIRES, 0, OLDDW, variable_registry );

    }

    i = _ave_y_flux_sum_names.begin();
    for (;i!=_ave_y_flux_sum_names.end();i++){

      register_variable( *i, FACEY, MODIFIES, variable_registry );
      register_variable( *i, FACEY, REQUIRES, 0, OLDDW, variable_registry );

    }

    i = _ave_z_flux_sum_names.begin();
    for (;i!=_ave_z_flux_sum_names.end();i++){

      register_variable( *i, FACEZ, MODIFIES, variable_registry );
      register_variable( *i, FACEZ, REQUIRES, 0, OLDDW, variable_registry );

    }

    i = _x_flux_sqr_sum_names.begin();
    for (;i!=_x_flux_sqr_sum_names.end();i++){

      register_variable( *i, FACEX, MODIFIES, variable_registry );
      register_variable( *i, FACEX, REQUIRES, 0, OLDDW, variable_registry );

    }

    i = _y_flux_sqr_sum_names.begin();
    for (;i!=_y_flux_sqr_sum_names.end();i++){

      register_variable( *i, FACEY, MODIFIES, variable_registry );
      register_variable( *i, FACEY, REQUIRES, 0, OLDDW, variable_registry );

    }

    i = _z_flux_sqr_sum_names.begin();
    for (;i!=_z_flux_sqr_sum_names.end();i++){

      register_variable( *i, FACEZ, MODIFIES, variable_registry );
      register_variable( *i, FACEZ, REQUIRES, 0, OLDDW, variable_registry );

    }

    register_variable( "uVelocitySPBC" , FACEX     , REQUIRES , 0 , NEWDW , variable_registry );
    register_variable( "vVelocitySPBC" , FACEY     , REQUIRES , 0 , NEWDW , variable_registry );
    register_variable( "wVelocitySPBC" , FACEZ     , REQUIRES , 0 , NEWDW , variable_registry );
    register_variable( _rho_name        , CC_DOUBLE , REQUIRES , 1 , NEWDW , variable_registry );

    std::vector<FluxInfo>::iterator ii = _flux_sum_info.begin();
    for (;ii!=_flux_sum_info.end();ii++){

      if ( (*ii).do_phi )
        register_variable( (*ii).phi , CC_DOUBLE, REQUIRES , 1 , NEWDW , variable_registry );

    }
  }
}

//--------------------------------------------------------------------------------------------------
void VariableStats::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                          SpatialOps::OperatorDatabase& opr )
{

  using SpatialOps::operator *;

  const double dt = tsk_info->get_dt();
  std::vector<std::string>::iterator i = _ave_sum_names.begin();

  int N = _ave_sum_names.size();

  //----------NEBO----------------
  // (Uintah implementation pasted below)
  //NOTE: For single variables, we will leave them in situ with regards to
  //       their respective variable type (ie, T)
  //

  //Single variables
  for ( int i = 0; i < N; i++ ){

    SVolFP sum               = tsk_info->get_so_field<SVolF>( _ave_sum_names[i] );
    SVolFP const var         = tsk_info->get_const_so_field<SVolF>( _base_var_names[i] );
    SVolFP const old_sum     = tsk_info->get_const_so_field<SVolF>( _ave_sum_names[i] );
    SVolFP sqr_sum           = tsk_info->get_so_field<SVolF>( _sqr_variable_names[i] );
    SVolFP const old_sqr_sum = tsk_info->get_const_so_field<SVolF>( _sqr_variable_names[i] );

    *sum <<= *old_sum + dt * (*var);
    *sqr_sum <<= *old_sqr_sum + dt * (*var) * (*var);

  }

  if ( !_no_flux ){

    //Fluxes
    XVolFP const u = tsk_info->get_const_so_field<XVolF>( "uVelocitySPBC" );
    YVolFP const v = tsk_info->get_const_so_field<YVolF>( "vVelocitySPBC" );
    ZVolFP const w = tsk_info->get_const_so_field<ZVolF>( "wVelocitySPBC" );
    SVolFP const rho = tsk_info->get_const_so_field<SVolF>( _rho_name );

    const SVtoXV* const ix = opr.retrieve_operator<SVtoXV>();
    const SVtoYV* const iy = opr.retrieve_operator<SVtoYV>();
    const SVtoZV* const iz = opr.retrieve_operator<SVtoZV>();

    //X FLUX
    N = _ave_x_flux_sum_names.size();
    for ( int i = 0; i < N; i++ ){

      XVolFP sum            = tsk_info->get_so_field<XVolF>( _ave_x_flux_sum_names[i] );
      XVolFP const old_sum  = tsk_info->get_const_so_field<XVolF>( _ave_x_flux_sum_names[i] );

      XVolFP sqr_sum         = tsk_info->get_so_field<XVolF>( _x_flux_sqr_sum_names[i] );
      XVolFP const old_sqr_sum = tsk_info->get_const_so_field<XVolF>( _x_flux_sqr_sum_names[i] );

      SpatialOps::SpatFldPtr<XVolF> flux = SpatialOps::SpatialFieldStore::get<XVolF>(*u);
      *flux <<= 0.0;

      if ( _flux_sum_info[i].do_phi ){

        SVolFP const phi = tsk_info->get_const_so_field<SVolF>( _flux_sum_info[i].phi );

        *flux <<= ( (*u) * (*ix)(*rho * *phi) );
        *sum <<= *old_sum + dt * ( *flux );

        *sqr_sum <<= *old_sqr_sum + dt * ( *flux * *flux );

      } else {

        *flux <<= ( (*ix)(*rho) * (*u) );

        *sum <<= *old_sum + dt * ( *flux );

        *sqr_sum <<= *old_sqr_sum + dt * ( *flux * *flux );

      }
    }

    //Y FLUX
    N = _ave_y_flux_sum_names.size();
    for ( int i = 0; i < N; i++ ){

      YVolFP sum            = tsk_info->get_so_field<YVolF>( _ave_y_flux_sum_names[i] );
      YVolFP const old_sum  = tsk_info->get_const_so_field<YVolF>( _ave_y_flux_sum_names[i] );

      YVolFP sqr_sum         = tsk_info->get_so_field<YVolF>( _y_flux_sqr_sum_names[i] );
      YVolFP const old_sqr_sum = tsk_info->get_const_so_field<YVolF>( _y_flux_sqr_sum_names[i] );

      SpatialOps::SpatFldPtr<YVolF> flux = SpatialOps::SpatialFieldStore::get<YVolF>(*v);
      *flux <<= 0.0;

      if ( _flux_sum_info[i].do_phi ){

        SVolFP const phi = tsk_info->get_const_so_field<SVolF>( _flux_sum_info[i].phi );

        *flux <<= ( *v * (*iy)(*rho * *phi) );
        *sum <<= *old_sum + dt * ( *flux );

        *sqr_sum <<= *old_sqr_sum + dt * ( *flux * *flux );

      } else {

        *flux <<= ( (*iy)(*rho) * (*v) );

        *sum <<= *old_sum + dt * ( *flux );

        *sqr_sum <<= *old_sqr_sum + dt * ( *flux * *flux );

      }
    }

    //Z FLUX
    N = _ave_z_flux_sum_names.size();
    for ( int i = 0; i < N; i++ ){

      ZVolFP sum            = tsk_info->get_so_field<ZVolF>( _ave_z_flux_sum_names[i] );
      ZVolFP const old_sum  = tsk_info->get_const_so_field<ZVolF>( _ave_z_flux_sum_names[i] );

      ZVolFP sqr_sum         = tsk_info->get_so_field<ZVolF>( _z_flux_sqr_sum_names[i] );
      ZVolFP const old_sqr_sum = tsk_info->get_const_so_field<ZVolF>( _z_flux_sqr_sum_names[i] );

      SpatialOps::SpatFldPtr<ZVolF> flux = SpatialOps::SpatialFieldStore::get<ZVolF>(*w);
      *flux <<= 0.0;

      if ( _flux_sum_info[i].do_phi ){

        SVolFP const phi = tsk_info->get_const_so_field<SVolF>( _flux_sum_info[i].phi );

        *flux <<= ( *w * (*iz)(*rho * *phi) );
        *sum <<= *old_sum + dt * ( *flux );

        *sqr_sum <<= *old_sqr_sum + dt * ( *flux * *flux );

      } else {

        *flux <<= ( (*iz)(*rho) * *w );

        *sum <<= *old_sum + dt * ( *flux );

        *sqr_sum <<= *old_sqr_sum + dt * ( *flux * *flux );

      }
    }
  }
}
