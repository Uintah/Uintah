#include <CCA/Components/Arches/PropertyModelsV2/VariableStats.h>
#include <CCA/Components/Arches/GridTools.h>
#include <CCA/Components/Arches/ChemMix/ChemHelper.h>
#include <Core/Exceptions/InternalError.h>

namespace Uintah{

//--------------------------------------------------------------------------------------------------
VariableStats::VariableStats( std::string task_name,
                              int matl_index ) :
TaskInterface( task_name, matl_index )
{}

//--------------------------------------------------------------------------------------------------
VariableStats::~VariableStats()
{}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace VariableStats::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace VariableStats::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &VariableStats::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &VariableStats::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &VariableStats::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace VariableStats::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &VariableStats::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &VariableStats::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &VariableStats::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace VariableStats::loadTaskTimestepInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                     , &VariableStats::timestep_init<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &VariableStats::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &VariableStats::timestep_init<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace VariableStats::loadTaskRestartInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::RESTART_INITIALIZE>( this
                                     , &VariableStats::restart_initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &VariableStats::restart_initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &VariableStats::restart_initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
void VariableStats::problemSetup( ProblemSpecP& db ){

  ChemHelper& chem_helper = ChemHelper::self();

  for ( ProblemSpecP var_db = db->findBlock("single_variable"); var_db != nullptr; var_db = var_db->findNextBlock("single_variable") ){

    std::string var_name;
    var_db->getAttribute("label", var_name);

    if ( var_db->findBlock( "table_variable" ) )  {
      chem_helper.add_lookup_species( var_name );
    }

    if ( var_name == "uVelocitySPBC" ||
         var_name == "vVelocitySPBC" ||
         var_name == "wVelocitySPBC" ){
      throw InvalidValue("Error: Cannot average velocities. Try a flux variable instead.",__FILE__,__LINE__);
    }

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

  for ( ProblemSpecP var_db = db->findBlock("flux_variable"); var_db != nullptr; var_db = var_db->findNextBlock("flux_variable") ){

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

    if ( var_db->findBlock( "table_variable" ) )  {
      chem_helper.add_lookup_species( phi_name );
    }

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

  for ( ProblemSpecP var_db = db->findBlock("new_single_variable"); var_db != nullptr; var_db = var_db->findNextBlock("new_single_variable") ){

    std::string name;
    var_db->getAttribute("label", name);
    std::string final_name = name + "_running_sum";
    _new_variables.push_back( final_name );

  }

  for ( ProblemSpecP var_db = db->findBlock("new_flux_variable"); var_db != nullptr; var_db = var_db->findNextBlock("new_flux_variable") ){

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
    register_new_variable<CCVariable<double> >( *i );
  }

  i = _sqr_variable_names.begin();
  for (; i!= _sqr_variable_names.end(); i++ ){
    register_new_variable<CCVariable<double> >( *i );
  }

  i = _ave_x_flux_sum_names.begin();
  for (; i!= _ave_x_flux_sum_names.end(); i++ ){
    register_new_variable<SFCXVariable<double> >( *i );
  }

  i = _ave_y_flux_sum_names.begin();
  for (; i!= _ave_y_flux_sum_names.end(); i++ ){
    register_new_variable<SFCYVariable<double> >( *i );
  }

  i = _ave_z_flux_sum_names.begin();
  for (; i!= _ave_z_flux_sum_names.end(); i++ ){
    register_new_variable<SFCZVariable<double> >( *i );
  }

  i = _x_flux_sqr_sum_names.begin();
  for (; i!= _x_flux_sqr_sum_names.end(); i++ ){
    register_new_variable<SFCXVariable<double> >( *i );
  }

  i = _y_flux_sqr_sum_names.begin();
  for (; i!= _y_flux_sqr_sum_names.end(); i++ ){
    register_new_variable<SFCYVariable<double> >( *i );
  }

  i = _z_flux_sqr_sum_names.begin();
  for (; i!= _z_flux_sqr_sum_names.end(); i++ ){
    register_new_variable<SFCZVariable<double> >( *i );
  }

}

//--------------------------------------------------------------------------------------------------
void VariableStats::register_initialize( VIVec& variable_registry , const bool pack_tasks){

  std::vector<std::string>::iterator i = _ave_sum_names.begin();
  for (;i!=_ave_sum_names.end();i++){

    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry );

  }

  i = _sqr_variable_names.begin();
  for (;i!=_sqr_variable_names.end();i++){

    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry );

  }

  i = _ave_x_flux_sum_names.begin();
  for (;i!=_ave_x_flux_sum_names.end();i++){

    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry );

  }
  i = _ave_y_flux_sum_names.begin();
  for (;i!=_ave_y_flux_sum_names.end();i++){

    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry );

  }
  i = _ave_z_flux_sum_names.begin();
  for (;i!=_ave_z_flux_sum_names.end();i++){

    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry );

  }
  i = _x_flux_sqr_sum_names.begin();
  for (;i!=_x_flux_sqr_sum_names.end();i++){

    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry );

  }
  i = _y_flux_sqr_sum_names.begin();
  for (;i!=_y_flux_sqr_sum_names.end();i++){

    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry );

  }
  i = _z_flux_sqr_sum_names.begin();
  for (;i!=_z_flux_sqr_sum_names.end();i++){

    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry );

  }
}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void VariableStats::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  auto i = _ave_sum_names.begin();
  for (;i!=_ave_sum_names.end();i++){
    CCVariable<double>& var = tsk_info->get_field<CCVariable<double> >( *i );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      var(i,j,k) = 0.0;
    });
  }

  i = _sqr_variable_names.begin();
  for (;i!=_sqr_variable_names.end();i++){
    CCVariable<double>& var = tsk_info->get_field<CCVariable<double> >( *i );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      var(i,j,k) = 0.0;
    });
  }

  i = _ave_x_flux_sum_names.begin();
  for (;i!=_ave_x_flux_sum_names.end();i++){
    SFCXVariable<double>& var = tsk_info->get_field<SFCXVariable<double> >( *i );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      var(i,j,k) = 0.0;
    });
  }

  i = _ave_y_flux_sum_names.begin();
  for (;i!=_ave_y_flux_sum_names.end();i++){
    SFCYVariable<double>& var = tsk_info->get_field<SFCYVariable<double> >( *i );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      var(i,j,k) = 0.0;
    });
  }

  i = _ave_z_flux_sum_names.begin();
  for (;i!=_ave_z_flux_sum_names.end();i++){
    SFCZVariable<double>& var = tsk_info->get_field<SFCZVariable<double> >( *i );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      var(i,j,k) = 0.0;
    });
  }
  i = _x_flux_sqr_sum_names.begin();
  for (;i!=_x_flux_sqr_sum_names.end();i++){
    SFCXVariable<double>& var = tsk_info->get_field<SFCXVariable<double> >( *i );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      var(i,j,k) = 0.0;
    });
  }
  i = _y_flux_sqr_sum_names.begin();
  for (;i!=_y_flux_sqr_sum_names.end();i++){
    SFCYVariable<double>& var = tsk_info->get_field<SFCYVariable<double> >( *i );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      var(i,j,k) = 0.0;
    });
  }
  i = _z_flux_sqr_sum_names.begin();
  for (;i!=_z_flux_sqr_sum_names.end();i++){
    SFCZVariable<double>& var = tsk_info->get_field<SFCZVariable<double> >( *i );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      var(i,j,k) = 0.0;
    });
  }

}

//--------------------------------------------------------------------------------------------------
void VariableStats::register_restart_initialize( VIVec& variable_registry , const bool packed_tasks){

  typedef std::vector<std::string> StrVec;

  for ( StrVec::iterator i = _new_variables.begin(); i != _new_variables.end(); i++ ){
    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry );
  }
}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace> 
void VariableStats::restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  typedef std::vector<std::string> StrVec;

  for ( StrVec::iterator i = _new_variables.begin(); i != _new_variables.end(); i++ ){
    CCVariable<double>& var = tsk_info->get_field<CCVariable<double> >( *i );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      var(i,j,k) = 0.0;
    });
  }
}

//--------------------------------------------------------------------------------------------------
void VariableStats::register_timestep_init( VIVec& variable_registry , const bool packed_tasks){

  std::vector<std::string>::iterator i = _ave_sum_names.begin();
  for (;i!=_ave_sum_names.end();i++){

    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry );

  }

  i = _sqr_variable_names.begin();
  for (;i!=_sqr_variable_names.end();i++){

    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry );

  }

  i = _ave_x_flux_sum_names.begin();
  for (;i!=_ave_x_flux_sum_names.end();i++){

    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry );

  }

  i = _ave_y_flux_sum_names.begin();
  for (;i!=_ave_y_flux_sum_names.end();i++){

    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry );

  }

  i = _ave_z_flux_sum_names.begin();
  for (;i!=_ave_z_flux_sum_names.end();i++){

    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry );

  }

  i = _x_flux_sqr_sum_names.begin();
  for (;i!=_x_flux_sqr_sum_names.end();i++){

    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry );

  }

  i = _y_flux_sqr_sum_names.begin();
  for (;i!=_y_flux_sqr_sum_names.end();i++){

    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry );

  }

  i = _z_flux_sqr_sum_names.begin();
  for (;i!=_z_flux_sqr_sum_names.end();i++){

    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry );

  }
}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace> void
VariableStats::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  auto i = _ave_sum_names.begin();
  for (;i!=_ave_sum_names.end();i++){
    CCVariable<double>& var = tsk_info->get_field<CCVariable<double> >( *i );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      var(i,j,k) = 0.0;
    });
  }

  i = _sqr_variable_names.begin();
  for (;i!=_sqr_variable_names.end();i++){
    CCVariable<double>& var = tsk_info->get_field<CCVariable<double> >( *i );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      var(i,j,k) = 0.0;
    });
  }

  i = _ave_x_flux_sum_names.begin();
  for (;i!=_ave_x_flux_sum_names.end();i++){
    SFCXVariable<double>& var = tsk_info->get_field<SFCXVariable<double> >( *i );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      var(i,j,k) = 0.0;
    });
  }

  i = _ave_y_flux_sum_names.begin();
  for (;i!=_ave_y_flux_sum_names.end();i++){
    SFCYVariable<double>& var = tsk_info->get_field<SFCYVariable<double> >( *i );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      var(i,j,k) = 0.0;
    });
  }

  i = _ave_z_flux_sum_names.begin();
  for (;i!=_ave_z_flux_sum_names.end();i++){
    SFCZVariable<double>& var = tsk_info->get_field<SFCZVariable<double> >( *i );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      var(i,j,k) = 0.0;
    });
  }
  i = _x_flux_sqr_sum_names.begin();
  for (;i!=_x_flux_sqr_sum_names.end();i++){
    SFCXVariable<double>& var = tsk_info->get_field<SFCXVariable<double> >( *i );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      var(i,j,k) = 0.0;
    });
  }
  i = _y_flux_sqr_sum_names.begin();
  for (;i!=_y_flux_sqr_sum_names.end();i++){
    SFCYVariable<double>& var = tsk_info->get_field<SFCYVariable<double> >( *i );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      var(i,j,k) = 0.0;
    });
  }
  i = _z_flux_sqr_sum_names.begin();
  for (;i!=_z_flux_sqr_sum_names.end();i++){
    SFCZVariable<double>& var = tsk_info->get_field<SFCZVariable<double> >( *i );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      var(i,j,k) = 0.0;
    });
  }
}

//--------------------------------------------------------------------------------------------------
void VariableStats::register_timestep_eval( VIVec& variable_registry, const int time_substep , const bool packed_tasks){

  std::vector<std::string>::iterator i = _ave_sum_names.begin();
  for (;i!=_ave_sum_names.end();i++){

    register_variable( *i, ArchesFieldContainer::MODIFIES, variable_registry );
    register_variable( *i, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );

  }

  i = _sqr_variable_names.begin();
  for (;i!=_sqr_variable_names.end();i++){

    register_variable( *i, ArchesFieldContainer::MODIFIES, variable_registry );
    register_variable( *i, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );

  }

  i = _base_var_names.begin();
  for (;i!=_base_var_names.end();i++){

    register_variable( *i, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry );

  }

  if ( !_no_flux ){
    i = _ave_x_flux_sum_names.begin();
    for (;i!=_ave_x_flux_sum_names.end();i++){

      register_variable( *i, ArchesFieldContainer::MODIFIES, variable_registry );
      register_variable( *i, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );

    }

    i = _ave_y_flux_sum_names.begin();
    for (;i!=_ave_y_flux_sum_names.end();i++){

      register_variable( *i, ArchesFieldContainer::MODIFIES, variable_registry );
      register_variable( *i, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );

    }

    i = _ave_z_flux_sum_names.begin();
    for (;i!=_ave_z_flux_sum_names.end();i++){

      register_variable( *i, ArchesFieldContainer::MODIFIES, variable_registry );
      register_variable( *i, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );

    }

    i = _x_flux_sqr_sum_names.begin();
    for (;i!=_x_flux_sqr_sum_names.end();i++){

      register_variable( *i, ArchesFieldContainer::MODIFIES, variable_registry );
      register_variable( *i, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );

    }

    i = _y_flux_sqr_sum_names.begin();
    for (;i!=_y_flux_sqr_sum_names.end();i++){

      register_variable( *i, ArchesFieldContainer::MODIFIES, variable_registry );
      register_variable( *i, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );

    }

    i = _z_flux_sqr_sum_names.begin();
    for (;i!=_z_flux_sqr_sum_names.end();i++){

      register_variable( *i, ArchesFieldContainer::MODIFIES, variable_registry );
      register_variable( *i, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );

    }

    register_variable( "uVelocitySPBC" , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( "vVelocitySPBC" , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( "wVelocitySPBC" , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( _rho_name       , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::NEWDW , variable_registry );

    std::vector<FluxInfo>::iterator ii = _flux_sum_info.begin();
    for (;ii!=_flux_sum_info.end();ii++){

      if ( (*ii).do_phi )
        register_variable( (*ii).phi , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::NEWDW , variable_registry );

    }
  }
}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void VariableStats::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  const double dt = tsk_info->get_dt();

  int N = _ave_sum_names.size();

  //----------NEBO----------------
  // (Uintah implementation pasted below)
  //NOTE: For single variables, we will leave them in situ with regards to
  //       their respective variable type (ie, T)
  //

  //Single variables
  for ( int i = 0; i < N; i++ ){


    CCVariable<double>& sum = tsk_info->get_field<CCVariable<double> >( _ave_sum_names[i] );
    constCCVariable<double>& old_sum = tsk_info->get_field<constCCVariable<double> >( _ave_sum_names[i] );
    CCVariable<double>& sqr_sum = tsk_info->get_field<CCVariable<double> >( _sqr_variable_names[i] );
    constCCVariable<double>& old_sqr_sum = tsk_info->get_field<constCCVariable<double> >( _sqr_variable_names[i] );

    // a base variable can be a double or a float
    const VarLabel* varlabel = VarLabel::find( _base_var_names[i], "ERROR  VariableStats::eval"  );
    const Uintah::TypeDescription* td = varlabel->typeDescription();
    const Uintah::TypeDescription::Type subtype = td->getSubType()->getType();

    //______double
    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    if ( subtype == TypeDescription::double_type ) {
      constCCVariable<double>& var = tsk_info->get_field<constCCVariable<double> >( _base_var_names[i] );

      Uintah::parallel_for( range, [&](int i, int j, int k){
        const double v = var(i,j,k);
        sum(i,j,k) = old_sum(i,j,k) + dt * v;
        sqr_sum(i,j,k) = old_sqr_sum(i,j,k) + dt * v * v;
      });
    }
    //______float
    else if ( subtype == TypeDescription::float_type ) {
      constCCVariable<float>& var = tsk_info->get_field<constCCVariable<float> >( _base_var_names[i] );

      Uintah::parallel_for( range, [&](int i, int j, int k){
        const double v = (double)var(i,j,k);
        sum(i,j,k) = old_sum(i,j,k) + dt * v;
        sqr_sum(i,j,k) = old_sqr_sum(i,j,k) + dt * v * v;
      });
    } else {
      throw InternalError("ERROR : Invalid variable type in VariableStats::eval(...) yet.", __FILE__, __LINE__);
    }
  }

  if ( !_no_flux ){

    //Fluxes
    constSFCXVariable<double>& u = tsk_info->get_field<constSFCXVariable<double> >( "uVelocitySPBC" );
    constSFCYVariable<double>& v = tsk_info->get_field<constSFCYVariable<double> >( "vVelocitySPBC" );
    constSFCZVariable<double>& w = tsk_info->get_field<constSFCZVariable<double> >( "wVelocitySPBC" );
    constCCVariable<double>& rho = tsk_info->get_field<constCCVariable<double> >( _rho_name );

    //X FLUX
    N = _ave_x_flux_sum_names.size();
    for ( int iname = 0; iname < N; iname++ ){

      SFCXVariable<double>&          sum = tsk_info->get_field<SFCXVariable<double> >( _ave_x_flux_sum_names[iname] );
      constSFCXVariable<double>& old_sum = tsk_info->get_field<constSFCXVariable<double> >( _ave_x_flux_sum_names[iname] );

      SFCXVariable<double>& sqr_sum = tsk_info->get_field<SFCXVariable<double> >( _x_flux_sqr_sum_names[iname] );
      constSFCXVariable<double>& old_sqr_sum = tsk_info->get_field<constSFCXVariable<double> >( _x_flux_sqr_sum_names[iname] );

      if ( _flux_sum_info[iname].do_phi ){

        constCCVariable<double>& phi = tsk_info->get_field<constCCVariable<double> >( _flux_sum_info[iname].phi );
        GET_EXTRACELL_FX_BUFFERED_PATCH_RANGE(0,1);
        Uintah::BlockRange range(low_fx_patch_range, high_fx_patch_range);
        Uintah::parallel_for(range, [&](int i, int j, int k){

          const double flux = u(i,j,k)/4. * ( (rho(i,j,k) + rho(i-1,j,k)) * (phi(i,j,k) + phi(i-1,j,k)) );
          sum(i,j,k)  = old_sum(i,j,k) + dt * flux;
          sqr_sum(i,j,k) = old_sqr_sum(i,j,k) + dt * ( flux * flux );

        });

      } else {

        GET_EXTRACELL_FX_BUFFERED_PATCH_RANGE(0,1);
        Uintah::BlockRange range(low_fx_patch_range, high_fx_patch_range);
        Uintah::parallel_for(range, [&](int i, int j, int k){

          const double flux = u(i,j,k) * ( (rho(i,j,k) + rho(i-1,j,k))/2. );
          sum(i,j,k)  = old_sum(i,j,k) + dt * flux;
          sqr_sum(i,j,k) = old_sqr_sum(i,j,k) + dt * ( flux * flux );
        });

      }
    }


    //Y FLUX
    N = _ave_y_flux_sum_names.size();
    for ( int iname = 0; iname < N; iname++ ){

      SFCYVariable<double>&          sum = tsk_info->get_field<SFCYVariable<double> >( _ave_y_flux_sum_names[iname] );
      constSFCYVariable<double>& old_sum = tsk_info->get_field<constSFCYVariable<double> >( _ave_y_flux_sum_names[iname] );

      SFCYVariable<double>& sqr_sum = tsk_info->get_field<SFCYVariable<double> >( _y_flux_sqr_sum_names[iname] );
      constSFCYVariable<double>& old_sqr_sum = tsk_info->get_field<constSFCYVariable<double> >( _y_flux_sqr_sum_names[iname] );

      if ( _flux_sum_info[iname].do_phi ){

        constCCVariable<double>& phi = tsk_info->get_field<constCCVariable<double> >( _flux_sum_info[iname].phi );
        GET_EXTRACELL_FY_BUFFERED_PATCH_RANGE(0,1);
        Uintah::BlockRange range(low_fy_patch_range, high_fy_patch_range);
        Uintah::parallel_for(range, [&](int i, int j, int k){

          const double flux = v(i,j,k)/4. * ( (rho(i,j,k) + rho(i,j-1,k)) * (phi(i,j,k) + phi(i,j-1,k)) );
          sum(i,j,k)  = old_sum(i,j,k) + dt * flux;
          sqr_sum(i,j,k) = old_sqr_sum(i,j,k) + dt * ( flux * flux );

        });

      } else {

        GET_EXTRACELL_FY_BUFFERED_PATCH_RANGE(0,1);
        Uintah::BlockRange range(low_fy_patch_range, high_fy_patch_range);
        Uintah::parallel_for(range, [&](int i, int j, int k){

          const double flux = v(i,j,k) * ( (rho(i,j,k) + rho(i,j-1,k))/2. );
          sum(i,j,k)  = old_sum(i,j,k) + dt * flux;
          sqr_sum(i,j,k) = old_sqr_sum(i,j,k) + dt * ( flux * flux );
        });

      }
    }

    //Z FLUX
    N = _ave_z_flux_sum_names.size();
    for ( int iname = 0; iname < N; iname++ ){

      SFCZVariable<double>&          sum = tsk_info->get_field<SFCZVariable<double> >( _ave_z_flux_sum_names[iname] );
      constSFCZVariable<double>& old_sum = tsk_info->get_field<constSFCZVariable<double> >( _ave_z_flux_sum_names[iname] );

      SFCZVariable<double>& sqr_sum = tsk_info->get_field<SFCZVariable<double> >( _z_flux_sqr_sum_names[iname] );
      constSFCZVariable<double>& old_sqr_sum = tsk_info->get_field<constSFCZVariable<double> >( _z_flux_sqr_sum_names[iname] );

      if ( _flux_sum_info[iname].do_phi ){

        constCCVariable<double>& phi = tsk_info->get_field<constCCVariable<double> >( _flux_sum_info[iname].phi );
        GET_EXTRACELL_FZ_BUFFERED_PATCH_RANGE(0,1);
        Uintah::BlockRange range(low_fz_patch_range, high_fz_patch_range);
        Uintah::parallel_for(range, [&](int i, int j, int k){

          const double flux = w(i,j,k)/4. * ( (rho(i,j,k) + rho(i,j,k-1)) * (phi(i,j,k) + phi(i,j,k-1)) );
          sum(i,j,k)  = old_sum(i,j,k) + dt * flux;
          sqr_sum(i,j,k) = old_sqr_sum(i,j,k) + dt * ( flux * flux );

        });

      } else {

        GET_EXTRACELL_FZ_BUFFERED_PATCH_RANGE(0,1);
        Uintah::BlockRange range(low_fz_patch_range, high_fz_patch_range);
        Uintah::parallel_for(range, [&](int i, int j, int k){

          const double flux = w(i,j,k) * ( (rho(i,j,k) + rho(i,j,k-1))/2. );
          sum(i,j,k)  = old_sum(i,j,k) + dt * flux;
          sqr_sum(i,j,k) = old_sqr_sum(i,j,k) + dt * ( flux * flux );
        });

      }
    }
  }
}
}
