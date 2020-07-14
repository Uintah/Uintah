#ifndef Uintah_Component_Arches_DSmaMMML_h
#define Uintah_Component_Arches_DSmaMMML_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/TurbulenceModels/DynamicSmagorinskyHelper.h>

namespace Uintah{
  template <typename TT>
  class DSmaMMML : public TaskInterface {

public:

  DSmaMMML( std::string task_name, int matl_index, const std::string turb_model_name );
  ~DSmaMMML();

  TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

  TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

  TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

  TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

  TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();

  void problemSetup( ProblemSpecP& db );

  void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){}

  void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){}

  void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

  void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){}

  template <typename ExecSpace, typename MemSpace>
  void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

  template <typename ExecSpace, typename MemSpace>
  void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

  template <typename ExecSpace, typename MemSpace>
  void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

  template <typename ExecSpace, typename MemSpace>
  void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

  void create_local_labels();

  //Build instructions for this (DSmaMMML) class.
  class Builder : public TaskInterface::TaskBuilder {

    public:

    Builder( std::string task_name, int matl_index, const std::string turb_model_name )
      : m_task_name(task_name), m_matl_index(matl_index), m_turb_model_name(turb_model_name){}
    ~Builder(){}

    DSmaMMML* build()
    { return scinew DSmaMMML<TT>( m_task_name, m_matl_index, m_turb_model_name ); }

    private:

    std::string m_task_name;
    int m_matl_index;
    const std::string m_turb_model_name;

  };

private:

  std::string m_u_vel_name;
  //double m_epsilon;
  Uintah::ArchesCore::FILTER Type_filter;
  std::string m_IsI_name;
  std::string m_volFraction_name;
  Uintah::ArchesCore::TestFilter m_Filter;
  const std::string m_turb_model_name;

};

//-------------------- CLASS DEFINITIONS -----------------------------------------------------------

template<typename TT>
DSmaMMML<TT>::DSmaMMML( std::string task_name, int matl_index, const std::string turb_model_name ) :
TaskInterface( task_name, matl_index ), m_turb_model_name(turb_model_name) {
}

//--------------------------------------------------------------------------------------------------
template<typename TT>
DSmaMMML<TT>::~DSmaMMML(){}

//--------------------------------------------------------------------------------------------------
template<typename TT>
TaskAssignedExecutionSpace DSmaMMML<TT>::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
template<typename TT>
TaskAssignedExecutionSpace DSmaMMML<TT>::loadTaskInitializeFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
template<typename TT>
TaskAssignedExecutionSpace DSmaMMML<TT>::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &DSmaMMML<TT>::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &DSmaMMML<TT>::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &DSmaMMML<TT>::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
template<typename TT>
TaskAssignedExecutionSpace DSmaMMML<TT>::loadTaskTimestepInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
template<typename TT>
TaskAssignedExecutionSpace DSmaMMML<TT>::loadTaskRestartInitFunctionPointers()
{
 return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
template<typename TT> void
DSmaMMML<TT>::problemSetup( ProblemSpecP& db ){

  using namespace Uintah::ArchesCore;
  // u, v , w velocities

  std::string m_Type_filter_name;
  db->findBlock("filter")->getAttribute("type",m_Type_filter_name);
  Type_filter = get_filter_from_string( m_Type_filter_name );
  m_Filter.get_w(Type_filter);

  const ProblemSpecP params_root = db->getRootNode();
  //db->require("epsilon",m_epsilon);

  std::string u_vel_name = parse_ups_for_role( UVELOCITY_ROLE, db, ArchesCore::default_uVel_name );

  std::stringstream composite_name;
  composite_name << "strainMagnitude_" << m_turb_model_name;
  m_IsI_name = composite_name.str();

  //** HACK **//
  if ( u_vel_name == "uVelocitySPBC"){
    m_IsI_name = "strainMagnitudeLabel";
  }

  m_volFraction_name = "volFraction";

}
//--------------------------------------------------------------------------------------------------
template<typename TT> void
DSmaMMML<TT>::create_local_labels(){
  register_new_variable< CCVariable<double>  > ( "filterbeta11");
  register_new_variable< CCVariable<double>  > ( "filterbeta12");
  register_new_variable< CCVariable<double>  > ( "filterbeta13");
  register_new_variable< CCVariable<double>  > ( "filterbeta22");
  register_new_variable< CCVariable<double>  > ( "filterbeta23");
  register_new_variable< CCVariable<double>  > ( "filterbeta33");

  register_new_variable< CCVariable<double> > ("filterIsI");
  register_new_variable< CCVariable<double> > ("filters11");
  register_new_variable< CCVariable<double> > ("filters12");
  register_new_variable< CCVariable<double> > ("filters13");
  register_new_variable< CCVariable<double> > ("filters22");
  register_new_variable< CCVariable<double> > ("filters23");
  register_new_variable< CCVariable<double> > ("filters33");

  register_new_variable< CCVariable<double> > ( "alpha11");
  register_new_variable< CCVariable<double> > ( "alpha12");
  register_new_variable< CCVariable<double> > ( "alpha13");
  register_new_variable< CCVariable<double> > ( "alpha22");
  register_new_variable< CCVariable<double> > ( "alpha23");
  register_new_variable< CCVariable<double> > ( "alpha33");

  register_new_variable< CCVariable<double> > ( "filterrhoUU");
  register_new_variable< CCVariable<double> > ( "filterrhoVV");
  register_new_variable< CCVariable<double> > ( "filterrhoWW");
  register_new_variable< CCVariable<double> > ( "filterrhoUV");
  register_new_variable< CCVariable<double> > ( "filterrhoUW");
  register_new_variable< CCVariable<double> > ( "filterrhoVW");
  register_new_variable< CCVariable<double> > ( "filterrhoU");
  register_new_variable< CCVariable<double> > ( "filterrhoV");
  register_new_variable< CCVariable<double> > ( "filterrhoW");

  register_new_variable< CCVariable<double> > ( "MM");
  register_new_variable< CCVariable<double> > ( "ML");
}

//--------------------------------------------------------------------------------------------------
template<typename TT> void
DSmaMMML<TT>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>&
                                      variable_registry, const int time_substep ,
                                      const bool packed_tasks ){

  int nG3 = 1;
  int nG1 = 0;
  if (packed_tasks ){
   nG3 = 3;
   nG1 = 1;
  }

  register_variable( m_volFraction_name, ArchesFieldContainer::REQUIRES, nG3, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

  register_variable( "filterbeta11", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable( "filterbeta12", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable( "filterbeta13", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable( "filterbeta22", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable( "filterbeta23", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable( "filterbeta33", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);

  register_variable( "filterIsI", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable( "filters11", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable( "filters12", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable( "filters13", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable( "filters22", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable( "filters23", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable( "filters33", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);

  register_variable( "alpha11", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable( "alpha12", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable( "alpha13", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable( "alpha22", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable( "alpha23", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable( "alpha33", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);

  register_variable( "filterrhoUU", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable( "filterrhoVV", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable( "filterrhoWW", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable( "filterrhoUV", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable( "filterrhoUW", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable( "filterrhoVW", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable( "filterrhoU", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable( "filterrhoV", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable( "filterrhoW", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);

  register_variable(  "MM", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);
  register_variable(  "ML", ArchesFieldContainer::COMPUTESCRATCHGHOST, nG1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks);

  register_variable( "Beta11", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );
  register_variable( "Beta12", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );
  register_variable( "Beta13", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );
  register_variable( "Beta22", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );
  register_variable( "Beta23", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );
  register_variable( "Beta33", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );

  register_variable( "Filterrho", ArchesFieldContainer::REQUIRES,1 , ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );
  register_variable( "Filterrhou", ArchesFieldContainer::REQUIRES,1 , ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );
  register_variable( "Filterrhov", ArchesFieldContainer::REQUIRES,1 , ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );
  register_variable( "Filterrhow", ArchesFieldContainer::REQUIRES,1 , ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );

  register_variable( m_IsI_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );
  register_variable( "s11", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );
  register_variable( "s12", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );
  register_variable( "s13", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );
  register_variable( "s22", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );
  register_variable( "s23", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );
  register_variable( "s33", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );

  register_variable("rhoUU" , ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );
  register_variable("rhoVV" , ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );
  register_variable("rhoWW" , ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );
  register_variable("rhoUV" , ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );
  register_variable("rhoUW" , ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );
  register_variable("rhoVW" , ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );
  register_variable( "rhoU", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );
  register_variable( "rhoV", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );
  register_variable( "rhoW", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );


}

//--------------------------------------------------------------------------------------------------
template<typename TT>
template <typename ExecSpace, typename MemSpace>
void DSmaMMML<TT>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  const Vector Dx = patch->dCell(); //
  const double filter   = pow(Dx.x()*Dx.y()*Dx.z(),1.0/3.0);
  const double filter2  = filter*filter;
  const double fhat     = 3.; //Mystery value for tilde(bar(delta))
  auto vol_fraction = tsk_info->get_field<constCCVariable<double>, const double, MemSpace >(m_volFraction_name);


  int nG = 0;
  if ( tsk_info->packed_tasks() ){
    nG = 1;
  }

  IntVector low_filter = patch->getCellLowIndex() + IntVector(-nG,-nG,-nG);
  IntVector high_filter = patch->getCellHighIndex() + IntVector(nG,nG,nG);
  Uintah::BlockRange range1(low_filter, high_filter );

  //typedef typename ArchesCore::VariableHelper< TT >::ConstXFaceType TX;
  //typedef typename ArchesCore::VariableHelper< TT >::ConstYFaceType TY;
  //typedef typename ArchesCore::VariableHelper< TT >::ConstZFaceType TZ;

  typedef typename ArchesCore::VariableHelper< TT >::XFaceType TX;
  typedef typename ArchesCore::VariableHelper< TT >::YFaceType TY;
  typedef typename ArchesCore::VariableHelper< TT >::ZFaceType TZ;

  typedef typename ArchesCore::VariableHelper< TT >::PODType TTPODType;
  typedef typename ArchesCore::VariableHelper< TX >::PODType TXPODType;
  typedef typename ArchesCore::VariableHelper< TY >::PODType TYPODType;
  typedef typename ArchesCore::VariableHelper< TZ >::PODType TZPODType;


  auto Beta11 = tsk_info->get_field<TT, TTPODType, MemSpace >("Beta11");
  auto Beta12 = tsk_info->get_field<TT, TTPODType, MemSpace >("Beta12");
  auto Beta13 = tsk_info->get_field<TT, TTPODType, MemSpace >("Beta13");
  auto Beta22 = tsk_info->get_field<TT, TTPODType, MemSpace >("Beta22");
  auto Beta23 = tsk_info->get_field<TT, TTPODType, MemSpace >("Beta23");
  auto Beta33 = tsk_info->get_field<TT, TTPODType, MemSpace >("Beta33");


  auto filterRho = tsk_info->get_field<TT, TTPODType, MemSpace >("Filterrho");
  auto filterRhoU = tsk_info->get_field<TX, TXPODType, MemSpace >("Filterrhou");
  auto filterRhoV = tsk_info->get_field<TY, TYPODType, MemSpace >("Filterrhov");
  auto filterRhoW = tsk_info->get_field<TZ, TZPODType, MemSpace >("Filterrhow");

  // Filter Beta
  auto filterBeta11 = tsk_info->get_field< CCVariable<double>, double, MemSpace >("filterbeta11");
  auto filterBeta12 = tsk_info->get_field< CCVariable<double>, double, MemSpace >("filterbeta12");
  auto filterBeta13 = tsk_info->get_field< CCVariable<double>, double, MemSpace >("filterbeta13");
  auto filterBeta22 = tsk_info->get_field< CCVariable<double>, double, MemSpace >("filterbeta22");
  auto filterBeta23 = tsk_info->get_field< CCVariable<double>, double, MemSpace >("filterbeta23");
  auto filterBeta33 = tsk_info->get_field< CCVariable<double>, double, MemSpace >("filterbeta33");

  auto filterIsI = tsk_info->get_field< CCVariable<double>, double, MemSpace >("filterIsI");
  auto filters11 = tsk_info->get_field< CCVariable<double>, double, MemSpace >("filters11");
  auto filters12 = tsk_info->get_field< CCVariable<double>, double, MemSpace >("filters12");
  auto filters13 = tsk_info->get_field< CCVariable<double>, double, MemSpace >("filters13");
  auto filters22 = tsk_info->get_field< CCVariable<double>, double, MemSpace >("filters22");
  auto filters23 = tsk_info->get_field< CCVariable<double>, double, MemSpace >("filters23");
  auto filters33 = tsk_info->get_field< CCVariable<double>, double, MemSpace >("filters33");

  Uintah::parallel_initialize(execObj, 0.0, filterBeta11, filterBeta12, filterBeta13, filterBeta22, filterBeta23, filterBeta33,
		  filterIsI, filters11, filters12, filters13, filters22, filters23, filters33 );


  m_Filter.applyFilter(Beta11,filterBeta11,vol_fraction,range1, execObj);
  m_Filter.applyFilter(Beta22,filterBeta22,vol_fraction,range1, execObj);
  m_Filter.applyFilter(Beta33,filterBeta33,vol_fraction,range1, execObj);
  m_Filter.applyFilter(Beta12,filterBeta12,vol_fraction,range1, execObj);
  m_Filter.applyFilter(Beta13,filterBeta13,vol_fraction,range1, execObj);
  m_Filter.applyFilter(Beta23,filterBeta23,vol_fraction,range1, execObj);
  // Filter IsI and sij then compute alpha

  Uintah::parallel_for(execObj, range1, KOKKOS_LAMBDA(int i, int j, int k){

    const double SMALL = 1E-16;
    const double fuep = filterRhoU(i+1,j,k) /
           (0.5 * (filterRho(i,j,k) + filterRho(i+1,j,k)) + SMALL);

    const double fuwp = filterRhoU(i,j,k)/
           (0.5 * (filterRho(i,j,k) + filterRho(i-1,j,k)) + SMALL);

    //note: we have removed the (1/2) from the denom. because
    //we are multiplying by (1/2) for Sij
    const double funp = ( 0.5 * filterRhoU(i+1,j+1,k) /
           ( (filterRho(i,j+1,k) + filterRho(i+1,j+1,k)) + SMALL)
           + 0.5 * filterRhoU(i,j+1,k) /
           ( (filterRho(i,j+1,k) + filterRho(i-1,j+1,k))+ SMALL) );

    const double fusp = ( 0.5 * filterRhoU(i+1,j-1,k) /
           ( (filterRho(i,j-1,k) + filterRho(i+1,j-1,k)) + SMALL )
           + 0.5 * filterRhoU(i,j-1,k) /
           ( (filterRho(i,j-1,k) + filterRho(i-1,j-1,k))+ SMALL) );

    const double futp = ( 0.5 * filterRhoU(i+1,j,k+1) /
           ( (filterRho(i,j,k+1) + filterRho(i+1,j,k+1)) + SMALL )
           + 0.5 * filterRhoU(i,j,k+1) /
           ( (filterRho(i,j,k+1) + filterRho(i-1,j,k+1))+ SMALL));

    const double fubp = ( 0.5 * filterRhoU(i+1,j,k-1) /
           ( ( filterRho(i,j,k-1) + filterRho(i+1,j,k-1))+ SMALL)
           + 0.5 * filterRhoU(i,j,k-1) /
           ( (filterRho(i,j,k-1) + filterRho(i-1,j,k-1))+ SMALL));

    const double fvnp = filterRhoV(i,j+1,k) /
           ( 0.5 * (filterRho(i,j,k) + filterRho(i,j+1,k))+ SMALL);

    const double fvsp = filterRhoV(i,j,k) /
           ( 0.5 * (filterRho(i,j,k) + filterRho(i,j-1,k))+ SMALL);

    const double fvep = ( 0.5 * filterRhoV(i+1,j+1,k)/
           ( (filterRho(i+1,j,k) +filterRho(i+1,j+1,k))+ SMALL)
           + 0.5 * filterRhoV(i+1,j,k)/
           ( (filterRho(i+1,j,k) + filterRho(i+1,j-1,k))+ SMALL));

    const double fvwp = ( 0.5 * filterRhoV(i-1,j+1,k)/
           ( (filterRho(i-1,j,k) + filterRho(i-1,j+1,k))+ SMALL)
           + 0.5 * filterRhoV(i-1,j,k)/
           ( (filterRho(i-1,j,k) + filterRho(i-1,j-1,k))+ SMALL));

    const double fvtp = ( 0.5 * filterRhoV(i,j+1,k+1) /
           ( (filterRho(i,j,k+1) + filterRho(i,j+1,k+1))+ SMALL)
           + 0.5 * filterRhoV(i,j,k+1) /
           ( (filterRho(i,j,k+1) + filterRho(i,j-1,k+1))+ SMALL));

    const double fvbp = ( 0.5 * filterRhoV(i,j+1,k-1)/
           ( (filterRho(i,j,k-1) + filterRho(i,j+1,k-1))+ SMALL)
           + 0.5 * filterRhoV(i,j,k-1) /
           ( (filterRho(i,j,k-1) + filterRho(i,j-1,k-1))+ SMALL));

    const double fwtp = filterRhoW(i,j,k+1) /
           ( 0.5 * (filterRho(i,j,k) + filterRho(i,j,k+1))+ SMALL);

    const double fwbp = filterRhoW(i,j,k) /
           ( 0.5 * (filterRho(i,j,k) + filterRho(i,j,k-1))+ SMALL);

    const double fwep = ( 0.5 * filterRhoW(i+1,j,k+1) /
           ( (filterRho(i+1,j,k) + filterRho(i+1,j,k+1))+ SMALL)
           + 0.5 * filterRhoW(i+1,j,k) /
           ( (filterRho(i+1,j,k) + filterRho(i+1,j,k-1))+ SMALL));

    const double fwwp = ( 0.5 * filterRhoW(i-1,j,k+1) /
           ( (filterRho(i-1,j,k) + filterRho(i-1,j,k+1))+ SMALL)
           + 0.5 * filterRhoW(i-1,j,k) /
           ( (filterRho(i-1,j,k) + filterRho(i-1,j,k-1))+ SMALL));

    const double fwnp = ( 0.5 * filterRhoW(i,j+1,k+1)/
           ( (filterRho(i,j+1,k) + filterRho(i,j+1,k+1))+ SMALL)
           + 0.5 * filterRhoW(i,j+1,k) /
           ( (filterRho(i,j+1,k) + filterRho(i,j+1,k-1))+ SMALL));

    const double fwsp = ( 0.5 * filterRhoW(i,j-1,k+1)/
           ( (filterRho(i,j-1,k) + filterRho(i,j-1,k+1))+ SMALL)
           + 0.5 * filterRhoW(i,j-1,k)/
               ( (filterRho(i,j-1,k) + filterRho(i,j-1,k-1))+ SMALL));

    //calculate the filtered strain rate tensor
    filters11(i,j,k) = (fuep-fuwp)/Dx.x();
    filters22(i,j,k) = (fvnp-fvsp)/Dx.y();
    filters33(i,j,k) = (fwtp-fwbp)/Dx.z();
    filters12(i,j,k) = 0.5*((funp-fusp)/Dx.y() + (fvep-fvwp)/Dx.x());
    filters13(i,j,k) = 0.5*((futp-fubp)/Dx.z() + (fwep-fwwp)/Dx.x());
    filters23(i,j,k) = 0.5*((fvtp-fvbp)/Dx.z() + (fwnp-fwsp)/Dx.y());
    filterIsI(i,j,k) = std::sqrt(2.0*(filters11(i,j,k)*filters11(i,j,k)
                       + filters22(i,j,k)*filters22(i,j,k) + filters33(i,j,k)*filters33(i,j,k)+
                       2.0*(filters12(i,j,k)*filters12(i,j,k) +
                        filters13(i,j,k)*filters13(i,j,k) + filters23(i,j,k)*filters23(i,j,k))));

  });

  auto alpha11 = tsk_info->get_field< CCVariable<double>, double, MemSpace >("alpha11");
  auto alpha12 = tsk_info->get_field< CCVariable<double>, double, MemSpace >("alpha12");
  auto alpha13 = tsk_info->get_field< CCVariable<double>, double, MemSpace >("alpha13");
  auto alpha22 = tsk_info->get_field< CCVariable<double>, double, MemSpace >("alpha22");
  auto alpha23 = tsk_info->get_field< CCVariable<double>, double, MemSpace >("alpha23");
  auto alpha33 = tsk_info->get_field< CCVariable<double>, double, MemSpace >("alpha33");

  auto rhoUU = tsk_info->get_field<TT, TTPODType, MemSpace >("rhoUU" );
  auto rhoVV = tsk_info->get_field<TT, TTPODType, MemSpace >("rhoVV" );
  auto rhoWW = tsk_info->get_field<TT, TTPODType, MemSpace >("rhoWW" );
  auto rhoUV = tsk_info->get_field<TT, TTPODType, MemSpace >("rhoUV" );
  auto rhoUW = tsk_info->get_field<TT, TTPODType, MemSpace >("rhoUW" );
  auto rhoVW = tsk_info->get_field<TT, TTPODType, MemSpace >("rhoVW" );
  auto rhoU = tsk_info->get_field<TT, TTPODType, MemSpace >("rhoU");
  auto rhoV = tsk_info->get_field<TT, TTPODType, MemSpace >("rhoV");
  auto rhoW = tsk_info->get_field<TT, TTPODType, MemSpace >("rhoW");

  auto filter_rhoUU = tsk_info->get_field< CCVariable<double>, double, MemSpace >("filterrhoUU");
  auto filter_rhoVV = tsk_info->get_field< CCVariable<double>, double, MemSpace >("filterrhoVV");
  auto filter_rhoWW = tsk_info->get_field< CCVariable<double>, double, MemSpace >("filterrhoWW");
  auto filter_rhoUV = tsk_info->get_field< CCVariable<double>, double, MemSpace >("filterrhoUV");
  auto filter_rhoUW = tsk_info->get_field< CCVariable<double>, double, MemSpace >("filterrhoUW");
  auto filter_rhoVW = tsk_info->get_field< CCVariable<double>, double, MemSpace >("filterrhoVW");
  auto filter_rhoU = tsk_info->get_field< CCVariable<double>, double, MemSpace >("filterrhoU");
  auto filter_rhoV = tsk_info->get_field< CCVariable<double>, double, MemSpace >("filterrhoV");
  auto filter_rhoW = tsk_info->get_field< CCVariable<double>, double, MemSpace >("filterrhoW");

  Uintah::parallel_initialize(execObj, 0.0, alpha11, alpha12, alpha13, alpha22, alpha23, alpha33,
		  filter_rhoUU, filter_rhoVV, filter_rhoWW, filter_rhoUV, filter_rhoUW, filter_rhoVW, filter_rhoU, filter_rhoV, filter_rhoW);

  Uintah::parallel_for(execObj, range1, KOKKOS_LAMBDA(int i, int j, int k){
    alpha11(i,j,k) = filterRho(i,j,k)*filterIsI(i,j,k)*filters11(i,j,k);
    alpha22(i,j,k) = filterRho(i,j,k)*filterIsI(i,j,k)*filters22(i,j,k);
    alpha33(i,j,k) = filterRho(i,j,k)*filterIsI(i,j,k)*filters33(i,j,k);
    alpha12(i,j,k) = filterRho(i,j,k)*filterIsI(i,j,k)*filters12(i,j,k);
    alpha13(i,j,k) = filterRho(i,j,k)*filterIsI(i,j,k)*filters13(i,j,k);
    alpha23(i,j,k) = filterRho(i,j,k)*filterIsI(i,j,k)*filters23(i,j,k);
  });


  // Filter rhouiuj and rhoui at cc

  m_Filter.applyFilter(rhoUU,filter_rhoUU,vol_fraction,range1, execObj);
  m_Filter.applyFilter(rhoVV,filter_rhoVV,vol_fraction,range1, execObj);
  m_Filter.applyFilter(rhoWW,filter_rhoWW,vol_fraction,range1, execObj);
  m_Filter.applyFilter(rhoUW,filter_rhoUW,vol_fraction,range1, execObj);
  m_Filter.applyFilter(rhoUV,filter_rhoUV,vol_fraction,range1, execObj);
  m_Filter.applyFilter(rhoVW,filter_rhoVW,vol_fraction,range1, execObj);
  m_Filter.applyFilter(rhoV,filter_rhoV,vol_fraction,range1, execObj);
  m_Filter.applyFilter(rhoW,filter_rhoW,vol_fraction,range1, execObj);
  m_Filter.applyFilter(rhoU,filter_rhoU,vol_fraction,range1, execObj);


  auto ML = tsk_info->get_field< CCVariable<double>, double, MemSpace >("ML");
  auto MM = tsk_info->get_field< CCVariable<double>, double, MemSpace >("MM");
  Uintah::parallel_initialize(execObj, 0.0, ML, MM);

 const double SMALL = 1e-16;
  Uintah::parallel_for(execObj, range1, KOKKOS_LAMBDA(int i, int j, int k){
    double M11 = 2.0*filter2*(filterBeta11(i,j,k) - 2.0*fhat*alpha11(i,j,k));
    double M22 = 2.0*filter2*(filterBeta22(i,j,k) - 2.0*fhat*alpha22(i,j,k));
    double M33 = 2.0*filter2*(filterBeta33(i,j,k) - 2.0*fhat*alpha33(i,j,k));
    double M12 = 2.0*filter2*(filterBeta12(i,j,k) - 2.0*fhat*alpha12(i,j,k));
    double M13 = 2.0*filter2*(filterBeta13(i,j,k) - 2.0*fhat*alpha13(i,j,k));
    double M23 = 2.0*filter2*(filterBeta23(i,j,k) - 2.0*fhat*alpha23(i,j,k));

    double L11 = filter_rhoUU(i,j,k) - filter_rhoU(i,j,k)*filter_rhoU(i,j,k)/(filterRho(i,j,k) + SMALL);
    double L22 = filter_rhoVV(i,j,k) - filter_rhoV(i,j,k)*filter_rhoV(i,j,k)/(filterRho(i,j,k) + SMALL);
    double L33 = filter_rhoWW(i,j,k) - filter_rhoW(i,j,k)*filter_rhoW(i,j,k)/(filterRho(i,j,k) + SMALL);
    double L12 = filter_rhoUV(i,j,k) - filter_rhoU(i,j,k)*filter_rhoV(i,j,k)/(filterRho(i,j,k) + SMALL);
    double L13 = filter_rhoUW(i,j,k) - filter_rhoU(i,j,k)*filter_rhoW(i,j,k)/(filterRho(i,j,k) + SMALL);
    double L23 = filter_rhoVW(i,j,k) - filter_rhoV(i,j,k)*filter_rhoW(i,j,k)/(filterRho(i,j,k) + SMALL);

    ML(i,j,k) = M11*L11 + M22*L22 + M33*L33 + 2.0*(M12*L12 + M13*L13 + M23*L23);
    MM(i,j,k) = M11*M11 + M22*M22 + M33*M33 + 2.0*(M12*M12 + M13*M13 + M23*M23);
  });

}
}
#endif
