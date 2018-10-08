#ifndef Uintah_Component_Arches_DSmaCs_h
#define Uintah_Component_Arches_DSmaCs_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/TurbulenceModels/DynamicSmagorinskyHelper.h>

namespace Uintah{

  template <typename TT>
  class DSmaCs : public TaskInterface {

public:

    DSmaCs( std::string task_name, int matl_index );
    ~DSmaCs();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();

    //Build instructions for this (DSmaCs) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) : m_task_name(task_name), m_matl_index(matl_index){}
      ~Builder(){}

      DSmaCs* build()
      { return scinew DSmaCs<TT>( m_task_name, m_matl_index ); }

      private:

      std::string m_task_name;
      int m_matl_index;
    };

private:

    std::string m_Cs_name; //DSmaCs constant
    std::string m_turb_viscosity_name;
    double m_molecular_visc;
    //std::string m_t_vis_name_production;
    std::string m_t_vis_name;
    //int Type_filter ;
    Uintah::ArchesCore::FILTER Type_filter;
    std::string m_volFraction_name;
    std::string m_density_name;
    std::string m_IsI_name;
    bool m_create_labels_IsI_t_viscosity{true};
    Uintah::ArchesCore::TestFilter m_Filter;
  };

//--------------------------------------------------------------------------------------------------
template<typename TT>
DSmaCs<TT>::DSmaCs( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {

}

//--------------------------------------------------------------------------------------------------
template<typename TT>
DSmaCs<TT>::~DSmaCs(){
}

//--------------------------------------------------------------------------------------------------
template<typename TT> void
DSmaCs<TT>::problemSetup( ProblemSpecP& db ){

  using namespace Uintah::ArchesCore;
  // u, v , w velocities

  //db->findBlock("Smagorinsky_constant_name")->getAttribute("Cs",m_Cs_name);
  m_Cs_name = "CsLabel";
  m_turb_viscosity_name = "turb_viscosity";
  if (db->findBlock("use_my_name_viscosity")){
    db->findBlock("use_my_name_viscosity")->getAttribute("label",m_t_vis_name);
  } else{
    m_t_vis_name = parse_ups_for_role( TOTAL_VISCOSITY, db, "viscosityCTS" );
  }

  //m_t_vis_name_production = "viscosityCTS";
  const ProblemSpecP params_root = db->getRootNode();
  if (params_root->findBlock("PhysicalConstants")) {
    params_root->findBlock("PhysicalConstants")->require("viscosity", m_molecular_visc);
    if( m_molecular_visc == 0 ) {
      throw InvalidValue("ERROR: Constant DSmaCs: problemSetup(): Zero viscosity specified in <PhysicalConstants> section of input file.",__FILE__,__LINE__);
    }
  } else {
    throw InvalidValue("ERROR: Constant DSmaCs: problemSetup(): Missing <PhysicalConstants> section in input file!",__FILE__,__LINE__);
  }

  std::string m_Type_filter_name;
  db->findBlock("filter")->getAttribute("type",m_Type_filter_name);
  Type_filter = get_filter_from_string( m_Type_filter_name );
  m_Filter.get_w(Type_filter);

  m_density_name     = parse_ups_for_role( DENSITY, db, "density" );
  m_volFraction_name = "volFraction";
  m_IsI_name = "strainMagnitudeLabel";

  if (m_t_vis_name == "viscosityCTS") { // this is production code
    m_create_labels_IsI_t_viscosity = false;
  }

}

//--------------------------------------------------------------------------------------------------
template<typename TT> void
DSmaCs<TT>::create_local_labels(){

  if (m_create_labels_IsI_t_viscosity) {
    register_new_variable<CCVariable<double> >( m_t_vis_name);
    register_new_variable<CCVariable<double> >( m_turb_viscosity_name);
    register_new_variable<CCVariable<double> >( m_Cs_name);
  }

  register_new_variable<CCVariable<double> >( "filterML");
  register_new_variable<CCVariable<double> >( "filterMM");

}

//--------------------------------------------------------------------------------------------------
template<typename TT> void
DSmaCs<TT>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>&
                                       variable_registry , const bool packed_tasks){
  if (m_create_labels_IsI_t_viscosity) {
    register_variable( m_t_vis_name, ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( m_turb_viscosity_name, ArchesFieldContainer::COMPUTES, variable_registry );
  }

  register_variable( m_Cs_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( "filterMM", ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( "filterML", ArchesFieldContainer::COMPUTES, variable_registry );
}

//--------------------------------------------------------------------------------------------------
template<typename TT> void
DSmaCs<TT>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){
  if (m_create_labels_IsI_t_viscosity) {
    CCVariable<double>& mu_sgc = *(tsk_info->get_uintah_field<CCVariable<double> >(m_t_vis_name));
    CCVariable<double>& mu_turb = *(tsk_info->get_uintah_field<CCVariable<double> >(m_turb_viscosity_name));
    mu_sgc.initialize(0.0);
    mu_turb.initialize(0.0);
  }
  CCVariable<double>& Cs = *(tsk_info->get_uintah_field<CCVariable<double> >(m_Cs_name));
  CCVariable<double>& filterMM = *(tsk_info->get_uintah_field<CCVariable<double> >("filterMM"));
  CCVariable<double>& filterML = *(tsk_info->get_uintah_field<CCVariable<double> >("filterML"));
  Cs.initialize(0.0);
  filterMM.initialize(0.0);
  filterML.initialize(0.0);
}
//--------------------------------------------------------------------------------------------------
template<typename TT> void
DSmaCs<TT>::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry , const bool packed_tasks){
  //register_variable( m_t_vis_name, ArchesFieldContainer::COMPUTES, variable_registry );
  //register_variable( m_Cs_name, ArchesFieldContainer::COMPUTES, variable_registry );
}

//--------------------------------------------------------------------------------------------------
template<typename TT> void
DSmaCs<TT>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){
  //CCVariable<double>& mu_sgc = *(tsk_info->get_uintah_field<CCVariable<double> >(m_t_vis_name));
  //CCVariable<double>& Cs = *(tsk_info->get_uintah_field<CCVariable<double> >(m_Cs_name));
}

//--------------------------------------------------------------------------------------------------
template<typename TT> void
DSmaCs<TT>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry, const int time_substep , const bool packed_tasks){

  if (m_create_labels_IsI_t_viscosity) {
    register_variable( m_t_vis_name, ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
    register_variable( m_turb_viscosity_name, ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  } else {
    register_variable( m_t_vis_name, ArchesFieldContainer::MODIFIES ,  variable_registry, time_substep );
    register_variable( m_turb_viscosity_name, ArchesFieldContainer::MODIFIES ,  variable_registry, time_substep );
    //register_variable( m_Cs_name, ArchesFieldContainer::MODIFIES ,  variable_registry, time_substep );
  }

  register_variable( m_Cs_name, ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  //register_variable( m_t_vis_name_production, ArchesFieldContainer::MODIFIES ,  variable_registry, time_substep );

  int nG = 1;
  if (packed_tasks ){
   nG = 3;
  }

  register_variable( m_density_name, ArchesFieldContainer::REQUIRES, nG, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  register_variable( m_volFraction_name, ArchesFieldContainer::REQUIRES, nG, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

  register_variable( "filterML" , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "filterMM" , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);

  register_variable("MM" , ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );
  register_variable("ML" , ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );
  register_variable(m_IsI_name , ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name, packed_tasks );


}

//--------------------------------------------------------------------------------------------------
template<typename TT> void
DSmaCs<TT>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& mu_sgc = *(tsk_info->get_uintah_field<CCVariable<double> >(m_t_vis_name));
  CCVariable<double>& mu_turb = *(tsk_info->get_uintah_field<CCVariable<double> >(m_turb_viscosity_name));
  CCVariable<double>& Cs = *(tsk_info->get_uintah_field<CCVariable<double> >(m_Cs_name));
  constCCVariable<double>& rho = *(tsk_info->get_const_uintah_field<constCCVariable<double> >(m_density_name));
  constCCVariable<double>& vol_fraction = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_volFraction_name);
  Cs.initialize(0.0);
  mu_sgc.initialize(0.0);
  mu_turb.initialize(0.0);

  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
  int nG = -1;
  if ( tsk_info->packed_tasks() ){
    nG = 0;
  }


  const Vector Dx = patch->dCell(); //
  double filter = pow(Dx.x()*Dx.y()*Dx.z(),1.0/3.0);
  double filter2 = filter*filter;

  FieldTool< TT > c_field_tool(tsk_info);

  TT* ML;
  TT* MM;
  TT* IsI;

  ML = c_field_tool.get("ML");
  MM = c_field_tool.get("MM");
  IsI = c_field_tool.get(m_IsI_name);

  CCVariable<double>& filterML = tsk_info->get_uintah_field_add< CCVariable<double> >("filterML", nG);
  //CCVariable<double>& filterML = tsk_info->get_uintah_field_add< CCVariable<double> >("filterML");
  //CCVariable<double>& filterMM = tsk_info->get_uintah_field_add< CCVariable<double> >("filterMM");
  CCVariable<double>& filterMM = tsk_info->get_uintah_field_add< CCVariable<double> >("filterMM", nG);
  filterML.initialize(0.0);
  filterMM.initialize(0.0);

  m_Filter.applyFilter<TT>((*MM),filterMM,range,vol_fraction);
  m_Filter.applyFilter<TT>((*ML),filterML,range,vol_fraction);

  const double m_MM_lower_value = 1.0e-14;
  const double m_ML_lower_value = 1.0e-14;
  Uintah::parallel_for( range, [&](int i, int j, int k){
    double value = 0;
    //if ( (*MM)(i,j,k) < m_MM_lower_value || (*ML)(i,j,k) < m_ML_lower_value) {
    if ( filterMM(i,j,k) < m_MM_lower_value || filterML(i,j,k) < m_ML_lower_value) {
    // value = 0.0;
    }else {
     //value  = (*ML)(i,j,k)/(*MM)(i,j,k);
      value  = filterML(i,j,k)/filterMM(i,j,k);
    }

    //double value  = filterML(i,j,k)/filterMM(i,j,k);
    //if (value < 0 || filterMM(i,j,k) < m_MM_lower_value) {
    //  value = 0;
    //}


    Cs(i,j,k) = vol_fraction(i,j,k)*Min(value,10.0);
    mu_sgc(i,j,k) = (Cs(i,j,k)*filter2*(*IsI)(i,j,k)*rho(i,j,k) + m_molecular_visc)*vol_fraction(i,j,k); //
    mu_turb(i,j,k) = mu_sgc(i,j,k) - m_molecular_visc; //

  });
  Uintah::ArchesCore::BCFilter bcfilter;
  bcfilter.apply_zero_neumann(patch,mu_sgc,vol_fraction);
  bcfilter.apply_zero_neumann(patch,mu_turb,vol_fraction);
  bcfilter.apply_zero_neumann(patch,Cs,vol_fraction);

  //Uintah::parallel_for( range, [&](int i, int j, int k){
  //  mu_sgc_p(i,j,k) = mu_sgc(i,j,k);
  //});
}
//--------------------------------------------------------------------------------------------------
}
#endif
