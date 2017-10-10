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

      Builder( std::string task_name, int matl_index ) : _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      DSmaCs* build()
      { return scinew DSmaCs<TT>( _task_name, _matl_index ); }

      private:

      std::string _task_name;
      int _matl_index;
    };

private:

    std::string m_Cs_name; //DSmaCs constant
    double m_molecular_visc;
    std::string m_t_vis_name;
    //int Type_filter ;
    Uintah::FILTER Type_filter;

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
  m_Cs_name = "Cs";

  if (db->findBlock("use_my_name_viscosity")){
    db->findBlock("use_my_name_viscosity")->getAttribute("label",m_t_vis_name);
  } else{
    m_t_vis_name = parse_ups_for_role( TOTAL_VISCOSITY, db );
  }

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
}

//--------------------------------------------------------------------------------------------------
template<typename TT> void
DSmaCs<TT>::create_local_labels(){
  register_new_variable<CCVariable<double> >( m_t_vis_name);
  register_new_variable<CCVariable<double> >( m_Cs_name);

  register_new_variable<CCVariable<double> >( "filterML");
  register_new_variable<CCVariable<double> >( "filterMM");

}

//--------------------------------------------------------------------------------------------------
template<typename TT> void
DSmaCs<TT>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>&
                                       variable_registry , const bool packed_tasks){
  register_variable( m_t_vis_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_Cs_name, ArchesFieldContainer::COMPUTES, variable_registry );

}

//--------------------------------------------------------------------------------------------------
template<typename TT> void
DSmaCs<TT>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){
  CCVariable<double>& mu_sgc = *(tsk_info->get_uintah_field<CCVariable<double> >(m_t_vis_name));
  CCVariable<double>& Cs = *(tsk_info->get_uintah_field<CCVariable<double> >(m_Cs_name));
  mu_sgc.initialize(0.0);
  Cs.initialize(0.0);
}
//--------------------------------------------------------------------------------------------------
template<typename TT> void
DSmaCs<TT>::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry , const bool packed_tasks){
  register_variable( m_t_vis_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_Cs_name, ArchesFieldContainer::COMPUTES, variable_registry );
}

//--------------------------------------------------------------------------------------------------
template<typename TT> void
DSmaCs<TT>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){
  CCVariable<double>& mu_sgc = *(tsk_info->get_uintah_field<CCVariable<double> >(m_t_vis_name));
  CCVariable<double>& Cs = *(tsk_info->get_uintah_field<CCVariable<double> >(m_Cs_name));
}

//--------------------------------------------------------------------------------------------------
template<typename TT> void
DSmaCs<TT>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry, const int time_substep , const bool packed_tasks){

  register_variable( m_t_vis_name, ArchesFieldContainer::MODIFIES ,  variable_registry, time_substep );
  register_variable( m_Cs_name, ArchesFieldContainer::MODIFIES ,  variable_registry, time_substep );


  register_variable( "filterML" , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , _task_name, packed_tasks);
  register_variable( "filterMM" , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , _task_name, packed_tasks);

  register_variable("MM" , ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, _task_name, packed_tasks );
  register_variable("ML" , ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, _task_name, packed_tasks );
  register_variable("IsI" , ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep, _task_name, packed_tasks );


}

//--------------------------------------------------------------------------------------------------
template<typename TT> void
DSmaCs<TT>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& mu_sgc = *(tsk_info->get_uintah_field<CCVariable<double> >(m_t_vis_name));
  CCVariable<double>& Cs = *(tsk_info->get_uintah_field<CCVariable<double> >(m_Cs_name));

  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );

  const Vector Dx = patch->dCell(); //
  double filter = pow(Dx.x()*Dx.y()*Dx.z(),1.0/3.0);
  double filter2 = filter*filter;

  FieldTool< TT > c_field_tool(tsk_info);

  TT* ML;
  TT* MM;
  TT* IsI;

  ML = c_field_tool.get("ML");
  MM = c_field_tool.get("MM");
  IsI = c_field_tool.get("IsI");

  CCVariable<double>& filterML = tsk_info->get_uintah_field_add< CCVariable<double> >("filterML", 0);
  CCVariable<double>& filterMM = tsk_info->get_uintah_field_add< CCVariable<double> >("filterMM", 0);
  filterML.initialize(0.0);
  filterMM.initialize(0.0);

  Uintah::FilterVarT<TT> get_fMM((*MM), filterMM, Type_filter);
  Uintah::FilterVarT<TT> get_fML((*ML), filterML, Type_filter);

  Uintah::parallel_for(range,get_fMM);
  Uintah::parallel_for(range,get_fML);

  Uintah::parallel_for( range, [&](int i, int j, int k){
    double value;
    if ( (*MM)(i,j,k) < 1.0e-14 || (*ML)(i,j,k) < 1.0e-14) {
//     value =0.04 ;
//     value =0.0289 ;
     value = 0.0;
    }else {
     value  = (*ML)(i,j,k)/(*MM)(i,j,k);
    }

    Cs(i,j,k) = Min(value,10.0);
    mu_sgc(i,j,k) = Cs(i,j,k)*filter2*(*IsI)(i,j,k) + m_molecular_visc; // I need to times density

  });
}
//--------------------------------------------------------------------------------------------------
}
#endif
