#ifndef Uintah_Component_Arches_DSmaMMML_h
#define Uintah_Component_Arches_DSmaMMML_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/TurbulenceModels/DynamicSmagorinskyHelper.h>

namespace Uintah{
  template <typename TT>
  class DSmaMMML : public TaskInterface {

public:

    DSmaMMML( std::string task_name, int matl_index );
    ~DSmaMMML();

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

    //Build instructions for this (DSmaMMML) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) : m_task_name(task_name), m_matl_index(matl_index){}
      ~Builder(){}

      DSmaMMML* build()
      { return scinew DSmaMMML<TT>( m_task_name, m_matl_index ); }

      private:

      std::string m_task_name;
      int m_matl_index;
    };

private:
  std::string m_u_vel_name;
  //int Type_filter;
  double m_epsilon;
  Uintah::ArchesCore::FILTER Type_filter;
  std::string m_IsI_name;
  std::string m_volFraction_name;
  Uintah::ArchesCore::TestFilter m_Filter;
  };

//--------------------------------------------------------------------------------------------------
template<typename TT>
DSmaMMML<TT>::DSmaMMML( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

//--------------------------------------------------------------------------------------------------
template<typename TT>
DSmaMMML<TT>::~DSmaMMML(){}
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
  db->require("epsilon",m_epsilon);


  m_IsI_name = "strainMagnitudeLabel";
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
DSmaMMML<TT>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>&
                                       variable_registry , const bool packed_tasks){
}
//--------------------------------------------------------------------------------------------------

template<typename TT> void
DSmaMMML<TT>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

}
//--------------------------------------------------------------------------------------------------
template<typename TT> void
DSmaMMML<TT>::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry , const bool packed_tasks){

}

//--------------------------------------------------------------------------------------------------
template<typename TT> void
DSmaMMML<TT>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

}

//--------------------------------------------------------------------------------------------------
template<typename TT> void
DSmaMMML<TT>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry, const int time_substep , const bool packed_tasks){

  int nG = 1;
  //int nGrho = 1;
  if (packed_tasks ){
   nG = 3;
   //nGrho = 2;
  }
 register_variable( m_volFraction_name, ArchesFieldContainer::REQUIRES, nG, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

  register_variable( "filterbeta11"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "filterbeta12"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "filterbeta13"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "filterbeta22"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "filterbeta23"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "filterbeta33"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);

  register_variable( "filterIsI"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "filters11"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "filters12"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "filters13"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "filters22"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "filters23"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "filters33"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);

  register_variable( "alpha11"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "alpha12"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "alpha13"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "alpha22"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "alpha23"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "alpha33"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);

  register_variable( "filterrhoUU"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "filterrhoVV"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "filterrhoWW"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "filterrhoUV"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "filterrhoUW"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "filterrhoVW"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "filterrhoU"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "filterrhoV"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "filterrhoW"  , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);

  register_variable(  "MM" , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable(  "ML" , ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);

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
template<typename TT> void
DSmaMMML<TT>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  const Vector Dx = patch->dCell(); //
  double filter   = pow(Dx.x()*Dx.y()*Dx.z(),1.0/3.0);
  double filter2  = filter*filter;
  double fhat     = m_epsilon;
  constCCVariable<double>& vol_fraction = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_volFraction_name);


  int nGhosts1 = -1; //not using a temp field but rather the DW (ie, if nGhost < 0 then DW var)
  int nG = 0;
  if ( tsk_info->packed_tasks() ){
    nGhosts1 = 1;
    nG = 1;
  }

  IntVector low_filter = patch->getCellLowIndex() + IntVector(-nG,-nG,-nG);
  IntVector high_filter = patch->getCellHighIndex() + IntVector(nG,nG,nG);
  Uintah::BlockRange range1(low_filter, high_filter );

  FieldTool< TT > c_field_tool(tsk_info);

  //typedef typename ArchesCore::VariableHelper< TT >::ConstXFaceType TX;
  //typedef typename ArchesCore::VariableHelper< TT >::ConstYFaceType TY;
  //typedef typename ArchesCore::VariableHelper< TT >::ConstZFaceType TZ;

  typedef typename ArchesCore::VariableHelper< TT >::XFaceType TX;
  typedef typename ArchesCore::VariableHelper< TT >::YFaceType TY;
  typedef typename ArchesCore::VariableHelper< TT >::ZFaceType TZ;

  FieldTool< TX > x_field_tool(tsk_info);
  FieldTool< TY > y_field_tool(tsk_info);
  FieldTool< TZ > z_field_tool(tsk_info);

  TT* Beta11;
  TT* Beta12;
  TT* Beta13;
  TT* Beta22;
  TT* Beta23;
  TT* Beta33;

  Beta11 = c_field_tool.get("Beta11");
  Beta12 = c_field_tool.get("Beta12");
  Beta13 = c_field_tool.get("Beta13");
  Beta22 = c_field_tool.get("Beta22");
  Beta23 = c_field_tool.get("Beta23");
  Beta33 = c_field_tool.get("Beta33");


  TT* filterRho;
  filterRho = c_field_tool.get("Filterrho");
  TX* filterRhoU;
  TY* filterRhoV;
  TZ* filterRhoW;

  filterRhoU = x_field_tool.get("Filterrhou");
  filterRhoV = y_field_tool.get("Filterrhov");
  filterRhoW = z_field_tool.get("Filterrhow");

  // Filter Beta
  CCVariable<double>& filterBeta11 = tsk_info->get_uintah_field_add< CCVariable<double> >("filterbeta11", nGhosts1);
  CCVariable<double>& filterBeta12 = tsk_info->get_uintah_field_add< CCVariable<double> >("filterbeta12", nGhosts1);
  CCVariable<double>& filterBeta13 = tsk_info->get_uintah_field_add< CCVariable<double> >("filterbeta13", nGhosts1);
  CCVariable<double>& filterBeta22 = tsk_info->get_uintah_field_add< CCVariable<double> >("filterbeta22", nGhosts1);
  CCVariable<double>& filterBeta23 = tsk_info->get_uintah_field_add< CCVariable<double> >("filterbeta23", nGhosts1);
  CCVariable<double>& filterBeta33 = tsk_info->get_uintah_field_add< CCVariable<double> >("filterbeta33", nGhosts1);

  filterBeta11.initialize(0.0);
  filterBeta12.initialize(0.0);
  filterBeta13.initialize(0.0);
  filterBeta22.initialize(0.0);
  filterBeta23.initialize(0.0);
  filterBeta33.initialize(0.0);

  m_Filter.applyFilter<TT>((*Beta11),filterBeta11,vol_fraction,range1);
  m_Filter.applyFilter<TT>((*Beta22),filterBeta22,vol_fraction,range1);
  m_Filter.applyFilter<TT>((*Beta33),filterBeta33,vol_fraction,range1);
  m_Filter.applyFilter<TT>((*Beta12),filterBeta12,vol_fraction,range1);
  m_Filter.applyFilter<TT>((*Beta13),filterBeta13,vol_fraction,range1);
  m_Filter.applyFilter<TT>((*Beta23),filterBeta23,vol_fraction,range1);
  // Filter IsI and sij then compute alpha

  CCVariable<double>& filterIsI = tsk_info->get_uintah_field_add< CCVariable<double> >("filterIsI",nGhosts1 );
  CCVariable<double>& filters11 = tsk_info->get_uintah_field_add< CCVariable<double> >("filters11",nGhosts1 );
  CCVariable<double>& filters12 = tsk_info->get_uintah_field_add< CCVariable<double> >("filters12",nGhosts1 );
  CCVariable<double>& filters13 = tsk_info->get_uintah_field_add< CCVariable<double> >("filters13",nGhosts1 );
  CCVariable<double>& filters22 = tsk_info->get_uintah_field_add< CCVariable<double> >("filters22",nGhosts1 );
  CCVariable<double>& filters23 = tsk_info->get_uintah_field_add< CCVariable<double> >("filters23",nGhosts1 );
  CCVariable<double>& filters33 = tsk_info->get_uintah_field_add< CCVariable<double> >("filters33",nGhosts1 );


  filterIsI.initialize(0.0);
  filters11.initialize(0.0);
  filters12.initialize(0.0);
  filters13.initialize(0.0);
  filters22.initialize(0.0);
  filters23.initialize(0.0);
  filters33.initialize(0.0);

  Uintah::ArchesCore::computeFilterIsInsijv2 get_filterIsIsij(filterIsI, filters11, filters22,
                                                              filters33, filters12, filters13,
                                                              filters23, (*filterRhoU), (*filterRhoV),
                                                              (*filterRhoW), (*filterRho), Dx,vol_fraction);
  Uintah::parallel_for(range1,get_filterIsIsij);

  CCVariable<double>& alpha11 = tsk_info->get_uintah_field_add< CCVariable<double> >("alpha11",nGhosts1 );
  CCVariable<double>& alpha12 = tsk_info->get_uintah_field_add< CCVariable<double> >("alpha12",nGhosts1 );
  CCVariable<double>& alpha13 = tsk_info->get_uintah_field_add< CCVariable<double> >("alpha13",nGhosts1 );
  CCVariable<double>& alpha22 = tsk_info->get_uintah_field_add< CCVariable<double> >("alpha22",nGhosts1 );
  CCVariable<double>& alpha23 = tsk_info->get_uintah_field_add< CCVariable<double> >("alpha23",nGhosts1 );
  CCVariable<double>& alpha33 = tsk_info->get_uintah_field_add< CCVariable<double> >("alpha33",nGhosts1 );

  alpha11.initialize(0.0);
  alpha12.initialize(0.0);
  alpha13.initialize(0.0);
  alpha22.initialize(0.0);
  alpha23.initialize(0.0);
  alpha33.initialize(0.0);

  Uintah::parallel_for( range1, [&](int i, int j, int k){
    alpha11(i,j,k) = (*filterRho)(i,j,k)*filterIsI(i,j,k)*filters11(i,j,k);
    alpha22(i,j,k) = (*filterRho)(i,j,k)*filterIsI(i,j,k)*filters22(i,j,k);
    alpha33(i,j,k) = (*filterRho)(i,j,k)*filterIsI(i,j,k)*filters33(i,j,k);
    alpha12(i,j,k) = (*filterRho)(i,j,k)*filterIsI(i,j,k)*filters12(i,j,k);
    alpha13(i,j,k) = (*filterRho)(i,j,k)*filterIsI(i,j,k)*filters13(i,j,k);
    alpha23(i,j,k) = (*filterRho)(i,j,k)*filterIsI(i,j,k)*filters23(i,j,k);
  });


  // Filter rhouiuj and rhoui at cc

  TT* rhoUU ;
  TT* rhoVV ;
  TT* rhoWW ;
  TT* rhoUV ;
  TT* rhoUW ;
  TT* rhoVW ;
  TT* rhoU ;
  TT* rhoV ;
  TT* rhoW ;

  rhoUU = c_field_tool.get("rhoUU" );
  rhoVV = c_field_tool.get("rhoVV" );
  rhoWW = c_field_tool.get("rhoWW" );
  rhoUV = c_field_tool.get("rhoUV" );
  rhoUW = c_field_tool.get("rhoUW" );
  rhoVW = c_field_tool.get("rhoVW" );
  rhoU = c_field_tool.get("rhoU");
  rhoV = c_field_tool.get("rhoV");
  rhoW = c_field_tool.get("rhoW");

  CCVariable<double>& filter_rhoUU = tsk_info->get_uintah_field_add< CCVariable<double> >("filterrhoUU",nGhosts1 );
  CCVariable<double>& filter_rhoVV = tsk_info->get_uintah_field_add< CCVariable<double> >("filterrhoVV",nGhosts1 );
  CCVariable<double>& filter_rhoWW = tsk_info->get_uintah_field_add< CCVariable<double> >("filterrhoWW",nGhosts1 );
  CCVariable<double>& filter_rhoUV = tsk_info->get_uintah_field_add< CCVariable<double> >("filterrhoUV",nGhosts1 );
  CCVariable<double>& filter_rhoUW = tsk_info->get_uintah_field_add< CCVariable<double> >("filterrhoUW",nGhosts1 );
  CCVariable<double>& filter_rhoVW = tsk_info->get_uintah_field_add< CCVariable<double> >("filterrhoVW",nGhosts1 );
  CCVariable<double>& filter_rhoU = tsk_info->get_uintah_field_add< CCVariable<double> >("filterrhoU",nGhosts1 );
  CCVariable<double>& filter_rhoV = tsk_info->get_uintah_field_add< CCVariable<double> >("filterrhoV",nGhosts1 );
  CCVariable<double>& filter_rhoW = tsk_info->get_uintah_field_add< CCVariable<double> >("filterrhoW",nGhosts1 );

  filter_rhoUU.initialize(0.0);
  filter_rhoVV.initialize(0.0);
  filter_rhoWW.initialize(0.0);
  filter_rhoUV.initialize(0.0);
  filter_rhoUW.initialize(0.0);
  filter_rhoVW.initialize(0.0);
  filter_rhoU.initialize(0.0);
  filter_rhoV.initialize(0.0);
  filter_rhoW.initialize(0.0);

  m_Filter.applyFilter<TT>((*rhoUU),filter_rhoUU,vol_fraction,range1);
  m_Filter.applyFilter<TT>((*rhoVV),filter_rhoVV,vol_fraction,range1);
  m_Filter.applyFilter<TT>((*rhoWW),filter_rhoWW,vol_fraction,range1);
  m_Filter.applyFilter<TT>((*rhoUW),filter_rhoUW,vol_fraction,range1);
  m_Filter.applyFilter<TT>((*rhoUV),filter_rhoUV,vol_fraction,range1);
  m_Filter.applyFilter<TT>((*rhoVW),filter_rhoVW,vol_fraction,range1);
  m_Filter.applyFilter<TT>((*rhoV),filter_rhoV,vol_fraction,range1);
  m_Filter.applyFilter<TT>((*rhoW),filter_rhoW,vol_fraction,range1);
  m_Filter.applyFilter<TT>((*rhoU),filter_rhoU,vol_fraction,range1);


  CCVariable<double>& ML = tsk_info->get_uintah_field_add< CCVariable<double> >("ML",nGhosts1);
  CCVariable<double>& MM = tsk_info->get_uintah_field_add< CCVariable<double> >("MM",nGhosts1);
  ML.initialize(0.0);
  MM.initialize(0.0);

 const double SMALL = 1e-16;
  Uintah::parallel_for( range1, [&](int i, int j, int k){
    double M11 = 2.0*filter2*(filterBeta11(i,j,k) - 2.0*fhat*alpha11(i,j,k));
    double M22 = 2.0*filter2*(filterBeta22(i,j,k) - 2.0*fhat*alpha22(i,j,k));
    double M33 = 2.0*filter2*(filterBeta33(i,j,k) - 2.0*fhat*alpha33(i,j,k));
    double M12 = 2.0*filter2*(filterBeta12(i,j,k) - 2.0*fhat*alpha12(i,j,k));
    double M13 = 2.0*filter2*(filterBeta13(i,j,k) - 2.0*fhat*alpha13(i,j,k));
    double M23 = 2.0*filter2*(filterBeta23(i,j,k) - 2.0*fhat*alpha23(i,j,k));

    double L11 = filter_rhoUU(i,j,k) - filter_rhoU(i,j,k)*filter_rhoU(i,j,k)/((*filterRho)(i,j,k) + SMALL);
    double L22 = filter_rhoVV(i,j,k) - filter_rhoV(i,j,k)*filter_rhoV(i,j,k)/((*filterRho)(i,j,k) + SMALL);
    double L33 = filter_rhoWW(i,j,k) - filter_rhoW(i,j,k)*filter_rhoW(i,j,k)/((*filterRho)(i,j,k) + SMALL);
    double L12 = filter_rhoUV(i,j,k) - filter_rhoU(i,j,k)*filter_rhoV(i,j,k)/((*filterRho)(i,j,k) + SMALL);
    double L13 = filter_rhoUW(i,j,k) - filter_rhoU(i,j,k)*filter_rhoW(i,j,k)/((*filterRho)(i,j,k) + SMALL);
    double L23 = filter_rhoVW(i,j,k) - filter_rhoV(i,j,k)*filter_rhoW(i,j,k)/((*filterRho)(i,j,k) + SMALL);

    ML(i,j,k) = M11*L11 + M22*L22 + M33*L33 + 2.0*(M12*L12 + M13*L13 + M23*L23);
    MM(i,j,k) = M11*M11 + M22*M22 + M33*M33 + 2.0*(M12*M12 + M13*M13 + M23*M23);
  });

}
}
#endif
