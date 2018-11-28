#ifndef Uintah_Component_Arches_MMS_ShunnP3_h
#define Uintah_Component_Arches_MMS_ShunnP3_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <Core/Grid/MaterialManager.h>

namespace Uintah{

  template <typename T>
  class MMS_ShunnP3 : public TaskInterface {

public:

  MMS_ShunnP3<T>( std::string task_name, int matl_index, MaterialManagerP materialManager  );
  ~MMS_ShunnP3<T>();

  void problemSetup( ProblemSpecP& db );

  //Build instructions for this (MMS_ShunnP3) class.
  class Builder : public TaskInterface::TaskBuilder {

    public:

    Builder( std::string task_name, int matl_index, MaterialManagerP materialManager ) :
      m_task_name(task_name), m_matl_index(matl_index), _materialManager(materialManager){}
    ~Builder(){}

    MMS_ShunnP3* build()
    { return scinew MMS_ShunnP3<T>( m_task_name, m_matl_index, _materialManager  ); }

    private:

    std::string m_task_name;
    int m_matl_index;

    MaterialManagerP _materialManager;
  };

protected:

  typedef ArchesFieldContainer::VariableInformation VarInfo;

  void register_initialize( std::vector<VarInfo>& variable_registry, const bool pack_tasks);

  void register_timestep_init( std::vector<VarInfo>& variable_registry, const bool pack_tasks);

  void register_timestep_eval( std::vector<VarInfo>& variable_registry, const int time_substep,
                                 const bool pack_tasks);

  void register_compute_bcs( std::vector<VarInfo>& variable_registry, const int time_substep,
                             const bool pack_tasks){}

  void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

  void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

  void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

  void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

  void create_local_labels();

private:

  const double m_pi = acos(-1.0);
  double m_k ;
  double m_w0 ;
  double m_rho0;
  double m_rho1;
  double m_uf;
  double m_vf;
  double m_D;

  std::string m_x_name;
  std::string m_y_name;

  std::string m_MMS_label;
  std::string m_rho_u_label;
  //std::string m_MMS_scalar_label;
  std::string m_MMS_source_label;
  //std::string m_MMS_source_scalar_label;
  //std::string m_MMS_rho_scalar_label;
  std::string m_MMS_rho_label;
  //std::string m_MMS_rho_face_label;
  std::string m_MMS_drhodt_label;
  std::string m_MMS_continuity_label;
  std::string m_which_vel;

  //std::string m_MMS_source_diff_label;
  //std::string m_MMS_source_t_label;

  MaterialManagerP _materialManager;

  void compute_source( const Patch* patch, ArchesTaskInfoManager* tsk_info );

};

//--------------------------------------------------------------------------------------------------
template <typename T>
MMS_ShunnP3<T>::MMS_ShunnP3( std::string task_name, int matl_index, MaterialManagerP materialManager ) :
TaskInterface( task_name, matl_index ) , _materialManager(materialManager){

}

//--------------------------------------------------------------------------------------------------
template <typename T>
MMS_ShunnP3<T>::~MMS_ShunnP3(){
}

//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_ShunnP3<T>::problemSetup( ProblemSpecP& db ){

  // Going to grab density from the cold flow properties list.
  // Note that the original Shunn paper mapped the fuel density (f=1) to rho1 and the air density (f=0)
  // to rho0. This is opposite of what Arches traditionally has done. So, in this file, we stick to
  // the Shunn notation but remap the density names for convienence.

  //NOTE: We are going to assume that the property the code is looking for is called "density"
  //      (as specified by the user)
  ProblemSpecP db_prop = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("StateProperties");
  bool found_coldflow_density = false;

  for ( ProblemSpecP db_p = db_prop->findBlock("model");
        db_p.get_rep() != nullptr;
        db_p = db_p->findNextBlock("model")){

    std::string label;
    std::string type;

    db_p->getAttribute("label", label);
    db_p->getAttribute("type", type);

    if ( type == "coldflow" ){

      for ( ProblemSpecP db_cf = db_p->findBlock("property");
            db_cf.get_rep() != nullptr;
            db_cf = db_cf->findNextBlock("property") ){

        std::string label;
        double value0;
        double value1;

        db_cf->getAttribute("label", label);

        std::cout << "LABEL = " << label << std::endl;

        if ( label == "density" ){
          db_cf->getAttribute("stream_0", value0);
          db_cf->getAttribute("stream_1", value1);

          found_coldflow_density = true;

          //NOTICE: We are inverting the mapping here. See note above.
          m_rho0 = value1;
          m_rho1 = value0;

        }
      }
    }
  }

  if ( !found_coldflow_density ){
    throw InvalidValue("Error: Cold flow property specification wasnt found which is needed to use the ShunnP3 source term.", __FILE__, __LINE__);
  }

  db->getWithDefault( "k", m_k, 2.0);
  db->getWithDefault( "w0", m_w0, 2.0);
  db->getWithDefault( "D", m_D, 0.001);
  db->getWithDefault( "uf", m_uf, 0.0);
  db->getWithDefault( "vf", m_vf, 0.0);


  db->require("which_vel", m_which_vel);
  ProblemSpecP db_coord = db->findBlock("coordinates");
  if ( db_coord ){
    db_coord->getAttribute("x", m_x_name);
    db_coord->getAttribute("y", m_y_name);
  } else {
    throw InvalidValue("Error: must have coordinates specified for almgren MMS init condition",
      __FILE__, __LINE__);
  }

  m_MMS_label             = m_task_name;
  m_rho_u_label           = m_task_name + "_rho_u";
  //m_MMS_scalar_label      = m_task_name+"_scalar";
  m_MMS_source_label      = m_task_name + "_source";
  //m_MMS_source_scalar_label = m_task_name + "_source_scalar";

  //m_MMS_rho_scalar_label   = m_task_name+"_rho_scalar";
  m_MMS_rho_label          = m_task_name+"_rho";
  //m_MMS_rho_face_label          = m_task_name+"_rho_face";

  m_MMS_drhodt_label       = m_task_name+"_drhodt";
  m_MMS_continuity_label       = m_task_name+"_continuity";

  //m_MMS_source_diff_label = m_task_name + "_source_diff";
  //m_MMS_source_t_label    = m_task_name + "_source_time";

}

//------------------------------------------------------------------------------
template <typename T>
void MMS_ShunnP3<T>::create_local_labels(){

  register_new_variable< T >( m_MMS_label);
  register_new_variable< T >( m_rho_u_label);
  //register_new_variable< CCVariable<double> >( m_MMS_scalar_label);
  //register_new_variable< CCVariable<double> >( m_MMS_source_scalar_label);
  register_new_variable< T >( m_MMS_drhodt_label);
  register_new_variable< T >( m_MMS_continuity_label);
  register_new_variable< T >( m_MMS_source_label);
  // register_new_variable< T >( m_MMS_source_diff_label);
  //register_new_variable< T >( m_MMS_source_t_label);
  //register_new_variable< CCVariable<double> >( m_MMS_rho_scalar_label);
  register_new_variable< CCVariable<double> >( m_MMS_rho_label);
  //register_new_variable< T >( m_MMS_rho_face_label);


}

//------------------------------------------------------------------------------
template <typename T>
void MMS_ShunnP3<T>::register_initialize( std::vector<VarInfo>&
                                      variable_registry , const bool pack_tasks){

  register_variable( m_MMS_label,               ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_rho_u_label,             ArchesFieldContainer::COMPUTES, variable_registry );
  //register_variable( m_MMS_scalar_label,        ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_MMS_source_label,        ArchesFieldContainer::COMPUTES, variable_registry );
  //register_variable( m_MMS_source_scalar_label, ArchesFieldContainer::COMPUTES, variable_registry );
  //register_variable( m_MMS_source_diff_label, ArchesFieldContainer::COMPUTES, variable_registry );
  //register_variable( m_MMS_source_t_label,    ArchesFieldContainer::COMPUTES, variable_registry );
  //register_variable( m_MMS_rho_scalar_label,    ArchesFieldContainer::COMPUTES ,  variable_registry);
  register_variable( m_MMS_rho_label,           ArchesFieldContainer::COMPUTES ,  variable_registry );
  //register_variable( m_MMS_rho_face_label,      ArchesFieldContainer::COMPUTES ,  variable_registry);

  register_variable( m_MMS_drhodt_label,        ArchesFieldContainer::COMPUTES ,  variable_registry );
  register_variable( m_MMS_continuity_label,        ArchesFieldContainer::COMPUTES ,  variable_registry );

  register_variable( m_x_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry );
  register_variable( m_y_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry );

}

//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_ShunnP3<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  compute_source(patch, tsk_info);

}
//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_ShunnP3<T>::register_timestep_init( std::vector<VarInfo>&
                                          variable_registry , const bool pack_tasks){
}

//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_ShunnP3<T>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

}

//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_ShunnP3<T>::register_timestep_eval( std::vector<VarInfo>&
                                          variable_registry, const int time_substep , const bool pack_tasks){

  register_variable( m_MMS_label,               ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  register_variable( m_rho_u_label,             ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  //register_variable( m_MMS_scalar_label,        ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  register_variable( m_MMS_source_label,        ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  //register_variable( m_MMS_source_scalar_label, ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  //register_variable( m_MMS_rho_scalar_label,    ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  register_variable( m_MMS_rho_label,           ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  //register_variable( m_MMS_rho_face_label,      ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  register_variable( m_MMS_drhodt_label,        ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  register_variable( m_MMS_continuity_label,        ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );

 // register_variable( m_MMS_source_diff_label, ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
 // register_variable( m_MMS_source_t_label,    ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );

  register_variable( m_x_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  register_variable( m_y_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

}

//--------------------------------------------------------------------------------------------------
template <typename T>
void MMS_ShunnP3<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  compute_source( patch, tsk_info );

}

template <typename T>
void MMS_ShunnP3<T>::compute_source( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  double time_d      = tsk_info->get_time(); //_materialManager->getElapsedSimTime();
  int   time_substep = tsk_info->get_time_substep();
  double factor      = tsk_info->get_ssp_time_factor(time_substep);
  double dt          = tsk_info->get_dt();
  time_d = time_d + factor*dt;

  T& f_mms        = tsk_info->get_uintah_field_add<T>(m_MMS_label);
  T& rho_f_mms    = tsk_info->get_uintah_field_add<T>(m_rho_u_label);
  CCVariable<double>& rho_mms      = tsk_info->get_uintah_field_add<CCVariable<double> >(m_MMS_rho_label);
  T& f_source_mms = tsk_info->get_uintah_field_add<T>(m_MMS_source_label);
  T& drhodt_mms   = tsk_info->get_uintah_field_add<T>(m_MMS_drhodt_label);
  T& continuity_mms   = tsk_info->get_uintah_field_add<T>(m_MMS_continuity_label);

  constCCVariable<double>& x = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_x_name);
  constCCVariable<double>& y = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_y_name);

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  if ( m_which_vel == "u" ){
    // for velocity
      Uintah::parallel_for( range, [&](int i, int j, int k){
        const double phi_f = (1.0 + std::sin(m_k*m_pi*(x(i,j,k)-m_uf*time_d))*
                        std::sin(m_k*m_pi*(y(i,j,k)-m_vf*time_d))*std::cos(m_w0*m_pi*time_d))/(1.0 +
                        m_rho0/m_rho1+(1.0-m_rho0/m_rho1)*std::sin(m_k*m_pi*(x(i,j,k)-m_uf*time_d))*
                        std::sin(m_k*m_pi*(y(i,j,k)-m_vf*time_d))*std::cos(m_w0*m_pi*time_d));

        rho_mms(i,j,k) = 1.0/(phi_f/m_rho1 + (1.0- phi_f )/m_rho0);

        rho_f_mms(i,j,k) = m_uf*rho_mms(i,j,k) - m_w0/m_k/4.0*std::cos(m_k*m_pi*(x(i,j,k)-m_uf*time_d))*std::sin(m_k*m_pi*(y(i,j,k)-m_vf*time_d))*std::sin(m_w0*m_pi*time_d)*(m_rho1-m_rho0);
        f_mms(i,j,k)     = rho_f_mms(i,j,k)/rho_mms(i,j,k);

        const double t2 = m_pi*m_w0*time_d;
        const double t3 = cos(t2);
        const double t12 = m_uf*time_d;
        const double t4 = -t12+x(i,j,k);
        const double t5 = m_k*m_pi*t4;
        const double t6 = sin(t5);
        const double t13 = m_vf*time_d;
        const double t7 = -t13+y(i,j,k);
        const double t8 = m_k*m_pi*t7;
        const double t9 = sin(t8);
        const double t10 = 1.0/m_rho1;
        const double t11 = m_rho0*t10;
        const double t14 = t3*t6*t9;
        const double t15 = t14+1.0;
        const double t16 = t11-1.0;
        const double t22 = t3*t6*t9*t16;
        const double t17 = t11-t22+1.0;
        const double t18 = 1.0/t17;
        const double t19 = 1.0/m_rho0;
        const double t20 = sin(t2);
        const double t21 = cos(t5);
        const double t23 = cos(t8);
        const double t24 = m_pi*m_w0*t6*t9*t20;
        const double t25 = m_k*m_pi*m_uf*t3*t9*t21;
        const double t26 = m_k*m_pi*m_vf*t3*t6*t23;
        const double t27 = t24+t25+t26;
        const double t28 = m_pi*m_w0*t6*t9*t16*t20;
        const double t29 = m_k*m_pi*m_uf*t3*t9*t16*t21;
        const double t30 = m_k*m_pi*m_vf*t3*t6*t16*t23;
        const double t31 = t28+t29+t30;
        const double t32 = 1.0/(t17*t17);
        const double t33 = 1.0/m_k;
        const double t34 = m_rho0-m_rho1;
        const double t35 = t15*t18;
        const double t36 = t35-1.0;
        const double t37 = t19*t36;
        const double t39 = t10*t15*t18;
        const double t38 = t37-t39;
        const double t40 = m_k*m_k;
        const double t41 = m_pi*m_pi;
        const double t42 = t3*t3;
        const double t43 = t23*t23;
        const double t44 = t6*t6;
        const double t45 = t16*t16;
        const double t46 = 1.0/(t17*t17*t17);
        const double t47 = t3*t6*t9*t18*t40*t41;
        const double t48 = t3*t6*t9*t15*t16*t32*t40*t41;
        const double t49 = t21*t21;
        const double t50 = t9*t9;
        const double t51 = m_k*m_w0*t9*t20*t21*t34*t38*t41*(1.0/4.0);
        const double t67 = m_w0*t9*t20*t21*t33*t34*t38*(1.0/4.0);
        const double t52 = m_uf-t67;
        const double t53 = m_k*m_pi*t3*t9*t18*t21;
        const double t54 = m_k*m_pi*t3*t9*t15*t16*t21*t32;
        const double t55 = t53+t54;
        const double t56 = m_k*m_pi*t3*t9*t10*t18*t21;
        const double t57 = m_k*m_pi*t3*t9*t10*t15*t16*t21*t32;
        const double t66 = t19*t55;
        const double t58 = t56+t57-t66;
        const double t59 = m_k*m_pi*t3*t6*t18*t23;
        const double t60 = m_k*m_pi*t3*t6*t15*t16*t23*t32;
        const double t61 = t59+t60;
        const double t62 = m_k*m_pi*t3*t6*t10*t18*t23;
        const double t63 = m_k*m_pi*t3*t6*t10*t15*t16*t23*t32;
        const double t69 = t19*t61;
        const double t64 = t62+t63-t69;
        const double t65 = 1.0/t38;
        const double t68 = m_pi*m_w0*t6*t9*t20*t34*t38*(1.0/4.0);
        const double t70 = 1.0/(t38*t38);
        const double t71 = t18*t27;
        const double t72 = t15*t31*t32;
        const double t73 = t71+t72;
        const double t74 = t10*t18*t27;
        const double t75 = t10*t15*t31*t32;
        const double t76 = t74+t75-t19*t73;
        const double t77 = m_vf-m_w0*t6*t20*t23*t33*t34*t38*(1.0/4.0);
        f_source_mms(i,j,k) =  -m_D*(t51+m_pi*m_w0*t20*t21*t23*t34*t64*(1.0/2.0)+
                               m_w0*t9*t20*t21*t33*t34*(t19*(t47+t48-t16*t32*t40*
                               t41*t42*t43*t44*2.0-t15*t40*t41*t42*t43*t44*t45*
                               t46*2.0)-t3*t6*t9*t10*t18*t40*t41+t10*t16*t32*
                               t40*t41*t42*t43*t44*2.0-t3*t6*t9*t10*t15*t16*
                               t32*t40*t41+t10*t15*t40*t41*t42*t43*t44*t45*t46*2.0)*
                               (1.0/4.0))-m_D*(t51-m_pi*m_w0*t6*t9*t20*t34*t58*(1.0/2.0)+
                               m_w0*t9*t20*t21*t33*t34*(t19*(t47+t48-t16*t32*t40*t41*t42*
                               t49*t50*2.0-t15*t40*t41*t42*t45*t46*t49*t50*2.0)-t3*t6*t9*
                               t10*t18*t40*t41+t10*t16*t32*t40*t41*t42*t49*t50*2.0-t3*t6*
                               t9*t10*t15*t16*t32*t40*t41+t10*t15*t40*t41*t42*t45*t46*t49*
                               t50*2.0)*(1.0/4.0))+t65*(m_w0*t9*t20*t21*t33*t34*t76*(1.0/4.0)+
                               m_pi*(m_w0*m_w0)*t3*t9*t21*t33*t34*t38*(1.0/4.0)+m_pi*m_uf*
                               m_w0*t6*t9*t20*t34*t38*(1.0/4.0)-m_pi*m_vf*m_w0*t20*t21*
                               t23*t34*t38*(1.0/4.0))-(t52*t52)*t58*t70-t52*t65*(t68+
                               m_w0*t9*t20*t21*t33*t34*t58*(1.0/4.0))*2.0-t52*t65*
                               (t68+m_w0*t6*t20*t23*t33*t34*t64*(1.0/4.0))+t65*t77*
                               (m_pi*m_w0*t20*t21*t23*t34*t38*(1.0/4.0)-m_w0*t9*t20*
                               t21*t33*t34*t64*(1.0/4.0))+t52*t70*t76-t52*t64*t70*t77;

      });
  } else {
    // for scalar
      Uintah::parallel_for( range, [&](int i, int j, int k){
        f_mms(i,j,k) = (1.0 + std::sin(m_k*m_pi*(x(i,j,k)-m_uf*time_d))*
                        std::sin(m_k*m_pi*(y(i,j,k)-m_vf*time_d))*std::cos(m_w0*m_pi*time_d))/(1.0 +
                        m_rho0/m_rho1+(1.0-m_rho0/m_rho1)*std::sin(m_k*m_pi*(x(i,j,k)-m_uf*time_d))*
                        std::sin(m_k*m_pi*(y(i,j,k)-m_vf*time_d))*std::cos(m_w0*m_pi*time_d));

        rho_mms(i,j,k) =1.0/(f_mms(i,j,k)/m_rho1 + (1.0- f_mms(i,j,k) )/m_rho0);

        rho_f_mms(i,j,k) = f_mms(i,j,k)*rho_mms(i,j,k);

        const double t2 = m_pi*m_w0*time_d;
        const double t3 = cos(t2);
        const double t14 = m_uf*time_d;
        const double t4 = -t14+x(i,j,k);
        const double t5 = m_k*m_pi*t4;
        const double t6 = sin(t5);
        const double t16 = m_vf*time_d;
        const double t7 = -t16+y(i,j,k);
        const double t8 = m_k*m_pi*t7;
        const double t9 = sin(t8);
        const double t10 = 1.0/m_rho1;
        const double t11 = m_rho0*t10;
        const double t12 = m_k*m_k;
        const double t13 = m_pi*m_pi;
        const double t15 = cos(t5);
        const double t17 = t11-1.0;
        const double t19 = t3*t6*t9*t17;
        const double t18 = t11-t19+1.0;
        const double t20 = 1.0/(t18*t18);
        const double t21 = t3*t3;
        const double t22 = t15*t15;
        const double t23 = t9*t9;
        const double t24 = t3*t6*t9;
        const double t25 = t24+1.0;
        const double t26 = 1.0/t18;
        const double t27 = t3*t6*t9*t12*t13*t26;
        const double t28 = cos(t8);
        const double t29 = t3*t6*t9*t12*t13*t17*t20*t25;
        const double t30 = t28*t28;
        const double t31 = t6*t6;
        const double t32 = t17*t17;
        const double t33 = 1.0/(t18*t18*t18);
        const double t34 = sin(t2);
        const double t35 = 1.0/m_rho0;
        const double t36 = t25*t26;
        const double t37 = t36-1.0;
        const double t38 = t35*t37;
        const double t41 = t10*t25*t26;
        const double t39 = t38-t41;
        const double t40 = m_rho0-m_rho1;
        const double t42 = 1.0/t39;
        const double t43 = m_pi*m_w0*t6*t9*t34*t39*t40*(1.0/4.0);
        const double t44 = 1.0/m_k;
        const double t45 = m_pi*m_w0*t6*t9*t34;
        const double t46 = m_k*m_pi*m_uf*t3*t9*t15;
        const double t47 = m_k*m_pi*m_vf*t3*t6*t28;
        const double t48 = t45+t46+t47;
        const double t49 = m_pi*m_w0*t6*t9*t17*t34;
        const double t50 = m_k*m_pi*m_uf*t3*t9*t15*t17;
        const double t51 = m_k*m_pi*m_vf*t3*t6*t17*t28;
        const double t52 = t49+t50+t51;
        const double t53 = 1.0/(t39*t39);
        const double t54 = m_k*m_pi*t3*t9*t15*t26;
        const double t55 = m_k*m_pi*t3*t9*t15*t17*t20*t25;
        const double t56 = t54+t55;
        const double t57 = m_k*m_pi*t3*t9*t10*t15*t26;
        const double t58 = m_k*m_pi*t3*t9*t10*t15*t17*t20*t25;
        const double t59 = t57+t58-t35*t56;
        const double t60 = m_k*m_pi*t3*t6*t26*t28;
        const double t61 = m_k*m_pi*t3*t6*t17*t20*t25*t28;
        const double t62 = t60+t61;
        const double t63 = m_k*m_pi*t3*t6*t10*t26*t28;
        const double t64 = m_k*m_pi*t3*t6*t10*t17*t20*t25*t28;
        const double t65 = t63+t64-t35*t62;
        const double t68 = m_w0*t9*t15*t34*t39*t40*t44*(1.0/4.0);
        const double t66 = m_uf-t68;
        const double t69 = m_w0*t6*t28*t34*t39*t40*t44*(1.0/4.0);
        const double t67 = m_vf-t69;
        f_source_mms(i,j,k) =  m_D*(t27+t29-t12*t13*t17*t20*t21*
                               t22*t23*2.0-t12*t13*t21*t22*t23*t25*
                               t32*t33*2.0)+m_D*(t27+t29-t12*t13*
                               t17*t20*t21*t30*t31*2.0-t12*t13*
                               t21*t25*t30*t31*t32*t33*2.0)+
                               t26*t42*t48-t25*t26*t42*(t43+
                               m_w0*t9*t15*t34*t40*t44*t59*(1.0/4.0))-
                               t25*t26*t42*(t43+m_w0*t6*t28*t34*t40*t44*
                               t65*(1.0/4.0))+t20*t25*t42*t52+t25*t26*
                               t53*(-t35*(t26*t48+t20*t25*t52)+t10*t26*
                               t48+t10*t20*t25*t52)-t25*t26*t53*t59*t66-
                               t25*t26*t53*t65*t67-m_k*m_pi*t3*t9*t15*
                               t26*t42*t66-m_k*m_pi*t3*t6*t26*t28*t42*
                               t67-m_k*m_pi*t3*t9*t15*t17*t20*t25*t42*
                               t66-m_k*m_pi*t3*t6*t17*t20*t25*t28*t42*t67;

    });

    // drhodt
    Uintah::parallel_for( range, [&](int i, int j, int k){

        const double t2 = m_pi*m_w0*time_d;
        const double t3 = cos(t2);
        const double t12 = m_uf*time_d;
        const double t4 = -t12+x(i,j,k);
        const double t5 = m_k*m_pi*t4;
        const double t6 = sin(t5);
        const double t13 = m_vf*time_d;
        const double t7 = -t13+y(i,j,k);
        const double t8 = m_k*m_pi*t7;
        const double t9 = sin(t8);
        const double t10 = 1.0/m_rho1;
        const double t11 = m_rho0*t10;
        const double t14 = t3*t6*t9;
        const double t15 = t14+1.0;
        const double t16 = t11-1.0;
        const double t20 = t3*t6*t9*t16;
        const double t17 = t11-t20+1.0;
        const double t18 = 1.0/t17;
        const double t19 = 1.0/m_rho0;
        const double t21 = sin(t2);
        const double t22 = cos(t5);
        const double t23 = cos(t8);
        const double t24 = m_pi*m_w0*t6*t9*t21;
        const double t25 = m_k*m_pi*m_uf*t3*t9*t22;
        const double t26 = m_k*m_pi*m_vf*t3*t6*t23;
        const double t27 = t24+t25+t26;
        const double t28 = m_pi*m_w0*t6*t9*t16*t21;
        const double t29 = m_k*m_pi*m_uf*t3*t9*t16*t22;
        const double t30 = m_k*m_pi*m_vf*t3*t6*t16*t23;
        const double t31 = t28+t29+t30;
        const double t32 = 1.0/(t17*t17);
        drhodt_mms(i,j,k) = 1.0/pow(t19*(t15*t18-1.0)-t10*
                            t15*t18,2.0)*(-t19*(t18*t27+t15*t31*t32)+t10*t18*t27+t10*t15*t31*t32);

    });

    Uintah::parallel_for( range, [&](int i, int j, int k){
        const double t2 = m_pi*m_w0*time_d;
        const double t3 = cos(t2);
        const double t12 = m_uf*time_d;
        const double t4 = -t12+x(i,j,k);
        const double t5 = m_k*m_pi*t4;
        const double t6 = sin(t5);
        const double t13 = m_vf*time_d;
        const double t7 = -t13+y(i,j,k);
        const double t8 = m_k*m_pi*t7;
        const double t9 = sin(t8);
        const double t10 = 1.0/m_rho1;
        const double t11 = m_rho0*t10;
        const double t14 = t3*t6*t9;
        const double t15 = t14+1.0;
        const double t16 = t11-1.0;
        const double t20 = t3*t6*t9*t16;
        const double t17 = t11-t20+1.0;
        const double t18 = 1.0/t17;
        const double t19 = 1.0/m_rho0;
        const double t21 = sin(t2);
        const double t22 = cos(t5);
        const double t23 = cos(t8);
        const double t24 = m_pi*m_w0*t6*t9*t21;
        const double t25 = m_k*m_pi*m_uf*t3*t9*t22;
        const double t26 = m_k*m_pi*m_vf*t3*t6*t23;
        const double t27 = t24+t25+t26;
        const double t28 = m_pi*m_w0*t6*t9*t16*t21;
        const double t29 = m_k*m_pi*m_uf*t3*t9*t16*t22;
        const double t30 = m_k*m_pi*m_vf*t3*t6*t16*t23;
        const double t31 = t28+t29+t30;
        const double t32 = 1.0/(t17*t17);
        const double t33 = t15*t18;
        const double t34 = t33-1.0;
        const double t35 = t19*t34;
        const double t38 = t10*t15*t18;
        const double t36 = t35-t38;
        const double t37 = m_rho0-m_rho1;
        const double t39 = m_pi*m_w0*t6*t9*t21*t36*t37*(1.0/4.0);
        const double t40 = 1.0/m_k;
        const double t41 = 1.0/t36;
        const double t42 = 1.0/(t36*t36);
        const double t43 = m_k*m_pi*t3*t9*t18*t22;
        const double t44 = m_k*m_pi*t3*t9*t15*t16*t22*t32;
        const double t45 = t43+t44;
        const double t46 = m_k*m_pi*t3*t9*t10*t18*t22;
        const double t47 = m_k*m_pi*t3*t9*t10*t15*t16*t22*t32;
        const double t48 = t46+t47-t19*t45;
        const double t49 = m_k*m_pi*t3*t6*t18*t23;
        const double t50 = m_k*m_pi*t3*t6*t15*t16*t23*t32;
        const double t51 = t49+t50;
        const double t52 = m_k*m_pi*t3*t6*t10*t18*t23;
        const double t53 = m_k*m_pi*t3*t6*t10*t15*t16*t23*t32;
        const double t54 = t52+t53-t19*t51;
        continuity_mms(i,j,k) = -t41*(t39+m_w0*t9*t21*t22*t37*t40*
                                t48*(1.0/4.0))-t41*(t39+m_w0*t6*t21*
                                t23*t37*t40*t54*(1.0/4.0))+t42*(-t19*
                                (t18*t27+t15*t31*t32)+t10*t18*t27+t10*
                                t15*t31*t32)-t42*t48*(m_uf-m_w0*t9*t21*
                                t22*t36*t37*t40*(1.0/4.0))-t42*t54*
                                (m_vf-m_w0*t6*t21*t23*t36*t37*t40*(1.0/4.0));

    });

  }
} } //Uintah::MMS_ShunnP3

#endif
