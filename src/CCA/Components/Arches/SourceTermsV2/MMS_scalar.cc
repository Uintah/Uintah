#include <CCA/Components/Arches/SourceTermsV2/MMS_scalar.h>


namespace Uintah{

//--------------------------------------------------------------------------------------------------
MMS_scalar::MMS_scalar( std::string task_name, int matl_index, MaterialManagerP materialManager ) :
TaskInterface( task_name, matl_index ) , _materialManager(materialManager)
{}

//--------------------------------------------------------------------------------------------------
MMS_scalar::~MMS_scalar()
{}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace MMS_scalar::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace MMS_scalar::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &MMS_scalar::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &MMS_scalar::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &MMS_scalar::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace MMS_scalar::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &MMS_scalar::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &MMS_scalar::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &MMS_scalar::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace MMS_scalar::loadTaskTimestepInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace MMS_scalar::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void
MMS_scalar::problemSetup( ProblemSpecP& db ){

  std::string wave_type;

  db->findBlock("wave")->getAttribute("type",wave_type);
  db->findBlock("wave")->findBlock("independent_variable")->getAttribute("label",ind_var_name);

  if ( wave_type == "sine"){

    ProblemSpecP db_sine = db->findBlock("wave")->findBlock("sine");

    _wtype = SINE;
    db_sine->getAttribute("A",A);
    db_sine->getAttribute("f",F);
    db_sine->getAttribute("offset",offset);

  }else if ( wave_type == "t1"){

    ProblemSpecP db_sine = db->findBlock("wave")->findBlock("t1");
    _wtype = T1;
    db_sine->getAttribute("f",F);

  }else if ( wave_type == "t3"){

    ProblemSpecP db_sine = db->findBlock("wave")->findBlock("t3");
    _wtype = T3;
    db_sine->getAttribute("f",F);

  }else if ( wave_type == "t2"){

    ProblemSpecP db_sine = db->findBlock("wave")->findBlock("t2");
    _wtype = T2;

  }else if ( wave_type == "sine_t"){

    ProblemSpecP db_sine = db->findBlock("wave")->findBlock("sine_t");

    _wtype = SINE_T;
    db_sine->getAttribute("A",A);
    db_sine->getAttribute("f",F);
    db_sine->getAttribute("offset",offset);

  } else if ( wave_type == "gcosine"){

    ProblemSpecP db_square= db->findBlock("wave")->findBlock("gcosine");

    _wtype = GCOSINE;
    db_square->getAttribute("sigma",sigma);

  } else {

    throw InvalidValue("Error: Wave type not recognized.",__FILE__,__LINE__);

  }

  m_MMS_label             = m_task_name;
  m_MMS_source_label      = m_task_name + "_source";
  m_MMS_source_diff_label = m_task_name + "_source_diff";
  m_MMS_source_t_label    = m_task_name + "_source_time";

}

//--------------------------------------------------------------------------------------------------
void
MMS_scalar::create_local_labels(){

  register_new_variable<CCVariable<double> >( m_MMS_label);
  register_new_variable<CCVariable<double> >( m_MMS_source_label);
  register_new_variable<CCVariable<double> >( m_MMS_source_diff_label);
  register_new_variable<CCVariable<double> >( m_MMS_source_t_label);

}

//--------------------------------------------------------------------------------------------------
void
MMS_scalar::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>&
                                variable_registry, const bool packed_tasks ){

  register_variable( m_MMS_label,             ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_MMS_source_label,      ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_MMS_source_diff_label, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_MMS_source_t_label,    ArchesFieldContainer::COMPUTES, variable_registry );

  register_variable( ind_var_name, ArchesFieldContainer::REQUIRES, 0 ,ArchesFieldContainer::NEWDW,
                    variable_registry );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void MMS_scalar::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  CCVariable<double>& f_mms = tsk_info->get_field<CCVariable<double> >(m_MMS_label);
  CCVariable<double>& s_mms = tsk_info->get_field<CCVariable<double> >(m_MMS_source_label);
  CCVariable<double>& s_diff_mms = tsk_info->get_field<CCVariable<double> >(m_MMS_source_diff_label);
  CCVariable<double>& s_t_mms = tsk_info->get_field<CCVariable<double> >(m_MMS_source_t_label);
  constCCVariable<double>& X = tsk_info->get_field<constCCVariable<double > >( ind_var_name );

  double time_d      = tsk_info->get_time(); //_materialManager->getElapsedSimTime();
  //  Vector Dx = patch->dCell();

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  switch (_wtype){
    case SINE:
    Uintah::parallel_for( range, [&](int i, int j, int k){
      MMS_SINE(f_mms(i,j,k), s_mms(i,j,k), s_diff_mms(i,j,k), X(i,j,k), time_d );
    });
    break;
    case T1:
    Uintah::parallel_for( range, [&](int i, int j, int k){
      MMS_T1(f_mms(i,j,k), s_t_mms(i,j,k), time_d );
    });
    break;
    case T2:
    Uintah::parallel_for( range, [&](int i, int j, int k){
      MMS_T2(f_mms(i,j,k), s_t_mms(i,j,k), time_d );
    });
    break;
    case T3:
    Uintah::parallel_for( range, [&](int i, int j, int k){
      double f = 0.0;
      double s_t = 0.0;
      MMS_T1(f, s_t, time_d );
      MMS_T2(f_mms(i,j,k), s_t_mms(i,j,k), time_d );
      f_mms(i,j,k)   += f;
      s_t_mms(i,j,k) += s_t;
    });
    break;
    case SINE_T:
    Uintah::parallel_for( range, [&](int i, int j, int k){
      MMS_SINE_T(f_mms(i,j,k), s_mms(i,j,k), s_diff_mms(i,j,k), s_t_mms(i,j,k), X(i,j,k), time_d );
    });
    break;
    case GCOSINE:
    Uintah::parallel_for( range, [&](int i, int j, int k){
      MMS_GCOSINE(f_mms(i,j,k), s_mms(i,j,k), X(i,j,k), time_d );
    });
    break;
    default:
    break;
  }
}

//--------------------------------------------------------------------------------------------------
void
MMS_scalar::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>&
                                   variable_registry, const int time_substep , const bool packed_tasks){

  register_variable( m_MMS_label,             ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  register_variable( m_MMS_source_label,      ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  register_variable( m_MMS_source_diff_label, ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );
  register_variable( m_MMS_source_t_label,    ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep );

  register_variable(ind_var_name,     ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::LATEST , variable_registry , time_substep );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void MMS_scalar::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  CCVariable<double>& f_mms = tsk_info->get_field<CCVariable<double> >(m_MMS_label);
  CCVariable<double>& s_mms = tsk_info->get_field<CCVariable<double> >(m_MMS_source_label);
  CCVariable<double>& s_diff_mms = tsk_info->get_field<CCVariable<double> >(m_MMS_source_diff_label);
  CCVariable<double>& s_t_mms = tsk_info->get_field<CCVariable<double> >(m_MMS_source_t_label);

  f_mms.initialize(0.0);
  s_mms.initialize(0.0);
  s_diff_mms.initialize(0.0);
  s_t_mms.initialize(0.0);

  constCCVariable<double>& X = tsk_info->get_field<constCCVariable<double > >( ind_var_name );

  double time_d      = tsk_info->get_time(); //_materialManager->getElapsedSimTime();
  int   time_substep = tsk_info->get_time_substep();
  double factor      = tsk_info->get_ssp_time_factor(time_substep);
  double dt          = tsk_info->get_dt();

  //  std::cout << "dt "         << dt   << std::endl; // OD
  //  std::cout << "Time  "         << time_d   << std::endl; // OD
  //  std::cout << "factor  "       << factor   << std::endl; // OD
  //  std::cout << "time_substep  " << time_substep   << std::endl; // OD
  time_d = time_d + factor*dt;
  //  Vector Dx = patch->dCell();

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  switch (_wtype){
    case SINE:
    Uintah::parallel_for( range, [&](int i, int j, int k){
      MMS_SINE(f_mms(i,j,k), s_mms(i,j,k), s_diff_mms(i,j,k),X(i,j,k), time_d );
    });
    break;
    case T1:
    Uintah::parallel_for( range, [&](int i, int j, int k){
      MMS_T1(f_mms(i,j,k), s_t_mms(i,j,k), time_d );
      // it is only for f_mms, because we need f_mms(t+dt),
      //without this we are going to get f_mms(t+0.5*dt)  **/
      if (time_substep==2){
        double s_t_dummy  = 0.0;
        double current_time = time_d + factor*dt;
        MMS_T1(f_mms(i,j,k), s_t_dummy, current_time );
      }
    });
    break;
    case T2:
    Uintah::parallel_for( range, [&](int i, int j, int k){

      MMS_T2(f_mms(i,j,k), s_t_mms(i,j,k), time_d );
      //it is only for f_mms, because we need f_mms(t+dt),
      //without this we are going to get f_mms(t+0.5*dt)  **/
      if (time_substep==2){
        double s_t_dummy  = 0.0;
        double current_time = time_d + factor*dt;
        MMS_T2(f_mms(i,j,k), s_t_dummy, current_time );
      }
    });
    break;
    case T3:
    Uintah::parallel_for( range, [&](int i, int j, int k){

      double f_m   = 0.0;
      double s_tm  = 0.0;
      MMS_T1(f_m, s_tm, time_d );
      MMS_T2(f_mms(i,j,k), s_t_mms(i,j,k), time_d );
      f_mms(i,j,k)   += f_m;
      s_t_mms(i,j,k) += s_tm;

      // it is only for f_mms, because we need f_mms(t+dt),
      //without this we are going to get f_mms(t+0.5*dt)  **/
      if (time_substep==2){
        double f_m   = 0.0;
        double s_t_dummy  = 0.0;
        double current_time = time_d + factor*dt;
        MMS_T1(f_m, s_t_dummy, current_time );
        MMS_T2(f_mms(i,j,k), s_t_dummy, current_time );
        f_mms(i,j,k)   += f_m;
      }


    });
    break;
    case SINE_T:
      Uintah::parallel_for( range, [&](int i, int j, int k){
        MMS_SINE_T(f_mms(i,j,k), s_mms(i,j,k), s_diff_mms(i,j,k), s_t_mms(i,j,k), X(i,j,k), time_d );
      });
      break;
    case GCOSINE: // I need to add diff source term in gcosine OD
      Uintah::parallel_for( range, [&](int i, int j, int k){
        MMS_GCOSINE(f_mms(i,j,k), s_mms(i,j,k), X(i,j,k), time_d );
      });
      break;
    default:
      break;
    }

}

//--------------------------------------------------------------------------------------------------
void
MMS_scalar::MMS_SINE( double& f_mms, double& s_mms, double& s_diff_mms,const double& x , double t)
{
  f_mms = A*sin(F*2.0*pi*x) + offset ; //MMS
  s_mms = 2.0*A*F*pi*cos(2.0*pi*F*x); // Source for conv assuming ux = 1
  s_diff_mms = 4.0*A*F*F*pi*pi*sin(2.0*pi*F*x); // Source for diff asuming D = 1
}

//--------------------------------------------------------------------------------------------------
void
MMS_scalar::MMS_T1( double& f_mms, double& s_t_mms, double t)
{
  f_mms      = -1.*cos(2.0*F*pi*t)/(2.0*F*pi) + 1./(2.*F*pi) ; //MMS
  s_t_mms    = sin(2.*F*pi*t); // time source term
}

//--------------------------------------------------------------------------------------------------
void
MMS_scalar::MMS_T2( double& f_mms, double& s_t_mms, double t)
{
  f_mms      = t*t*t/3.0 ; //MMS
  s_t_mms    = t*t; // time source term
}

//--------------------------------------------------------------------------------------------------
void
MMS_scalar::MMS_SINE_T( double& f_mms, double& s_mms, double& s_diff_mms, double& s_t_mms, const double& x , double t)
{
  double u_x = 1.0;
  f_mms      = (A*sin(F*2.0*pi*x) + offset)*(t+1.0) ; //MMS
  s_mms      = 2.0*A*F*pi*u_x*cos(2.0*pi*F*x)*(t + 1.0); // Source for conv assuming ux = 1
  s_diff_mms = 4.0*A*F*F*pi*pi*sin(2.0*pi*F*x)*(t + 1.0);// Source for diff asuming D = 1
  s_t_mms    = offset + A*sin(2.0*pi*F*x); // time source term
}

//--------------------------------------------------------------------------------------------------
void
MMS_scalar::MMS_GCOSINE( double& f_mms, double& s_mms, const double& x , double t)
{
  double u_x = 1.0;
  f_mms = cos(pi*(x-0.5)/sigma)*exp(-(x-0.5)*(x-0.5)/(2.0*sigma*sigma))  ;
  s_mms = - (u_x*exp(-(x - 0.5)*(x-0.5)/(2.0*sigma*sigma))*cos((pi*(x - 0.5))/sigma)*(2.0*x - 1.0))/(2.0*sigma*sigma) - (pi*u_x*exp(-(x - 0.5)*(x-0.5)/(2*sigma*sigma))*sin((pi*(x - 0.5))/sigma))/sigma;

}

} //namespace Uintah
