#ifndef Uintah_Component_Arches_KScalarRHS_h
#define Uintah_Component_Arches_KScalarRHS_h

/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Transport/TransportHelper.h>
#include <CCA/Components/Arches/GridTools.h>
#include <CCA/Components/Arches/ConvectionHelper.h>
#include <CCA/Components/Arches/Directives.h>
#include <CCA/Components/Arches/BoundaryConditions/BoundaryFunctors.h>
#include <CCA/Components/Arches/UPSHelper.h>
#include <Core/Util/Timers/Timers.hpp>

#define  IMAX_SIZE 10 

namespace Uintah{

  template<typename T, typename PT>
  class KScalarRHS : public TaskInterface {

public:

    KScalarRHS<T, PT>( std::string task_name, int matl_index );
    ~KScalarRHS<T, PT>();

    typedef std::vector<ArchesFieldContainer::VariableInformation> ArchesVIVector;

    TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

    TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

    TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

    TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

    TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( ArchesVIVector& variable_registry , const bool pack_tasks);

    void register_timestep_init( ArchesVIVector& variable_registry , const bool packed_tasks){}

    void register_timestep_eval( ArchesVIVector& variable_registry,
                                 const int time_substep , const bool packed_tasks);

    void register_compute_bcs( ArchesVIVector& variable_registry,
                               const int time_substep , const bool packed_tasks);

    template <typename ExecSpace, typename MemSpace>
    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    template <typename ExecSpace, typename MemSpace>
    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    void create_local_labels();

    //Build instructions for this (KScalarRHS) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, ProblemSpecP& db, int matl_index )
      : m_task_name(task_name), m_db(db), m_matl_index(matl_index){}
      ~Builder(){}

      KScalarRHS* build()
      { return scinew KScalarRHS<T, PT>( m_task_name, m_matl_index ); }

      private:

      std::string m_task_name;
      ProblemSpecP& m_db;
      int m_matl_index;

    };

private:

    typedef typename ArchesCore::VariableHelper<T>::ConstType CT;
    typedef typename ArchesCore::VariableHelper<T>::XFaceType FXT;
    typedef typename ArchesCore::VariableHelper<T>::YFaceType FYT;
    typedef typename ArchesCore::VariableHelper<T>::ZFaceType FZT;
    typedef typename ArchesCore::VariableHelper<CT>::XFaceType CFXT;
    typedef typename ArchesCore::VariableHelper<CT>::YFaceType CFYT;
    typedef typename ArchesCore::VariableHelper<CT>::ZFaceType CFZT;

    typedef typename ArchesCore::VariableHelper<PT>::XFaceType FluxXT;
    typedef typename ArchesCore::VariableHelper<PT>::YFaceType FluxYT;
    typedef typename ArchesCore::VariableHelper<PT>::ZFaceType FluxZT;

    std::string m_D_name;
    std::string m_x_velocity_name;
    std::string m_y_velocity_name;
    std::string m_z_velocity_name;
    std::string m_eps_name;

    ArchesCore::EQUATION_CLASS m_eqn_class;
    std::vector<std::string> m_transported_eqn_names;
    std::vector<std::string> m_eqn_names_BC;

    std::vector<std::string> m_eqn_names;
    std::vector<bool> m_do_diff;
    std::vector<bool> m_do_clip;
    std::vector<double> m_low_clip;
    std::vector<double> m_high_clip;
    std::vector<double> m_init_value;

    bool m_has_D;
    bool m_has_unweighted_rep;             ///< Has an unweighted version of itself (e.g., phi = rho*phi/rho)

    int m_total_eqns;
    int m_boundary_int{0};
    int m_dir{0};

    struct SourceInfo{
      std::string name;
      double weight;
    };

    std::vector<std::vector<SourceInfo> > m_source_info;
    struct Scaling_info {
      std::string unscaled_var; // unscaled value
      double constant; //
    };
    std::map<std::string, Scaling_info> m_scaling_info;
    //std::map<std::string, double> m_scaling_info;

    std::vector<LIMITER> m_conv_scheme;

    ArchesCore::BCFunctors<T>* m_boundary_functors;
    //std::string m_volFraction_name{"volFraction"};

enum cartSpace {x_direc,y_direc,z_direc};

template <typename ExecSpace, typename MemSpace, unsigned int Cscheme>
inline
typename std::enable_if<std::is_same<MemSpace, UintahSpaces::HostSpace>::value, void>::type
doConvection(       ExecutionObject<ExecSpace, MemSpace> & execObj
            ,       Uintah::BlockRange                   & range_conv
            , const Array3<double>                       & phi
            , const Array3<double>                       & rho_phi
            , const Array3<double>                       & xyzVel
            ,       Array3<double>                       & xyzFlux
            , const Array3<double>                       & eps
            ,       unsigned int                           xyzDir
            ,       int                                  & ieqn
            )
{
  if ( m_transported_eqn_names[ieqn] != m_eqn_names[ieqn] ) {
    Uintah::ComputeConvectiveFlux1D<ExecSpace, MemSpace, Array3<double>, const Array3<double>, Cscheme> partiallySpecializedTemplatedStruct;
    partiallySpecializedTemplatedStruct.get_flux( execObj, range_conv, rho_phi, xyzVel, xyzFlux, eps, xyzDir );
  }
  else {
    Uintah::ComputeConvectiveFlux1D<ExecSpace, MemSpace, Array3<double>, const Array3<double>, Cscheme> partiallySpecializedTemplatedStruct;
    partiallySpecializedTemplatedStruct.get_flux( execObj, range_conv, phi,xyzVel, xyzFlux, eps, xyzDir );
    }
}

#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
template <typename ExecSpace, typename MemSpace, unsigned int Cscheme>
inline
typename std::enable_if<std::is_same<MemSpace, Kokkos::HostSpace>::value, void>::type
doConvection( ExecutionObject<ExecSpace, MemSpace> & execObj
            , Uintah::BlockRange                   & range_conv
            , KokkosView3<const double, MemSpace>    phi
            , KokkosView3<const double, MemSpace>    rho_phi
            , KokkosView3<const double, MemSpace>    xyzVel
            , KokkosView3<      double, MemSpace>    xyzFlux
            , KokkosView3<const double, MemSpace>    eps
            , unsigned int                           xyzDir
            , int                                  & ieqn
            )
{
  if ( m_transported_eqn_names[ieqn] != m_eqn_names[ieqn] ) {
    Uintah::ComputeConvectiveFlux1D<ExecSpace, MemSpace, KokkosView3<double, MemSpace>, KokkosView3<const double, MemSpace>, Cscheme> partiallySpecializedTemplatedStruct;
    partiallySpecializedTemplatedStruct.get_flux( execObj, range_conv, rho_phi, xyzVel, xyzFlux, eps, xyzDir );
  }
  else {
    Uintah::ComputeConvectiveFlux1D<ExecSpace, MemSpace, KokkosView3<double, MemSpace>, KokkosView3<const double, MemSpace>, Cscheme> partiallySpecializedTemplatedStruct;
    partiallySpecializedTemplatedStruct.get_flux( execObj, range_conv, phi, xyzVel, xyzFlux, eps, xyzDir );
  }
}
#endif

#if defined( HAVE_CUDA ) && defined( KOKKOS_ENABLE_CUDA )
template <typename ExecSpace, typename MemSpace, unsigned int Cscheme>
inline
typename std::enable_if<std::is_same<MemSpace, Kokkos::CudaSpace>::value, void>::type
doConvection( ExecutionObject<ExecSpace, MemSpace> & execObj
            , Uintah::BlockRange                   & range_conv
            , KokkosView3<const double, MemSpace>    phi
            , KokkosView3<const double, MemSpace>    rho_phi
            , KokkosView3<const double, MemSpace>    xyzVel
            , KokkosView3<      double, MemSpace>    xyzFlux
            , KokkosView3<const double, MemSpace>    eps
            , unsigned int                           xyzDir
            , int                                  & ieqn
            )
{
  if ( m_transported_eqn_names[ieqn] != m_eqn_names[ieqn] ) {
    Uintah::ComputeConvectiveFlux1D<ExecSpace, MemSpace, KokkosView3<double, MemSpace>, KokkosView3<const double, MemSpace>, Cscheme> partiallySpecializedTemplatedStruct;
    partiallySpecializedTemplatedStruct.get_flux( execObj, range_conv, rho_phi, xyzVel, xyzFlux, eps, xyzDir );
  }
  else {
    Uintah::ComputeConvectiveFlux1D<ExecSpace, MemSpace, KokkosView3<double, MemSpace>, KokkosView3<const double, MemSpace>, Cscheme> partiallySpecializedTemplatedStruct;
    partiallySpecializedTemplatedStruct.get_flux( execObj, range_conv, phi, xyzVel, xyzFlux, eps, xyzDir );
  }
}
#endif

  };

  //------------------------------------------------------------------------------------------------
  template <typename T, typename PT>
  KScalarRHS<T, PT>::KScalarRHS( std::string task_name, int matl_index ) :
  TaskInterface( task_name, matl_index ) {

    m_boundary_functors = scinew ArchesCore::BCFunctors<T>();

    ArchesCore::VariableHelper<T>* helper = scinew ArchesCore::VariableHelper<T>;
    if ( helper->dir != ArchesCore::NODIR ) m_boundary_int = 1;
    m_dir = helper->dir;
    delete helper;

  }

  //------------------------------------------------------------------------------------------------
  template <typename T, typename PT>
  KScalarRHS<T, PT>::~KScalarRHS(){

    delete m_boundary_functors;

  }

  //--------------------------------------------------------------------------------------------------
  template <typename T, typename PT>
  TaskAssignedExecutionSpace KScalarRHS<T, PT>::loadTaskComputeBCsFunctionPointers()
  {
    return create_portable_arches_tasks<TaskInterface::BC>( this
                                       , &KScalarRHS<T, PT>::compute_bcs<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                       , &KScalarRHS<T, PT>::compute_bcs<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                       , &KScalarRHS<T, PT>::compute_bcs<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                       );
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T, typename PT>
  TaskAssignedExecutionSpace KScalarRHS<T, PT>::loadTaskInitializeFunctionPointers()
  {
    return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                       , &KScalarRHS<T, PT>::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                       , &KScalarRHS<T, PT>::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                       , &KScalarRHS<T, PT>::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                       );
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T, typename PT>
  TaskAssignedExecutionSpace KScalarRHS<T, PT>::loadTaskEvalFunctionPointers()
  {
    return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                       , &KScalarRHS<T, PT>::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                       , &KScalarRHS<T, PT>::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                       , &KScalarRHS<T, PT>::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                       );
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T, typename PT>
  TaskAssignedExecutionSpace KScalarRHS<T, PT>::loadTaskTimestepInitFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T, typename PT>
  TaskAssignedExecutionSpace KScalarRHS<T, PT>::loadTaskRestartInitFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //------------------------------------------------------------------------------------------------
  template <typename T, typename PT> void
  KScalarRHS<T, PT>::problemSetup( ProblemSpecP& input_db ){

    using namespace ArchesCore;

    std::string eqn_grp_name = strip_class_name();

    m_has_unweighted_rep = false;
    if ( input_db->findBlock("weight_factor") != nullptr ){
      m_has_unweighted_rep = true;
    }

    m_total_eqns = 0;

    ConvectionHelper* conv_helper = scinew ConvectionHelper();

    std::string eqn_class = "density_weighted";
    if ( input_db->findAttribute("class") ){
      input_db->getAttribute("class", eqn_class);
    }

    m_eqn_class = assign_eqn_class_enum( eqn_class );
    std::string premultiplier_name = get_premultiplier_name(m_eqn_class);
    std::string postmultiplier_name = get_postmultiplier_name(m_eqn_class);

    std::string env_number="NA";
    if (m_eqn_class == DQMOM) {
      input_db->findBlock("env_number")->getAttribute("number", env_number);
    }

    for( ProblemSpecP db = input_db->findBlock("eqn"); db != nullptr; db = db->findNextBlock("eqn") ) {

      //Equation name
      std::string eqn_name;
      db->getAttribute("label", eqn_name);
      m_eqn_names.push_back(eqn_name);

      std::string rho_phi_name;
      if ((m_eqn_class == DQMOM) && ( input_db->findBlock("no_weight_factor") == nullptr )) {
        m_has_unweighted_rep = false;
        std::string delimiter = env_number ;
        std::string name_1    = eqn_name.substr(0, eqn_name.find(delimiter));
        rho_phi_name = name_1 + postmultiplier_name + env_number;
      } else {
        rho_phi_name = premultiplier_name + eqn_name;
      }

      if ( db->findBlock("no_weight_factor") != nullptr
           && m_eqn_class == DQMOM ){

        rho_phi_name = eqn_name;//"NA";// for weights in DQMOM
        //Scaling Constant only for weight
        if ( db->findBlock("scaling") ){

          double scaling_constant;
          db->findBlock("scaling")->getAttribute("value", scaling_constant);

          Scaling_info scaling_w ;
          scaling_w.unscaled_var = "w_" + env_number ;
          scaling_w.constant    = scaling_constant;
          m_scaling_info.insert(std::make_pair(eqn_name, scaling_w));

        }
      }

      m_transported_eqn_names.push_back(rho_phi_name);

      // Convection
      if ( db->findBlock("convection")){
        std::string conv_scheme;
        db->findBlock("convection")->getAttribute("scheme", conv_scheme);
        m_conv_scheme.push_back(conv_helper->get_limiter_from_string(conv_scheme));
      } else {
        m_conv_scheme.push_back(NOCONV);
      }

      // Diffusion
      m_has_D = false;
      if ( db->findBlock("diffusion")){
        m_do_diff.push_back(true);
        m_has_D = true;
      } else {
        m_do_diff.push_back(false);
      }

      // Clipping
      if ( db->findBlock("clip")){
        m_do_clip.push_back(true);
        double low; double high;
        db->findBlock("clip")->getAttribute("low", low);
        db->findBlock("clip")->getAttribute("high", high);
        m_low_clip.push_back(low);
        m_high_clip.push_back(high);
      } else {
        m_do_clip.push_back(false);
        m_low_clip.push_back(-999.9);
        m_high_clip.push_back(999.9);
      }

      // Initial Value
      if ( db->findBlock("initialize") ){
        double value;
        db->findBlock("initialize")->getAttribute("value",value);
        m_init_value.push_back(value);
      }
      else {
        m_init_value.push_back(0.0);
      }

      // Source Terms
      std::vector<SourceInfo> eqn_srcs;
      for ( ProblemSpecP src_db = db->findBlock("src"); src_db != nullptr; src_db = src_db->findNextBlock("src") ) {

        std::string src_label;
        double weight = 1.0;

        src_db->getAttribute("label",src_label);

        if ( src_db->findBlock("weight")){
          src_db->findBlock("weight")->getAttribute("value",weight);
        }

        SourceInfo info;
        info.name = src_label;
        info.weight = weight;

        eqn_srcs.push_back(info);

      }

      m_source_info.push_back(eqn_srcs);

    }

    // Setup the boundary conditions for this eqn set
    if (m_eqn_class == DQMOM) {
      for ( auto i = m_transported_eqn_names.begin(); i != m_transported_eqn_names.end(); i++ ){
        m_eqn_names_BC.push_back(*i);
      }
    } else {
      for ( auto i = m_eqn_names.begin(); i != m_eqn_names.end(); i++ ){
        m_eqn_names_BC.push_back(*i);
      }
    }

    m_boundary_functors->create_bcs( input_db, m_eqn_names_BC );

    GridVarMap<T> var_map;
    var_map.problemSetup( input_db );
    m_eps_name = var_map.vol_frac_name;
    m_x_velocity_name = var_map.uvel_name;
    m_y_velocity_name = var_map.vvel_name;
    m_z_velocity_name = var_map.wvel_name;

    if ( input_db->findBlock("velocity") ){
      // can overide the global velocity space with this:
      input_db->findBlock("velocity")->getAttribute("xlabel",m_x_velocity_name);
      input_db->findBlock("velocity")->getAttribute("ylabel",m_y_velocity_name);
      input_db->findBlock("velocity")->getAttribute("zlabel",m_z_velocity_name);
    }

    // Diffusion coeff -- assuming the same one across all eqns.
    m_D_name = "NA";

    if ( m_has_D ){
      if ( input_db->findBlock("diffusion_coef") ) {
        input_db->findBlock("diffusion_coef")->getAttribute("label",m_D_name);
      } else {
        std::stringstream msg;
        msg << "Error: Diffusion specified for task " << m_task_name << std::endl
        << "but no diffusion coefficient label specified." << std::endl;
        throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
      }
    }

    delete conv_helper;

  } // End problemSetup

  //------------------------------------------------------------------------------------------------
  template <typename T, typename PT>
  void
  KScalarRHS<T, PT>::create_local_labels(){

    const int istart = 0;
    int iend = m_eqn_names.size();
    for (int i = istart; i < iend; i++ ){
      register_new_variable<T>( m_eqn_names[i] );
      if ( m_transported_eqn_names[i] != m_eqn_names[i] ){
        register_new_variable<T>( m_transported_eqn_names[i] );
      }
      register_new_variable<T>(   m_transported_eqn_names[i]+"_RHS" );
      register_new_variable<FXT>( m_eqn_names[i]+"_x_flux" );
      register_new_variable<FYT>( m_eqn_names[i]+"_y_flux" );
      register_new_variable<FZT>( m_eqn_names[i]+"_z_flux" );
    }
    for ( auto i = m_scaling_info.begin(); i != m_scaling_info.end(); i++ ){
      register_new_variable<T>( (i->second).unscaled_var);
    }
  }

  //------------------------------------------------------------------------------------------------
  template <typename T, typename PT> void
  KScalarRHS<T, PT>::register_initialize(
    std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
    const bool packed_tasks ){

    const int istart = 0;
    const int iend = m_eqn_names.size();

    for (int ieqn = istart; ieqn < iend; ieqn++ ){
      //phi
      register_variable(  m_eqn_names[ieqn], ArchesFieldContainer::COMPUTES , variable_registry, m_task_name );
      //rho*phi
      register_variable(  m_transported_eqn_names[ieqn], ArchesFieldContainer::COMPUTES , variable_registry, m_task_name );
      //all the other transport terms
      register_variable(  m_transported_eqn_names[ieqn]+"_RHS", ArchesFieldContainer::COMPUTES , variable_registry, m_task_name );
      register_variable(  m_eqn_names[ieqn]+"_x_flux", ArchesFieldContainer::COMPUTES , variable_registry, m_task_name );
      register_variable(  m_eqn_names[ieqn]+"_y_flux", ArchesFieldContainer::COMPUTES , variable_registry, m_task_name );
      register_variable(  m_eqn_names[ieqn]+"_z_flux", ArchesFieldContainer::COMPUTES , variable_registry, m_task_name );
    }

    register_variable( m_eps_name, ArchesFieldContainer::REQUIRES, 1 , ArchesFieldContainer::NEWDW, variable_registry, m_task_name   );

    for ( auto ieqn = m_scaling_info.begin(); ieqn != m_scaling_info.end(); ieqn++ ){
      register_variable((ieqn->second).unscaled_var, ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );
    }
  }

  //------------------------------------------------------------------------------------------------
  template <typename T, typename PT>
  template <typename ExecSpace, typename MemSpace>
  void KScalarRHS<T, PT>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

    auto eps = tsk_info->get_field<CT,const double, MemSpace>(m_eps_name);

    const int istart = 0;
    const int iend = m_eqn_names.size();

    const int imax=1; 

    for (int ieqn = istart; ieqn < iend; ieqn++ ){

      double scalar_init_value = m_init_value[ieqn];

      auto rhs = tsk_info->get_field<T, double, MemSpace>(m_transported_eqn_names[ieqn]+"_RHS");
      auto phi = tsk_info->get_field<T, double, MemSpace>(m_eqn_names[ieqn]);

      auto x_flux = tsk_info->get_field<FXT, double, MemSpace>(m_eqn_names[ieqn]+"_x_flux");
      auto y_flux = tsk_info->get_field<FYT, double, MemSpace>(m_eqn_names[ieqn]+"_y_flux");
      auto z_flux = tsk_info->get_field<FZT, double, MemSpace>(m_eqn_names[ieqn]+"_z_flux");

      auto rho_phi = createContainer<T, double, imax, MemSpace>(m_transported_eqn_names[ieqn] != m_eqn_names[ieqn] ?  1 : 0);

      if ( m_transported_eqn_names[ieqn] != m_eqn_names[ieqn] ) {
         tsk_info->get_unmanaged_uintah_field<T, double, MemSpace>(m_transported_eqn_names[ieqn],rho_phi[0] );
      }

      Uintah::BlockRange range1( patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
      Uintah::parallel_for(execObj, range1, KOKKOS_LAMBDA (int i, int j, int k){
        rhs(i,j,k)=(0.0);
        phi(i,j,k)=(scalar_init_value);
        x_flux(i,j,k)=(0.0);
        y_flux(i,j,k)=(0.0);
        z_flux(i,j,k)=(0.0);
        for( unsigned int iter=0; iter < rho_phi.runTime_size ; iter++){
          rho_phi[iter](i,j,k)=(0.0);
        }
      });
    } //eqn loop

    for ( auto i = m_scaling_info.begin(); i != m_scaling_info.end(); i++ ){
      auto phi_unscaled = tsk_info->get_field<T, double, MemSpace>((i->second).unscaled_var);
      auto phi = tsk_info->get_field<T, double, MemSpace>(i->first);
      const double scalingConstant = i->second.constant;

      Uintah::BlockRange range2( patch->getCellLowIndex(), patch->getCellHighIndex() );

      Uintah::parallel_for(execObj, range2, KOKKOS_LAMBDA (int i, int j, int k){
        phi_unscaled(i,j,k) = phi(i,j,k) * scalingConstant * eps(i,j,k)  ;

      });
    }

  }

  //------------------------------------------------------------------------------------------------
  template <typename T, typename PT> void
  KScalarRHS<T, PT>::
  register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
                          const int time_substep , const bool packed_tasks ){

    const int istart = 0;
    const int iend = m_eqn_names.size();
    for (int ieqn = istart; ieqn < iend; ieqn++ ){

      if ( time_substep == 0 ){
        register_variable( m_transported_eqn_names[ieqn], ArchesFieldContainer::COMPUTES , variable_registry, time_substep, m_task_name );
        register_variable( m_eqn_names[ieqn], ArchesFieldContainer::COMPUTES, variable_registry, time_substep, m_task_name );
      }
      register_variable( m_eqn_names[ieqn], ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::LATEST, variable_registry, time_substep, m_task_name );
      register_variable( m_transported_eqn_names[ieqn], ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::LATEST, variable_registry, time_substep, m_task_name );
      register_variable( m_transported_eqn_names[ieqn]+"_RHS", ArchesFieldContainer::COMPUTES, variable_registry, time_substep, m_task_name );
      register_variable( m_eqn_names[ieqn]+"_x_flux", ArchesFieldContainer::COMPUTES, variable_registry, time_substep, m_task_name );
      register_variable( m_eqn_names[ieqn]+"_y_flux", ArchesFieldContainer::COMPUTES, variable_registry, time_substep, m_task_name );
      register_variable( m_eqn_names[ieqn]+"_z_flux", ArchesFieldContainer::COMPUTES, variable_registry, time_substep, m_task_name );
      if ( m_do_diff[ieqn] ){
        register_variable( m_eqn_names[ieqn]+"_x_dflux", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name );
        register_variable( m_eqn_names[ieqn]+"_y_dflux", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name );
        register_variable( m_eqn_names[ieqn]+"_z_dflux", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name );
      }

      typedef std::vector<SourceInfo> VS;
      for (typename VS::iterator i = m_source_info[ieqn].begin(); i != m_source_info[ieqn].end(); i++){
        register_variable( i->name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name );
      }

    }

    //globally common variables
    register_variable( m_x_velocity_name, ArchesFieldContainer::REQUIRES, 1 , ArchesFieldContainer::LATEST, variable_registry, time_substep, m_task_name );
    register_variable( m_y_velocity_name, ArchesFieldContainer::REQUIRES, 1 , ArchesFieldContainer::LATEST, variable_registry, time_substep, m_task_name );
    register_variable( m_z_velocity_name, ArchesFieldContainer::REQUIRES, 1 , ArchesFieldContainer::LATEST, variable_registry, time_substep, m_task_name );
    register_variable( m_eps_name, ArchesFieldContainer::REQUIRES, 2 , ArchesFieldContainer::OLDDW, variable_registry, time_substep, m_task_name );

  }

  //------------------------------------------------------------------------------------------------
  template <typename T, typename PT>
  template <typename ExecSpace, typename MemSpace>
  void KScalarRHS<T, PT>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

    Vector Dx = patch->dCell();
    double ax = Dx.y() * Dx.z();
    double ay = Dx.z() * Dx.x();
    double az = Dx.x() * Dx.y();
    const double V = Dx.x()*Dx.y()*Dx.z();

    auto uVel = tsk_info->get_field<CFXT,const double, MemSpace>(m_x_velocity_name);
    auto vVel = tsk_info->get_field<CFYT,const double, MemSpace>(m_y_velocity_name);
    auto wVel = tsk_info->get_field<CFZT,const double, MemSpace>(m_z_velocity_name);
    auto eps  = tsk_info->get_field<CT, const double, MemSpace>(m_eps_name);

    const int istart = 0;
    const int iend = m_eqn_names.size();
    const int time_substep = tsk_info->get_time_substep();

    for (int ieqn = istart; ieqn < iend; ieqn++ ){

      //Copy forward old values
      // Because old values have ghosts we have to explicitly copy values over.
      // Additionally, we want the carry forward to occur in the right place.
      if ( time_substep == 0 ){

        IntVector low  = patch->getExtraCellLowIndex();
        IntVector high = patch->getExtraCellHighIndex();
        Uintah::BlockRange range( low, high);
        if ( m_transported_eqn_names[ieqn] != m_eqn_names[ieqn]){
          auto rho_phi = tsk_info->get_field<T, double, MemSpace>(m_transported_eqn_names[ieqn]);
          auto old_rho_phi = tsk_info->get_field<CT, const double, MemSpace>(m_transported_eqn_names[ieqn]);

          Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k){
            rho_phi(i,j,k) = old_rho_phi(i,j,k);
          });

        }

        auto phi = tsk_info->get_field<T, double, MemSpace>(m_eqn_names[ieqn]);
        auto old_phi = tsk_info->get_field<CT, const double, MemSpace>(m_eqn_names[ieqn]);

        Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k){
          phi(i,j,k) = old_phi(i,j,k);
        });

      }

      auto rho_phi=createConstContainer<CT, const double, 1, MemSpace>(1);
      if ( m_transported_eqn_names[ieqn] != m_eqn_names[ieqn] ){
        rho_phi[0] = tsk_info->get_field<CT, const double, MemSpace>(m_transported_eqn_names[ieqn]);
      }

      auto phi = tsk_info->get_field<CT, const double, MemSpace>(m_eqn_names[ieqn]);
      auto rhs = tsk_info->get_field<T, double, MemSpace>(m_transported_eqn_names[ieqn]+"_RHS");

      Uintah::parallel_initialize( execObj, 0.0, rhs );

      auto x_flux = tsk_info->get_field<FXT, double, MemSpace>(m_eqn_names[ieqn]+"_x_flux");
      auto y_flux = tsk_info->get_field<FYT, double, MemSpace>(m_eqn_names[ieqn]+"_y_flux");
      auto z_flux = tsk_info->get_field<FZT, double, MemSpace>(m_eqn_names[ieqn]+"_z_flux");
      Uintah::parallel_initialize( execObj, 0.0, x_flux, y_flux, z_flux );

      if ( m_conv_scheme[ieqn] != NOCONV ){

        //Convection:
        IntVector low_x  = patch->getCellLowIndex();
        IntVector high_x = patch->getCellHighIndex();
        IntVector low_y  = patch->getCellLowIndex();
        IntVector high_y = patch->getCellHighIndex();
        IntVector low_z  = patch->getCellLowIndex();
        IntVector high_z = patch->getCellHighIndex();

        IntVector lbuffer(0,0,0), hbuffer(0,0,0);
        int  boundary_buffer_x = 0;
        int  boundary_buffer_y = 0;
        int  boundary_buffer_z = 0;

        if ( m_boundary_int > 0 && m_dir == 0 ) boundary_buffer_x = 1;
        if ( m_boundary_int > 0 && m_dir == 1 ) boundary_buffer_y = 1;
        if ( m_boundary_int > 0 && m_dir == 2 ) boundary_buffer_z = 1;

        if ( patch->getBCType(Patch::xminus) != Patch::Neighbor ) {
          low_x[0]  += 1 ;
          // xminus face is computed with central scheme
          IntVector low  = patch->getCellLowIndex();
          IntVector high = patch->getExtraCellHighIndex();
          high[0] = low[0] + 1 ;
          Uintah::BlockRange range( low, high);

          doConvection<ExecSpace, MemSpace, CentralConvection>(execObj,range,phi,rho_phi[0],uVel,x_flux,eps,x_direc,ieqn);
        }

        if ( patch->getBCType(Patch::xplus)  != Patch::Neighbor ) {
           // check this
          IntVector low  = patch->getCellLowIndex();
          IntVector high = patch->getExtraCellHighIndex();
          low[0] = high[0] - boundary_buffer_x -1 ;
          Uintah::BlockRange range( low, high);

          doConvection<ExecSpace, MemSpace, CentralConvection>(execObj,range,phi,rho_phi[0],uVel,x_flux,eps,x_direc,ieqn);

        }

        if ( patch->getBCType(Patch::yminus) != Patch::Neighbor ) {
          low_y[1] += 1 ;
          IntVector low = patch->getCellLowIndex();
          IntVector high = patch->getExtraCellHighIndex();
          high[1] = low[1] + 1 ;
          Uintah::BlockRange range( low, high);

          doConvection<ExecSpace, MemSpace, CentralConvection>(execObj,range,phi,rho_phi[0],vVel,y_flux,eps,y_direc,ieqn);

        }

        if ( patch->getBCType(Patch::yplus)  != Patch::Neighbor ) {
          IntVector low  = patch->getCellLowIndex();
          IntVector high = patch->getExtraCellHighIndex();
          low[1] = high[1] - boundary_buffer_y -1 ;
          Uintah::BlockRange range( low, high);

          doConvection<ExecSpace, MemSpace, CentralConvection>(execObj,range,phi,rho_phi[0],vVel,y_flux,eps,y_direc,ieqn);

        }


        if ( patch->getBCType(Patch::zminus) != Patch::Neighbor ) {
          low_z[2] += 1 ;
          IntVector low = patch->getCellLowIndex();
          IntVector high = patch->getExtraCellHighIndex();
          high[2] = low[2] +  1 ;
          Uintah::BlockRange range( low, high);

          doConvection<ExecSpace, MemSpace, CentralConvection>(execObj,range,phi,rho_phi[0],wVel,z_flux,eps,z_direc,ieqn);

        }

        if ( patch->getBCType(Patch::zplus)  != Patch::Neighbor ) {
          IntVector low  = patch->getCellLowIndex();
          IntVector high = patch->getExtraCellHighIndex();
          low[2] = high[2] - boundary_buffer_z -1 ;
          Uintah::BlockRange range( low, high);

          doConvection<ExecSpace, MemSpace, CentralConvection>(execObj,range,phi,rho_phi[0],wVel,z_flux,eps,z_direc,ieqn);

        }

        Uintah::BlockRange range_cl_to_ech_x( low_x, high_x);
        Uintah::BlockRange range_cl_to_ech_y( low_y, high_y);
        Uintah::BlockRange range_cl_to_ech_z( low_z, high_z);

        if ( m_conv_scheme[ieqn] == UPWIND ){

          doConvection<ExecSpace, MemSpace, UpwindConvection>(execObj,range_cl_to_ech_x,phi,rho_phi[0],uVel,x_flux,eps,x_direc,ieqn);
          doConvection<ExecSpace, MemSpace, UpwindConvection>(execObj,range_cl_to_ech_y,phi,rho_phi[0],vVel,y_flux,eps,y_direc,ieqn);
          doConvection<ExecSpace, MemSpace, UpwindConvection>(execObj,range_cl_to_ech_z,phi,rho_phi[0],wVel,z_flux,eps,z_direc,ieqn);

        } else if ( m_conv_scheme[ieqn] == CENTRAL ){

          doConvection<ExecSpace, MemSpace, CentralConvection>(execObj,range_cl_to_ech_x,phi,rho_phi[0],uVel,x_flux,eps,x_direc,ieqn);
          doConvection<ExecSpace, MemSpace, CentralConvection>(execObj,range_cl_to_ech_y,phi,rho_phi[0],vVel,y_flux,eps,y_direc,ieqn);
          doConvection<ExecSpace, MemSpace, CentralConvection>(execObj,range_cl_to_ech_z,phi,rho_phi[0],wVel,z_flux,eps,z_direc,ieqn);

        } else if ( m_conv_scheme[ieqn] == SUPERBEE ){

          doConvection<ExecSpace, MemSpace, SuperBeeConvection>(execObj,range_cl_to_ech_x,phi,rho_phi[0],uVel,x_flux,eps,x_direc,ieqn);
          doConvection<ExecSpace, MemSpace, SuperBeeConvection>(execObj,range_cl_to_ech_y,phi,rho_phi[0],vVel,y_flux,eps,y_direc,ieqn);
          doConvection<ExecSpace, MemSpace, SuperBeeConvection>(execObj,range_cl_to_ech_z,phi,rho_phi[0],wVel,z_flux,eps,z_direc,ieqn);

        } else if ( m_conv_scheme[ieqn] == VANLEER ){

          doConvection<ExecSpace, MemSpace, VanLeerConvection>(execObj,range_cl_to_ech_x,phi,rho_phi[0],uVel,x_flux,eps,x_direc,ieqn);
          doConvection<ExecSpace, MemSpace, VanLeerConvection>(execObj,range_cl_to_ech_y,phi,rho_phi[0],vVel,y_flux,eps,y_direc,ieqn);
          doConvection<ExecSpace, MemSpace, VanLeerConvection>(execObj,range_cl_to_ech_z,phi,rho_phi[0],wVel,z_flux,eps,z_direc,ieqn);

        } else if ( m_conv_scheme[ieqn] == ROE ){

          doConvection<ExecSpace, MemSpace, RoeConvection>(execObj,range_cl_to_ech_x,phi,rho_phi[0],uVel,x_flux,eps,x_direc,ieqn);
          doConvection<ExecSpace, MemSpace, RoeConvection>(execObj,range_cl_to_ech_y,phi,rho_phi[0],vVel,y_flux,eps,y_direc,ieqn);
          doConvection<ExecSpace, MemSpace, RoeConvection>(execObj,range_cl_to_ech_z,phi,rho_phi[0],wVel,z_flux,eps,z_direc,ieqn);

        } else if ( m_conv_scheme[ieqn] == FOURTH ){

          doConvection<ExecSpace, MemSpace, FourthConvection>(execObj,range_cl_to_ech_x,phi,rho_phi[0],uVel,x_flux,eps,x_direc,ieqn);
          doConvection<ExecSpace, MemSpace, FourthConvection>(execObj,range_cl_to_ech_y,phi,rho_phi[0],vVel,y_flux,eps,y_direc,ieqn);
          doConvection<ExecSpace, MemSpace, FourthConvection>(execObj,range_cl_to_ech_z,phi,rho_phi[0],wVel,z_flux,eps,z_direc,ieqn);

        } else {

          throw InvalidValue("Error: Convection scheme for eqn: "+m_transported_eqn_names[ieqn]+" not valid.", __FILE__, __LINE__);

        }
      }
      //Diffusion:
      if ( m_do_diff[ieqn] ) {

        auto x_dflux = tsk_info->get_field<CFXT, const double, MemSpace>(m_eqn_names[ieqn]+"_x_dflux");
        auto y_dflux = tsk_info->get_field<CFYT, const double, MemSpace>(m_eqn_names[ieqn]+"_y_dflux");
        auto z_dflux = tsk_info->get_field<CFZT, const double, MemSpace>(m_eqn_names[ieqn]+"_z_dflux");

        GET_EXTRACELL_BUFFERED_PATCH_RANGE(0,0);

        Uintah::BlockRange range_diff(low_patch_range, high_patch_range);

        Uintah::parallel_for(execObj, range_diff, KOKKOS_LAMBDA (int i, int j, int k){

          rhs(i,j,k) += ax * ( x_dflux(i+1,j,k) - x_dflux(i,j,k) ) +
                        ay * ( y_dflux(i,j+1,k) - y_dflux(i,j,k) ) +
                        az * ( z_dflux(i,j,k+1) - z_dflux(i,j,k) );

        });
      }

      //Sources:
      typedef std::vector<SourceInfo> VS;
      for (typename VS::iterator isrc = m_source_info[ieqn].begin();
        isrc != m_source_info[ieqn].end(); isrc++){

        auto src = tsk_info->get_field<CT, const double, MemSpace>((*isrc).name);
        const double weight = (*isrc).weight;
        Uintah::BlockRange src_range(patch->getCellLowIndex(), patch->getCellHighIndex());

        Uintah::parallel_for(execObj , src_range, KOKKOS_LAMBDA (int i, int j, int k){

          rhs(i,j,k) += weight * src(i,j,k) * V;

        });
      }
    } // equation loop
  }

//--------------------------------------------------------------------------------------------------
  template <typename T, typename PT> void
  KScalarRHS<T, PT>::register_compute_bcs(
    std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
    const int time_substep , const bool packed_tasks){

    for ( auto i = m_eqn_names_BC.begin(); i != m_eqn_names_BC.end(); i++ ){
      register_variable( *i, ArchesFieldContainer::MODIFIES, variable_registry, m_task_name );
    }

    ArchesCore::FunctorDepList bc_dep;
    m_boundary_functors->get_bc_dependencies( m_eqn_names_BC, m_bcHelper, bc_dep );
    for ( auto i = bc_dep.begin(); i != bc_dep.end(); i++ ){
      register_variable( (*i).variable_name, ArchesFieldContainer::REQUIRES, (*i).n_ghosts , (*i).dw,
                         variable_registry, m_task_name );
    }

    std::vector<std::string> bc_mod;
    m_boundary_functors->get_bc_modifies( m_eqn_names_BC, m_bcHelper, bc_mod );
    for ( auto i = bc_mod.begin(); i != bc_mod.end(); i++ ){
      register_variable( *i, ArchesFieldContainer::MODIFIES, variable_registry, m_task_name );
    }

  }

//--------------------------------------------------------------------------------------------------
  template <typename T, typename PT>
  template <typename ExecSpace, typename MemSpace>
  void KScalarRHS<T, PT>::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){
    m_boundary_functors->apply_bc( m_eqn_names_BC, m_bcHelper, tsk_info, patch ,execObj);
  }
}
#endif
