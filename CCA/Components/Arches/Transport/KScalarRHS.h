#ifndef Uintah_Component_Arches_KScalarRHS_h
#define Uintah_Component_Arches_KScalarRHS_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/DiscretizationTools.h>
#include <CCA/Components/Arches/ConvectionHelper.h>
#include <CCA/Components/Arches/Directives.h>
#include <spatialops/util/TimeLogger.h>

namespace Uintah{

  class Operators;
  template<typename T>
  class KScalarRHS : public TaskInterface {

public:

    KScalarRHS<T>( std::string task_name, int matl_index );
    ~KScalarRHS<T>();

    typedef std::vector<ArchesFieldContainer::VariableInformation> ArchesVIVector;

    void problemSetup( ProblemSpecP& db );

    void register_initialize( ArchesVIVector& variable_registry );

    void register_timestep_init( ArchesVIVector& variable_registry );

    void register_timestep_eval( ArchesVIVector& variable_registry,
                                 const int time_substep );

    void register_compute_bcs( ArchesVIVector& variable_registry,
                               const int time_substep );

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                      SpatialOps::OperatorDatabase& opr );

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                     SpatialOps::OperatorDatabase& opr );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                        SpatialOps::OperatorDatabase& opr );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
               SpatialOps::OperatorDatabase& opr );

    void create_local_labels();

    //Build instructions for this (KScalarRHS) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index )
      : _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      KScalarRHS* build()
      { return scinew KScalarRHS<T>( _task_name, _matl_index ); }

      private:

      std::string _task_name;
      int _matl_index;

    };

private:

    typedef typename VariableHelper<T>::ConstType CT;
    typedef typename VariableHelper<T>::XFaceType FXT;
    typedef typename VariableHelper<T>::YFaceType FYT;
    typedef typename VariableHelper<T>::ZFaceType FZT;
    typedef typename VariableHelper<T>::ConstXFaceType CFXT;
    typedef typename VariableHelper<T>::ConstYFaceType CFYT;
    typedef typename VariableHelper<T>::ConstZFaceType CFZT;

    std::string _rhs_name;
    std::string _D_name;
    std::string _X_flux_name;
    std::string _Y_flux_name;
    std::string _Z_flux_name;
    std::string _X_psi_name;
    std::string _Y_psi_name;
    std::string _Z_psi_name;
    std::string _Xvelocity_name;
    std::string _Yvelocity_name;
    std::string _Zvelocity_name;

    bool _do_conv;
    bool _do_diff;
    bool _do_clip;

    double _low_clip;
    double _high_clip;
    double _init_value;

    struct SourceInfo{
      std::string name;
      double weight;
    };

    std::vector<SourceInfo> _source_info;

    LIMITER _conv_scheme;

  };

  //------------------------------------------------------------------------------------------------
  template <typename T>
  KScalarRHS<T>::KScalarRHS( std::string task_name, int matl_index ) :
  TaskInterface( task_name, matl_index ) {

    _rhs_name    = task_name+"_rhs";
    _X_flux_name = task_name+"_x_flux";
    _Y_flux_name = task_name+"_y_flux";
    _Z_flux_name = task_name+"_z_flux";
    _X_psi_name  = task_name+"_x_psi";
    _Y_psi_name  = task_name+"_y_psi";
    _Z_psi_name  = task_name+"_z_psi";

  }

  template <typename T>
  KScalarRHS<T>::~KScalarRHS(){}

  template <typename T> void
  KScalarRHS<T>::problemSetup( ProblemSpecP& db ){

    ConvectionHelper* helper;

    _do_conv = false;
    if ( db->findBlock("convection")){
      std::string conv_scheme;
      db->findBlock("convection")->getAttribute("scheme", conv_scheme);
      _conv_scheme = helper->get_limiter_from_string(conv_scheme);
      _do_conv = true;
    }

    _do_diff = false;
    if ( db->findBlock("diffusion")){
      _do_diff = true;
    }

    _do_clip = false;
    if ( db->findBlock("clip")){
      _do_clip = true;
      db->findBlock("clip")->getAttribute("low", _low_clip);
      db->findBlock("clip")->getAttribute("high", _high_clip);
    }

    _init_value = 0.0;
    if ( db->findBlock("initialize")){
      db->findBlock("initialize")->getAttribute("value",_init_value);
    }

    for ( ProblemSpecP src_db = db->findBlock("src"); src_db != 0;
          src_db = src_db->findNextBlock("src") ){

      std::string src_label;
      double weight = 1.0;

      src_db->getAttribute("label",src_label);

      if ( src_db->findBlock("weight")){
        src_db->findBlock("weight")->getAttribute("value",weight);
      }

      SourceInfo info;
      info.name = src_label;
      info.weight = weight;

      _source_info.push_back(info);

    }

    // Default velocity names:
    _Xvelocity_name = "uVelocitySPBC";
    _Yvelocity_name = "vVelocitySPBC";
    _Zvelocity_name = "wVelocitySPBC";
    if ( db->findBlock("velocity") ){
      db->findBlock("velocity")->getAttribute("xlabel",_Xvelocity_name);
      db->findBlock("velocity")->getAttribute("ylabel",_Yvelocity_name);
      db->findBlock("velocity")->getAttribute("zlabel",_Zvelocity_name);
    }

    // Diffusion coeff:
    _D_name = "NA";
    if ( _do_diff ) {
      if ( !db->findBlock("diffusion_coef")){
        throw InvalidValue("Error: when using diffusion for transport, you must specify the diffusion coefficient for eqn "+_task_name, __FILE__, __LINE__);
      }
      db->findBlock("diffusion_coef")->getAttribute("label",_D_name);
    }

  }

  template <typename T>
  void
  KScalarRHS<T>::create_local_labels(){

    register_new_variable<T>( _rhs_name );
    register_new_variable<T>( _task_name );
    register_new_variable<T>( _X_flux_name );
    register_new_variable<T>( _Y_flux_name );
    register_new_variable<T>( _Z_flux_name );

  }

  template <typename T> void
  KScalarRHS<T>::register_initialize(
    std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

    register_variable(  _rhs_name    , ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable(  _task_name   , ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable(  _X_flux_name , ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable(  _Y_flux_name , ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable(  _Z_flux_name , ArchesFieldContainer::COMPUTES , variable_registry );

  }

  template <typename T> void
  KScalarRHS<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                         SpatialOps::OperatorDatabase& opr ){

    T& rhs    = *(tsk_info->get_uintah_field<T>(_rhs_name));
    T& phi    = *(tsk_info->get_uintah_field<T>(_task_name));
    T& x_flux = *(tsk_info->get_uintah_field<T>(_X_flux_name));
    T& y_flux = *(tsk_info->get_uintah_field<T>(_Y_flux_name));
    T& z_flux = *(tsk_info->get_uintah_field<T>(_Z_flux_name));

    rhs.initialize(0.0);
    phi.initialize(_init_value);
    x_flux.initialize(0.0);
    y_flux.initialize(0.0);
    z_flux.initialize(0.0);
    phi.initialize(_init_value);

  }

  //------------------------------------------------------------------------------------------------
  template <typename T> void
  KScalarRHS<T>::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
    register_variable( _task_name   , ArchesFieldContainer::COMPUTES , variable_registry  );
    register_variable( _task_name   , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::OLDDW , variable_registry  );
    register_variable( _X_flux_name , ArchesFieldContainer::COMPUTES , variable_registry  );
    register_variable( _Y_flux_name , ArchesFieldContainer::COMPUTES , variable_registry  );
    register_variable( _Z_flux_name , ArchesFieldContainer::COMPUTES , variable_registry  );
    register_variable( _rhs_name    , ArchesFieldContainer::COMPUTES , variable_registry  );
  }

  template <typename T> void
  KScalarRHS<T>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                            SpatialOps::OperatorDatabase& opr ){

    T& phi = *(tsk_info->get_uintah_field<T>( _task_name ));
    typedef typename VariableHelper<T>::ConstType CT;
    CT& old_phi = *(tsk_info->get_const_uintah_field<CT>( _task_name ));

    T& rhs    = *(tsk_info->get_uintah_field<T>(_rhs_name));
    T& x_flux = *(tsk_info->get_uintah_field<T>(_X_flux_name));
    T& y_flux = *(tsk_info->get_uintah_field<T>(_Y_flux_name));
    T& z_flux = *(tsk_info->get_uintah_field<T>(_Z_flux_name));

    x_flux.initialize(0.0);
    y_flux.initialize(0.0);
    z_flux.initialize(0.0);
    rhs.initialize(0.0);
    phi.copyData(old_phi);

  }

  //------------------------------------------------------------------------------------------------
  template <typename T> void
  KScalarRHS<T>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){

    register_variable( _rhs_name       , ArchesFieldContainer::MODIFIES , variable_registry , time_substep );
    register_variable( _X_flux_name    , ArchesFieldContainer::MODIFIES , variable_registry , time_substep );
    register_variable( _Y_flux_name    , ArchesFieldContainer::MODIFIES , variable_registry , time_substep );
    register_variable( _Z_flux_name    , ArchesFieldContainer::MODIFIES , variable_registry , time_substep );
    register_variable( _X_psi_name     , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::NEWDW  , variable_registry , time_substep );
    register_variable( _Y_psi_name     , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::NEWDW  , variable_registry , time_substep );
    register_variable( _Z_psi_name     , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::NEWDW  , variable_registry , time_substep );
    if ( _do_diff )
      register_variable( _D_name         , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::NEWDW  , variable_registry , time_substep );
    register_variable( _task_name      , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
    register_variable( _Xvelocity_name , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
    register_variable( _Yvelocity_name , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
    register_variable( _Zvelocity_name , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
    register_variable( "areaFractionX" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::OLDDW  , variable_registry , time_substep );
    register_variable( "areaFractionY" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::OLDDW  , variable_registry , time_substep );
    register_variable( "areaFractionZ" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::OLDDW  , variable_registry , time_substep );

    typedef std::vector<SourceInfo> VS;
    for (typename VS::iterator i = _source_info.begin(); i != _source_info.end(); i++){
      register_variable( i->name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    }

  }

  template <typename T> void
  KScalarRHS<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                   SpatialOps::OperatorDatabase& opr ){

    T& rhs      = *(tsk_info->get_uintah_field<T>(_rhs_name));
    CT& phi     = *(tsk_info->get_const_uintah_field<CT>(_task_name));
    FXT& x_flux = *(tsk_info->get_uintah_field<FXT>(_X_flux_name));
    FYT& y_flux = *(tsk_info->get_uintah_field<FYT>(_Y_flux_name));
    FZT& z_flux = *(tsk_info->get_uintah_field<FZT>(_Z_flux_name));
    CFXT& x_psi = *(tsk_info->get_const_uintah_field<CFXT>(_X_psi_name));
    CFYT& y_psi = *(tsk_info->get_const_uintah_field<CFYT>(_Y_psi_name));
    CFZT& z_psi = *(tsk_info->get_const_uintah_field<CFZT>(_Z_psi_name));
    CFXT& u     = *(tsk_info->get_const_uintah_field<CFXT>(_Xvelocity_name));
    CFYT& v     = *(tsk_info->get_const_uintah_field<CFYT>(_Yvelocity_name));
    CFZT& w     = *(tsk_info->get_const_uintah_field<CFZT>(_Zvelocity_name));
    CFXT& af_x  = *(tsk_info->get_const_uintah_field<CFXT>("areaFractionX"));
    CFYT& af_y  = *(tsk_info->get_const_uintah_field<CFYT>("areaFractionY"));
    CFZT& af_z  = *(tsk_info->get_const_uintah_field<CFZT>("areaFractionZ"));

    Vector Dx = patch->dCell();
    double ax = Dx.y() * Dx.z();
    double ay = Dx.z() * Dx.x();
    double az = Dx.x() * Dx.y();
    double V = Dx.x()*Dx.y()*Dx.z();

    Uintah::BlockRange range(patch->getCellLowIndex(), patch->getExtraCellHighIndex());

#ifdef DO_TIMINGS
    SpatialOps::TimeLogger timer("kokkos_scalar_assemble.out."+_task_name);
    timer.start("work");
#endif

    //Convection:
    if ( _do_conv ){
      Uintah::ComputeConvectiveFlux<T> get_flux( phi, u, v, w, x_psi, y_psi, z_psi,
                                                 x_flux, y_flux, z_flux, af_x, af_y, af_z );
      Uintah::BlockRange range(patch->getCellLowIndex(), patch->getExtraCellHighIndex());
      Uintah::parallel_for( range, get_flux );
    }

    //Diffusion:
    if ( _do_diff ) {

      CT& D       = *(tsk_info->get_const_uintah_field<CT>(_D_name));

      //NOTE: No diffusion allowed on boundaries.
      GET_BUFFERED_PATCH_RANGE(1,-1);

      Uintah::BlockRange range_diff(low_patch_range, high_patch_range);

      Uintah::parallel_for( range_diff, [&phi, &D, &rhs, &ax, &ay, &az,
                                         &af_x, &af_y, &af_z, &Dx](int i, int j, int k){

        rhs(i,j,k) += ax/(2.*Dx.x()) * ( af_x(i+1,j,k) * ( D(i+1,j,k) + D(i,j,k))   * (phi(i+1,j,k) - phi(i,j,k))
                                       - af_x(i,j,k)   * ( D(i,j,k)   + D(i-1,j,k)) * (phi(i,j,k)   - phi(i-1,j,k)) ) +
                      ay/(2.*Dx.y()) * ( af_y(i,j+1,k) * ( D(i,j+1,k) + D(i,j,k))   * (phi(i,j+1,k) - phi(i,j,k))
                                       - af_y(i,j,k)   * ( D(i,j,k)   + D(i,j-1,k)) * (phi(i,j,k)   - phi(i,j-1,k)) ) +
                      az/(2.*Dx.z()) * ( af_z(i,j,k+1) * ( D(i,j,k+1) + D(i,j,k))   * (phi(i,j,k+1) - phi(i,j,k))
                                       - af_z(i,j,k)   * ( D(i,j,k)   + D(i,j,k-1)) * (phi(i,j,k)   - phi(i,j,k-1)) );


      });
    }

    //Sources:
    typedef std::vector<SourceInfo> VS;
    for (typename VS::iterator i = _source_info.begin(); i != _source_info.end(); i++){

      CT& src = *(tsk_info->get_const_uintah_field<CT>((*i).name));
      double weight = (*i).weight;
      Uintah::BlockRange src_range(patch->getCellLowIndex(), patch->getCellHighIndex());

      Uintah::parallel_for( src_range, [&rhs, &src, &V, &weight](int i, int j, int k){

        rhs(i,j,k) += weight * src(i,j,k) * V;

      });
    }

#ifdef DO_TIMINGS
    timer.stop("work");
#endif

  }

  template <typename T> void
  KScalarRHS<T>::register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){
    //register_variable( _task_name, ArchesFieldContainer::MODIFIES, variable_registry, time_substep );
  }

  template <typename T> void
  KScalarRHS<T>::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                          SpatialOps::OperatorDatabase& opr ){
  }
}
#endif
