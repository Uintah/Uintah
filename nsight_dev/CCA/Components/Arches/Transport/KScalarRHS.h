#ifndef Uintah_Component_Arches_KScalarRHS_h
#define Uintah_Component_Arches_KScalarRHS_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>
#include <CCA/Components/Arches/ConvectionHelper.h>
#include <CCA/Components/Arches/Directives.h>
#include <spatialops/util/TimeLogger.h>

namespace Uintah{

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

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

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

    typedef typename ArchesCore::VariableHelper<T>::ConstType CT;
    typedef typename ArchesCore::VariableHelper<T>::XFaceType FXT;
    typedef typename ArchesCore::VariableHelper<T>::YFaceType FYT;
    typedef typename ArchesCore::VariableHelper<T>::ZFaceType FZT;
    typedef typename ArchesCore::VariableHelper<T>::ConstXFaceType CFXT;
    typedef typename ArchesCore::VariableHelper<T>::ConstYFaceType CFYT;
    typedef typename ArchesCore::VariableHelper<T>::ConstZFaceType CFZT;

    std::string _D_name;
    std::string _x_velocity_name;
    std::string _y_velocity_name;
    std::string _z_velocity_name;

    std::vector<std::string> _eqn_names;
    std::vector<bool> _do_diff;
    std::vector<bool> _do_clip;
    std::vector<double> _low_clip;
    std::vector<double> _high_clip;
    std::vector<double> _init_value;

    bool _has_D;

    int _total_eqns;

    struct SourceInfo{
      std::string name;
      double weight;
    };

    std::vector<std::vector<SourceInfo> > _source_info;

    std::vector<LIMITER> _conv_scheme;

  };

  //------------------------------------------------------------------------------------------------
  template <typename T>
  KScalarRHS<T>::KScalarRHS( std::string task_name, int matl_index ) :
  TaskInterface( task_name, matl_index ) {}

  template <typename T>
  KScalarRHS<T>::~KScalarRHS(){}

  template <typename T> void
  KScalarRHS<T>::problemSetup( ProblemSpecP& input_db ){

    _total_eqns = 0;

    ConvectionHelper* conv_helper;
    for (ProblemSpecP db = input_db->findBlock("eqn"); db != 0;
         db = db->findNextBlock("eqn")){

      //Equation name
      std::string eqn_name;
      db->getAttribute("label", eqn_name);
      _eqn_names.push_back(eqn_name);

      //Convection
      if ( db->findBlock("convection")){
        std::string conv_scheme;
        db->findBlock("convection")->getAttribute("scheme", conv_scheme);
        _conv_scheme.push_back(conv_helper->get_limiter_from_string(conv_scheme));
      } else {
        _conv_scheme.push_back(NOCONV);
      }

      //Diffusion
      if ( db->findBlock("diffusion")){
        _do_diff.push_back(true);
      } else {
        _do_diff.push_back(false);
      }

      //Clipping
      if ( db->findBlock("clip")){
        _do_clip.push_back(true);
        double low; double high;
        db->findBlock("clip")->getAttribute("low", low);
        db->findBlock("clip")->getAttribute("high", high);
        _low_clip.push_back(low);
        _high_clip.push_back(high);
      } else {
        _do_clip.push_back(false);
        _low_clip.push_back(-999.9);
        _high_clip.push_back(999.9);
      }

      //Initial Value
      if ( db->findBlock("initialize") ){
        double value;
        db->findBlock("initialize")->getAttribute("value",value);
        _init_value.push_back(value);
      } else {
        _init_value.push_back(0.0);
      }

      std::vector<SourceInfo> eqn_srcs;
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

        eqn_srcs.push_back(info);

      }

      _source_info.push_back(eqn_srcs);

    }

    // Default velocity names:
    _x_velocity_name = "uVelocitySPBC";
    _y_velocity_name = "vVelocitySPBC";
    _z_velocity_name = "wVelocitySPBC";
    if ( input_db->findBlock("velocity") ){
      input_db->findBlock("velocity")->getAttribute("xlabel",_x_velocity_name);
      input_db->findBlock("velocity")->getAttribute("ylabel",_y_velocity_name);
      input_db->findBlock("velocity")->getAttribute("zlabel",_z_velocity_name);
    }

    // Diffusion coeff -- assuming the same one across all eqns.
    _D_name = "NA";
    _has_D = false;

    if ( input_db->findBlock("diffusion_coef") ) {
      input_db->findBlock("diffusion_coef")->getAttribute("label",_D_name);
      _has_D = true;
    }

  }

  template <typename T>
  void
  KScalarRHS<T>::create_local_labels(){

    const int istart = 0;
    int iend = _eqn_names.size();
    for (int i = istart; i < iend; i++ ){
      register_new_variable<T>( _eqn_names[i] );
      register_new_variable<T>( _eqn_names[i]+"_rhs" );
      register_new_variable<FXT>( _eqn_names[i]+"_x_flux" );
      register_new_variable<FYT>( _eqn_names[i]+"_y_flux" );
      register_new_variable<FZT>( _eqn_names[i]+"_z_flux" );
    }
  }

  template <typename T> void
  KScalarRHS<T>::register_initialize(
    std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

    const int istart = 0;
    const int iend = _eqn_names.size();
    for (int ieqn = istart; ieqn < iend; ieqn++ ){
      register_variable(  _eqn_names[ieqn], ArchesFieldContainer::COMPUTES , variable_registry );
      register_variable(  _eqn_names[ieqn]+"_rhs", ArchesFieldContainer::COMPUTES , variable_registry );
      //if ( _conv_scheme[ieqn] != NOCONV ){
        register_variable(  _eqn_names[ieqn]+"_x_flux", ArchesFieldContainer::COMPUTES , variable_registry );
        register_variable(  _eqn_names[ieqn]+"_y_flux", ArchesFieldContainer::COMPUTES , variable_registry );
        register_variable(  _eqn_names[ieqn]+"_z_flux", ArchesFieldContainer::COMPUTES , variable_registry );
      //}
    }
  }

  template <typename T> void
  KScalarRHS<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    const int istart = 0;
    const int iend = _eqn_names.size();
    for (int ieqn = istart; ieqn < iend; ieqn++ ){

      double scalar_init_value = _init_value[ieqn];

      T& phi    = *(tsk_info->get_uintah_field<T>(_eqn_names[ieqn]+"_rhs"));
      T& rhs    = *(tsk_info->get_uintah_field<T>(_eqn_names[ieqn]));
      Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
      Uintah::parallel_for( range, [&](int i, int j, int k){
        phi(i,j,k) = scalar_init_value;
        rhs(i,j,k) = 0.0;
      });

      FXT& x_flux = *(tsk_info->get_uintah_field<FXT>(_eqn_names[ieqn]+"_x_flux"));
      FYT& y_flux = *(tsk_info->get_uintah_field<FYT>(_eqn_names[ieqn]+"_y_flux"));
      FZT& z_flux = *(tsk_info->get_uintah_field<FZT>(_eqn_names[ieqn]+"_z_flux"));

      Uintah::parallel_for( range, [&](int i, int j, int k){
        x_flux(i,j,k) = 0.0;
        y_flux(i,j,k) = 0.0;
        z_flux(i,j,k) = 0.0;
      });

    } //eqn loop
  }

  //------------------------------------------------------------------------------------------------
  template <typename T> void
  KScalarRHS<T>::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
    const int istart = 0;
    const int iend = _eqn_names.size();
    for (int ieqn = istart; ieqn < iend; ieqn++ ){
      register_variable( _eqn_names[ieqn], ArchesFieldContainer::COMPUTES , variable_registry  );
      register_variable( _eqn_names[ieqn], ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::OLDDW , variable_registry  );
    }
  }

  template <typename T> void
  KScalarRHS<T>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    const int istart = 0;
    const int iend = _eqn_names.size();
    for (int ieqn = istart; ieqn < iend; ieqn++ ){
      T& phi = *(tsk_info->get_uintah_field<T>( _eqn_names[ieqn] ));
      typedef typename ArchesCore::VariableHelper<T>::ConstType CT;
      CT& old_phi = *(tsk_info->get_const_uintah_field<CT>( _eqn_names[ieqn] ));

      phi.copyData(old_phi);

    } //equation loop
  }

  //------------------------------------------------------------------------------------------------
  template <typename T> void
  KScalarRHS<T>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){

    const int istart = 0;
    const int iend = _eqn_names.size();
    for (int ieqn = istart; ieqn < iend; ieqn++ ){

      register_variable( _eqn_names[ieqn], ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( _eqn_names[ieqn]+"_rhs", ArchesFieldContainer::COMPUTES, variable_registry, time_substep );
      register_variable( _eqn_names[ieqn]+"_x_flux", ArchesFieldContainer::COMPUTES, variable_registry, time_substep );
      register_variable( _eqn_names[ieqn]+"_y_flux", ArchesFieldContainer::COMPUTES, variable_registry, time_substep );
      register_variable( _eqn_names[ieqn]+"_z_flux", ArchesFieldContainer::COMPUTES, variable_registry, time_substep );
      if ( _conv_scheme[ieqn] != NOCONV ){
        register_variable( _eqn_names[ieqn]+"_x_psi", ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
        register_variable( _eqn_names[ieqn]+"_y_psi", ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
        register_variable( _eqn_names[ieqn]+"_z_psi", ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      }

      typedef std::vector<SourceInfo> VS;
      for (typename VS::iterator i = _source_info[ieqn].begin(); i != _source_info[ieqn].end(); i++){
        register_variable( i->name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      }

    }

    //globally common variables
    if ( _has_D )
      register_variable( _D_name       , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::NEWDW  , variable_registry , time_substep );
    register_variable( _x_velocity_name, ArchesFieldContainer::REQUIRES, 0 , ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( _y_velocity_name, ArchesFieldContainer::REQUIRES, 0 , ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( _z_velocity_name, ArchesFieldContainer::REQUIRES, 0 , ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( "areaFractionX", ArchesFieldContainer::REQUIRES, 1 , ArchesFieldContainer::OLDDW, variable_registry, time_substep );
    register_variable( "areaFractionY", ArchesFieldContainer::REQUIRES, 1 , ArchesFieldContainer::OLDDW, variable_registry, time_substep );
    register_variable( "areaFractionZ", ArchesFieldContainer::REQUIRES, 1 , ArchesFieldContainer::OLDDW, variable_registry, time_substep );

  }

  template <typename T> void
  KScalarRHS<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    Vector Dx = patch->dCell();
    double ax = Dx.y() * Dx.z();
    double ay = Dx.z() * Dx.x();
    double az = Dx.x() * Dx.y();
    double V = Dx.x()*Dx.y()*Dx.z();

    CFXT& u     = *(tsk_info->get_const_uintah_field<CFXT>(_x_velocity_name));
    CFYT& v     = *(tsk_info->get_const_uintah_field<CFYT>(_y_velocity_name));
    CFZT& w     = *(tsk_info->get_const_uintah_field<CFZT>(_z_velocity_name));
    CFXT& af_x  = *(tsk_info->get_const_uintah_field<CFXT>("areaFractionX"));
    CFYT& af_y  = *(tsk_info->get_const_uintah_field<CFYT>("areaFractionY"));
    CFZT& af_z  = *(tsk_info->get_const_uintah_field<CFZT>("areaFractionZ"));
    Uintah::BlockRange range_cl_to_ech(patch->getCellLowIndex(), patch->getExtraCellHighIndex());


#ifdef DO_TIMINGS
    SpatialOps::TimeLogger timer("kokkos_scalar_assemble.out."+_task_name);
    timer.start("work");
#endif
    const int istart = 0;
    const int iend = _eqn_names.size();
    for (int ieqn = istart; ieqn < iend; ieqn++ ){
      CT& phi     = *(tsk_info->get_const_uintah_field<CT>(_eqn_names[ieqn]));
      T& rhs      = *(tsk_info->get_uintah_field<T>(_eqn_names[ieqn]+"_rhs"));

      //Convection:
      FXT& x_flux = *(tsk_info->get_uintah_field<FXT>(_eqn_names[ieqn]+"_x_flux"));
      FYT& y_flux = *(tsk_info->get_uintah_field<FYT>(_eqn_names[ieqn]+"_y_flux"));
      FZT& z_flux = *(tsk_info->get_uintah_field<FZT>(_eqn_names[ieqn]+"_z_flux"));

      Uintah::BlockRange init_range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
      Uintah::parallel_for( init_range, [&](int i, int j, int k){

        rhs(i,j,k) = 0.;
        x_flux(i,j,k) = 0.;
        y_flux(i,j,k) = 0.;
        z_flux(i,j,k) = 0.;

      });

      if ( _conv_scheme[ieqn] != NOCONV ){

        CFXT& x_psi = *(tsk_info->get_const_uintah_field<CFXT>(_eqn_names[ieqn]+"_x_psi"));
        CFYT& y_psi = *(tsk_info->get_const_uintah_field<CFYT>(_eqn_names[ieqn]+"_y_psi"));
        CFZT& z_psi = *(tsk_info->get_const_uintah_field<CFZT>(_eqn_names[ieqn]+"_z_psi"));

        Uintah::ComputeConvectiveFlux<T> get_flux( phi, u, v, w, x_psi, y_psi, z_psi,
                                                   x_flux, y_flux, z_flux, af_x, af_y, af_z );
        Uintah::parallel_for( range_cl_to_ech, get_flux );
      }

      //Diffusion:
      if ( _do_diff[ieqn] ) {

        CT& D = *(tsk_info->get_const_uintah_field<CT>(_D_name));

        //NOTE: No diffusion allowed on boundaries.
        GET_BUFFERED_PATCH_RANGE(1,-1);

        Uintah::BlockRange range_diff(low_patch_range, high_patch_range);

        Uintah::parallel_for( range_diff, [&](int i, int j, int k){

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
      for (typename VS::iterator isrc = _source_info[ieqn].begin(); isrc != _source_info[ieqn].end(); isrc++){

        CT& src = *(tsk_info->get_const_uintah_field<CT>((*isrc).name));
        double weight = (*isrc).weight;
        Uintah::BlockRange src_range(patch->getCellLowIndex(), patch->getCellHighIndex());

        Uintah::parallel_for( src_range, [&](int i, int j, int k){

          rhs(i,j,k) += weight * src(i,j,k) * V;

        });
      }
    } // equation loop
#ifdef DO_TIMINGS
    timer.stop("work");
#endif

  }

  template <typename T> void
  KScalarRHS<T>::register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){
    //register_variable( _task_name, ArchesFieldContainer::MODIFIES, variable_registry, time_substep );
  }

  template <typename T> void
  KScalarRHS<T>::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){ }

}
#endif
