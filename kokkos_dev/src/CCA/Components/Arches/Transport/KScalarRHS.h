#ifndef Uintah_Component_Arches_KScalarRHS_h
#define Uintah_Component_Arches_KScalarRHS_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/DiscretizationTools.h>
#include <Kokkos_Core.hpp>

namespace Uintah{

  class Operators;
  template<typename T>
  class KScalarRHS : public TaskInterface {

public:

    KScalarRHS<T>( std::string task_name, int matl_index );
    ~KScalarRHS<T>();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry );

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry );

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep );

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep );

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

      Builder( std::string task_name, int matl_index ) : _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      KScalarRHS* build()
      { return scinew KScalarRHS<T>( _task_name, _matl_index ); }

      private:

      std::string _task_name;
      int _matl_index;

    };

private:

    std::string _rhs_name;
    std::string _D_name;
    std::string _Fconv_name;
    std::string _Fdiff_name;
    std::string _conv_scheme;

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

  };

  //------------------------------------------------------------------------------------------------
  template <typename T>
  KScalarRHS<T>::KScalarRHS( std::string task_name, int matl_index ) :
  TaskInterface( task_name, matl_index ) {

    _rhs_name = task_name+"_RHS";
    _D_name = task_name+"_D";
    _Fconv_name = task_name+"_Fconv";
    _Fdiff_name = task_name+"_Fdiff";

  }

  template <typename T>
  KScalarRHS<T>::~KScalarRHS(){

  }

  template <typename T> void
  KScalarRHS<T>::problemSetup( ProblemSpecP& db ){

    _do_conv = false;
    if ( db->findBlock("convection")){
      db->findBlock("convection")->getAttribute("scheme", _conv_scheme);
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

    for (ProblemSpecP src_db = db->findBlock("src"); src_db != 0; src_db = src_db->findNextBlock("src")){

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

  }

  template <typename T>
  void
  KScalarRHS<T>::create_local_labels(){

    register_new_variable<T>( _rhs_name );
    register_new_variable<T>( _task_name );
    register_new_variable<CCVariable<double> >( _D_name );
    register_new_variable<T>( _Fconv_name );
    register_new_variable<T>( _Fdiff_name );

  }

  template <typename T> void
  KScalarRHS<T>::register_initialize(
    std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

    register_variable(  _rhs_name   , ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable(  _task_name  , ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable(  _D_name     , ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable(  _Fconv_name , ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable(  _Fdiff_name , ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable( "gridX", ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry ); 

  }

  template <typename T> void
  KScalarRHS<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                         SpatialOps::OperatorDatabase& opr ){

    T& rhs = *(tsk_info->get_uintah_field<T>(_rhs_name));
    T& phi = *(tsk_info->get_uintah_field<T>(_task_name));
    CCVariable<double>& gamma = *(tsk_info->get_uintah_field<CCVariable<double> >(_D_name));
    T& Fdiff = *(tsk_info->get_uintah_field<T>(_Fdiff_name));
    T& Fconv = *(tsk_info->get_uintah_field<T>(_Fconv_name));
    constCCVariable<double>& x = *(tsk_info->get_const_uintah_field<constCCVariable<double> >("gridX"));

    rhs.initialize(0.0);
    phi.initialize(_init_value);
    gamma.initialize(.0001);
    Fdiff.initialize(0.0);
    Fconv.initialize(0.0);

    VariableInitializeFunctor<T> init_variable(phi,x,_init_value);

    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();
    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::yminus) == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::zminus) == Patch::Neighbor?0:1);
    h -= IntVector(patch->getBCType(Patch::xplus)  == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::yplus)  == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::zplus)  == Patch::Neighbor?0:1);

    if ( _task_name == "phi" )
      Kokkos::parallel_for( Kokkos::Range3Policy<int>(l[0],l[1],l[2], h[0],h[1],h[2]), init_variable );


  }

  //------------------------------------------------------------------------------------------------
  template <typename T> void
  KScalarRHS<T>::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
    register_variable( _D_name     , ArchesFieldContainer::COMPUTES , variable_registry  );
    register_variable( _D_name     , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::OLDDW , variable_registry );
    register_variable( _task_name  , ArchesFieldContainer::COMPUTES , variable_registry  );
    register_variable( _task_name  , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::OLDDW , variable_registry  );
    register_variable( _Fconv_name  , ArchesFieldContainer::COMPUTES , variable_registry  );
  }

  template <typename T> void
  KScalarRHS<T>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                            SpatialOps::OperatorDatabase& opr ){

    CCVariable<double>& gamma = *(tsk_info->get_uintah_field<CCVariable<double> >( _D_name ));
    constCCVariable<double>& old_gamma = *(tsk_info->get_const_uintah_field<constCCVariable<double> >( _D_name ));
    T& phi = *(tsk_info->get_uintah_field<T>( _task_name ));
    typedef typename VariableHelper<T>::ConstType CONST_TYPE;
    CONST_TYPE& old_phi = *(tsk_info->get_const_uintah_field<CONST_TYPE>( _task_name ));

    CCVariable<double>& Fconv = *(tsk_info->get_uintah_field<CCVariable<double> >(_Fconv_name));
    Fconv.initialize(0.0);

    gamma.copyData(old_gamma);
    phi.copyData(old_phi);

  }

  //------------------------------------------------------------------------------------------------
  template <typename T> void
  KScalarRHS<T>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){

  //  //FUNCITON CALL     STRING NAME(VL)     DEPENDENCY    GHOST DW     VR
    register_variable( _rhs_name        , ArchesFieldContainer::COMPUTES , variable_registry , time_substep );
    register_variable( _Fconv_name      , ArchesFieldContainer::MODIFIES , variable_registry , time_substep );
    register_variable( _Fdiff_name      , ArchesFieldContainer::COMPUTES , variable_registry , time_substep );
    register_variable( _D_name          , ArchesFieldContainer::REQUIRES,  1 , ArchesFieldContainer::NEWDW  , variable_registry , time_substep );
    register_variable( _task_name       , ArchesFieldContainer::REQUIRES , 2 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
    register_variable( "uVelocitySPBC"  , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
    register_variable( "vVelocitySPBC"  , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
    register_variable( "wVelocitySPBC"  , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
    // register_variable( "areaFractionFX" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::OLDDW  , variable_registry , time_substep );
    // register_variable( "areaFractionFY" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::OLDDW  , variable_registry , time_substep );
    // register_variable( "areaFractionFZ" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::OLDDW  , variable_registry , time_substep );
    // register_variable( "volFraction"    , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::OLDDW  , variable_registry , time_substep );
    register_variable( "density"        , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
  //  //register_variable( "areaFraction"   , ArchesFieldContainer::REQUIRES , 2 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
  //
  //  typedef std::vector<SourceInfo> VS;
  //  for (VS::iterator i = _source_info.begin(); i != _source_info.end(); i++){
  //    register_variable( i->name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  //  }

  }

  template <typename T> void
  KScalarRHS<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                   SpatialOps::OperatorDatabase& opr ){


    T& rhs = *(tsk_info->get_uintah_field<T>(_rhs_name));
    typedef typename VariableHelper<T>::ConstType CONST_TYPE;
    CONST_TYPE& phi = *(tsk_info->get_const_uintah_field<CONST_TYPE>(_task_name));
    constCCVariable<double>& rho = *(tsk_info->get_const_uintah_field<constCCVariable<double> >("density"));
    constCCVariable<double>& gamma = *(tsk_info->get_const_uintah_field<constCCVariable<double> >(_D_name));
    constSFCXVariable<double>& u = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("uVelocitySPBC"));
    constSFCYVariable<double>& v = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >("vVelocitySPBC"));
    constSFCZVariable<double>& w = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >("wVelocitySPBC"));

    rhs.initialize(0.0);

    Vector DX = patch->dCell();
    double A = DX.y() * DX.z();

    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();
    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::yminus) == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::zminus) == Patch::Neighbor?0:1);
    h -= IntVector(patch->getBCType(Patch::xplus)  == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::yplus)  == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::zplus)  == Patch::Neighbor?0:1);

    //better to combine all directions into a single grid loop?
    IntVector dir(1,0,0);
    ComputeConvection<T, constSFCXVariable<double> > x_conv( phi, rhs, rho, u, dir, A);
    ComputeDiffusion<T> x_diff( phi, gamma, rhs, dir, A, DX.x() );

    if ( _do_conv )
      Kokkos::parallel_for( Kokkos::Range3Policy<int>(l[0],l[1],l[2], h[0],h[1],h[2]), x_conv );
    if ( _do_diff )
      Kokkos::parallel_for( Kokkos::Range3Policy<int>(l[0],l[1],l[2], h[0],h[1],h[2]), x_diff );

    dir = IntVector(0,1,0);
    if ( _do_conv )
      ComputeConvection<T, constSFCYVariable<double> > y_conv( phi, rhs, rho, v, dir, A);
    if ( _do_diff )
      ComputeDiffusion<T> y_diff( phi, gamma, rhs, dir, A, DX.y() );

    dir = IntVector(0,0,1);
    if ( _do_conv )
      ComputeConvection<T, constSFCZVariable<double> > z_conv( phi, rhs, rho, w, dir, A);
    if ( _do_diff )
      ComputeDiffusion<T> z_diff( phi, gamma, rhs, dir, A, DX.z() );

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
