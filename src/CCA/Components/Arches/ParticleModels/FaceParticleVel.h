#ifndef Uintah_Component_Arches_FaceParticleVel_h
#define Uintah_Component_Arches_FaceParticleVel_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>

namespace Uintah{

  //IT is the independent variable type
  template <typename T>
  class FaceParticleVel : public TaskInterface {

public:

    FaceParticleVel<T>( std::string task_name, int matl_index, const std::string var_name );
    ~FaceParticleVel<T>();

    void problemSetup( ProblemSpecP& db );



    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index, std::string base_var_name ) :
        m_task_name(task_name), m_matl_index(matl_index), _base_var_name(base_var_name){}
      ~Builder(){}

      FaceParticleVel* build()
      { return scinew FaceParticleVel<T>( m_task_name, m_matl_index, _base_var_name ); }

      private:

      std::string m_task_name;
      int m_matl_index;
      std::string _base_var_name;

    };

protected:

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();

private:

    typedef typename ArchesCore::VariableHelper<T>::XFaceType FXT;
    typedef typename ArchesCore::VariableHelper<T>::YFaceType FYT;
    typedef typename ArchesCore::VariableHelper<T>::ZFaceType FZT;
    typedef typename ArchesCore::VariableHelper<T>::ConstType CT;

    const std::string _base_var_name;
    std::string _temperature_var_name;
    std::string _conc_var_name;
    ArchesCore::INTERPOLANT m_int_scheme;

    int m_N;                 //<<< The number of "environments"
    int m_ghost_cells;
    std::string up_root;
    std::string vp_root;
    std::string wp_root;
    std::string up_face;
    std::string vp_face;
    std::string wp_face;

  };

  //Function definitions:

  template <typename T>
  void FaceParticleVel<T>::create_local_labels(){

    for ( int i = 0; i < m_N; i++ ){

      std::string up_face_i = ArchesCore::append_env(up_face,i);
      std::string vp_face_i = ArchesCore::append_env(vp_face,i);
      std::string wp_face_i = ArchesCore::append_env(wp_face,i);
      register_new_variable<FXT>( up_face_i );
      register_new_variable<FYT>( vp_face_i );
      register_new_variable<FZT>( wp_face_i );
    }

  }


  template <typename T>
  FaceParticleVel<T>::FaceParticleVel( std::string task_name, int matl_index,
                                                      const std::string base_var_name ) :
  TaskInterface( task_name, matl_index ), _base_var_name(base_var_name){
  }

  template <typename T>
  FaceParticleVel<T>::~FaceParticleVel()
  {}

  template <typename T>
  void FaceParticleVel<T>::problemSetup( ProblemSpecP& db ){

    m_N = ArchesCore::get_num_env(db,ArchesCore::DQMOM_METHOD);
    up_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_XVEL );
    vp_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_YVEL );
    wp_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_ZVEL );
    up_face = "face_pvel_x";
    vp_face = "face_pvel_y";
    wp_face = "face_pvel_z";
    
    std::string scheme = "second";
    m_int_scheme = ArchesCore::get_interpolant_from_string( scheme );

    m_ghost_cells = 1;
    if (m_int_scheme== ArchesCore::FOURTHCENTRAL){
      m_ghost_cells = 2;
    }
  }

  //======INITIALIZATION:
  template <typename T>
  void FaceParticleVel<T>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

    for ( int i = 0; i < m_N; i++ ){
      std::string up_face_i = ArchesCore::append_env(up_face,i);
      std::string vp_face_i = ArchesCore::append_env(vp_face,i);
      std::string wp_face_i = ArchesCore::append_env(wp_face,i);
      register_variable( up_face_i, ArchesFieldContainer::COMPUTES, variable_registry );
      register_variable( vp_face_i, ArchesFieldContainer::COMPUTES, variable_registry );
      register_variable( wp_face_i, ArchesFieldContainer::COMPUTES, variable_registry );

      std::string up_i = ArchesCore::append_env(up_root,i);
      std::string vp_i = ArchesCore::append_env(vp_root,i);
      std::string wp_i = ArchesCore::append_env(wp_root,i);

      register_variable( up_i, ArchesFieldContainer::REQUIRES, m_ghost_cells, ArchesFieldContainer::NEWDW, variable_registry );
      register_variable( vp_i, ArchesFieldContainer::REQUIRES, m_ghost_cells, ArchesFieldContainer::NEWDW, variable_registry );
      register_variable( wp_i, ArchesFieldContainer::REQUIRES, m_ghost_cells, ArchesFieldContainer::NEWDW, variable_registry );

    }

  }

  template <typename T>
  void FaceParticleVel<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){


  Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );


  for ( int ienv = 0; ienv < m_N; ienv++ ){

    std::string up_face_i = ArchesCore::append_env(up_face,ienv);
    std::string vp_face_i = ArchesCore::append_env(vp_face,ienv);
    std::string wp_face_i = ArchesCore::append_env(wp_face,ienv);

    FXT& up_f = tsk_info->get_uintah_field_add<FXT>(up_face_i);
    FYT& vp_f = tsk_info->get_uintah_field_add<FYT>(vp_face_i);
    FZT& wp_f = tsk_info->get_uintah_field_add<FZT>(wp_face_i);

    std::string up_i = ArchesCore::append_env(up_root,ienv);
    std::string vp_i = ArchesCore::append_env(vp_root,ienv);
    std::string wp_i = ArchesCore::append_env(wp_root,ienv);

    CT& up = tsk_info->get_const_uintah_field_add<CT>(up_i);
    CT& vp = tsk_info->get_const_uintah_field_add<CT>(vp_i);
    CT& wp = tsk_info->get_const_uintah_field_add<CT>(wp_i);

    ArchesCore::OneDInterpolator my_interpolant_up( up_f, up, -1, 0, 0 );
    ArchesCore::OneDInterpolator my_interpolant_vp( vp_f, vp, 0, -1, 0 );
    ArchesCore::OneDInterpolator my_interpolant_wp( wp_f, wp, 0, 0, -1 );

    if ( m_int_scheme == ArchesCore::SECONDCENTRAL ) {

      ArchesCore::SecondCentral ci;
      Uintah::parallel_for( range, my_interpolant_up, ci );
      Uintah::parallel_for( range, my_interpolant_vp, ci );
      Uintah::parallel_for( range, my_interpolant_wp, ci );

    } else if ( m_int_scheme== ArchesCore::FOURTHCENTRAL ){

      ArchesCore::FourthCentral ci;
      Uintah::parallel_for( range, my_interpolant_up, ci );
      Uintah::parallel_for( range, my_interpolant_vp, ci );
      Uintah::parallel_for( range, my_interpolant_wp, ci );

  }


  }
  }

  //======TIME STEP INITIALIZATION:
  template <typename T>
  void FaceParticleVel<T>::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){
  }

  template <typename T>
  void FaceParticleVel<T>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

  //======TIME STEP EVALUATION:
  template <typename T>
  void FaceParticleVel<T>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){


    for ( int ienv = 0; ienv < m_N; ienv++ ){

      std::string up_face_i = ArchesCore::append_env(up_face,ienv);
      std::string vp_face_i = ArchesCore::append_env(vp_face,ienv);
      std::string wp_face_i = ArchesCore::append_env(wp_face,ienv);
      register_variable( up_face_i, ArchesFieldContainer::COMPUTES, variable_registry , m_task_name  );
      register_variable( vp_face_i, ArchesFieldContainer::COMPUTES, variable_registry , m_task_name  );
      register_variable( wp_face_i, ArchesFieldContainer::COMPUTES, variable_registry , m_task_name  );

      std::string up_i = ArchesCore::append_env(up_root,ienv);
      std::string vp_i = ArchesCore::append_env(vp_root,ienv);
      std::string wp_i = ArchesCore::append_env(wp_root,ienv);

      register_variable( up_i, ArchesFieldContainer::REQUIRES, m_ghost_cells, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( vp_i, ArchesFieldContainer::REQUIRES, m_ghost_cells, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( wp_i, ArchesFieldContainer::REQUIRES, m_ghost_cells, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    }


  }

  template <typename T>
  void FaceParticleVel<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){


  Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );


  for ( int ienv = 0; ienv < m_N; ienv++ ){

    std::string up_face_i = ArchesCore::append_env(up_face,ienv);
    std::string vp_face_i = ArchesCore::append_env(vp_face,ienv);
    std::string wp_face_i = ArchesCore::append_env(wp_face,ienv);
    FXT& up_f = tsk_info->get_uintah_field_add<FXT>(up_face_i);
    FYT& vp_f = tsk_info->get_uintah_field_add<FYT>(vp_face_i);
    FZT& wp_f = tsk_info->get_uintah_field_add<FZT>(wp_face_i);

    std::string up_i = ArchesCore::append_env(up_root,ienv);
    std::string vp_i = ArchesCore::append_env(vp_root,ienv);
    std::string wp_i = ArchesCore::append_env(wp_root,ienv);

    CT& up = tsk_info->get_const_uintah_field_add<CT>(up_i);
    CT& vp = tsk_info->get_const_uintah_field_add<CT>(vp_i);
    CT& wp = tsk_info->get_const_uintah_field_add<CT>(wp_i);

    ArchesCore::OneDInterpolator my_interpolant_up( up_f, up, -1, 0, 0 );
    ArchesCore::OneDInterpolator my_interpolant_vp( vp_f, vp, 0, -1, 0 );
    ArchesCore::OneDInterpolator my_interpolant_wp( wp_f, wp, 0, 0, -1 );

    if ( m_int_scheme == ArchesCore::SECONDCENTRAL ) {

      ArchesCore::SecondCentral ci;
      Uintah::parallel_for( range, my_interpolant_up, ci );
      Uintah::parallel_for( range, my_interpolant_vp, ci );
      Uintah::parallel_for( range, my_interpolant_wp, ci );

    } else if ( m_int_scheme== ArchesCore::FOURTHCENTRAL ){

      ArchesCore::FourthCentral ci;
      Uintah::parallel_for( range, my_interpolant_up, ci );
      Uintah::parallel_for( range, my_interpolant_vp, ci );
      Uintah::parallel_for( range, my_interpolant_wp, ci );

  }


  }
  }
}
#endif
