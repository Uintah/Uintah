#ifndef Uintah_Component_Arches_BodyForce_h
#define Uintah_Component_Arches_BodyForce_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
//-------------------------------------------------------

/**
 * @class    Body Force
 * @author   Alex Abboud
 * @date     August 2014
 *
 * @brief    This class sets up the body force on particles due to gravity
 *
 * @details  The class calculates the body force for particles as the simple \f$ du_i/dt = g_i (\rho_p-\rho_g)/\rho_p \f$
 *           When using lagrangian particles N = 1. This generalization should allow for the same code to be utilized for
 *           any particle method - DQMOM, CQMOM, or Lagrangian.
 *
 */

//-------------------------------------------------------

namespace Uintah{

  //IT is the independent variable type
  //DT is the dependent variable type
  template <typename IT, typename DT>
  class BodyForce : public TaskInterface {

  public:

    BodyForce<IT, DT>( std::string task_name, int matl_index, const std::string var_name, const int N );
    ~BodyForce<IT, DT>();

    void problemSetup( ProblemSpecP& db );

    void create_local_labels();

    class Builder : public TaskInterface::TaskBuilder {

    public:

      Builder( std::string task_name, int matl_index, std::string base_var_name, const int N ) :
      m_task_name(task_name), m_matl_index(matl_index), m_base_var_name(base_var_name), m_N(N){}
      ~Builder(){}

      BodyForce* build()
      { return scinew BodyForce<IT, DT>( m_task_name, m_matl_index, m_base_var_name, m_N ); }

    private:

      std::string m_task_name;
      int m_matl_index;
      std::string m_base_var_name;
      const int m_N;

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

  private:

    const std::string m_base_var_name;
    std::string m_base_density_name;
    std::string m_gas_density_name;
    std::string m_direction;

    const int m_N;                 //<<< The number of "environments"
    double m_gravity_component;

    const std::string get_name(const int i, const std::string base_name){
      std::stringstream out;
      std::string env;
      out << i;
      env = out.str();
      return base_name + "_" + env;
    }

  };

  //Function definitions:

  template <typename IT, typename DT>
  void BodyForce<IT,DT>::create_local_labels(){
    for ( int i = 0; i < m_N; i++ ){
      const std::string name = get_name(i, m_base_var_name);
      register_new_variable<DT>(name );
    }
  }

  template <typename IT, typename DT>
  BodyForce<IT, DT>::BodyForce( std::string task_name, int matl_index,
                                const std::string base_var_name, const int N ) :
  TaskInterface( task_name, matl_index ), m_base_var_name(base_var_name), m_N(N){
  }

  template <typename IT, typename DT>
  BodyForce<IT, DT>::~BodyForce()
  {}

  template <typename IT, typename DT>
  void BodyForce<IT, DT>::problemSetup( ProblemSpecP& db ){
    proc0cout << "WARNING: ParticleModels BodyForce needs to be made consistent with DQMOM models and use correct DW, use model at your own risk."
      << "\n" << "\n" << "\n" << "\n" << "\n" << "\n" << "\n" << "\n" << "\n" << "\n"<< std::endl;
    m_base_density_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_DENSITY);

    db->require("direction",m_direction);
    m_gas_density_name = "densityCP";
    const ProblemSpecP params_root = db->getRootNode();
    Vector gravity;
    if (params_root->findBlock("PhysicalConstants")) {
      ProblemSpecP db_phys = params_root->findBlock("PhysicalConstants");
      db_phys->require("gravity", gravity);
    }

    if ( m_direction == "x" ) {
      m_gravity_component = gravity.x();
    } else if ( m_direction == "y" ) {
      m_gravity_component = gravity.y();
    } else if ( m_direction == "z" ) {
      m_gravity_component = gravity.z();
    }
  }

  //======INITIALIZATION:
  template <typename IT, typename DT>
  void BodyForce<IT, DT>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

    for ( int i = 0; i < m_N; i++ ){
      const std::string name = get_name(i, m_base_var_name);
      register_variable( name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );

    }
  }

  template <typename IT, typename DT>
  void BodyForce<IT,DT>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    for ( int ienv = 0; ienv < m_N; ienv++ ){
      const std::string name = get_name(ienv, m_base_var_name);

      DT& model_value = *(tsk_info->get_uintah_field<DT>(name));
      Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
      Uintah::parallel_for( range, [&](int i, int j, int k){
        model_value(i,j,k) = 0.0;
      });

    }
  }

  //======TIME STEP INITIALIZATION:
  template <typename IT, typename DT>
  void BodyForce<IT, DT>::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){
  }

  template <typename IT, typename DT>
  void BodyForce<IT,DT>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

  //======TIME STEP EVALUATION:
  template <typename IT, typename DT>
  void BodyForce<IT, DT>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){

    for ( int i = 0; i < m_N; i++ ){
      //dependent variables(s) or model values
      const std::string name = get_name(i, m_base_var_name);
      register_variable( name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

      //independent variables
      const std::string density_name = get_name( i, m_base_density_name );
      register_variable( density_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    }

    register_variable( m_gas_density_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  }

  template <typename IT, typename DT>
  void BodyForce<IT,DT>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    typedef typename ArchesCore::VariableHelper<IT>::ConstType CIT;
    typedef typename ArchesCore::VariableHelper<DT>::ConstType CDT;

    CIT& rhoG = *(tsk_info->get_const_uintah_field<CIT>(m_gas_density_name));

    for ( int ienv = 0; ienv < m_N; ienv++ ){

      const std::string name = get_name(ienv, m_base_var_name);
      DT& model_value = *(tsk_info->get_uintah_field<DT>(name));

      const std::string density_name = get_name( ienv, m_base_density_name );
      CDT& rhoP = *(tsk_info->get_const_uintah_field<CDT>(density_name));


      Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
      Uintah::parallel_for( range, [&](int i, int j, int k){

        model_value(i,j,k) = m_gravity_component * ( rhoP(i,j,k)  - rhoG(i,j,k) ) / rhoP(i,j,k);

      });

    }
  }
}
#endif
