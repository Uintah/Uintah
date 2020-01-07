#ifndef UT_TransportFactory_h
#define UT_TransportFactory_h

#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <string>

namespace Uintah{

  class TransportFactory : public TaskFactoryBase {

  public:

    TransportFactory( const ApplicationCommon* arches );
    ~TransportFactory();

    void register_all_tasks( ProblemSpecP& db );

    void build_all_tasks( ProblemSpecP& db );

    std::vector<std::string> retrieve_task_subset( const std::string subset ) {

    //SCALARS--------------------------------------------------------------------------------
      if ( subset == "scalar_rhs_builders" ){
        return _scalar_builders;
      } else if ( subset == "prop_scalar_rhs_builders"){
        return _prop_scalar_builders;
      } else if ( subset == "prop_diffusion_flux_builders"){
        return _prop_scalar_diffusion;
      } else if ( subset == "diffusion_flux_builders"){
        return _scalar_diffusion;
      } else if ( subset == "prop_scalar_update"){
        return _prop_scalar_update;
      } else if ( subset == "scalar_update" ){
        return _scalar_update;
      } else if ( subset == "rk_time_ave" ){
        return _rk_time_ave;
      } else if ( subset == "prop_rk_time_ave"){
        return _prop_rk_time_ave;
    //MOMENTUM------------------------------------------------------------------------------
      } else if ( subset == "momentum_stress_tensor" ){
        return _momentum_stress_tensor;
      } else if ( subset == "mom_rhs_builders"){
        return _momentum_builders;
      } else if ( subset == "momentum_fe_update" ){
        return _momentum_update;
      } else if ( subset == "mom_ssp"){
        return _momentum_spp;
      } else if ( subset == "momentum_conv" ){
        return _momentum_conv;
    //PRESSURE------------------------------------------------------------------------------
      } else if ( subset == "pressure_eqn" ){
        return _pressure_eqn;
    //DQMOM---------------------------------------------------------------------------------
      } else if ( subset == "dqmom_rhs_builders"){
        return _dqmom_builders;
      } else if ( subset == "dqmom_diffusion_flux_builders"){
        return _dqmom_compute_diff;
      } else if ( subset == "dqmom_fe_update"){
        return _dqmom_fe_update;
    //SCALING/WEIGHTING----------------------------------------------------------------------
      } else if ( subset == "dqmom_ic_from_wic" ){
        return _ic_from_w_ic;
      } else if ( subset == "phi_from_rho_phi"){
        return _phi_from_rho_phi;
      } else if ( subset == "prop_phi_from_rho_phi"){
        return _prop_phi_from_rho_phi;
      } else if ( subset == "u_from_rhou"){
        return _u_from_rho_u;
      } else if ( subset == _all_tasks_str ){
        return _active_tasks;
      } else {
        throw InvalidValue("Error: Task subset not recognized for TransportFactory: "+subset,
          __FILE__,__LINE__);
      }

    }

    void schedule_initialization( const LevelP& level,
                                  SchedulerP& sched,
                                  const MaterialSet* matls,
                                  bool doing_restart );

  private:

    std::vector<std::string> _momentum_stress_tensor;
    std::vector<std::string> _rk_time_ave;
    std::vector<std::string> _scalar_builders;
    std::vector<std::string> _momentum_builders;
    std::vector<std::string> _scalar_update;
    std::vector<std::string> _momentum_update;
    std::vector<std::string> _scalar_ssp;
    std::vector<std::string> _momentum_spp;
    std::vector<std::string> _pressure_eqn;
    std::vector<std::string> _momentum_conv;
    std::vector<std::string> _dqmom_builders;
    std::vector<std::string> _dqmom_fe_update;
    std::vector<std::string> _dqmom_compute_diff;
    std::vector<std::string> _scalar_diffusion;
    std::vector<std::string> _ic_from_w_ic;
    std::vector<std::string> _u_from_rho_u;
    std::vector<std::string> _phi_from_rho_phi;
    std::vector<std::string> _prop_phi_from_rho_phi;
    std::vector<std::string> _prop_rk_time_ave;
    std::vector<std::string> _prop_scalar_update;
    std::vector<std::string> _prop_scalar_builders;
    std::vector<std::string> _prop_scalar_diffusion;

    std::string m_u_vel_name;                             ///<Name of u velocity
    std::string m_v_vel_name;                             ///<Name of v velocity
    std::string m_w_vel_name;                             ///<Name of w velocity
    std::string m_density_name;                           ///<Name of rho

    bool m_pack_transport_construction_tasks{false};

    void register_DQMOM( ProblemSpecP db );

    std::string m_dqmom_grp_name{"dqmom_eqns"};

  };
}
#endif
