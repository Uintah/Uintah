#ifndef UT_TransportFactory_h
#define UT_TransportFactory_h

#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <string>

namespace Uintah{

  class TransportFactory : public TaskFactoryBase {

  public:

    TransportFactory();
    ~TransportFactory();

    void register_all_tasks( ProblemSpecP& db );

    void build_all_tasks( ProblemSpecP& db );

    std::vector<std::string> retrieve_task_subset( const std::string subset ) {

      if ( subset == "scalar_rhs_builders" ){
        return _scalar_builders;
      } else if ( subset == "scalar_fe_update" ){
        return _scalar_update;
      } else if ( subset == "scalar_ssp" ){
        return _scalar_ssp;
      } else if ( subset == "mom_rhs_builders"){
        return _momentum_builders;
      } else if ( subset == "momentum_fe_update" ){
        return _momentum_update;
      } else if ( subset == "mom_ssp"){
        return _momentum_spp;
      } else if ( subset == "momentum_construction" ){
        return _momentum_solve;
      } else if ( subset == "scalar_psi_builders" ){
        return _scalar_compute_psi;
      } else if ( subset == "momentum_psi_builders" ){
        return _momentum_compute_psi;
      } else if ( subset == "pressure_eqn" ){
        return _pressure_eqn;
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

    std::vector<std::string> _scalar_builders;
    std::vector<std::string> _momentum_builders;
    std::vector<std::string> _scalar_update;
    std::vector<std::string> _momentum_update;
    std::vector<std::string> _scalar_ssp;
    std::vector<std::string> _momentum_spp;
    std::vector<std::string> _scalar_compute_psi;
    std::vector<std::string> _momentum_compute_psi;
    std::vector<std::string> _pressure_eqn;
    std::vector<std::string> _momentum_solve;

    bool m_pack_transport_construction_tasks{false}; 

  };
}
#endif
