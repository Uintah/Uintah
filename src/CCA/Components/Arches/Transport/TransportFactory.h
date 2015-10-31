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
      } else if ( subset == "mom_fe_update" ){
        return _momentum_update;
      } else if ( subset == "mom_ssp"){
        return _momentum_spp;
      } else {
        throw InvalidValue("Error: Task subset not recognized for TransportFactory.",__FILE__,__LINE__);
      }

    }

  private:

    std::vector<std::string> _scalar_builders;
    std::vector<std::string> _momentum_builders;
    std::vector<std::string> _scalar_update;
    std::vector<std::string> _momentum_update;
    std::vector<std::string> _scalar_ssp;
    std::vector<std::string> _momentum_spp;

  };
}
#endif
