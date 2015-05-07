#ifndef UT_LagrangianParticleFactory_h
#define UT_LagrangianParticleFactory_h

#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <string>

namespace Uintah{ 

  class LagrangianParticleFactory : public TaskFactoryBase { 

  public: 

    LagrangianParticleFactory(); 
    ~LagrangianParticleFactory(); 

    void register_all_tasks( ProblemSpecP& db ); 

    void build_all_tasks( ProblemSpecP& db ); 

    std::vector<std::string> retrieve_task_subset(const std::string subset){
      throw InvalidValue("Error: Accessing the base class implementation of retrieve_task_subset, which means there is no implementation for this factory.",__FILE__,__LINE__); 
    } 

  protected: 


  private: 

  
  };
}
#endif 
