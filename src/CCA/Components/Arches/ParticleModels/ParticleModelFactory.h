#ifndef UT_ParticleModelFactory_h
#define UT_ParticleModelFactory_h

#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <string>

namespace Uintah{ 

  class ParticleModelFactory : public TaskFactoryBase { 

  public: 

    ParticleModelFactory(); 
    ~ParticleModelFactory(); 

    void register_all_tasks( ProblemSpecP& db ); 

    void build_all_tasks( ProblemSpecP& db ); 

  protected: 


  private: 
  
  
  
  };
}
#endif 
