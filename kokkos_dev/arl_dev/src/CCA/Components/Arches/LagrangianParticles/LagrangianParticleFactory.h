#ifndef UT_LagrangianParticleFactory_h
#define UT_LagrangianParticleFactory_h

#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <string>

namespace Uintah{ 

  class ArchesParticlesHelper; 

  class LagrangianParticleFactory : public TaskFactoryBase { 

  public: 

    LagrangianParticleFactory(); 
    ~LagrangianParticleFactory(); 

    void register_all_tasks( ProblemSpecP& db ); 

    void build_all_tasks( ProblemSpecP& db ); 

    void set_particle_helper( ArchesParticlesHelper* part_helper ){ _part_helper = part_helper; }


  protected: 


  private: 

    ArchesParticlesHelper* _part_helper;  
  
  
  };
}
#endif 
