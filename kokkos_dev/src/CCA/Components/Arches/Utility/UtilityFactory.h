#ifndef UT_UtilityFactory_h
#define UT_UtilityFactory_h

#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <string>

namespace Uintah{ 

  class UtilityFactory : public TaskFactoryBase { 

  public: 

    UtilityFactory(); 
    ~UtilityFactory(); 

    void register_all_tasks( ProblemSpecP& db ); 

    void build_all_tasks( ProblemSpecP& db ); 

  protected: 


  private: 
  
  
  
  };
}
#endif 
