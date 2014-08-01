#ifndef UT_InitializeFactory_h
#define UT_InitializeFactory_h

#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <string>

namespace Uintah{ 

  class InitializeFactory : public TaskFactoryBase { 

  public: 

    InitializeFactory(); 
    ~InitializeFactory(); 

    void register_all_tasks( ProblemSpecP& db ); 

    void build_all_tasks( ProblemSpecP& db ); 

  protected: 


  private: 
  
  
  
  };
}
#endif 
