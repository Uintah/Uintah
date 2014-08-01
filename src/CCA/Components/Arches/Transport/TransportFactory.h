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

  protected: 


  private: 
  
  
  
  };
}
#endif 
