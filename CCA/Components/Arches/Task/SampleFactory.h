#ifndef UT_SampleFactory_h
#define UT_SampleFactory_h

#include <CCA/Components/Arches/Task/SampleFactory.h>
#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <string>

namespace Uintah{ 

  class SampleFactory : public TaskFactoryBase { 

  public: 

    SampleFactory(); 
    ~SampleFactory(); 

    void register_all_tasks( ProblemSpecP& db ); 

    void build_all_tasks( ProblemSpecP& db ); 

  protected: 


  private: 
  
  
  
  };
}
#endif 
