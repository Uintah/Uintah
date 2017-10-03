#ifndef UT_TaskFactoryHelper_h
#define UT_TaskFactoryHelper_h

#include <CCA/Components/Arches/Task/TaskFactoryBase.h>

namespace Uintah { namespace ArchesCore{ 


  void inline find_and_schedule_task( const std::string task_name, 
                                      TaskInterface::TASK_TYPE type,
                                      const LevelP& level,
                                      SchedulerP& sched, 
                                      const MaterialSet* matls, 
                                      std::map<std::string,std::shared_ptr<TaskFactoryBase> > factories, 
                                      const int time_substep = 0 ){ 

    bool did_task = false; 

    for ( auto ifac = factories.begin(); ifac != factories.end(); ifac++ ){ 
  
      bool has_task = ifac->second->has_task( task_name ); 

      if ( has_task ){ 

        if ( !did_task ){ 

          ifac->second->schedule_task( task_name, type, level, sched, matls, time_substep ); 
          proc0cout << " Scheduling task: " << task_name << std::endl;
          did_task = true; 

        } else { 

          throw InvalidValue( "Error: This task is in two different factories: "+task_name, __FILE__, __LINE__ ); 

        } 

      } 

    }
  } 
}} //Uintah::ArchesCore 
#endif
