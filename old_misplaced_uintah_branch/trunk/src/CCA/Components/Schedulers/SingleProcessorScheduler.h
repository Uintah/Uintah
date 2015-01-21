#ifndef UINTAH_HOMEBREW_SinglePROCESSORSCHEDULER_H
#define UINTAH_HOMEBREW_SinglePROCESSORSCHEDULER_H

#include <CCA/Components/Schedulers/SchedulerCommon.h>
#include <CCA/Ports/DataWarehouseP.h>
 
#include <Core/Grid/Task.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <map>
#include <sgi_stl_warnings_off.h>

namespace Uintah {
  using std::vector;

  class OnDemandDataWarehouse;
  class Task;

/**************************************

CLASS
   SingleProcessorScheduler
   
   Short description...

GENERAL INFORMATION

   SingleProcessorScheduler.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Scheduler_Brain_Damaged

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class SingleProcessorScheduler : public SchedulerCommon {
  public:
    SingleProcessorScheduler(const ProcessorGroup* myworld, Output* oport, SingleProcessorScheduler* parent = NULL);
    virtual ~SingleProcessorScheduler();

    //////////
    // Insert Documentation Here:
    virtual void execute(int tgnum = 0, int iteration = 0);
    
    virtual SchedulerP createSubScheduler();

  private:
    SingleProcessorScheduler& operator=(const SingleProcessorScheduler&);

    virtual void verifyChecksum();

    SingleProcessorScheduler* m_parent;
  };
} // End namespace Uintah
   
#endif
