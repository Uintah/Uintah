#ifndef UINTAH_HOMEBREW_SinglePROCESSORSCHEDULER_H
#define UINTAH_HOMEBREW_SinglePROCESSORSCHEDULER_H

#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/CCA/Components/Schedulers/Relocate.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/TaskProduct.h>
#include <Packages/Uintah/Core/Grid/Task.h>
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
    virtual void execute();
    
    virtual SchedulerP createSubScheduler();
    //////////
    // Insert Documentation Here:
    virtual void scheduleParticleRelocation(const LevelP& level,
					    const VarLabel* old_posLabel,
					    const vector<vector<const VarLabel*> >& old_labels,
					    const VarLabel* new_posLabel,
					    const vector<vector<const VarLabel*> >& new_labels,
					    const VarLabel* particleIDLabel,
					    const MaterialSet* matls);

    virtual const MaterialSet* getMaterialSet(){return reloc_.getMaterialSet();}
  protected:
    virtual void actuallyCompile();
  private:
    SPRelocate reloc_;
    SingleProcessorScheduler& operator=(const SingleProcessorScheduler&);

    virtual void verifyChecksum();

    SingleProcessorScheduler* m_parent;
  };
} // End namespace Uintah
   
#endif
