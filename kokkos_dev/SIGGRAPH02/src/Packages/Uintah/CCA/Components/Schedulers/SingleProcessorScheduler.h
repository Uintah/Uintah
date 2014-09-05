#ifndef UINTAH_HOMEBREW_SinglePROCESSORSCHEDULER_H
#define UINTAH_HOMEBREW_SinglePROCESSORSCHEDULER_H

#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/CCA/Components/Schedulers/Relocate.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/TaskProduct.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <vector>
#include <map>

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
    virtual void execute( const ProcessorGroup * pg );
    virtual void executeTimestep( const ProcessorGroup * pc );
    virtual void executeRefine( const ProcessorGroup * pc );
    virtual void executeCoarsen( const ProcessorGroup * pc );
    virtual void finalizeTimestep(const GridP& grid);
    
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

  protected:
    virtual void actuallyCompile(const ProcessorGroup* pg);
  private:
    SPRelocate reloc;
    SingleProcessorScheduler& operator=(const SingleProcessorScheduler&);
    virtual void execute_tasks( const ProcessorGroup * pc, DataWarehouse* oldDW, DataWarehouse* newDW );

    virtual void verifyChecksum();

    SingleProcessorScheduler* m_parent;
  };
} // End namespace Uintah
   
#endif
