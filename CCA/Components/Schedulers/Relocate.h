#ifndef UINTAH_HOMEBREW_RELOCATE_H
#define UINTAH_HOMEBREW_RELOCATE_H

#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  class DataWarehouse;
  class LoadBalancer;
  class ProcessorGroup;
  class Scheduler;
  class VarLabel;
  using namespace std;

/**************************************

CLASS
   MPIRelocate
   
   Short description...

GENERAL INFORMATION

   MPIScheduler.h

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

  class Relocate {
  public:
    Relocate();
    virtual ~Relocate();
    
    //////////
    // Insert Documentation Here:
    virtual void scheduleParticleRelocation(Scheduler*,
					    const ProcessorGroup* pg,
					    LoadBalancer* lb,
					    const LevelP& level,
					    const VarLabel* old_posLabel,
					    const vector<vector<const VarLabel*> >& old_labels,
					    const VarLabel* new_posLabel,
					    const vector<vector<const VarLabel*> >& new_labels,
					    const VarLabel* particleIDLabel,
					    const MaterialSet* matls);

    const MaterialSet* getMaterialSet() { return reloc_matls;}

  protected:
    virtual void relocateParticles(const ProcessorGroup*,
				   const PatchSubset* patches,
				   const MaterialSubset* matls,
				   DataWarehouse* old_dw,
				   DataWarehouse* new_dw) = 0;
    const VarLabel* reloc_old_posLabel;
    vector<vector<const VarLabel*> > reloc_old_labels;
    const VarLabel* reloc_new_posLabel;
    vector<vector<const VarLabel*> > reloc_new_labels;
    const VarLabel* particleIDLabel_;
    const MaterialSet* reloc_matls;
    LoadBalancer* lb;
  };

  class MPIRelocate : public Relocate {
  public:
    MPIRelocate();
    virtual ~MPIRelocate();
    
  private:
    virtual void relocateParticles(const ProcessorGroup*,
				   const PatchSubset* patches,
				   const MaterialSubset* matls,
				   DataWarehouse* old_dw,
				   DataWarehouse* new_dw);
  };

  class SPRelocate : public Relocate {
  public:
    SPRelocate();
    virtual ~SPRelocate();
    
  private:
    virtual void relocateParticles(const ProcessorGroup*,
				   const PatchSubset* patches,
				   const MaterialSubset* matls,
				   DataWarehouse* old_dw,
				   DataWarehouse* new_dw);
  };
} // End namespace Uintah
   
#endif
