#ifndef UINTAH_HOMEBREW_RELOCATE_H
#define UINTAH_HOMEBREW_RELOCATE_H

#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

#include <mpi.h>

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

   Relocate.h

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

  class MPIScatterRecords; // defined in .cc

  class Relocate {
  public:
    Relocate();
    virtual ~Relocate();
    
    //////////
    // Insert Documentation Here:
    void scheduleParticleRelocation(Scheduler*,
			const ProcessorGroup* pg,
			LoadBalancer* lb,
			const LevelP& level,
			const VarLabel* old_posLabel,
			const vector<vector<const VarLabel*> >& old_labels,
			const VarLabel* new_posLabel,
			const vector<vector<const VarLabel*> >& new_labels,
			const VarLabel* particleIDLabel,
			const MaterialSet* matls);

    const MaterialSet* getMaterialSet() const { return reloc_matls;}

  private:
    void relocateParticles(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw);
    void exchangeParticles(const ProcessorGroup*, 
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw, MPIScatterRecords* scatter_records, 
                           int total_reloc[3]);
    void finalizeCommunication();
    const VarLabel* reloc_old_posLabel;
    vector<vector<const VarLabel*> > reloc_old_labels;
    const VarLabel* reloc_new_posLabel;
    vector<vector<const VarLabel*> > reloc_new_labels;
    const VarLabel* particleIDLabel_;
    const MaterialSet* reloc_matls;
    LoadBalancer* lb;
    vector<char*> recvbuffers;
    vector<char*> sendbuffers;
    vector<MPI_Request> sendrequests;


  };
} // End namespace Uintah
   
#endif
