/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_HOMEBREW_RELOCATE_H
#define UINTAH_HOMEBREW_RELOCATE_H

#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <vector>

#include <sci_defs/mpi_defs.h> // For MPIPP_H on SGI

namespace Uintah {
  class DataWarehouse;
  class LoadBalancer;
  class ProcessorGroup;
  class Scheduler;
  class VarLabel;
  
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
                        const std::vector<vector<const VarLabel*> >& old_labels,
			const VarLabel* new_posLabel,
			const std::vector<std::vector<const VarLabel*> >& new_labels,
			const VarLabel* particleIDLabel,
			const MaterialSet* matls);

    const MaterialSet* getMaterialSet() const { return reloc_matls;}

  private:
    void relocateParticles(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw,
                           const Level* coarsestLevelwithParticles);
    void exchangeParticles(const ProcessorGroup*, 
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw, MPIScatterRecords* scatter_records, 
                           int total_reloc[3]);
    
    void findNeighboringPatches(const Patch* patch,
                                const Level* level,
                                const bool findFiner,
                                const bool findCoarser,
                                Patch::selectType& AllNeighborPatches);
   
    void finalizeCommunication();
    const VarLabel* reloc_old_posLabel;
    std::vector<std::vector<const VarLabel*> > reloc_old_labels;
    const VarLabel* reloc_new_posLabel;
    std::vector<std::vector<const VarLabel*> > reloc_new_labels;
    const VarLabel* particleIDLabel_;
    const MaterialSet* reloc_matls;
    LoadBalancer* lb;
    std::vector<char*> recvbuffers;
    std::vector<char*> sendbuffers;
    std::vector<MPI_Request> sendrequests;


  };
} // End namespace Uintah
   
#endif
