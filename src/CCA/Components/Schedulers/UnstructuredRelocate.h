/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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

#ifndef UINTAH_HOMEBREW_UNSTRUCTURED_RELOCATE_H
#define UINTAH_HOMEBREW_UNSTRUCTURED_RELOCATE_H

#include <Core/Grid/UnstructuredLevelP.h>
#include <Core/Grid/UnstructuredPatch.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Parallel/UintahMPI.h>

#include <vector>

namespace Uintah {
  class UnstructuredDataWarehouse;
  class UnstructuredLoadBalancer;
  class ProcessorGroup;
  class UnstructuredScheduler;
  class UnstructuredVarLabel;
  
/**************************************

CLASS
   MPIRelocate
   
   Short description...

GENERAL INFORMATION

   UnstructuredRelocate.h

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

class UnstructuredMPIScatterRecords; // defined in .cc

class UnstructuredRelocate {

  public:

    UnstructuredRelocate(){};

    virtual ~UnstructuredRelocate();
    
    //////////
    // Insert Documentation Here:
    void scheduleParticleRelocation( UnstructuredScheduler                                        *,
                                     const ProcessorGroup                             * pg,
                                     UnstructuredLoadBalancer                                     * lb,
                                     const UnstructuredLevelP                                     & level,
                                     const UnstructuredVarLabel                                   * old_posLabel,
                                     const std::vector<std::vector<const UnstructuredVarLabel*> > & old_labels,
                                     const UnstructuredVarLabel                                   * new_posLabel,
                                     const std::vector<std::vector<const UnstructuredVarLabel*> > & new_labels,
                                     const UnstructuredVarLabel                                   * particleIDLabel,
                                     const MaterialSet                                * matls );
    //////////
    // Schedule particle relocation without the need to provide pre-relocation labels. Warning: This
    // is experimental and has not been fully tested yet. Use with caution (tsaad).
    void scheduleParticleRelocation(UnstructuredScheduler                                        *,
                                    const ProcessorGroup                             * pg,
                                    UnstructuredLoadBalancer                                 * lb,
                                    const UnstructuredLevelP                                     & level,
                                    const UnstructuredVarLabel                                   * posLabel,
                                    const std::vector<std::vector<const UnstructuredVarLabel*> > & otherLabels,
                                    const MaterialSet                                * matls);

    const MaterialSet* getMaterialSet() const { return reloc_matls;}


  private:

    // varlabels created for the modifies version of relocation
    std::vector<const Uintah::UnstructuredVarLabel*> destroyMe_;
    
    //////////
    // Callback function for particle relocation that doesn't use pre-Relocation variables.
    void relocateParticlesModifies(const ProcessorGroup*,
                                   const UnstructuredPatchSubset* patches,
                                   const MaterialSubset* matls,
                                   UnstructuredDataWarehouse* old_dw,
                                   UnstructuredDataWarehouse* new_dw,
                                   const UnstructuredLevel* coarsestLevelwithParticles);

    void relocateParticles(const ProcessorGroup*,
                           const UnstructuredPatchSubset* patches,
                           const MaterialSubset* matls,
                           UnstructuredDataWarehouse* old_dw,
                           UnstructuredDataWarehouse* new_dw,
                           const UnstructuredLevel* coarsestLevelwithParticles);

    void exchangeParticles( const ProcessorGroup    *, 
                            const UnstructuredPatchSubset       * patches,
                            const MaterialSubset    * matls,
                                  UnstructuredDataWarehouse     * old_dw,
                                  UnstructuredDataWarehouse     * new_dw,
                                  UnstructuredMPIScatterRecords * scatter_records, 
                                  int                 total_reloc[3] );
    
    void findNeighboringPatches( const UnstructuredPatch       * patch,
                                 const UnstructuredLevel       * level,
                                 const bool          findFiner,
                                 const bool          findCoarser,
                                 UnstructuredPatch::selectType & AllNeighborPatches);
   
    void finalizeCommunication();

    const UnstructuredVarLabel                             * reloc_old_posLabel{ nullptr };
    std::vector<std::vector<const UnstructuredVarLabel*> >   reloc_old_labels;
    const UnstructuredVarLabel                             * reloc_new_posLabel{ nullptr };
    std::vector<std::vector<const UnstructuredVarLabel*> >   reloc_new_labels;
    const UnstructuredVarLabel                             * particleIDLabel_{   nullptr };
    const MaterialSet                          * reloc_matls{        nullptr };
    UnstructuredLoadBalancer                               * m_lb{               nullptr };
    std::vector<char*>                          recvbuffers;
    std::vector<char*>                          sendbuffers;
    std::vector<MPI_Request>                    sendrequests;

};

} // End namespace Uintah
   
#endif
