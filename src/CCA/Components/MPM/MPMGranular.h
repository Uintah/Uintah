/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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


#ifndef UINTAH_HOMEBREW_MPM_GRANULAR_H
#define UINTAH_HOMEBREW_MPM_GRANULAR_H



#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/MaterialManager.h>
#include <CCA/Ports/Scheduler.h>
#include <vector>
#include <map>

/**************************************

CLASS
   MPMGranular
   
   Short description...

GENERAL INFORMATION

   MPMGranular.h

    

KEYWORDS
   MPMGranular

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/
namespace Uintah {
  typedef int particleIndex;
  typedef int particleId;

  class GeometryObject;
  class Patch;
  class DataWarehouse;
  class MPMFlags;
  class MPMLabel;
  class ParticleSubset;
  class VarLabel;

  class MPMGranular {
  public:
    
    MPMGranular(MaterialManagerP& ss, MPMFlags* flags);

    virtual ~MPMGranular();

    virtual void MPMGranularProblemSetup(const ProblemSpecP& prob_spec,
                                          MPMFlags* flags);
    void scheduleGranularMPM(SchedulerP           & sched,
                             const PatchSet       * patches,
                             const MaterialSet    * matls );

    void GranularMPM(const ProcessorGroup  * pg,
                            const PatchSubset     * patches,
                            const MaterialSubset  * matls,
                            DataWarehouse         * old_dw,
                            DataWarehouse         * new_dw );
                            
                            
    virtual void insertGranularParticles(const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw);  
                               
    virtual void scheduleInsertGranularParticles(SchedulerP&, 
                                       const PatchSet*,
                                       const MaterialSet*);   
                                       
    void readInsertGranularParticlesFile(std::string filename);                                                                               
    


       // The following are used iff the d_insertParticles flag is true.
  std::vector<double> d_IPTimes;
  std::vector<double> d_IPColor;
  std::vector<Vector> d_IPTranslate;
  std::vector<Vector> d_IPVelNew;
   std::vector<double> d_IPdwi;//
   
  protected:

    MPMLabel* d_lb;
    MPMFlags* d_flags;
    MaterialManagerP d_materialManager;
    int NGN, NGP;
        
  };

} // End of namespace Uintah

#endif // __UINTAH_HOMEBREW_MPM_GRANULAR_H

















