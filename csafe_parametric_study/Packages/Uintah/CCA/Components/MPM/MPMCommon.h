#ifndef UINTAH_HOMEBREW_MPM_COMMON_H
#define UINTAH_HOMEBREW_MPM_COMMON_H


#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Core/Util/DebugStream.h>

#include <Packages/Uintah/CCA/Components/MPM/share.h>
namespace Uintah {

  class ProcessorGroup;

  using namespace SCIRun;
  
  class SCISHARE MPMCommon {

  public:

    MPMCommon(const ProcessorGroup* myworld);
    virtual ~MPMCommon();

    virtual void materialProblemSetup(const ProblemSpecP& prob_spec,
                                      SimulationStateP& sharedState,
                                      MPMFlags* flags);


    virtual void printSchedule(const PatchSet* patches,
                               DebugStream& dbg,
                               const string& where);
  
    virtual void printSchedule(const LevelP& level,
                               DebugStream& dbg,
                               const string& where);
                     
    virtual void printTask(const PatchSubset* patches,
                           const Patch* patch,
                           DebugStream& dbg,
                           const string& where);

   protected:
    const ProcessorGroup* d_myworld;
    
  };
}

#endif
