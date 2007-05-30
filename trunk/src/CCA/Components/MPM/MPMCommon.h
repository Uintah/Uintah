#ifndef UINTAH_HOMEBREW_MPM_COMMON_H
#define UINTAH_HOMEBREW_MPM_COMMON_H


#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <SCIRun/Core/Util/DebugStream.h>

#include <CCA/Components/MPM/share.h>
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
