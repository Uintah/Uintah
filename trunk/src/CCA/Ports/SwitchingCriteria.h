#ifndef UINTAH_HOMEBREW_SwitchingCriteria_H
#define UINTAH_HOMEBREW_SwitchingCriteria_H

#include <Core/Parallel/UintahParallelPort.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Ports/SchedulerP.h>
#include <Core/Grid/LevelP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <CCA/Ports/share.h>

namespace Uintah {

  class SCISHARE SwitchingCriteria : public UintahParallelPort {
    
  public:
    
    SwitchingCriteria();
    virtual ~SwitchingCriteria();

    virtual void problemSetup(const ProblemSpecP& params,
                              const ProblemSpecP& restart_prob_spec,
                              SimulationStateP& state) = 0;

    virtual void scheduleInitialize(const LevelP& level, SchedulerP& sched)
      {};
    virtual void scheduleSwitchTest(const LevelP& level, SchedulerP& sched)
      {};

  private:
    
    SwitchingCriteria(const SwitchingCriteria&);
    SwitchingCriteria& operator=(const SwitchingCriteria&);
    
  };
}

#endif
