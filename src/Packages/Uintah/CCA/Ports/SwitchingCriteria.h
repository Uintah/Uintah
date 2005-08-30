#ifndef UINTAH_HOMEBREW_SwitchingCriteria_H
#define UINTAH_HOMEBREW_SwitchingCriteria_H

#include <Packages/Uintah/Core/Parallel/UintahParallelPort.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  class SwitchingCriteria : public UintahParallelPort {
    
  public:
    
    SwitchingCriteria();
    virtual ~SwitchingCriteria();

    virtual void problemSetup(const ProblemSpecP& params,
                              SimulationStateP& state) = 0;

    virtual void scheduleSwitchTest(const LevelP& level, SchedulerP& sched)
      {};

  private:
    
    SwitchingCriteria(const SwitchingCriteria&);
    SwitchingCriteria& operator=(const SwitchingCriteria&);
    
  };
}

#endif
