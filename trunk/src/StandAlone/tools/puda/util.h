
#include <vector>

namespace Uintah {

  void findTimestep_loopLimits( const bool tslow_set, 
                                const bool tsup_set,
                                const std::vector<double> times,
                                unsigned long & time_step_lower,
                                unsigned long & time_step_upper );
}

