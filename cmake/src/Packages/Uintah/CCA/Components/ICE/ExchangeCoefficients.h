#ifndef __EXCHANGE_COEFFICIENTS_H__
#define __EXCHANGE_COEFFICIENTS_H__

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <vector>

namespace Uintah {
  class ExchangeCoefficients {
  public:
    ExchangeCoefficients();
    ~ExchangeCoefficients();

    void problemSetup(ProblemSpecP& ps,
                      SimulationStateP& sharedState);
    void outputProblemSpec(ProblemSpecP& ps);
    
    
    bool convective();
    int conv_fluid_matlindex();
    int conv_solid_matlindex();
    vector<double> K_mom();
    vector<double> K_heat();
  private:
    vector<double> d_K_mom, d_K_heat;
    bool d_convective;
    int d_conv_fluid_matlindex;
    int d_conv_solid_matlindex;
    
  };


}



#endif
