#ifndef Wasatch_TurbulenceParameters_h
#define Wasatch_TurbulenceParameters_h

#include <Core/ProblemSpec/ProblemSpec.h>

namespace Wasatch{
  /**
   *  \ingroup WasatchCore
   *  \struct TurbulenceModelsNames
   *  \author Tony Saad
   *  \date   June, 2012
   *
   *  \brief An enum listing the supported turbulence models. 
   */  
  enum TurbulenceModelsNames {
    SMAGORINSKY,
    DYNAMIC,
    WALE,
    NONE
  };
  
  /**
   *  \ingroup WasatchCore
   *  \struct TurbulenceParameters
   *  \author Tony Saad
   *  \date   June, 2012
   *
   *  \brief Holds some key parameters for supported turbulence models.
   */
  struct TurbulenceParameters {
    double eddyViscosityConstant;
    double kolmogorovScale;
    TurbulenceModelsNames turbulenceModelName;
  };
  
  void parse_turbulence_input(Uintah::ProblemSpecP turbulenceInputParams,
                              TurbulenceParameters& turbParams);
  
}

#endif // Wasatch_Turbulence_Parameters_h
