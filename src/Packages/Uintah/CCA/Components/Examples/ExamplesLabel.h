
#ifndef Packages_Uintah_CCA_Components_Examples_ExamplesLabel_h
#define Packages_Uintah_CCA_Components_Examples_ExamplesLabel_h

namespace Uintah {
  class VarLabel;
  class ExamplesLabel {
  public:
    // For Poisson1
    const VarLabel* phi;
    const VarLabel* residual;

    // For Smoke
    const VarLabel* uvel;
    const VarLabel* vvel;
    const VarLabel* wvel;
    const VarLabel* pressure;

    ExamplesLabel();
    ~ExamplesLabel();
  };
}

#endif


