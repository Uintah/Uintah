
#ifndef Packages_Uintah_CCA_Components_Examples_ExamplesLabel_h
#define Packages_Uintah_CCA_Components_Examples_ExamplesLabel_h

namespace Uintah {
  class VarLabel;
  class ExamplesLabel {
  public:
    // For Poisson1
    const VarLabel* phi;
    const VarLabel* residual;

    // For SimpleCFD
    const VarLabel* bctype;
    const VarLabel* xvelocity;
    const VarLabel* yvelocity;
    const VarLabel* zvelocity;
    const VarLabel* density;

    // For Burger
    const VarLabel* u;

    ExamplesLabel();
    ~ExamplesLabel();
  };
}

#endif


