
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
    const VarLabel* temperature;
    const VarLabel* pressure;
    const VarLabel* ccvelocity;

    const VarLabel* xvelocity_matrix;
    const VarLabel* xvelocity_rhs;
    const VarLabel* yvelocity_matrix;
    const VarLabel* yvelocity_rhs;
    const VarLabel* zvelocity_matrix;
    const VarLabel* zvelocity_rhs;
    const VarLabel* density_matrix;
    const VarLabel* density_rhs;
    const VarLabel* pressure_matrix;
    const VarLabel* pressure_rhs;
    const VarLabel* temperature_matrix;
    const VarLabel* temperature_rhs;

    const VarLabel* ccvorticity;
    const VarLabel* ccvorticitymag;
    const VarLabel* vcforce;
    const VarLabel* NN;

    // For AMRSimpleCFD
    const VarLabel* pressure2;
    const VarLabel* pressure2_matrix;
    const VarLabel* pressure2_rhs;

    const VarLabel* pressure_gradient_mag;
    const VarLabel* temperature_gradient_mag;
    const VarLabel* density_gradient_mag;

    // For Burger
    const VarLabel* u;

    ExamplesLabel();
    ~ExamplesLabel();
  };
}

#endif
