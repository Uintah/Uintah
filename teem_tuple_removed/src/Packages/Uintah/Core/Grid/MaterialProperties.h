
#ifndef Uintah_MaterialProperties_h
#define Uintah_MaterialProperties_h

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {
  class MaterialProperties {
  public:
    MaterialProperties();
    ~MaterialProperties();
    void parse(ProblemSpecP&);

    double Cp;
    //double Cv;
    //double gamma;
    //double viscosity;
    //double thermalConductivity;
    double molecularWeight;
  };
}

#endif
