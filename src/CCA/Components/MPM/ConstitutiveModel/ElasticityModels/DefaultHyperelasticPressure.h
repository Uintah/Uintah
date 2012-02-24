/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef __DEFAULT_HYPERELASTIC_PRESSURE_MODEL_H__
#define __DEFAULT_HYPERELASTIC_PRESSURE_MODEL_H__

#include "PressureModel.h"

namespace Uintah {

  ////////////////////////////////////////////////////////////////////////////
  /*! 
    \class DefaultHyperelasticPressure
    \brief Compute pressure for a hyperelastic material with
            W = D1 (J-1)^2 = 0.5*kappa*(J-1)^2
            p = dW/dJ = kappa*(J-1)
            dp/dJ = d2W/dJ^2 = kappa
            k = p + J dp/dJ = kappa*(2J-1)
            c^2 = k/rho
    \author Biswajit Banerjee, \n
  */
  ////////////////////////////////////////////////////////////////////////////

  class DefaultHyperelasticPressure : public PressureModel {

  private:

    double d_kappa;    // Bulk modulus
  
    DefaultHyperelasticPressure& operator=(const DefaultHyperelasticPressure& pm);

  public:
         
    DefaultHyperelasticPressure(ProblemSpecP& ps);
    DefaultHyperelasticPressure(const DefaultHyperelasticPressure* pm);

    virtual ~DefaultHyperelasticPressure();

    virtual void outputProblemSpec(ProblemSpecP& ps);
         
    // Calculate the hydrostatic component of stress (pressure)
    double computePressure(const DeformationState* state);

    // Calculate the pressure without considering internal energy (option 1)
    double computePressure(const double& rho_orig,
                           const double& rho_cur);

    // Calculate the pressure without considering internal energy (option 2).  
    //   Also compute dp/drho and c^2. 
    void computePressure(const double& rho_orig,
                         const double& rho_cur,
                         double& pressure,
                         double& dp_drho,
                         double& csquared);

    // Calculate the tangent bulk modulus 
    double computeTangentBulkModulus(const double& rho_orig,
                                     const double& rho_cur);

    // Calculate the accumulated strain energy 
    double computeStrainEnergy(const double& pressure,
                               const DeformationState* state);

    // Calculate the mass density at a given pressure 
    double computeDensity(const double& rho_orig,
                          const double& pressure);

  };
} // End namespace Uintah
      


#endif  // __DEFAULT_HYPERELASTIC_PRESSURE_MODEL_H__

