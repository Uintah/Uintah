/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __PRESSURE_MODEL_H__
#define __PRESSURE_MODEL_H__

#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include "DeformationState.h"
#include <Core/Math/Matrix3.h>


namespace Uintah {

  ////////////////////////////////////////////////////////////////////////////
  /*! 
    \class PressureModel
    \brief Abstract base class for solid equations of state and other models
           for computing the pressure
    \author Biswajit Banerjee, \n
  */
  ////////////////////////////////////////////////////////////////////////////

  class PressureModel {

  public:
         
    PressureModel();
    virtual ~PressureModel();

    virtual void outputProblemSpec(ProblemSpecP& ps) = 0;
         
    // Calculate the hydrostatic component of stress (pressure)
    virtual double computePressure(const DeformationState* state) = 0;

    // Calculate the pressure without considering internal energy (option 1)
    virtual double computePressure(const double& rho_orig,
                                   const double& rho_cur) = 0;

    // Calculate the pressure without considering internal energy (option 2).  
    //   Also compute dp/drho and c^2. 
    virtual void computePressure(const double& rho_orig,
                                 const double& rho_cur,
                                 double& pressure,
                                 double& dp_drho,
                                 double& csquared) = 0;

    // Calculate the tangent bulk modulus 
    virtual double computeTangentBulkModulus(const double& rho_orig,
                                             const double& rho_cur) = 0;

    // Calculate the accumulated strain energy 
    virtual double computeStrainEnergy(const double& pressure,
                                       const DeformationState* state) = 0;

    // Calculate the mass density at a given pressure 
    virtual double computeDensity(const double& rho_orig,
                                  const double& pressure) = 0;

  };
} // End namespace Uintah
      


#endif  // __PRESSURE_MODEL_H__

