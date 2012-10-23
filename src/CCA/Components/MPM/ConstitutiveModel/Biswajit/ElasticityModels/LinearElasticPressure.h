/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __LINEAR_ELASTIC_PRESSURE_MODEL_H__
#define __LINEAR_ELASTIC_PRESSURE_MODEL_H__

#include "PressureModel.h"

namespace Uintah {

  ////////////////////////////////////////////////////////////////////////////
  /*! 
    \class LinearElasticPressure
    \brief Compute pressure for ain isotropic linear elastic (hypoelastic) material.
    \author Biswajit Banerjee, \n
  */
  ////////////////////////////////////////////////////////////////////////////

  class LinearElasticPressure : public PressureModel {

  private:

    double d_kappa;    // Bulk modulus
  
    LinearElasticPressure& operator=(const LinearElasticPressure& pm);

  public:
         
    LinearElasticPressure(ProblemSpecP& ps);
    LinearElasticPressure(const LinearElasticPressure* pm);

    virtual ~LinearElasticPressure();

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
      


#endif  // __LINEAR_ELASTIC_PRESSURE_MODEL_H__

