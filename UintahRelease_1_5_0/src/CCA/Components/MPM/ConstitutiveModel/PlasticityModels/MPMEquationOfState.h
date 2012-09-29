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

#ifndef __EQUATION_OF_STATE_H__
#define __EQUATION_OF_STATE_H__

#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include "PlasticityState.h"
#include <Core/Math/Matrix3.h>


namespace Uintah {

  ////////////////////////////////////////////////////////////////////////////
  /*! 
    \class MPMEquationOfState
    \brief Abstract base class for solid equations of state
    \author Biswajit Banerjee, \n
    C-SAFE and Department of Mechanical Engineering, \n
    University of Utah \n

  */
  ////////////////////////////////////////////////////////////////////////////

  class MPMEquationOfState {

  protected:
    double d_bulk;

  public:
         
    MPMEquationOfState();
    virtual ~MPMEquationOfState();

    virtual void outputProblemSpec(ProblemSpecP& ps) = 0;
         
    void setBulkModulus(const double& bulk) {d_bulk = bulk;}
    double initialBulkModulus() {return d_bulk;}

    ////////////////////////////////////////////////////////////////////////
    /*! Calculate the hydrostatic component of stress (pressure)
        using an equation of state */
    ////////////////////////////////////////////////////////////////////////
    virtual double computePressure(const MPMMaterial* matl,
                                   const PlasticityState* state,
                                   const Matrix3& deformGrad,
                                   const Matrix3& rateOfDeformation,
                                   const double& delT) = 0;

    ////////////////////////////////////////////////////////////////////////
    /*! Calculate the pressure without considering internal energy 
        (option 1)*/
    ////////////////////////////////////////////////////////////////////////
    virtual double computePressure(const double& rho_orig,
                                   const double& rho_cur) = 0;

    ////////////////////////////////////////////////////////////////////////
    /*! Calculate the pressure without considering internal energy 
        (option 2).  Also compute dp/drho and c^2. */
    ////////////////////////////////////////////////////////////////////////
    virtual void computePressure(const double& rho_orig,
                                 const double& rho_cur,
                                 double& pressure,
                                 double& dp_drho,
                                 double& csquared) = 0;


    /*! Calculate the derivative of \f$p(J)\f$ wrt \f$J\f$ 
        where \f$J = det(F) = rho_0/rho\f$ */
    virtual double eval_dp_dJ(const MPMMaterial* matl,
                              const double& delF,
                              const PlasticityState* state) = 0;

    // Calculate rate of temperature change due to compression/expansion
    virtual double computeIsentropicTemperatureRate(const double T,
                                                    const double rho_0,
                                                    const double rho_cur,
                                                    const double Dtrace);

    ////////////////////////////////////////////////////////////////////////
    /*! Calculate the tangent bulk modulus */
    ////////////////////////////////////////////////////////////////////////
    virtual double computeBulkModulus(const double& rho_orig,
                                      const double& rho_cur) = 0;

    ////////////////////////////////////////////////////////////////////////
    /*! Calculate the accumulated strain energy */
    ////////////////////////////////////////////////////////////////////////
    virtual double computeStrainEnergy(const double& rho_orig,
                                       const double& rho_cur) = 0;

    ////////////////////////////////////////////////////////////////////////
    /*! Calculate the mass density given a pressure */
    ////////////////////////////////////////////////////////////////////////
    virtual double computeDensity(const double& rho_orig,
                                  const double& pressure) = 0;

  };
} // End namespace Uintah
      


#endif  // __EQUATION_OF_STATE_H__

