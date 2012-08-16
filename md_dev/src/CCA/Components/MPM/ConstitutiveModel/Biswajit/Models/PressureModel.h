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


#ifndef __BB_EQUATION_OF_STATE_H__
#define __BB_EQUATION_OF_STATE_H__

#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include "ModelState.h"
#include <Core/Math/Matrix3.h>


namespace UintahBB {

  ////////////////////////////////////////////////////////////////////////////
  /*! 
    \class PressureModel
    \brief Abstract base class for solid equations of state
    \author Biswajit Banerjee, \n
    C-SAFE and Department of Mechanical Engineering, \n
    University of Utah \n
    Copyright (C) 2002-2003 University of Utah

  */
  ////////////////////////////////////////////////////////////////////////////

  class PressureModel {

  protected:
    double d_bulk;

  public:
         
    PressureModel();
    virtual ~PressureModel();

    virtual void outputProblemSpec(Uintah::ProblemSpecP& ps) = 0;
         
    void setBulkModulus(const double& bulk) {d_bulk = bulk;}
    double initialBulkModulus() {return d_bulk;}

    ////////////////////////////////////////////////////////////////////////
    /*! Calculate the hydrostatic component of stress (pressure)
        using an equation of state */
    ////////////////////////////////////////////////////////////////////////
    virtual double computePressure(const Uintah::MPMMaterial* matl,
                                   const ModelState* state,
                                   const Uintah::Matrix3& deformGrad,
                                   const Uintah::Matrix3& rateOfDeformation,
                                   const double& delT) = 0;

    // Compute bulk modulus
    virtual double computeBulkModulus(const ModelState* state) = 0;

    // Compute strain energy
    virtual double computeStrainEnergy(const ModelState* state)  = 0;

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
    virtual double eval_dp_dJ(const Uintah::MPMMaterial* matl,
                              const double& delF,
                              const ModelState* state) = 0;

    ////////////////////////////////////////////////////////////////////////
    /*! Calculate the derivative of p with respect to epse_v
        where epse_v = tr(epse)
              epse = total elastic strain */
    ////////////////////////////////////////////////////////////////////////
    virtual double computeDpDepse_v(const ModelState* state) const = 0;

    ////////////////////////////////////////////////////////////////////////
    /*! Calculate the derivative of p with respect to epse_s
        where epse_s = sqrt{2}{3} ||ee||
              ee = epse - 1/3 tr(epse) I
              epse = total elastic strain */
    ////////////////////////////////////////////////////////////////////////
    virtual double computeDpDepse_s(const ModelState* state) const = 0;

    // Calculate rate of temperature change due to compression/expansion
    virtual double computeIsentropicTemperatureRate(const double T,
                                                    const double rho_0,
                                                    const double rho_cur,
                                                    const double Dtrace);

    ////////////////////////////////////////////////////////////////////////
    /*! Calculate the tangent bulk modulus */
    ////////////////////////////////////////////////////////////////////////
    virtual double computeInitialBulkModulus() = 0;
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
      


#endif  // __BB_EQUATION_OF_STATE_H__

