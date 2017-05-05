/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 * Copyright (c) 2013-2014 Callaghan Innovation, New Zealand
 * Copyright (c) 2015 Parresia Research Limited, New Zealand
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

#ifndef __BB_EQUATION_OF_STATE_H__
#define __BB_EQUATION_OF_STATE_H__

#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/ConstitutiveModel/ArenaSoilBanerjeeBrannon/Models/ModelStateBase.h>

#include <Core/Math/Matrix3.h>


namespace Vaango {

  using ParameterDict = std::map<std::string, double>;

  ////////////////////////////////////////////////////////////////////////////
  /*! 
    \class PressureModel
    \brief Abstract base class for solid equations of state
    \author Biswajit Banerjee, \n
    C-SAFE and Department of Mechanical Engineering, \n
    University of Utah \n

  */
  ////////////////////////////////////////////////////////////////////////////

  class PressureModel {

  protected:
    double d_bulkModulus;

  public:
         
    PressureModel();
    virtual ~PressureModel();

    virtual void outputProblemSpec(Uintah::ProblemSpecP& ps) = 0;
         
    /////////////////////////////////////////////////////////////////////////
    /*!
      \brief Get the model parameters
    */
    /////////////////////////////////////////////////////////////////////////
    virtual std::map<std::string, double> getParameters() const = 0 ;

    void setBulkModulus(const double& bulk) {d_bulkModulus = bulk;}
    double initialBulkModulus() {return d_bulkModulus;}
    double getBulkModulus() const {return d_bulkModulus;}

    ////////////////////////////////////////////////////////////////////////
    /*! Calculate the hydrostatic component of stress (pressure)
      using an equation of state */
    ////////////////////////////////////////////////////////////////////////
    virtual
    double computePressure(const Uintah::MPMMaterial* matl,
                           const ModelStateBase* state,
                           const Uintah::Matrix3& deformGrad,
                           const Uintah::Matrix3& rateOfDeformation,
                           const double& delT) = 0;

    // Compute bulk modulus
    virtual
    double computeBulkModulus(const ModelStateBase* state) = 0;

    // Compute strain energy
    virtual
    double computeStrainEnergy(const ModelStateBase* state) = 0;

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
    virtual
    double eval_dp_dJ(const Uintah::MPMMaterial* matl,
                      const double& delF,
                      const ModelStateBase* state) = 0;

    ////////////////////////////////////////////////////////////////////////
    /*! Calculate the derivative of p with respect to epse_v
      where epse_v = tr(epse)
      epse = total elastic strain */
    ////////////////////////////////////////////////////////////////////////
    virtual
    double computeDpDepse_v(const ModelStateBase* state) const = 0;

    ////////////////////////////////////////////////////////////////////////
    /*! Calculate the derivative of p with respect to epse_s
      where epse_s = sqrt{2}{3} ||ee||
      ee = epse - 1/3 tr(epse) I
      epse = total elastic strain */
    ////////////////////////////////////////////////////////////////////////
    virtual
    double computeDpDepse_s(const ModelStateBase* state) const = 0;

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

    ////////////////////////////////////////////////////////////////////////
    /**
     * Function: computeElasticVolumetricStrain
     *
     * Purpose:
     *   Compute the volumetric strain given a pressure (p)
     *
     * Inputs:
     *   pp  = current pressure
     *   p0 = initial pressure
     *
     * Returns:
     *   eps_e_v = current elastic volume strain 
     */ 
    ////////////////////////////////////////////////////////////////////////
    virtual
    double computeElasticVolumetricStrain(const double& pp,
                                          const double& p0) {return -1.0;}

    ////////////////////////////////////////////////////////////////////////
    /**
     * Function: computeExpElasticVolumetricStrain
     *
     * Purpose:
     *   Compute the exponential of volumetric strain given a pressure (p)
     *
     * Inputs:
     *   pp  = current pressure
     *   p0 = initial pressure
     *
     * Returns:
     *   exp(eps_e_v) = exponential of the current elastic volume strain 
     */ 
    ////////////////////////////////////////////////////////////////////////
    virtual
    double computeExpElasticVolumetricStrain(const double& pp,
                                             const double& p0) {return -1.0;}

    ////////////////////////////////////////////////////////////////////////
    /**
     * Function: computeDerivExpElasticVolumetricStrain
     *
     * Purpose:
     *   Compute the pressure drivative of the exponential of 
     *   the volumetric strain at a given pressure (p)
     *
     * Inputs:
     *   pp  = current pressure
     *   p0 = initial pressure
     *
     * Outputs:
     *   exp_eps_e_v = exp(eps_e_v) = exponential of elastic volumeric strain
     *
     * Returns:
     *   deriv = d/dp[exp(eps_e_v)] = derivative of the exponential of
     *                                current elastic volume strain 
     */ 
    ////////////////////////////////////////////////////////////////////////
    virtual
    double computeDerivExpElasticVolumetricStrain(const double& pp,
                                                  const double& p0,
                                                  double& exp_eps_e_v) {return -1.0;}

  };
} // End namespace Uintah
      


#endif  // __BB_EQUATION_OF_STATE_H__

