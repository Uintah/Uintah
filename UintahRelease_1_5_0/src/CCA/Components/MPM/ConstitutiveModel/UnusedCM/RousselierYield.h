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

#ifndef __ROUSSELIER_YIELD_MODEL_H__
#define __ROUSSELIER_YIELD_MODEL_H__

#include "YieldCondition.h"     
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class  RousselierYield
   *  \brief  Rousselier Yield Condition.
   *  \author Biswajit Banerjee
   *  \author C-SAFE and Department of Mechanical Engineering
   *  \author University of Utah
   *  \warning The stress tensor is the Cauchy stress and not the 
   *           Kirchhoff stress.

   References:

   1) Bernauer, G. and Brocks, W., 2002, Fatigue Fract. Engg. Mater. Struct.,
   25, 363-384.

   The yield condition is given by
   \f[ 
   \Phi(\sigma,k,T) = 
   \frac{\sigma_{eq}}{1-f} + 
   D \sigma_1 f \exp \left(\frac{(1/3)Tr(\sigma)}{(1-f)\sigma_1}\right) -
   \sigma_f = 0 
   \f]
   where \f$\Phi(\sigma,k,T)\f$ is the yield condition,
   \f$\sigma\f$ is the Cauchy stress,
   \f$k\f$ is a set of internal variable that evolve with time,
   \f$T\f$ is the temperature,
   \f$\sigma_{eq}\f$ is the von Mises equivalent stress given by
   \f$ \sigma_{eq} = \sqrt{\frac{3}{2}\sigma^{d}:\sigma^{d}}\f$ where
   \f$\sigma^{d}\f$ is the deviatoric part of the Cauchy stress, 
   \f$\sigma_{f}\f$ is the flow stress,
   \f$D,\sigma_1\f$ are material constants, and
   \f$f\f$ is the porosity (void volume fraction).
  */

  class RousselierYield : public YieldCondition {

  public:

    /*! \struct CMData
      \brief Constants needed for Rousselier model */
    struct CMData {
      double D;  /**< Constant D */
      double sig_1;  /**< Constant \sigma_1 */
    };

  private:

    CMData d_constant;

    // Prevent copying of this class
    // copy constructor
    //RousselierYield(const RousselierYield &);
    RousselierYield& operator=(const RousselierYield &);

  public:

    //! Constructor
    /*! Creates a Rousselier Yield Function object */
    RousselierYield(ProblemSpecP& ps);
    RousselierYield(const RousselierYield* cm);
         
    //! Destructor 
    ~RousselierYield();
         
    //! Evaluate the yield function.
    double evalYieldCondition(const double equivStress,
                              const double flowStress,
                              const double traceOfCauchyStress,
                              const double porosity,
                              double& sig);

    /////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Evaluate the derivative of the yield function \f$(\Phi)\f$
      with respect to \f$\sigma_{ij}\f$.

      This is for the associated flow rule.
    */
    /////////////////////////////////////////////////////////////////////////
    void evalDerivOfYieldFunction(const Matrix3& stress,
                                  const double flowStress,
                                  const double porosity,
                                  Matrix3& derivative);

    /////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Evaluate the derivative of the yield function \f$(\Phi)\f$
      with respect to \f$s_{ij}\f$.

      This is for the associated flow rule with \f$s_{ij}\f$ being
      the deviatoric stress.
    */
    /////////////////////////////////////////////////////////////////////////
    void evalDevDerivOfYieldFunction(const Matrix3& stress,
                                     const double flowStress,
                                     const double porosity,
                                     Matrix3& derivative);

    /////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate the derivative of the yield function \f$ \Phi \f$
      with respect to a scalar variable.

      \return derivative 

      \warning Not yet implemented
    */
    /////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTPlasticityScalar(double trSig,
                                             double porosity,
                                             double sigY,
                                             double dsigYdV);

    /////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate the derivative of the yield function \f$ \Phi \f$
      with respect to the porosity

      \return derivative 

      \warning Not yet implemented.
    */
    /////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTPorosity(double trSig,
                                     double porosity,
                                     double sigY);

    /////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate the factor \f$h_1\f$ for porosity

      \f[
      h_1 = (1-f) Tr(\sigma) + A \frac{\sigma : f_{\sigma}}{(1-f) \sigma_Y}
      \f]
      
      \return factor 
    */
    /////////////////////////////////////////////////////////////////////////
    double computePorosityFactor_h1(double sigma_f_sigma,
                                    double tr_f_sigma,
                                    double porosity,
                                    double sigma_Y,
                                    double A);
 
    /////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate the factor \f$h_2\f$ for plastic strain

      \f[
      h_2 = \frac{\sigma : f_{\sigma}}{(1-f) \sigma_Y}
      \f]
      
      \return factor 
    */
    /////////////////////////////////////////////////////////////////////////
    double computePlasticStrainFactor_h2(double sigma_f_sigma,
                                         double porosity,
                                         double sigma_Y);

    /////////////////////////////////////////////////////////////////////////
    /*!
      \brief Compute the continuum elasto-plastic tangent modulus
      assuming associated flow rule.

      \f[
      C_{ep} = C_{e} - \frac{(C_e:f_{\sigma})\otimes(f_{\sigma}:C_e)}
      {-f_q.h_q + f_{\sigma}:C_e:f_{\sigma}}
      \f]
      
      \return TangentModulusTensor \f$ C_{ep} \f$.
    */
    /////////////////////////////////////////////////////////////////////////
    void computeTangentModulus(const TangentModulusTensor& Ce,
                               const Matrix3& f_sigma, 
                               double f_q1, 
                               double f_q2,
                               double h_q1,
                               double h_q2,
                               TangentModulusTensor& Cep);

    /////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Compute the elastic-plastic tangent modulus.
    */
    /////////////////////////////////////////////////////////////////////////
    void computeElasPlasTangentModulus(const TangentModulusTensor& Ce,
                                       const Matrix3& sigma, 
                                       double sigY,
                                       double dsigYdV,
                                       double porosity,
                                       double voidNuclFac,
                                       TangentModulusTensor& Cep);
  };

} // End namespace Uintah

#endif  // __ROUSSELIER_YIELD_MODEL_H__ 
