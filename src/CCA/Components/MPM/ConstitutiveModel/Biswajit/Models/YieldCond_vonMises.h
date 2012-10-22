/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef __BB_VONMISES_YIELD_MODEL_H__
#define __BB_VONMISES_YIELD_MODEL_H__

#include "YieldCondition.h"     
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace UintahBB {

  /*! \class  YieldCond_vonMises
   *  \brief  von Mises-Huber Yield Condition (J2 plasticity).
   *  \author Biswajit Banerjee
   *  \author C-SAFE and Department of Mechanical Engineering
   *  \author University of Utah
   *  \warning The stress tensor is the Cauchy stress and not the 
   *           Kirchhoff stress.

   The yield condition is given by
   \f[
   \Phi(\sigma,k,T) = \sigma_{eq} - \sigma_{f} = 0 
   \f]
   where \f$\Phi(\sigma,k,T)\f$ is the yield condition,
   \f$\sigma\f$ is the Cauchy stress,
   \f$k\f$ is a set of internal variable that evolve with time,
   \f$T\f$ is the temperature,
   \f$\sigma_{eq}\f$ is the von Mises equivalent stress given by
   \f$ \sigma_{eq} = \sqrt{\frac{3}{2}\sigma^{d}:\sigma^{d}},\f$
   \f$\sigma^{d}\f$ is the deviatoric part of the Cauchy stress, and
   \f$\sigma^{f}\f$ is the flow stress. 
  */

  class YieldCond_vonMises : public YieldCondition {

  private:

    // Prevent copying of this class
    // copy constructor
    //YieldCond_vonMises(const YieldCond_vonMises &);
    YieldCond_vonMises& operator=(const YieldCond_vonMises &);

  public:

    //! Constructor
    /*! Creates a YieldCond_vonMises function object */
    YieldCond_vonMises(Uintah::ProblemSpecP& ps);
    YieldCond_vonMises(const YieldCond_vonMises* cm);
         
    //! Destructor 
    ~YieldCond_vonMises();

    virtual void outputProblemSpec(Uintah::ProblemSpecP& ps);
         
    //! Evaluate the yield function.
    double evalYieldCondition(const double equivStress,
                              const double flowStress,
                              const double traceOfCauchyStress,
                              const double porosity,
                              double& sig);

    double evalYieldCondition(const Uintah::Matrix3& xi,
                              const ModelState* state);

    /////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Evaluate the derivative of the yield function \f$(\Phi)\f$
      with respect to \f$\sigma_{ij}\f$.

      This is for the associated flow rule.
    */
    /////////////////////////////////////////////////////////////////////////
    void evalDerivOfYieldFunction(const Uintah::Matrix3& stress,
                                  const double flowStress,
                                  const double porosity,
                                  Uintah::Matrix3& derivative);

    /////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Evaluate the derivative of the yield function \f$(\Phi)\f$
      with respect to \f$s_{ij}\f$.

      This is for the associated flow rule with \f$s_{ij}\f$ being
      the deviatoric stress.
    */
    /////////////////////////////////////////////////////////////////////////
    void evalDevDerivOfYieldFunction(const Uintah::Matrix3& stress,
                                     const double flowStress,
                                     const double porosity,
                                     Uintah::Matrix3& derivative);

    /*! Derivative with respect to the Cauchy stress (\f$\sigma \f$)*/
    void eval_df_dsigma(const Uintah::Matrix3& xi,
                        const ModelState* state,
                        Uintah::Matrix3& df_dsigma);

    /*! Derivative with respect to the \f$xi\f$ where \f$\xi = s - \beta \f$  
        where \f$s\f$ is deviatoric part of Cauchy stress and 
        \f$\beta\f$ is the backstress */
    void eval_df_dxi(const Uintah::Matrix3& xi,
                     const ModelState* state,
                     Uintah::Matrix3& df_xi);

    /* Derivative with respect to \f$ s \f$ and \f$ \beta \f$ */
    void eval_df_ds_df_dbeta(const Uintah::Matrix3& xi,
                             const ModelState* state,
                             Uintah::Matrix3& df_ds,
                             Uintah::Matrix3& df_dbeta);

    /*! Derivative with respect to the plastic strain (\f$\epsilon^p \f$)*/
    double eval_df_dep(const Uintah::Matrix3& xi,
                       const double& d_sigy_dep,
                       const ModelState* state);

    /*! Derivative with respect to the porosity (\f$\epsilon^p \f$)*/
    double eval_df_dphi(const Uintah::Matrix3& xi,
                        const ModelState* state);

    /*! Compute h_alpha  where \f$d/dt(ep) = d/dt(gamma)~h_{\alpha}\f$ */
    double eval_h_alpha(const Uintah::Matrix3& xi,
                        const ModelState* state);

    /*! Compute h_phi  where \f$d/dt(phi) = d/dt(gamma)~h_{\phi}\f$ */
    double eval_h_phi(const Uintah::Matrix3& xi,
                      const double& factorA,
                      const ModelState* state);

    /////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Compute the elastic-plastic tangent modulus.
    */
    /////////////////////////////////////////////////////////////////////////
    void computeElasPlasTangentModulus(const Uintah::TangentModulusTensor& Ce,
                                       const Uintah::Matrix3& sigma, 
                                       double sigY,
                                       double dsigYdep,
                                       double porosity,
                                       double voidNuclFac,
                                       Uintah::TangentModulusTensor& Cep);

    /////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate the factor \f$h_1\f$ for plastic strain

      \f[
      h_1 = \frac{\sigma : f_{\sigma}}{\sigma_Y}
      \f]
      
      \return factor 
    */
    /////////////////////////////////////////////////////////////////////////
    inline double computePlasticStrainFactor(double sigma_f_sigma,
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
    void computeTangentModulus(const Uintah::TangentModulusTensor& Ce,
                               const Uintah::Matrix3& f_sigma, 
                               double f_q1, 
                               double h_q1,
                               Uintah::TangentModulusTensor& Cep);

    //--------------------------------------------------------------
    // Compute value of yield function
    //--------------------------------------------------------------
    double evalYieldCondition(const ModelState* state) {return 0.0;};

    //--------------------------------------------------------------
    // Compute df/dp  where p = volumetric stress = 1/3 Tr(sigma)
    //--------------------------------------------------------------
    double computeVolStressDerivOfYieldFunction(const ModelState* state) {return 0.0;};

    //--------------------------------------------------------------
    // Compute df/dq  where q = sqrt(3 J_2), J_2 = 2nd invariant deviatoric stress
    //--------------------------------------------------------------
    double computeDevStressDerivOfYieldFunction(const ModelState* state) {return 0.0;};

    //--------------------------------------------------------------
    // Compute d/depse_v(df/dp)
    //--------------------------------------------------------------
    double computeVolStrainDerivOfDfDp(const ModelState* state,
                                       const PressureModel* eos,
                                       const ShearModulusModel* shear,
                                       const InternalVariableModel* intvar) {return 0.0;};

    //--------------------------------------------------------------
    // Compute d/depse_s(df/dp)
    //--------------------------------------------------------------
    double computeDevStrainDerivOfDfDp(const ModelState* state,
                                       const PressureModel* eos,
                                       const ShearModulusModel* shear,
                                       const InternalVariableModel* intvar) {return 0.0;};

    //--------------------------------------------------------------
    // Compute d/depse_v(df/dq)
    //--------------------------------------------------------------
    double computeVolStrainDerivOfDfDq(const ModelState* state,
                                       const PressureModel* eos,
                                       const ShearModulusModel* shear,
                                       const InternalVariableModel* intvar) {return 0.0;};

    //--------------------------------------------------------------
    // Compute d/depse_s(df/dq)
    //--------------------------------------------------------------
    double computeDevStrainDerivOfDfDq(const ModelState* state,
                                       const PressureModel* eos,
                                       const ShearModulusModel* shear,
                                       const InternalVariableModel* intvar) {return 0.0;};

    //--------------------------------------------------------------
    // Compute df/depse_v
    //--------------------------------------------------------------
    double computeVolStrainDerivOfYieldFunction(const ModelState* state,
                                                const PressureModel* eos,
                                                const ShearModulusModel* shear,
                                                const InternalVariableModel* intvar) {return 0.0;};

    //--------------------------------------------------------------
    // Compute df/depse_s
    //--------------------------------------------------------------
    double computeDevStrainDerivOfYieldFunction(const ModelState* state,
                                                const PressureModel* eos,
                                                const ShearModulusModel* shear,
                                                const InternalVariableModel* intvar) {return 0.0;};

  };

} // End namespace Uintah

#endif  // __BB_VONMISES_YIELD_MODEL_H__ 
