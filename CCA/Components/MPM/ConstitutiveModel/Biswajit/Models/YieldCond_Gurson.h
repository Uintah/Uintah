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


#ifndef __BB_GURSON_YIELD_MODEL_H__
#define __BB_GURSON_YIELD_MODEL_H__

#include "YieldCondition.h"     
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace UintahBB {

  //////////////////////////////////////////////////////////////////////
  /*!
    \class  YieldCond_Gurson
    \brief  Gurson-Tvergaard-Needleman Yield Condition.
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering
    University of Utah
    Copyright (C) 2003 Container Dynamics Group
    \warning The stress tensor is the Cauchy stress and not the 
    Kirchhoff stress.

    References:

    1) Bernauer, G. and Brocks, W., 2002, Fatigue Fract. Engg. Mater. Struct.,
    25, 363-384.
    2) Ramaswamy, S. and Aravas, N., 1998, Comput. Methods Appl. Mech. Engrg.,
    163, 33-53.
  
    The yield condition is given by
    \f[ 
    \Phi(\sigma,k,T) = 
    \frac{\sigma_{eq}^2}{\sigma_f^2} + 
    2 q_1 f_* \cosh \left(q_2 \frac{Tr(\sigma)}{2\sigma_f}\right) -
    (1+q_3 f_*^2) = 0 
    \f]
    where \f$\Phi(\sigma,k,T)\f$ is the yield condition,
    \f$\sigma\f$ is the Cauchy stress,
    \f$k\f$ is a set of internal variable that evolve with time,
    \f$T\f$ is the temperature,
    \f$\sigma_{eq}\f$ is the von Mises equivalent stress given by
    \f$ \sigma_{eq} = \sqrt{\frac{3}{2}\sigma^{d}:\sigma^{d}}\f$ where
    \f$\sigma^{d}\f$ is the deviatoric part of the Cauchy stress, 
    \f$\sigma_{f}\f$ is the flow stress,
    \f$q_1,q_2,q_3\f$ are material constants, and
    \f$f_*\f$ is the porosity (damage) function. 
  
    The damage function is given by
    \f$ f_* = f \f$ for \f$ f \le f_c \f$, 
    \f$ f_* = f_c + k (f - f_c) \f$ for \f$ f > f_c \f$, where
    \f$ k \f$ is constant, and \f$ f \f$ is the porosity (void volume
    fraction).
  */
  //////////////////////////////////////////////////////////////////////

  class YieldCond_Gurson : public YieldCondition {

  public:

    /*! \struct CMData
      \brief Constants needed for GTN model */
    struct CMData {
      double q1;  /*< Constant q_1 */
      double q2;  /*< Constant q_2 */
      double q3;  /*< Constant q_3 */
      double k;   /*< Constant k */
      double f_c; /*< Critical void volume fraction */
    };

  private:

    CMData d_CM;

    // Prevent copying of this class
    // copy constructor
    //YieldCond_Gurson(const YieldCond_Gurson &);
    YieldCond_Gurson& operator=(const YieldCond_Gurson &);

  public:

    /*! Constructor
      Creates a Gurson Yield Function object */
    YieldCond_Gurson(ProblemSpecP& ps);
    YieldCond_Gurson(const YieldCond_Gurson* cm);
         
    //! Destructor 
    virtual ~YieldCond_Gurson();
         
    virtual void outputProblemSpec(ProblemSpecP& ps);

    //! Evaluate the yield function.
    double evalYieldCondition(const double equivStress,
                              const double flowStress,
                              const double traceOfCauchyStress,
                              const double porosity,
                              double& sig);

    double evalYieldCondition(const Matrix3& xi,
                              const ModelState* state);

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

      \f[
      \Phi := \sigma^2_{eq} - 
      (A \cosh(\frac{B}{\sigma_Y(v_i)}) - C) \sigma_Y^2(v_i)
      \f]
      Therefore,
      \f[
      \frac{d\Phi}{dv_i} := -A \sinh(\frac{B}{\sigma_Y}) B 
      \frac{d\sigma_Y}{dv_i} + 
      2 (A \cosh(\frac{B}{\sigma_Y}) - C) \sigma_Y 
      \frac{d\sigma_Y}{dv_i} 
      \f]
      
      \return derivative 
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

      \f[
      \frac{d\Phi}{df} := \left[ 2 q_1 
      cosh\left(\frac{q_2 Tr(\sigma)}{2 \sigma_Y}\right)
      - 2 q_3 f^* \right] \sigma_Y^2
      \f]
      
      \return derivative 
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

    /*! Derivative with respect to the Cauchy stress (\f$\sigma \f$)*/
    void eval_df_dsigma(const Matrix3& xi,
                        const ModelState* state,
                        Matrix3& df_dsigma);

    /*! Derivative with respect to the \f$xi\f$ where \f$\xi = s - \beta \f$  
        where \f$s\f$ is deviatoric part of Cauchy stress and 
        \f$\beta\f$ is the backstress */
    void eval_df_dxi(const Matrix3& xi,
                     const ModelState* state,
                     Matrix3& df_xi);

    /* Derivative with respect to \f$ s \f$ and \f$ \beta \f$ */
    void eval_df_ds_df_dbeta(const Matrix3& xi,
                             const ModelState* state,
                             Matrix3& df_ds,
                             Matrix3& df_dbeta);

    /*! Derivative with respect to the plastic strain (\f$\epsilon^p \f$)*/
    double eval_df_dep(const Matrix3& xi,
                       const double& d_sigy_dep,
                       const ModelState* state);

    /*! Derivative with respect to the porosity (\f$\epsilon^p \f$)*/
    double eval_df_dphi(const Matrix3& xi,
                        const ModelState* state);

    /*! Compute h_alpha  where \f$d/dt(ep) = d/dt(gamma)~h_{\alpha}\f$ */
    double eval_h_alpha(const Matrix3& xi,
                        const ModelState* state);

    /*! Compute h_phi  where \f$d/dt(phi) = d/dt(gamma)~h_{\phi}\f$ */
    double eval_h_phi(const Matrix3& xi,
                      const double& factorA,
                      const ModelState* state);

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

#endif  // __BB_GURSON_YIELD_MODEL_H__ 
