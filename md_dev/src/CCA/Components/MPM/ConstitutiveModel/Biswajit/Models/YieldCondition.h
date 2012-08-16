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


#ifndef __BB_YIELD_CONDITION_H__
#define __BB_YIELD_CONDITION_H__

#include <Core/Math/Matrix3.h>
#include <Core/Math/TangentModulusTensor.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Math/TangentModulusTensor.h>
#include <CCA/Components/MPM/ConstitutiveModel/Biswajit/Models/ModelState.h>
#include <CCA/Components/MPM/ConstitutiveModel/Biswajit/Models/PressureModel.h>
#include <CCA/Components/MPM/ConstitutiveModel/Biswajit/Models/ShearModulusModel.h>
#include <CCA/Components/MPM/ConstitutiveModel/Biswajit/Models/InternalVariableModel.h>

namespace UintahBB {

  /*! \class YieldCondition
   *  \brief A generic wrapper for various yield conditions
   *  \author Biswajit Banerjee, 
   *  \author C-SAFE and Department of Mechanical Engineering,
   *  \author University of Utah.
   *  \author Copyright (C) 2003 Container Dynamics Group
   *  \warning Mixing and matching yield conditions with damage and plasticity 
   *           models should be done with care.  No checks are provided to stop
   *           the user from using the wrong combination of models.
   *
   * Provides an abstract base class for various yield conditions used
   * in the plasticity and damage models
  */
  class YieldCondition {

  public:
         
    //! Construct a yield condition.  
    /*! This is an abstract base class. */
    YieldCondition();

    //! Destructor of yield condition.  
    /*! Virtual to ensure correct behavior */
    virtual ~YieldCondition();

    virtual void outputProblemSpec(Uintah::ProblemSpecP& ps) = 0;
         
    /////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Evaluate the yield function \f$(\Phi)\f$.

      If \f$\Phi \le 0\f$ the state is elastic.
      If \f$\Phi > 0\f$ the state is plastic and a normal return 
      mapping algorithm is necessary. 

      Returns the appropriate value of sig(t+delT) that is on
      the flow surface.
    */
    /////////////////////////////////////////////////////////////////////////
    virtual double evalYieldCondition(const double equivStress,
                                      const double flowStress,
                                      const double traceOfCauchyStress,
                                      const double porosity,
                                      double& sig) = 0;

    /////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Evaluate the derivative of the yield function \f$(\Phi)\f$
      with respect to \f$\sigma_{ij}\f$.
    */
    /////////////////////////////////////////////////////////////////////////
    virtual void evalDerivOfYieldFunction(const Uintah::Matrix3& stress,
                                          const double flowStress,
                                          const double porosity,
                                          Uintah::Matrix3& derivative) = 0;

    /////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Evaluate the derivative of the yield function \f$(\Phi)\f$
      with respect to \f$s_{ij}\f$.

      This is for the associated flow rule with \f$s_{ij}\f$ being
      the deviatoric stress.
    */
    /////////////////////////////////////////////////////////////////////////
    virtual void evalDevDerivOfYieldFunction(const Uintah::Matrix3& stress,
                                             const double flowStress,
                                             const double porosity,
                                             Uintah::Matrix3& derivative) = 0;

    /*! Evaluate the yield condition - \f$ sigma \f$ is the Cauchy stress
    and \f$ \beta \f$ is the back stress */
    virtual double evalYieldCondition(const Uintah::Matrix3& xi,
                                      const ModelState* state) = 0;

    /*! Derivative with respect to the Cauchy stress (\f$\sigma \f$)*/
    virtual void eval_df_dsigma(const Uintah::Matrix3& xi,
                                const ModelState* state,
                                Uintah::Matrix3& df_dsigma) = 0;

    /*! Derivative with respect to the \f$xi\f$ where \f$\xi = s - \beta \f$  
        where \f$s\f$ is deviatoric part of Cauchy stress and 
        \f$\beta\f$ is the backstress */
    virtual void eval_df_dxi(const Uintah::Matrix3& xi,
                             const ModelState* state,
                             Uintah::Matrix3& df_xi) = 0;

    /* Derivative with respect to \f$ s \f$ and \f$ \beta \f$ */
    virtual void eval_df_ds_df_dbeta(const Uintah::Matrix3& xi,
                                     const ModelState* state,
                                     Uintah::Matrix3& df_ds,
                                     Uintah::Matrix3& df_dbeta) = 0;

    /*! Derivative with respect to the plastic strain (\f$\epsilon^p \f$)*/
    virtual double eval_df_dep(const Uintah::Matrix3& xi,
                               const double& d_sigy_dep,
                               const ModelState* state) = 0;

    /*! Derivative with respect to the porosity (\f$\epsilon^p \f$)*/
    virtual double eval_df_dphi(const Uintah::Matrix3& xi,
                                const ModelState* state) = 0;

    /*! Compute h_alpha  where \f$d/dt(ep) = d/dt(gamma)~h_{\alpha}\f$ */
    virtual double eval_h_alpha(const Uintah::Matrix3& xi,
                                const ModelState* state) = 0;

    /*! Compute h_phi  where \f$d/dt(phi) = d/dt(gamma)~h_{\phi}\f$ */
    virtual double eval_h_phi(const Uintah::Matrix3& xi,
                              const double& factorA,
                              const ModelState* state) = 0;

    /////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Compute the elastic-plastic tangent modulus.
    */
    /////////////////////////////////////////////////////////////////////////
    virtual void computeElasPlasTangentModulus(const Uintah::TangentModulusTensor& Ce,
                                               const Uintah::Matrix3& sigma, 
                                               double sigY,
                                               double dsigYdV,
                                               double porosity,
                                               double voidNuclFac,
                                               Uintah::TangentModulusTensor& Cep) = 0;

    /*! Compute continuum elastic-plastic tangent modulus.
       df_dsigma = r */ 
    virtual void computeElasPlasTangentModulus(const Uintah::Matrix3& r, 
                                               const Uintah::Matrix3& df_ds, 
                                               const Uintah::Matrix3& h_beta,
                                               const Uintah::Matrix3& df_dbeta, 
                                               const double& h_alpha,             
                                               const double& df_dep,
                                               const double& h_phi,             
                                               const double& df_phi,
                                               const double& J,
                                               const double& dp_dJ,
                                               const ModelState* state,
                                               Uintah::TangentModulusTensor& Cep);

    //--------------------------------------------------------------
    // Compute value of yield function
    //--------------------------------------------------------------
    virtual double evalYieldCondition(const ModelState* state) = 0;

    //--------------------------------------------------------------
    // Compute df/dp  where p = volumetric stress = 1/3 Tr(sigma)
    //--------------------------------------------------------------
    virtual double computeVolStressDerivOfYieldFunction(const ModelState* state) = 0;

    //--------------------------------------------------------------
    // Compute df/dq  where q = sqrt(3 J_2), J_2 = 2nd invariant deviatoric stress
    //--------------------------------------------------------------
    virtual double computeDevStressDerivOfYieldFunction(const ModelState* state) = 0;

    //--------------------------------------------------------------
    // Compute d/depse_v(df/dp)
    //--------------------------------------------------------------
    virtual double computeVolStrainDerivOfDfDp(const ModelState* state,
                                       const PressureModel* eos,
                                       const ShearModulusModel* shear,
                                       const InternalVariableModel* intvar) = 0;

    //--------------------------------------------------------------
    // Compute d/depse_s(df/dp)
    //--------------------------------------------------------------
    virtual double computeDevStrainDerivOfDfDp(const ModelState* state,
                                       const PressureModel* eos,
                                       const ShearModulusModel* shear,
                                       const InternalVariableModel* intvar) = 0;

    //--------------------------------------------------------------
    // Compute d/depse_v(df/dq)
    //--------------------------------------------------------------
    virtual double computeVolStrainDerivOfDfDq(const ModelState* state,
                                       const PressureModel* eos,
                                       const ShearModulusModel* shear,
                                       const InternalVariableModel* intvar) = 0;

    //--------------------------------------------------------------
    // Compute d/depse_s(df/dq)
    //--------------------------------------------------------------
    virtual double computeDevStrainDerivOfDfDq(const ModelState* state,
                                       const PressureModel* eos,
                                       const ShearModulusModel* shear,
                                       const InternalVariableModel* intvar) = 0;

    //--------------------------------------------------------------
    // Compute df/depse_v
    //--------------------------------------------------------------
    virtual double computeVolStrainDerivOfYieldFunction(const ModelState* state,
                                                const PressureModel* eos,
                                                const ShearModulusModel* shear,
                                                const InternalVariableModel* intvar) = 0;

    //--------------------------------------------------------------
    // Compute df/depse_s
    //--------------------------------------------------------------
    virtual double computeDevStrainDerivOfYieldFunction(const ModelState* state,
                                                const PressureModel* eos,
                                                const ShearModulusModel* shear,
                                                const InternalVariableModel* intvar) = 0;

  };
} // End namespace Uintah
      
#endif  // __BB_YIELD_CONDITION_H__

