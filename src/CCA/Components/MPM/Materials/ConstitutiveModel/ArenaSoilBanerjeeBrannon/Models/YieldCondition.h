/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
 * Copyright (c) 2013-2014 Callaghan Innovation, New Zealand
 * Copyright (c) 2015-2016 Parresia Research Limited, New Zealand
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

#ifndef __BB_YIELD_CONDITION_H__
#define __BB_YIELD_CONDITION_H__

#include <Core/Math/Matrix3.h>
#include <Core/Math/TangentModulusTensor.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ArenaSoilBanerjeeBrannon/Models/ModelStateBase.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/ParticleVariable.h>

namespace Uintah {
  class Task;
  class Patch;
  class VarLabel;
  class MPMMaterial;
  class DataWarehouse;
  class ParticleSubset;
  class ParticleVariableBase;
  class Matrix3;
}

namespace Vaango {

  using ParameterDict = std::map<std::string, double>;


  /*! \class YieldCondition
   *  \brief A generic wrapper for various yield conditions
   *  \author Biswajit Banerjee, 
   *  \author C-SAFE and Department of Mechanical Engineering,
   *  \author University of Utah.
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
      \brief Get/compute the parameters of the yield condition model
    */ 
    /////////////////////////////////////////////////////////////////////////
    virtual std::map<std::string, double> getParameters() const = 0;
    virtual void computeModelParameters(double factor) {};

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
    virtual
    double evalYieldCondition(const Uintah::Matrix3& xi,
                              const ModelStateBase* state) = 0;

    /*! Derivative with respect to the Cauchy stress (\f$\sigma \f$)*/
    virtual
    void eval_df_dsigma(const Uintah::Matrix3& xi,
                        const ModelStateBase* state,
                        Uintah::Matrix3& df_dsigma) = 0;

    /*! Derivative with respect to the \f$xi\f$ where \f$\xi = s - \beta \f$  
      where \f$s\f$ is deviatoric part of Cauchy stress and 
      \f$\beta\f$ is the backstress */
    virtual
    void eval_df_dxi(const Uintah::Matrix3& xi,
                     const ModelStateBase* state,
                     Uintah::Matrix3& df_xi) = 0;

    /* Derivative with respect to \f$ s \f$ and \f$ \beta \f$ */
    virtual
    void eval_df_ds_df_dbeta(const Uintah::Matrix3& xi,
                             const ModelStateBase* state,
                             Uintah::Matrix3& df_ds,
                             Uintah::Matrix3& df_dbeta) = 0;

    /*! Derivative with respect to the plastic strain (\f$\epsilon^p \f$)*/
    virtual
    double eval_df_dep(const Uintah::Matrix3& xi,
                       const double& d_sigy_dep,
                       const ModelStateBase* state) = 0;

    /*! Derivative with respect to the porosity (\f$\epsilon^p \f$)*/
    virtual
    double eval_df_dphi(const Uintah::Matrix3& xi,
                        const ModelStateBase* state) = 0;

    /*! Compute h_alpha  where \f$d/dt(ep) = d/dt(gamma)~h_{\alpha}\f$ */
    virtual
    double eval_h_alpha(const Uintah::Matrix3& xi,
                        const ModelStateBase* state) = 0;

    /*! Compute h_phi  where \f$d/dt(phi) = d/dt(gamma)~h_{\phi}\f$ */
    virtual
    double eval_h_phi(const Uintah::Matrix3& xi,
                      const double& factorA,
                      const ModelStateBase* state) = 0;

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
    void computeElasPlasTangentModulus(const Uintah::Matrix3& r, 
                                       const Uintah::Matrix3& df_ds, 
                                       const Uintah::Matrix3& h_beta,
                                       const Uintah::Matrix3& df_dbeta, 
                                       const double& h_alpha,             
                                       const double& df_dep,
                                       const double& h_phi,             
                                       const double& df_phi,
                                       const double& J,
                                       const double& dp_dJ,
                                       const ModelStateBase* state,
                                       Uintah::TangentModulusTensor& Cep);

    //--------------------------------------------------------------
    // Compute value of yield function
    //--------------------------------------------------------------
    virtual
    double evalYieldCondition(const ModelStateBase* state) = 0;

    //--------------------------------------------------------------
    // Compute max value of yield function for convergence tolerance check
    //--------------------------------------------------------------
    virtual
    double evalYieldConditionMax(const ModelStateBase* state) = 0;

    //--------------------------------------------------------------
    // Compute df/dp  where p = volumetric stress = 1/3 Tr(sigma)
    //--------------------------------------------------------------
    virtual
    double computeVolStressDerivOfYieldFunction(const ModelStateBase* state) = 0;

    //--------------------------------------------------------------
    // Compute df/dq  where q = sqrt(3 J_2), J_2 = 2nd invariant deviatoric stress
    //--------------------------------------------------------------
    virtual
    double computeDevStressDerivOfYieldFunction(const ModelStateBase* state) = 0;


    /**
     * Function: getInternalPoint
     *
     * Purpose: Get a point that is inside the yield surface
     *
     * Inputs:
     *  state_old = old state
     *  state_new = new state
     *
     * Returns:
     *   I1 = value of tr(stress) at a point inside the yield surface
     */
    virtual
    double getInternalPoint(const ModelStateBase* state_old,
                            const ModelStateBase* state_new) = 0;

    /**
     * Function: getClosestPoint
     *
     * Purpose: Get the point on the yield surface that is closest to a given point (2D)
     *
     * Inputs:
     *  state = current state
     *  px = x-coordinate of point
     *  py = y-coordinate of point
     *
     * Outputs:
     *  cpx = x-coordinate of closest point on yield surface
     *  cpy = y-coordinate of closest point
     *
     * Returns:
     *   true - if the closest point can be found
     *   false - otherwise
     */
    virtual
    bool getClosestPoint(const ModelStateBase* state,
                         const double& px, const double& py,
                         double& cpx, double& cpy) {return false;}

    /**
     * These are needed for keeping track of point-to-point material variability
     */
    virtual
    void addInitialComputesAndRequires(Uintah::Task* task,
                                       const Uintah::MPMMaterial* matl,
                                       const Uintah::PatchSet* patch) const {};

    virtual
    void initializeLocalVariables(const Uintah::Patch* patch,
                                  Uintah::ParticleSubset* pset,
                                  Uintah::DataWarehouse* new_dw,
                                  Uintah::constParticleVariable<double>& pVolume) {};

    virtual
    void addComputesAndRequires(Uintah::Task* task,
                                const Uintah::MPMMaterial* matl,
                                const Uintah::PatchSet* patches) const {};

    virtual
    void copyLocalVariables(Uintah::ParticleSubset* pset,
                            Uintah::DataWarehouse* old_dw,
                            Uintah::DataWarehouse* new_dw) {};

    virtual
    std::vector<std::string> getLocalVariableLabels() const { 
      std::vector<std::string> pYieldParamLabels;
      pYieldParamLabels.emplace_back("None");
      return pYieldParamLabels;
    }

    virtual
    std::vector<Uintah::constParticleVariable<double> >
    getLocalVariables(Uintah::ParticleSubset* pset,
                      Uintah::DataWarehouse* old_dw) {
      Uintah::constParticleVariable<double> pNull;
      std::vector<Uintah::constParticleVariable<double> > pYieldParams;
      pYieldParams.emplace_back(pNull);
      return pYieldParams;
    } 

    /**
     *  This is used to scale the yield parameters 
     */
    virtual
    void updateLocalVariables(Uintah::ParticleSubset* pset,
                              Uintah::DataWarehouse* old_dw,
                              Uintah::DataWarehouse* new_dw,
                              Uintah::constParticleVariable<double>& pCoherence_old,
                              const Uintah::ParticleVariable<double>& pCoherence_new) {};

    virtual
    void addParticleState(std::vector<const Uintah::VarLabel*>& from,
                          std::vector<const Uintah::VarLabel*>& to) {};

  };

} // End namespace Vaango
      
#endif  // __BB_YIELD_CONDITION_H__

