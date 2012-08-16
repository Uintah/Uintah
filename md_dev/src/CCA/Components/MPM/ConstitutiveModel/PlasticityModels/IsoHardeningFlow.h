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


#ifndef __ISOHARDENING_FLOW_MODEL_H__
#define __ISOHARDENING_FLOW_MODEL_H__


#include "FlowModel.h"    
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*! 
    \class IsoHardeningFlow
    \brief Isotropic Hardening flow model.
    (Simo and Hughes, 1998, Computational Inelasticity, p. 319)
    \author Biswajit Banerjee,
    \author Department of Mechanical Engineering,
    \author University of Utah,
    Copyright (C) 2003 University of Utah

    The flow rule is given by
    \f[
    f(\sigma) = K \alpha + \sigma_0
    \f]

    where \f$f(\sigma)\f$ = flow stress \n
    \f$K\f$ = hardening modulus \n
    \f$\alpha\f$ = evolution parameter for hardening law \n
    \f$\sigma_0\f$ = initial yield stress \n

    Evolution of alpha is given by
    \f[
    d\alpha/dt = \sqrt{2/3}*\gamma
    \f]
    where \f$\gamma\f$ = consistency parameter
  */  
  /////////////////////////////////////////////////////////////////////////////

  class IsoHardeningFlow : public FlowModel {

    // Create datatype for storing model parameters
  public:
    struct CMData {
      double K;
      double sigma_0;
    };   

    constParticleVariable<double> pAlpha;
    ParticleVariable<double> pAlpha_new;

    const VarLabel* pAlphaLabel;  // For Isotropic Hardening Plasticity
    const VarLabel* pAlphaLabel_preReloc;  // For Isotropic Hardening Plasticity

  private:

    CMData d_CM;
         
    // Prevent copying of this class
    // copy constructor
    //IsoHardeningFlow(const IsoHardeningFlow &cm);
    IsoHardeningFlow& operator=(const IsoHardeningFlow &cm);

  public:
    // constructors
    IsoHardeningFlow(ProblemSpecP& ps);
    IsoHardeningFlow(const IsoHardeningFlow* cm);
         
    // destructor 
    virtual ~IsoHardeningFlow();

    virtual void outputProblemSpec(ProblemSpecP& ps);
         
    // Computes and requires for internal evolution variables
    // Only one internal variable for Isotropic-Hardening :: plastic strain
    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* patches);

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches);

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches,
                                        bool recurse,
                                        bool SchedParent);

    virtual void allocateCMDataAddRequires(Task* task, const MPMMaterial* matl,
                                           const PatchSet* patch, 
                                           MPMLabel* lb);

    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
                                   ParticleSubset* addset,
                                   map<const VarLabel*, 
                                     ParticleVariableBase*>* newState,
                                   ParticleSubset* delset,
                                   DataWarehouse* old_dw);


    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to);

    virtual void initializeInternalVars(ParticleSubset* pset,
                                        DataWarehouse* new_dw);

    virtual void getInternalVars(ParticleSubset* pset,
                                 DataWarehouse* old_dw);

    virtual void allocateAndPutInternalVars(ParticleSubset* pset,
                                            DataWarehouse* new_dw); 

    virtual void allocateAndPutRigid(ParticleSubset* pset,
                                     DataWarehouse* new_dw); 

    virtual void updateElastic(const particleIndex idx);

    virtual void updatePlastic(const particleIndex idx, const double& delGamma);

    ///////////////////////////////////////////////////////////////////////////
    /*! compute the flow stress */
    ///////////////////////////////////////////////////////////////////////////
    virtual double computeFlowStress(const PlasticityState* state,
                                     const double& delT,
                                     const double& tolerance,
                                     const MPMMaterial* matl,
                                     const particleIndex idx);

    //////////
    /*! \brief Calculate the plastic strain rate [epdot(tau,ep,T)] */
    //////////
    virtual double computeEpdot(const PlasticityState* state,
                                const double& delT,
                                const double& tolerance,
                                const MPMMaterial* matl,
                                const particleIndex idx);
 
    ///////////////////////////////////////////////////////////////////////////
    /*! Compute the elastic-plastic tangent modulus 
    **WARNING** Assumes vonMises yield condition and the
    associated flow rule */
    ///////////////////////////////////////////////////////////////////////////
    virtual void computeTangentModulus(const Matrix3& stress,
                                       const PlasticityState* state,
                                       const double& delT,
                                       const MPMMaterial* matl,
                                       const particleIndex idx,
                                       TangentModulusTensor& Ce,
                                       TangentModulusTensor& Cep);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to scalar and
      internal variables.

      \return Three derivatives in Vector deriv 
      (deriv[0] = \f$d\sigma_Y/d\dot\epsilon\f$,
      deriv[1] = \f$d\sigma_Y/dT\f$, 
      deriv[2] = \f$d\sigma_Y/d\alpha\f$)
    */
    ///////////////////////////////////////////////////////////////////////////
    void evalDerivativeWRTScalarVars(const PlasticityState* state,
                                     const particleIndex idx,
                                     Vector& derivs);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to plastic
      strain.

      The Isotropic-Hardening yield stress is given by :
      \f[
      \sigma_Y(\alpha) := \sigma_0 + K\alpha
      \f]

      The derivative is given by
      \f[
      \frac{d\sigma_Y}{d\epsilon_p} := K
      \f]

      \return Derivative \f$ d\sigma_Y / d\alpha\f$.

      \warning Not implemented yet.

    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTPlasticStrain(const PlasticityState* state,
                                          const particleIndex idx);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to strain rate.

      The Isotropic-Hardening yield stress is given by :
      \f[
      \sigma_Y(\dot\epsilon_p) := C
      \f]

      The derivative is given by
      \f[
      \frac{d\sigma_Y}{d\dot\epsilon_p} := 0
      \f]

      \return Derivative \f$ d\sigma_Y / d\dot\epsilon_p \f$.
    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTStrainRate(const PlasticityState* state,
                                       const particleIndex idx);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Compute the shear modulus. 
    */
    ///////////////////////////////////////////////////////////////////////////
    double computeShearModulus(const PlasticityState* state);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Compute the melting temperature
    */
    ///////////////////////////////////////////////////////////////////////////
    double computeMeltingTemp(const PlasticityState* state);

  protected:

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to temperature.

      The Isotropic-Hardening yield stress is given by :
      \f[
      \sigma_Y(T) := C
      \f]

      The derivative is given by
      \f[
      \frac{d\sigma_Y}{dT} := 0
      \f]

      \return Derivative \f$ d\sigma_Y / dT \f$.
    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTTemperature(const PlasticityState* state,
                                        const particleIndex idx);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to alpha

      The Isotropic-Hardening yield stress is given by :
      \f[
      \sigma_Y(\alpha) := \sigma_0 + K\alpha
      \f]

      The derivative is given by
      \f[
      \frac{d\sigma_Y}{d\alpha} := K
      \f]

      \return Derivative \f$ d\sigma_Y / d\alpha\f$.

    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTAlpha(const PlasticityState* state,
                                  const particleIndex idx);

  };

} // End namespace Uintah

#endif  // __ISOHARDENING_FLOW_MODEL_H__ 
