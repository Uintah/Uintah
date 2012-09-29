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

#ifndef __FLOW_MODEL_H__
#define __FLOW_MODEL_H__

#include <Core/Math/Matrix3.h>
#include <vector>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Math/TangentModulusTensor.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include "PlasticityState.h"


namespace Uintah {

  ///////////////////////////////////////////////////////////////////////////
  /*!
    \class  FlowModel
    \brief  Abstract Base class for flow models (calculate yield stress)
    \author Biswajit Banerjee, \n
            C-SAFE and Department of Mechanical Engineering, \n
            University of Utah,\n
    \warn   Assumes vonMises yield condition and the associated flow rule for 
            all cases other than Gurson plasticity.
  */
  ///////////////////////////////////////////////////////////////////////////

  class FlowModel {

  private:

  public:
         
    FlowModel();
    virtual ~FlowModel();

    virtual void outputProblemSpec(ProblemSpecP& ps) = 0;
         
    // Computes and requires for internal evolution variables
    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* patches) {};

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) {};

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches,
                                        bool recurse,
                                        bool SchedParent) {};

    virtual void allocateCMDataAddRequires(Task* task, const MPMMaterial* matl,
                                           const PatchSet* patch, 
                                           MPMLabel* lb){};

    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
                                   ParticleSubset* addset,
                                   map<const VarLabel*, 
                                   ParticleVariableBase*>* newState,
                                   ParticleSubset* delset,
                                   DataWarehouse* old_dw){};

    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to){};

    virtual void initializeInternalVars(ParticleSubset* pset,
                                        DataWarehouse* new_dw){};

    virtual void getInternalVars(ParticleSubset* pset,
                                 DataWarehouse* old_dw){};

    virtual void allocateAndPutInternalVars(ParticleSubset* pset,
                                            DataWarehouse* new_dw){}; 

    virtual void allocateAndPutRigid(ParticleSubset* pset,
                                     DataWarehouse* new_dw){}; 

    virtual void updateElastic(const particleIndex idx){};

    virtual void updatePlastic(const particleIndex idx, 
                               const double& delGamma){};

    //////////
    /*! \brief Calculate the flow stress */
    //////////
    virtual double computeFlowStress(const PlasticityState* state,
                                     const double& delT,
                                     const double& tolerance,
                                     const MPMMaterial* matl,
                                     const particleIndex idx) = 0;
 
    //////////
    /*! \brief Calculate the plastic strain rate [epdot(tau,ep,T)] */
    //////////
    virtual double computeEpdot(const PlasticityState* state,
                                const double& delT,
                                const double& tolerance,
                                const MPMMaterial* matl,
                                const particleIndex idx) = 0;
 
    /*! Compute the elastic-plastic tangent modulus 
      This is given by
      \f[ 
      C_{ep} = C_e - \frac{(C_e:r) (x) (f_s:C_e)}
      {-f_q.h + f_s:C_e:r}
      \f]
      where \n
      \f$ C_{ep} \f$ is the continuum elasto-plastic tangent modulus \n
      \f$ C_{e} \f$ is the continuum elastic tangent modulus \n
      \f$ r \f$ is the plastic flow direction \f$ d\phi/d\sigma = r \f$\n
      \f$ h \f$ gives the evolution of \f$ q \f$ \n
      \f$ f_s = \partial f /\partial \sigma \f$ \n
      \f$ f_q = \partial f /\partial q \f$ 
    */
    virtual void computeTangentModulus(const Matrix3& stress,
                                       const PlasticityState* state,
                                       const double& delT,
                                       const MPMMaterial* matl,
                                       const particleIndex idx,
                                       TangentModulusTensor& Ce,
                                       TangentModulusTensor& Cep){};

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to scalar and
        internal variables.

      \return Three derivatives in Vector derivs 
        (derivs[0] = \f$d\sigma_Y/d\dot\epsilon\f$,
         derivs[1] = \f$d\sigma_Y/dT\f$, 
         derivs[2] = \f$d\sigma_Y/d(int. var.)\f$)
    */
    ///////////////////////////////////////////////////////////////////////////
    virtual void evalDerivativeWRTScalarVars(const PlasticityState* state,
                                             const particleIndex idx,
                                             Vector& derivs) {};

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to plastic
        strain.

      \return \f$d\sigma_Y/d\epsilon_p\f$
    */
    ///////////////////////////////////////////////////////////////////////////
    virtual double evalDerivativeWRTPlasticStrain(const PlasticityState* state, 
                                                  const particleIndex idx) = 0;

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to plastic
        strain rate.

      \return \f$d\sigma_Y/d\dot{\epsilon_p}\f$
    */
    ///////////////////////////////////////////////////////////////////////////
    virtual double evalDerivativeWRTStrainRate(const PlasticityState* state,
                                               const particleIndex idx) = 0;

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Compute the shear modulus. 
    */
    ///////////////////////////////////////////////////////////////////////////
    virtual double computeShearModulus(const PlasticityState* state) = 0;

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Compute the melting temperature
    */
    ///////////////////////////////////////////////////////////////////////////
    virtual double computeMeltingTemp(const PlasticityState* state) = 0;
  };
} // End namespace Uintah
      


#endif  // __FLOW_MODEL_H__

