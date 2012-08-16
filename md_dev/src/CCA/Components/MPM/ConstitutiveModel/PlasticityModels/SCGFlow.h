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


#ifndef __SCG_FLOW_MODEL_H__
#define __SCG_FLOW_MODEL_H__


#include "FlowModel.h"    
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  ////////////////////////////////////////////////////////////////////////////
  /*! 
    \class SCGFlow
    \brief Steinberg-Cochran-Guinan-Lund plasticity model 
    \author Biswajit Banerjee, \n
    C-SAFE and Department of Mechanical Engineering, \n
    University of Utah \n
    Copyright (C) 2002-2003 University of Utah

    Reference : \n
    Steinberg, D.J., Cochran, S.G., and Guinan, M.W., (1980),
    Journal of Applied Physics, 51(3), 1498-1504.
    Steinberg and Lund, 1989, J. App. Phys. 65(4), p.1528.

    The shear modulus (\f$ \mu \f$) is a function of hydrostatic pressure 
    (\f$ p \f$) and temperature (\f$ T \f$), but independent of 
    plastic strain rate (\f$ \epsilon_p \f$), and is given by
    \f[
       \mu = \mu_0\left[1 + A\frac{p}{\eta^{1/3}} - B(T - 300)\right]
    \f]
    where,\n
    \f$ \mu_0 \f$ is the shear modulus at the reference state
    (\f$ T \f$ = 300 K, \f$ p \f$ = 0, \f$ \epsilon_p \f$ = 0), \n
    \f$ \eta = \rho/\rho_0\f$ is the compression, and \n
    \f[ 
       A = \frac{1}{\mu_0} \frac{d\mu}{dp} ~~;~~
       B = \frac{1}{\mu_0} \frac{d\mu}{dT}
    \f]

    The flow stress (\f$ \sigma \f$) is given by
    \f[
    \sigma = \sigma_0 \left[1 + \beta(\epsilon_p + \epsilon_{p0})\right]^n
        \left(\frac{\mu}{\mu_0}\right)
    \f]
    where, \n
    \f$\sigma_0\f$ is the uniaxial yield strength in the reference state, \n
    \f$\beta,~n\f$ are work hardening parameters, and \n 
    \f$\epsilon_{p0}\f$ is the initial equivalent plastic strain. \n

    The value of the flow stress is limited by the condition
    \f[
      \sigma_0\left[1 + \beta(\epsilon_p + \epsilon_{p0})\right]^n \le Y_{max}
    \f]
    where, 
    \f$ Y_{max} \f$ is the maximum value of uniaxial yield at the reference
    temperature and pressure. \n

    The melt temperature (\f$T_m\f$) varies with pressure and is given by
    \f[
        T_m = T_{m0} \exp\left[2a\left(1-\frac{1}{\eta}\right)\right]
              \eta^{2(\Gamma_0-a-1/3)}
    \f]
    where, \n
    \f$ T_{m0} \f$ is the melt temperature at \f$ \rho = \rho_0 \f$, \n
    \f$ a \f$ is the coefficient of the first order volume correction to
     Gruneisen's gamma, and \n
    \f$ \Gamma_0 \f$ is the value of Gruneisen's gamma in the reference state.
  */
  ////////////////////////////////////////////////////////////////////////////

  class SCGFlow : public FlowModel {

  public:

    // Create datatype for storing model parameters
    struct CMData {
      double mu_0; 
      double A;
      double B;
      double sigma_0;
      double beta;
      double n;
      double epsilon_p0;
      double Y_max; 
      double T_m0; 
      double a;
      double Gamma_0;
      double C1;
      double C2;
      double dislocationDensity;
      double lengthOfDislocationSegment;
      double distanceBetweenPeierlsValleys;
      double lengthOfBurgerVector;
      double debyeFrequency;
      double widthOfKinkLoop;
      double dragCoefficient;
      double kinkPairEnergy;
      double boltzmannConstant;
      double peierlsStress;
    };   

  private:

    CMData d_CM;
         
    // Prevent copying of this class
    // copy constructor
    //SCGFlow(const SCGFlow &cm);
    SCGFlow& operator=(const SCGFlow &cm);

  public:
    // constructors
    SCGFlow(ProblemSpecP& ps);
    SCGFlow(const SCGFlow* cm);
         
    // destructor 
    virtual ~SCGFlow();

    virtual void outputProblemSpec(ProblemSpecP& ps);
         
    // Computes and requires for internal evolution variables
    // Only one internal variable for SCG model :: mechanical threshold stress
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
    /*! \brief Compute the flow stress */
    ///////////////////////////////////////////////////////////////////////////
    virtual double computeFlowStress(const PlasticityState* state,
                                     const double& delT,
                                     const double& tolerance,
                                     const MPMMaterial* matl,
                                     const particleIndex idx);

    double computeThermallyActivatedYieldStress(const double& epdot,
                                                const double& T,
                                                const double& tolerance);

    //////////
    /*! \brief Calculate the plastic strain rate [epdot(tau,ep,T)] */
    //////////
    virtual double computeEpdot(const PlasticityState* state,
                                const double& delT,
                                const double& tolerance,
                                const MPMMaterial* matl,
                                const particleIndex idx);
 
    ///////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Compute the elastic-plastic tangent modulus. 

      \warning Assumes vonMises yield condition and the associated flow rule .
    */
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
      (deriv[0] = \f$d\sigma_Y/dp\f$,
      deriv[1] = \f$d\sigma_Y/dT\f$, 
      deriv[2] = \f$d\sigma_Y/d\epsilon_p\f$)
    */
    ///////////////////////////////////////////////////////////////////////////
    void evalDerivativeWRTScalarVars(const PlasticityState* state,
                                     const particleIndex idx,
                                     Vector& derivs);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to plastic strain

      The derivative is given by
      \f[
         \frac{\partial \sigma}{\partial \epsilon_p} = 
         \left(\frac{\sigma_0\mu~n~\beta}{\mu_0}\right)
         \left[1+\beta~(\epsilon_p - \epsilon_{p0})\right]^{n-1}
      \f]

      \return Derivative \f$ d\sigma_Y / d\epsilon_p\f$.

      \warning Not sure what should be done at Y = Ymax.
    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTPlasticStrain(const PlasticityState* state,
                                          const particleIndex idx);

    ///////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Evaluate derivative of flow stress with respect to 
             the plastic strain rate

      The yield function is given by
      \f[
      Y = [Y_t(epdot,T) + Y_a(ep)]*\mu(p,T)/\mu_0
      \f]
      The derivative wrt epdot is
      \f[
      dY/depdot = dY_t/depdot*\mu/\mu_0
      \f]

      The equation for Y_t in terms of epdot can be expressed as
      \f[
      A(1 - B3 Y_t)^2 - ln(B1 Y_t - B2) + ln(Y_t) = 0
      \f]
      where \f$ A = 2*U_k/(\kappa T) \f$, \f$ B1 = C1/epdot \f$ \n
      \f$ B2 = C1 C2 \f$, and \f$ B3 = 1/Y_p \f$.\n

      The solution of this equation is 
      \f[
      Y_t(epdot,T) = exp(RootOf(Z + A - 2 A B3 exp(Z) (1 - B3 exp(Z))
      - ln(B1 exp(Z) - B2) = 0))
      \f]
      The root is determined using a Newton iterative technique.

      The derivative is given by
      \f[
      dY_t/depdot = -B1 X1^2/[X4(2 X2 - 1) - 2 X3(C1(1-B3 X1) + B3 X4)]
      \f]
      where \f$ X1 = exp(Z) \f$, \f$ X2 = A B3 X1 \f$, \f$ X3 = X2 X1 \f$, \n
      \f$ X4 = B2 epdot \f$.
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

      The derivative is given by
      \f[
         \frac{\partial \sigma}{\partial T} = 
         -B\sigma_0\left[1+\beta~(\epsilon_p - \epsilon_{p0})\right]^n
      \f]

      \return Derivative \f$ d\sigma_Y / dT \f$.
    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTTemperature(const PlasticityState* state,
                                        const particleIndex idx);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to pressure.

      The derivative is given by
      \f[
         \frac{\partial \sigma}{\partial p} = 
         \frac{A}{\eta^{1/3}}
         \sigma_0\left[1+\beta~(\epsilon_p - \epsilon_{p0})\right]^n
      \f]

      \return Derivative \f$ d\sigma_Y / dp \f$.
    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTPressure(const PlasticityState* state,
                                     const particleIndex idx);

  };

} // End namespace Uintah

#endif  // __SCG_FLOW_MODEL_H__ 
