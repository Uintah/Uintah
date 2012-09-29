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

#ifndef __ZERILLI_ARMSTRONG_MODEL_H__
#define __ZERILLI_ARMSTRONG_MODEL_H__

#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/FlowModel.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

////////////////////////////////////////////////////////////////////////////////
  /*!
    \class ZAFlow
    \brief Zerilli-Armstrong Strain rate dependent plasticity model
    \author Anup Bhawalkar, 
    Department of Mechanical Engineering, 
    University of Utah
   
    Zerilli-Armstrong Plasticity Model 
    (Zerilli, F.J. and Armstrong, R.W., 1987, J. Appl. Phys. 61(5), p.1816)
    (Zerilli, F.J., 2004, Metall. Materials Trans. A, v. 35A, p.2547)

    Flow rule: (the general form implemented in Uintah) 
	
	sigma = sigma_a + B*exp(-beta*T) + B_0*sqrt(ep)*exp(-alpha*T)

	where 
		if(c_0 == 0)
			sigma_a = sigma_g + (k_H/sqrt(l)) + K*(ep)^n;
		else
			sigma_a = c_0 + K*(ep)^n;
		end
		beta = beta_0 - beta_1*ln(epdot);
		alpha = alpha_0 - alpha1*ln(epdot)

   Flow rule : Original form (1987)

	Y =  A + (C1 + C2*sqrt(ep))*(exp(-C3 + C4*ln(epdot))*T) + C5*(ep)^n

  Corelation between these  two forms:
		A  = c_0
		C1 = B
		C2 = B_0
		C3 = beta_0 = alpha_0
		C4 = beta_1 = alpha_1
		C5 = K

   
  FCC Metals :

	General Form : B = 0 ; K = 0; beta_0 = beta_1 = 0
	Original From : C1 = 0; C5 = 0;

  BCC Metals :

	General Form : B_0 = 0 ; alpha_0 = alpha_1 = 0
	Original From : C2 = 0; 

  HCP Metals :

	All constants are non-zero


  Terms :

	ep    =  equvivalent plastic strain
	epdot =  equvivalent plastic strain rate
        C1, C2, C3, C4, C5, A =  Constants, choose appropriately according to the model (fcc or bcc or hcp)
	T     = Temperature
	sigma_a = athermal component of the flow stress
	k_H   = Microstructural stress intensity
	l     = Average grain diameter
	sigma_g = stres contribution due to solutes and initial dislocation density

  */
  /////////////////////////////////////////////////////////////////////////////

  class ZAFlow : public FlowModel {

  public:

    // Create datatype for storing model parameters
    struct CMData {
      double c_0;  // c_0 = sigma_g + k_H*l^(-1/2)
      double sigma_g;
      double k_H;
      double sqrt_l;
      double B;
      double beta_0;
      double beta_1;
      double B_0;
      double alpha_0;
      double alpha_1;
      double K;
      double n;
    };   

  private:

    CMData d_CM;
         
    // Prevent copying of this class
    // copy constructor
    ZAFlow& operator=(const ZAFlow &cm);

  public:

    // constructors
    ZAFlow(ProblemSpecP& ps);
    ZAFlow(const ZAFlow* cm);
         
    // destructor 
    virtual ~ZAFlow();

    virtual void outputProblemSpec(ProblemSpecP& ps);

    ///////////////////////////////////////////////////////////////////////////
    /*! \brief  compute the flow stress */
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
      deriv[2] = \f$d\sigma_Y/d\epsilon\f$)
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
       \frac{\partial\sigma}{\partial\epsilon_p} = 

        n K \epsilon_p^{n-1} + \frac{1}{2} B_0 exp(-alpha T) \epsilon_p^(-1/2)
      \f]

      \return Derivative \f$ d\sigma / d\epsilon\f$.

      \warning Expect error when \f$ \epsilon_p = 0 \f$. 
    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTPlasticStrain(const PlasticityState* state,
                                          const particleIndex idx);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to strain rate.

      The derivative is given by
      \f[
       \frac{\partial\sigma}{\partial\dot\epsilon_p} = 
         B \beta_1 T exp(-\beta T)/\dot\epsilon_p +
         B_0 \sqrt{\epsilon_p} \alpha_1 T exp(-\alpha T)/\dot\epsilon_p 
      \f]

      \return Derivative \f$ d\sigma / d\dot\epsilon \f$.
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
       \frac{\partial\sigma}{\partial T} = 
          -B_0 \alpha sqrt{\epsilon_p} exp(-\alpha T) -
          -B \beta exp(-\beta T)
      \f]

      \return Derivative \f$ d\sigma / dT \f$.
 
    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTTemperature(const PlasticityState* state,
                                        const particleIndex idx);

  };

} // End namespace Uintah

#endif  // __ZERILLI_ARMSTRONG_MODEL_H__
