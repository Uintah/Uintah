#ifndef __VONMISES_YIELD_MODEL_H__
#define __VONMISES_YIELD_MODEL_H__

#include "YieldCondition.h"	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class  VonMisesYield
   *  \brief  von Mises-Huber Yield Condition (J2 plasticity).
   *  \author Biswajit Banerjee
   *  \author C-SAFE and Department of Mechanical Engineering
   *  \author University of Utah
   *  \author Copyright (C) 2003 Container Dynamics Group
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

  class VonMisesYield : public YieldCondition {

  private:

    // Prevent copying of this class
    // copy constructor
    VonMisesYield(const VonMisesYield &);
    VonMisesYield& operator=(const VonMisesYield &);

  public:

    //! Constructor
    /*! Creates a VonMisesYield function object */
    VonMisesYield(ProblemSpecP& ps);
	 
    //! Destructor 
    ~VonMisesYield();
	 
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
      \brief Compute the elastic-plastic tangent modulus.
    */
    /////////////////////////////////////////////////////////////////////////
    void computeElasPlasTangentModulus(const TangentModulusTensor& Ce,
				       const Matrix3& sigma, 
				       double sigY,
				       double dsigYdep,
				       double porosity,
				       double voidNuclFac,
				       TangentModulusTensor& Cep);

    /////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate the factor \f$h_1\f$ for plastic strain

      \f[
      h_1 = \frac{\sigma : f_{\sigma}}{\sigma_Y}
      \f]
      
      \return factor 
    */
    /////////////////////////////////////////////////////////////////////////
    double computePlasticStrainFactor(double sigma_f_sigma,
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
			       double h_q1,
			       TangentModulusTensor& Cep);

  };

} // End namespace Uintah

#endif  // __VONMISES_YIELD_MODEL_H__ 
