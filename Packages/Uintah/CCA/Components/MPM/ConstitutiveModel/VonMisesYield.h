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
  };

} // End namespace Uintah

#endif  // __VONMISES_YIELD_MODEL_H__ 
