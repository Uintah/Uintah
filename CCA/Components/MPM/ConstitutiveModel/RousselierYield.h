#ifndef __ROUSSELIER_YIELD_MODEL_H__
#define __ROUSSELIER_YIELD_MODEL_H__

#include "YieldCondition.h"	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class  RousselierYield
   *  \brief  Rousselier Yield Condition.
   *  \author Biswajit Banerjee
   *  \author C-SAFE and Department of Mechanical Engineering
   *  \author University of Utah
   *  \author Copyright (C) 2003 Container Dynamics Group
   *  \warning The stress tensor is the Cauchy stress and not the 
   *           Kirchhoff stress.

      The yield condition is given by
      \f[ 
      \Phi(\sigma,k,T) = 
      \frac{\sigma_{eq}}{1-f} + 
      D \sigma_1 f \exp \left(\frac{(1/3)Tr(\sigma)}{(1-f)\sigma_1}\right) -
      \sigma_f = 0 
      \f]
      where \f$\Phi(\sigma,k,T)\f$ is the yield condition,
      \f$\sigma\f$ is the Cauchy stress,
      \f$k\f$ is a set of internal variable that evolve with time,
      \f$T\f$ is the temperature,
      \f$\sigma_{eq}\f$ is the von Mises equivalent stress given by
      \f$ \sigma_{eq} = \sqrt{\frac{3}{2}\sigma^{d}:\sigma^{d}}\f$ where
      \f$\sigma^{d}\f$ is the deviatoric part of the Cauchy stress, 
      \f$\sigma_{f}\f$ is the flow stress,
      \f$D,\sigma_1\f$ are material constants, and
      \f$f\f$ is the porosity (void volume fraction).
   */

  class RousselierYield : public YieldCondition {

  public:

    /*! \struct CMData
        \brief Constants needed for Rousselier model */
    struct CMData {
      double D;  /**< Constant D */
      double sig_1;  /**< Constant \sigma_1 */
    };

  private:

    CMData d_constant;

    // Prevent copying of this class
    // copy constructor
    RousselierYield(const RousselierYield &);
    RousselierYield& operator=(const RousselierYield &);

  public:

    //! Constructor
    /*! Creates a Rousselier Yield Function object */
    RousselierYield(ProblemSpecP& ps);
	 
    //! Destructor 
    ~RousselierYield();
	 
    //! Evaluate the yield function.
    double evalYieldCondition(const double equivStress,
			      const double flowStress,
			      const double traceOfCauchyStress,
			      const double porosity);
  };

} // End namespace Uintah

#endif  // __ROUSSELIER_YIELD_MODEL_H__ 
