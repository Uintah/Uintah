#ifndef __GURSON_YIELD_MODEL_H__
#define __GURSON_YIELD_MODEL_H__

#include "YieldCondition.h"	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class  GursonYield
   *  \brief  Gurson-Tvergaard-Needleman Yield Condition.
   *  \author Biswajit Banerjee
   *  \author C-SAFE and Department of Mechanical Engineering
   *  \author University of Utah
   *  \author Copyright (C) 2003 Container Dynamics Group
   *  \warning The stress tensor is the Cauchy stress and not the 
   *           Kirchhoff stress.

      The yield condition is given by
      \f[ 
      \Phi(\sigma,k,T) = 
      \frac{\sigma_{eq}^2}{\sigma_f} + 
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

  class GursonYield : public YieldCondition {

  public:

    /*! \struct CMData
        \brief Constants needed for GTN model */
    struct CMData {
      double q1;  /**< Constant q_1 */
      double q2;  /**< Constant q_2 */
      double q3;  /**< Constant q_3 */
      double k;   /**< Constant k */
      double f_c; /**< Critical void volume fraction */
    };

  private:

    CMData d_constant;

    // Prevent copying of this class
    // copy constructor
    GursonYield(const GursonYield &);
    GursonYield& operator=(const GursonYield &);

  public:

    //! Constructor
    /*! Creates a Gurson Yield Function object */
    GursonYield(ProblemSpecP& ps);
	 
    //! Destructor 
    ~GursonYield();
	 
    //! Evaluate the yield function.
    double evalYieldCondition(const double equivStress,
			      const double flowStress,
			      const double traceOfCauchyStress,
			      const double porosity);
  };

} // End namespace Uintah

#endif  // __GURSON_YIELD_MODEL_H__ 
