#ifndef __YIELD_CONDITION_H__
#define __YIELD_CONDITION_H__

#include <Packages/Uintah/Core/Math/Matrix3.h>

namespace Uintah {

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

      This is for the associated flow rule.
    */
    /////////////////////////////////////////////////////////////////////////
    virtual void evalDerivOfYieldFunction(const Matrix3& stress,
                                          const double flowStress,
                                          const double porosity,
                                          Matrix3& derivative) = 0;

    /////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Evaluate the derivative of the yield function \f$(\Phi)\f$
      with respect to \f$s_{ij}\f$.

      This is for the associated flow rule with \f$s_{ij}\f$ being
      the deviatoric stress.
    */
    /////////////////////////////////////////////////////////////////////////
    virtual void evalDevDerivOfYieldFunction(const Matrix3& stress,
					     const double flowStress,
					     const double porosity,
					     Matrix3& derivative) = 0;
  };
} // End namespace Uintah
      
#endif  // __YIELD_CONDITION_H__

