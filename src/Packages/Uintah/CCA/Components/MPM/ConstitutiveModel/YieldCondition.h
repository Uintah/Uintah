#ifndef __YIELD_CONDITION_H__
#define __YIELD_CONDITION_H__

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
	 
    //! Evaluate the yield condition \f$(\Phi)\f$.
    /*! If \f$\Phi \le 0\f$ the state is elastic.
      If \f$\Phi > 0\f$ the state is plastic and a normal return 
      mapping algorithm is necessary. */
    virtual double evalYieldCondition(const double equivStress,
                                      const double flowStress,
                                      const double traceOfCauchyStress,
                                      const double porosity) = 0;
  };
} // End namespace Uintah
      
#endif  // __YIELD_CONDITION_H__

