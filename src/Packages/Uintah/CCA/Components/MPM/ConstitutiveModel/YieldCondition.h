#ifndef __YIELD_CONDITION_H__
#define __YIELD_CONDITION_H__

#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/PlasticityModel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/DamageModel.h>


namespace Uintah {

  //! YieldCondition
  /*! 
    A generic wrapper for various yield conditions
 
    Biswajit Banerjee, 
    C-SAFE and Department of Mechanical Engineering,
    University of Utah.

    Copyright (C) 2003 Container Dynamics Group
 
    KEYWORDS :
    Yield Conditions, von Mises, Gurson-Tvergaard-Needleman, Rousselier
 
    DESCRIPTION :
    Provides an abstract base class for various yield conditions used
    in the plasticity and damage models
   
    WARNING :
    Mixing and matching yield conditions with damage and plasticity 
    models should be done with care.  No checks are provided to stop
    the user from using the wrong combination of models.
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

