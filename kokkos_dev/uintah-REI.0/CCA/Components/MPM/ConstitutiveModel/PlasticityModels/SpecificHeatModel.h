#ifndef __SPECIFIC_HEAT_MODEL_H__
#define __SPECIFIC_HEAT_MODEL_H__

#include "PlasticityState.h"
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

namespace Uintah {

  /*! \class SpecificHeatModel
   *  \brief A generic wrapper for various specific heat models
   *  \author Biswajit Banerjee, 
   *  \author C-SAFE and Department of Mechanical Engineering,
   *  \author University of Utah.
   *  \author Copyright (C) 2005 Container Dynamics Group
   *
   * Provides an abstract base class for various specific heat models
  */
  class SpecificHeatModel {

  public:
	 
    //! Construct a specific heat model.  
    /*! This is an abstract base class. */
    SpecificHeatModel();

    //! Destructor of specific heat model.  
    /*! Virtual to ensure correct behavior */
    virtual ~SpecificHeatModel();
	 
    virtual void outputProblemSpec(ProblemSpecP& ps) = 0;
	 
    /////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Compute the specific heat
    */
    /////////////////////////////////////////////////////////////////////////
    virtual double computeSpecificHeat(const PlasticityState* state) = 0;
  };
} // End namespace Uintah
      
#endif  // __SPECIFIC_HEAT_MODEL_H__

