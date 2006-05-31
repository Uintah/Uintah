#ifndef __CONSTANT_SPECIFIC_HEAT_MODEL_H__
#define __CONSTANT_SPECIFIC_HEAT_MODEL_H__

#include "SpecificHeatModel.h"
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class ConstantCp
   *  \brief The specfic heat does not vary with temperature
   *  \author Biswajit Banerjee, 
   *  \author C-SAFE and Department of Mechanical Engineering,
   *  \author University of Utah.
   *  \author Copyright (C) 2005 Container Dynamics Group
   *
  */
  class ConstantCp : public SpecificHeatModel {

  private:
    ConstantCp& operator=(const ConstantCp &smm);

  public:
	 
    /*! Construct a constant specfic heat model. */
    ConstantCp(ProblemSpecP& ps);

    /*! Construct a copy of constant specfic heat model. */
    ConstantCp(const ConstantCp* smm);

    /*! Destructor of constant specfic heat model.   */
    virtual ~ConstantCp();
	 
    virtual void outputProblemSpec(ProblemSpecP& ps);
	 
    /*! Compute the specfic heat */
    double computeSpecificHeat(const PlasticityState* state);
  };
} // End namespace Uintah
      
#endif  // __CONSTANT_SPECIFIC_HEAT_MODEL_H__

