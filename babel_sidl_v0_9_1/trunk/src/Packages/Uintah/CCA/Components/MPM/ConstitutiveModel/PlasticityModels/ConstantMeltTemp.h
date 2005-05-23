#ifndef __CONSTANT_MELT_TEMP_MODEL_H__
#define __CONSTANT_MELT_TEMP_MODEL_H__

#include "MeltingTempModel.h"
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class ConstantMeltTemp
   *  \brief The melting temperature does not vary with pressure
   *  \author Biswajit Banerjee, 
   *  \author C-SAFE and Department of Mechanical Engineering,
   *  \author University of Utah.
   *  \author Copyright (C) 2004 Container Dynamics Group
   *
  */
  class ConstantMeltTemp : public MeltingTempModel {

  private:
    ConstantMeltTemp& operator=(const ConstantMeltTemp &mtm);

  public:
	 
    /*! Construct a constant melt temp model. */
    ConstantMeltTemp(ProblemSpecP& ps);

    /*! Construct a copy of constant melt temp model. */
    ConstantMeltTemp(const ConstantMeltTemp* mtm);

    /*! Destructor of constant melt temp model.   */
    virtual ~ConstantMeltTemp();
	 
    /*! Compute the melt temp */
    double computeMeltingTemp(const PlasticityState* state);
  };
} // End namespace Uintah
      
#endif  // __CONSTANT_MELT_TEMP_MODEL_H__

