#ifndef __SCG_MELT_TEMP_MODEL_H__
#define __SCG_MELT_TEMP_MODEL_H__

#include "MeltingTempModel.h"
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class SCGMeltTemp
   *  \brief The melt temp model used by Steinberg,Cochran,Guinan in
   *         the SCG plasticity model.
   *  \author Biswajit Banerjee, 
   *  \author C-SAFE and Department of Mechanical Engineering,
   *  \author University of Utah.
   *  \author Copyright (C) 2004 Container Dynamics Group
   *
  */
  class SCGMeltTemp : public MeltingTempModel {

  private:

    double d_Gamma0; // Material constant (also in SCG model)
    double d_a;      // Material constant (also in SCG model)
    double d_Tm0;   // Material constant (also in SCG model)

    SCGMeltTemp& operator=(const SCGMeltTemp &mtm);

  public:
	 
    /*! Construct a constant melt temp model. */
    SCGMeltTemp(ProblemSpecP& ps);

    /*! Construct a copy of constant melt temp model. */
    SCGMeltTemp(const SCGMeltTemp* mtm);

    /*! Destructor of constant melt temp model.   */
    virtual ~SCGMeltTemp();
	 
    /*! Compute the melt temp */
    double computeMeltingTemp(const PlasticityState* state);
  };
} // End namespace Uintah
      
#endif  // __SCG_MELT_TEMP_MODEL_H__

