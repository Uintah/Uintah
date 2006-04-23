#ifndef __BPS_MELT_TEMP_MODEL_H__
#define __BPS_MELT_TEMP_MODEL_H__

#include "MeltingTempModel.h"
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class BPSMeltTemp
   *  \brief The melt temp model from Burakovsky, Preston, Siblar in
   *         the PTW plasticity model.
   *  \author Biswajit Banerjee, 
   *  \author C-SAFE and Department of Mechanical Engineering,
   *  \author University of Utah.
   *  \author Copyright (C) 2005 Container Dynamics Group
   *
  */
  class BPSMeltTemp : public MeltingTempModel {

  private:

    double d_B0;       // Bulk modulus (Guinan-Steinberg, 1974)
    double d_dB_dp0;   // Derivative of bulk modulus (Guinan-Steinberg, 1974)
    double d_G0;       // Shear modulus (Guinan-Steinberg, 1974)
    double d_dG_dp0;   // Derivative of shear modulus (Guinan-Steinberg, 1974)
    double d_kappa;  
    double d_z;        
    double d_b2rhoTm;
    double d_alpha;
    double d_lambda;
    double d_a;        
    double d_factor;        
    double d_kb;       // Boltzmann constant

    BPSMeltTemp& operator=(const BPSMeltTemp &mtm);

  public:
	 
    /*! Construct a constant melt temp model. */
    BPSMeltTemp(ProblemSpecP& ps);

    /*! Construct a copy of constant melt temp model. */
    BPSMeltTemp(const BPSMeltTemp* mtm);

    /*! Destructor of constant melt temp model.   */
    virtual ~BPSMeltTemp();
	 
    virtual void outputProblemSpec(ProblemSpecP& ps);

    /*! Compute the melt temp */
    double computeMeltingTemp(const PlasticityState* state);
  };
} // End namespace Uintah
      
#endif  // __BPS_MELT_TEMP_MODEL_H__

