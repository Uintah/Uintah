#ifndef __COPPER_SPECIFIC_HEAT_MODEL_H__
#define __COPPER_SPECIFIC_HEAT_MODEL_H__

#include "SpecificHeatModel.h"
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class CopperCp
   *  \brief The specific heat model for copper
   *  \author Biswajit Banerjee, 
   *  \author C-SAFE and Department of Mechanical Engineering,
   *  \author University of Utah.
   *  \author Copyright (C) 2005 Container Dynamics Group
   *
      The specific heat is given by
      for \f$(T < T_0)\f$
      \f[
          C_p = A0*T^3 - B0*T^2 + C0*T - D0
      \f]
      for \f$(T > T_0)\f$
      \f[
          C_p = A1*T + B1
      \f]
      where \f$T\f$ is the temperature.

      The input file should contain the following (in consistent
      units - the default is SI):
      <T_transition>   </T_transition> \n
      <A_LowT>         </A_LowT> \n
      <B_LowT>         </B_LowT> \n
      <C_LowT>         </C_LowT> \n
      <D_LowT>         </D_LowT> \n
      <A_HighT>        </A_HighT> \n
      <B_HighT>        </B_HighT> \n
  */

  class CopperCp : public SpecificHeatModel {

  private:

    double d_T0;
    double d_A0;
    double d_B0;
    double d_C0;
    double d_D0;
    double d_A1;
    double d_B1;
    CopperCp& operator=(const CopperCp &smm);

  public:
	 
    /*! Construct a copper specific heat model. */
    CopperCp(ProblemSpecP& ps);

    /*! Construct a copy of copper specific heat model. */
    CopperCp(const CopperCp* smm);

    /*! Destructor of copper specific heat model.   */
    virtual ~CopperCp();
	 
    virtual void outputProblemSpec(ProblemSpecP& ps);
	 
    /*! Compute the specific heat */
    double computeSpecificHeat(const PlasticityState* state);
  };
} // End namespace Uintah
      
#endif  // __COPPER_SPECIFIC_HEAT_MODEL_H__

