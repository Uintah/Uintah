#ifndef __STEEL_SPECIFIC_HEAT_MODEL_H__
#define __STEEL_SPECIFIC_HEAT_MODEL_H__

#include "SpecificHeatModel.h"
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class SteelCp
   *  \brief The specific heat model for steel
   *  \author Biswajit Banerjee, 
   *  \author C-SAFE and Department of Mechanical Engineering,
   *  \author University of Utah.
   *  \author Copyright (C) 2005 Container Dynamics Group
   *
      The specific heat is given by
      for \f$(T < T_c)\f$
      \f[
          C_p = A0 - B0*t + C0/t^{n0}
      \f]
      for \f$(T > T_c)\f$
      \f[
          C_p = A1 + B1*t + C1/t^{n1}
      \f]
      where \f$T\f$ is the temperature \n
            \f$ t = T/T_c - 1.0 \f$ for \f$ T > T_c \f$ \n
            \f$ t = 1.0 - T/T_c \f$ for \f$ T < T_c \f$.


      The input file should contain the following (in consistent
      units - the default is SI):
      <T_transition>   </T_transition> \n
      <A_LowT>         </A_LowT> \n
      <B_LowT>         </B_LowT> \n
      <C_LowT>         </C_LowT> \n
      <n_LowT>         </n_LowT> \n
      <A_HighT>        </A_HighT> \n
      <B_HighT>        </B_HighT> \n
      <C_HighT>        </C_HighT> \n
      <n_HighT>        </n_HighT> \n
  */
  class SteelCp : public SpecificHeatModel {

  private:

    double d_Tc;
    double d_A0;
    double d_B0;
    double d_C0;
    double d_n0;
    double d_A1;
    double d_B1;
    double d_C1;
    double d_n1;
    SteelCp& operator=(const SteelCp &smm);

  public:
	 
    /*! Construct a steel specific heat model. */
    SteelCp(ProblemSpecP& ps);

    /*! Construct a copy of steel specific heat model. */
    SteelCp(const SteelCp* smm);

    /*! Destructor of steel specific heat model.   */
    virtual ~SteelCp();
	 
    virtual void outputProblemSpec(ProblemSpecP& ps);
	 
    /*! Compute the specific heat */
    double computeSpecificHeat(const PlasticityState* state);
  };
} // End namespace Uintah
      
#endif  // __STEEL_SPECIFIC_HEAT_MODEL_H__

