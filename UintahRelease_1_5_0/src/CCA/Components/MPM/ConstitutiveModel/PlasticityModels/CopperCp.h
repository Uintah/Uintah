/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __COPPER_SPECIFIC_HEAT_MODEL_H__
#define __COPPER_SPECIFIC_HEAT_MODEL_H__

#include "SpecificHeatModel.h"
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class CopperCp
   *  \brief The specific heat model for copper
   *  \author Biswajit Banerjee, 
   *  \author C-SAFE and Department of Mechanical Engineering,
   *  \author University of Utah.
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

