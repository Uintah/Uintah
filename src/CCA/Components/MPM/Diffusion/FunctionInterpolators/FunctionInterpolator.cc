/*
 * FunctionInterpolator.cc
 *
 *  Created on: Feb 7, 2019
 *      Author: jbhooper
 */

#include <CCA/Components/MPM/Diffusion/FunctionInterpolators/FunctionInterpolator.h>

namespace Uintah {
  FunctionInterpolator::FunctionInterpolator(std::string interpType
                                            ) : d_interpType(interpType)
  {

  }

  FunctionInterpolator::~FunctionInterpolator() {

  }

  void FunctionInterpolator::baseOutputInterpProblemSpec(ProblemSpecP & probSpec
                                                        ,bool           doOutput) const
  {
    if (doOutput) {
      probSpec->setAttribute("type",d_interpType);
    }
  }
}


