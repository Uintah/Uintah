/*
 * The MIT License
 *
 * Copyright (c) 2012-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef Wasatch_VarDenParameters_h
#define Wasatch_VarDenParameters_h

#include <Core/ProblemSpec/ProblemSpec.h>

namespace Wasatch{
  
  /**
   *  \ingroup WasatchCore
   *  \struct VarDenParameters
   *  \author Tony Saad, Amir Biglari
   *  \date   June, 2012
   *
   *  \brief Holds some key parameters for supported variable density models.
   */
  struct VarDenParameters {
    VarDenParameters();

    /**
     *  \ingroup WasatchCore
     *  \enum VariableDensityModels
     *  \brief An enum listing the supported variable density models.
     */
    enum VariableDensityModels {
      CONSTANT,
      IMPULSE,
      DYNAMIC
    };

    double alpha0;
    VariableDensityModels model;
    bool onePredictor;
  };
  
  void parse_varden_input( Uintah::ProblemSpecP varDenSpec,
                               VarDenParameters& varDenParams );
  
}

#endif // Wasatch_VarDenParameters_h
