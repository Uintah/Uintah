/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef __SPECIFIC_HEAT_MODEL_H__
#define __SPECIFIC_HEAT_MODEL_H__

#include "PlasticityState.h"
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>

namespace Uintah {

  /*! \class SpecificHeatModel
   *  \brief A generic wrapper for various specific heat models
   *  \author Biswajit Banerjee, 
   *  \author C-SAFE and Department of Mechanical Engineering,
   *  \author University of Utah.
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

