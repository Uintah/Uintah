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

#ifndef __MELTING_TEMP_MODEL_H__
#define __MELTING_TEMP_MODEL_H__

#include "PlasticityState.h"
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>

namespace Uintah {

  /*! \class MeltingTempModel
   *  \brief A generic wrapper for various melting temp models
   *  \author Biswajit Banerjee, 
   *  \author C-SAFE and Department of Mechanical Engineering,
   *  \author University of Utah.
   *
   * Provides an abstract base class for various melting temp models used
   * in the plasticity and damage models
  */
  class MeltingTempModel {

  public:
         
    //! Construct a melting temp model.  
    /*! This is an abstract base class. */
    MeltingTempModel();

    //! Destructor of melting temp model.  
    /*! Virtual to ensure correct behavior */
    virtual ~MeltingTempModel();

    virtual void outputProblemSpec(ProblemSpecP& ps) = 0;
         
    /////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Compute the melting temperature
    */
    /////////////////////////////////////////////////////////////////////////
    virtual double computeMeltingTemp(const PlasticityState* state) = 0;

  };
} // End namespace Uintah
      
#endif  // __MELTING_TEMP_MODEL_H__

