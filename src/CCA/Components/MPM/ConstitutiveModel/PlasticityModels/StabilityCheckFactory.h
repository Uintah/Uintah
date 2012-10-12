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

#ifndef _STABILITYCHECKFACTORY_H_
#define _STABILITYCHECKFACTORY_H_

#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  // Forward declarations
  class StabilityCheck;

  /*! \class StabilityCheckFactory
   *  \brief Creates instances of stability check methods
   *  \author  Biswajit Banerjee,
   *  \author  C-SAFE and Department of Mechanical Engineering,
   *  \author  University of Utah.
   *  \warning Currently implemented stability checks
   *           Acoustic tensor (loss of ellipticity/hyperbolicity)
  */

  class StabilityCheckFactory {

  public:

    //! Create a yield condition from the input file problem specification.
    static StabilityCheck* create(ProblemSpecP& ps);
    static StabilityCheck* createCopy(const StabilityCheck* sc);
  };
} // End namespace Uintah
      
#endif /* _STABILITYCHECKFACTORY_H_ */
