/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#ifndef _SHEARMODULUSMODELFACTORY_H_
#define _SHEARMODULUSMODELFACTORY_H_

#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  // Forward declarations
  class ShearModulusModel;

  /*! \class ShearModulusModelFactory
   *  \brief Creates instances of Shear Modulus Models
   *  \author  Biswajit Banerjee,
   *  \author  C-SAFE and Department of Mechanical Engineering,
   *  \author  University of Utah.
  */

  class ShearModulusModelFactory {

  public:

    //! Create a shear modulus model from the input file problem specification.
    static ShearModulusModel* create(ProblemSpecP& ps);
    static ShearModulusModel* createCopy(const ShearModulusModel* yc);
  };
} // End namespace Uintah
      
#endif /* _SHEARMODULUSMODELFACTORY_H_ */
