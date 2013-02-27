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

#ifndef __NONE_CHECK_H__
#define __NONE_CHECK_H__

#include "StabilityCheck.h"     
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Math/FastMatrix.h>
#include <Core/Math/TangentModulusTensor.h>

namespace Uintah {

  /*! \class NoneCheck
   *  \brief Do not check for loss of ellipticity/hyperbolicity.
   *  \author  Biswajit Banerjee, \n
   *           C-SAFE and Department of Mechanical Engineering,\n
   *           University of Utah.\n
  */
  class NoneCheck : public StabilityCheck {

  public:
         
    //! Construct an object that can be used to check stability
    NoneCheck();
    NoneCheck(ProblemSpecP& ps);
    NoneCheck(const NoneCheck* cm);

    //! Destructor of stability check
    ~NoneCheck();

    virtual void outputProblemSpec(ProblemSpecP& ps);

    virtual bool doIt() {
      return false;
    };
         
    /*! Check the stability.

      \return true if unstable
      \return false if stable
    */
    bool checkStability(const Matrix3& stress,
                        const Matrix3& deformRate,
                        const TangentModulusTensor& tangentModulus,
                        Vector& direction);

  private:


    // Prevent copying of this class and copy constructor
    //NoneCheck(const NoneCheck &);
    NoneCheck& operator=(const NoneCheck &);
  };
} // End namespace Uintah
      
#endif  // __NONE_CHECK_H__

