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


/*
 *  IncorrectAllocation.h: 
 *
 *  Written by:
 *   Wayne Witzel
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 */

#ifndef UINTAH_COMPONENTS_SCHEDULERS_INCORRECT_ALLOCATION_H
#define UINTAH_COMPONENTS_SCHEDULERS_INCORRECT_ALLOCATION_H

#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Exceptions/Exception.h>
#include <string>

namespace Uintah {

  using SCIRun::Exception;

  class IncorrectAllocation : public Exception {
  public:
    IncorrectAllocation(const VarLabel* expectedLabel,
			const VarLabel* actualLabel);
    IncorrectAllocation(const IncorrectAllocation& copy);
    virtual ~IncorrectAllocation() {}

    static string makeMessage(const VarLabel* expectedLabel,
			      const VarLabel* actualLabel);
     
    virtual const char* message() const;
    virtual const char* type() const;
  protected:
  private:
    IncorrectAllocation& operator=(const IncorrectAllocation& copy);
    const VarLabel* expectedLabel_;
    const VarLabel* actualLabel_;
    string d_msg;
  };

} // End namespace Uintah

#endif


