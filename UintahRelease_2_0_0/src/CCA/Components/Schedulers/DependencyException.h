/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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


/*
 *  DependencyException.h
 *
 *  Written by:
 *   Wayne Witzel
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#ifndef UINTAH_COMPONENTS_SCHEDULERS_DEPENDENCY_EXCEPTION_H
#define UINTAH_COMPONENTS_SCHEDULERS_DEPENDENCY_EXCEPTION_H

#include <Core/Exceptions/Exception.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarLabel.h>

#include <string>

namespace Uintah {


  class DependencyException : public Exception {

  public:
    DependencyException(const Task* task,
                        const VarLabel* label,
                              int matlIndex,
                        const Patch* patch,
                              std::string has,
                              std::string needs,
                        const char* file,
                              int line);

    DependencyException(const DependencyException& copy);

    virtual ~DependencyException() {}

    static std::string
    makeMessage(const Task* task,
                const VarLabel* label,
                      int matlIndex,
                const Patch* patch,
                      std::string has,
                      std::string needs);
     
    virtual const char* message() const;
    virtual const char* type() const;

  protected:

  private:
    DependencyException& operator=(const DependencyException& copy);
    const Task*     task_;
    const VarLabel* label_;
    int             matlIndex_;
    const Patch*    patch_;
    std::string     d_msg;
  };

} // End namespace Uintah

#endif


