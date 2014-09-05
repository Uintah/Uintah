/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

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
 *  Copyright (C) 2002 SCI Group
 */

#ifndef UINTAH_COMPONENTS_SCHEDULERS_DEPENDENCY_EXCEPTION_H
#define UINTAH_COMPONENTS_SCHEDULERS_DEPENDENCY_EXCEPTION_H

#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Patch.h>
#include <Core/Exceptions/Exception.h>
#include <string>

namespace Uintah {

  using SCIRun::Exception;

  class DependencyException : public Exception {
  public:
    DependencyException(const Task* task, const VarLabel* label,
			int matlIndex, const Patch* patch,
			string has, string needs,
                        const char* file, int line);
    DependencyException(const DependencyException& copy);
    virtual ~DependencyException() {}

    static string
    makeMessage(const Task* task, const VarLabel* label, int matlIndex,
		const Patch* patch, string has, string needs);
     
    virtual const char* message() const;
    virtual const char* type() const;
  protected:
  private:
    DependencyException& operator=(const DependencyException& copy);
    const Task* task_;
    const VarLabel* label_;
    int matlIndex_;
    const Patch* patch_;
    string d_msg;
  };

} // End namespace Uintah

#endif


