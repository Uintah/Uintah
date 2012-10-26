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


#ifndef Uintah_TableInterface_h
#define Uintah_TableInterface_h

#include <Core/Grid/Variables/CCVariable.h>

namespace Uintah {

/****************************************

CLASS
   TableInterface
   
   Short description...

GENERAL INFORMATION

   TableInterface.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   TableInterface

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class TableInterface {
  public:
    TableInterface();
    virtual ~TableInterface();

    virtual void outputProblemSpec(ProblemSpecP& ps) = 0;
    virtual void addIndependentVariable(const string&) = 0;
    virtual int addDependentVariable(const string&) = 0;
    
    virtual void setup(const bool cerrSwitch) = 0;
    
    virtual void interpolate(int index, CCVariable<double>& result,
                             const CellIterator&,
                             vector<constCCVariable<double> >& independents) = 0;
    virtual double interpolate(int index, vector<double>& independents) = 0;

  private:
  };
} // End namespace Uintah
    
#endif
