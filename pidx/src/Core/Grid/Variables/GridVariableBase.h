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

#ifndef UINTAH_HOMEBREW_GRIDVARIABLE_H
#define UINTAH_HOMEBREW_GRIDVARIABLE_H

#include <Core/Grid/Variables/Variable.h>
#include <Core/Parallel/BufferInfo.h>
#include <Core/Geometry/IntVector.h>
namespace Uintah {

  using SCIRun::IntVector;

  /**************************************

CLASS
   GridVariable
   
   Short description...

GENERAL INFORMATION

   GridVariable.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   GridVariable

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class GridVariableBase : public Variable {
  public:
    virtual ~GridVariableBase() {}
      
    virtual bool rewindow(const IntVector& low, const IntVector& high) = 0;
    virtual void offset(const IntVector& offset) = 0;

    virtual GridVariableBase* cloneType() const = 0;

    using Variable::allocate; // Quiets PGI compiler warning about hidden virtual function...
    virtual void allocate(const IntVector& lowIndex, const IntVector& highIndex) = 0;
    virtual void allocate(const GridVariableBase* src) { allocate(src->getLow(), src->getHigh()); }
    virtual void allocate(const Patch* patch, const IntVector& boundary) = 0;
    
    virtual void getMPIBuffer(BufferInfo& buffer,
                              const IntVector& low, const IntVector& high);

    virtual void getSizes(IntVector& low, IntVector& high,
                          IntVector& siz) const = 0;
    virtual void getSizes(IntVector& low, IntVector& high,
                            IntVector& dataLow, IntVector& siz,
                            IntVector& strides) const = 0;
    //////////
    // Insert Documentation Here:
    virtual void copyPatch(const GridVariableBase* src,
                           const IntVector& lowIndex,
                           const IntVector& highIndex) = 0;
    
    virtual void copyData(const GridVariableBase* src) = 0;
    
    virtual void* getBasePointer() const = 0;
    virtual IntVector getLow() const = 0;
    virtual IntVector getHigh() const = 0;

    virtual const GridVariableBase* clone() const = 0;
    virtual GridVariableBase* clone() = 0;

  protected:
    GridVariableBase() {}
    GridVariableBase(const GridVariableBase&);
  private:
    GridVariableBase& operator=(const GridVariableBase&);    
  };

} // namespace Uintah
#endif
