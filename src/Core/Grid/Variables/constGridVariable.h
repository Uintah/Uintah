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

#ifndef UINTAH_HOMEBREW_CONSTGRIDVARIABLE_H
#define UINTAH_HOMEBREW_CONSTGRIDVARIABLE_H

#include <Core/Grid/Variables/GridVariable.h>
#include <Core/Grid/Variables/constVariable.h>
#include <Core/Grid/Variables/Array3.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Util/Assert.h>

namespace Uintah {

  class TypeDescription;

  /**************************************

CLASS
   constGridVariable
   
   constVariable-based class for the grid (array3) variables:
   CC, NC, SFCX, etc.

GENERAL INFORMATION

   constGridVariable.h

   Wayne Witzel
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   Variable, const, grid

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  template<class VariableBase, class Variable, class T> 
  class constGridVariable
    : public constVariable<VariableBase, Variable, T, const IntVector&> {
  public:
    constGridVariable()
      : constVariable<VariableBase, Variable, T, const IntVector&>() {}

    constGridVariable(const Variable& copy)
      : constVariable<VariableBase, Variable, T, const IntVector&>(copy) {}

    IntVector getLowIndex() const
    { return this->rep_.getLowIndex(); }

    IntVector getHighIndex() const
    { return this->rep_.getHighIndex(); }

    IntVector getFortLowIndex() const
    { return this->rep_.getFortLowIndex(); }

    IntVector getFortHighIndex() const
    { return this->rep_.getFortHighIndex(); }

    inline const Array3Window<T>* getWindow() const {
      return this->rep_.getWindow();
    }
    
    inline const T* getPointer() const {
      return this->rep_.getPointer();
    }

    operator const Array3<T>&() const
    { return this->rep_; }

    void print(std::ostream& out) const
    { this->rep_.print(out); }
  };

  typedef constVariableBase<GridVariableBase> constGridVariableBase;
} // end namespace Uintah


#endif

