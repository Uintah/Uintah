#ifndef UINTAH_HOMEBREW_CONSTGRIDVARIABLE_H
#define UINTAH_HOMEBREW_CONSTGRIDVARIABLE_H

#include <Packages/Uintah/Core/Grid/Variables/constVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/Array3.h>
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
  
   Copyright (C) 2001 SCI Group

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

} // end namespace Uintah


#endif

