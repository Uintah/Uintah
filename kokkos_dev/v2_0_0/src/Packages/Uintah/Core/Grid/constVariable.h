#ifndef UINTAH_HOMEBREW_CONSTVARIABLE_H
#define UINTAH_HOMEBREW_CONSTVARIABLE_H

#include <Packages/Uintah/Core/Grid/constVariableBase.h>
#include <Core/Util/Assert.h>

namespace Uintah {

  class TypeDescription;

  /**************************************

CLASS
   constVariable
   
   Version of *Variable that is const in the sense that you can't
   modify the data that it points to (although you can change what it
   points to if it is a non-const version of the constVariableBase).

GENERAL INFORMATION

   constVariable.h

   Wayne Witzel
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2001 SCI Group

KEYWORDS
   Variable, const

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  template<class VariableBase, class Variable, class T, class Index> 
  class constVariable : public constVariableBase<VariableBase> {
  public:
    typedef T value_type;

    constVariable()
      : rep_() {}

    constVariable(const Variable& copy)
      : rep_(copy) {}

    constVariable<VariableBase, Variable, T, Index>&
    operator=(const constVariable<VariableBase, Variable, T, Index>& v)
    { copyPointer(v.rep_); return *this; }

    constVariable<VariableBase, Variable, T, Index>& operator=(const Variable& v)
    { copyPointer(v); return *this; }
    
    constVariableBase<VariableBase>&
    operator=(const constVariableBase<VariableBase>& v)
    {
      const constVariable<VariableBase, Variable, T, Index>* cvp =
	dynamic_cast<const constVariable<VariableBase, Variable, T, Index>*>(&v);
      ASSERT(cvp != 0);
      copyPointer(cvp->rep_);
      return *this;
    }

    constVariableBase<VariableBase>&
    operator=(const VariableBase& v)
    { copyPointer(v); return *this; }

    Variable& castOffConst() {
      return rep_;
    }
   
    virtual ~constVariable() {}

    operator const Variable&() const
    { return rep_; }
    virtual const VariableBase& getBaseRep()
    { return rep_; }

    // It's ok for a constVariable to copyPointer of a const variable
    // (even though a non-const variable can't).
    inline void copyPointer(const Variable& copy)
    { rep_.copyPointer(const_cast<Variable&>(copy)); }
    virtual void copyPointer(const VariableBase& copy)
    { rep_.copyPointer(const_cast<VariableBase&>(copy)); }

    virtual const VariableBase* clone() const
    { return rep_.clone(); }

    virtual VariableBase* cloneType() const
    { return rep_.cloneType(); }

    inline const T& operator[](Index idx) const
    { return rep_[idx]; }
     
    virtual const TypeDescription* virtualGetTypeDescription() const
    { return rep_.virtualGetTypeDescription(); }

  protected:
    Variable rep_;
  };

} // end namespace Uintah


#endif

