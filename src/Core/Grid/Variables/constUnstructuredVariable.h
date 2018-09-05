/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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
#ifndef UINTAH_HOMEBREW_CONST_UNSTRUCTURED_VARIABLE_H
#define UINTAH_HOMEBREW_CONST_UNSTRUCTURED_VARIABLE_H

#include <Core/Grid/Variables/constUnstructuredVariableBase.h>
#include <Core/Util/Assert.h>

#include <sci_defs/kokkos_defs.h>

#ifdef UINTAH_ENABLE_KOKKOS
  #include <Core/Grid/Variables/Array3.h>
#endif // end UINTAH_ENABLE_KOKKOS

namespace Uintah {

  class UnstructuredTypeDescription;

  /**************************************

CLASS
   constUnstructuredVariable

   Version of *Variable that is const in the sense that you can't
   modify the data that it points to (although you can change what it
   points to if it is a non-const version of the constUnstructuredVariableBase).

GENERAL INFORMATION

   constUnstructuredVariable.h

   Wayne Witzel
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


KEYWORDS
   Variable, const

DESCRIPTION
   Long description...

WARNING

****************************************/

  template<class VariableBase, class Variable, class T, class Index>
  class constUnstructuredVariable : public constUnstructuredVariableBase<VariableBase> {
  public:
    typedef T value_type;

    constUnstructuredVariable()
      : rep_() {}

    constUnstructuredVariable(const Variable& copy)
      : rep_(copy) {}

    constUnstructuredVariable<VariableBase, Variable, T, Index>&
    operator=(const constUnstructuredVariable<VariableBase, Variable, T, Index>& v)
    { copyPointer(v.rep_); return *this; }

    constUnstructuredVariable<VariableBase, Variable, T, Index>& operator=(const Variable& v)
    { copyPointer(v); return *this; }

    constUnstructuredVariableBase<VariableBase>&
    operator=(const constUnstructuredVariableBase<VariableBase>& v)
    {
      const constUnstructuredVariable<VariableBase, Variable, T, Index>* cvp =
        dynamic_cast<const constUnstructuredVariable<VariableBase, Variable, T, Index>*>(&v);
      ASSERT(cvp != 0);
      copyPointer(cvp->rep_);
      return *this;
    }

    constUnstructuredVariableBase<VariableBase>&
    operator=(const VariableBase& v)
    { copyPointer(v); return *this; }

    // Steve writes: castOffConst() is evil.  It returns a CCVariable
    // from a constCCVariable that you can modify.  However, you
    // should NOT modify it or you will cause serious problems for
    // your simulation.  I used it in SimpleCFD as a hack to get
    // around some silliness, but if you feel tempted to use it please
    // let us know why and we will see if we can come up with a better
    // solution.  So the answer to how/where is never/nowhere.
    Variable& castOffConst() {
      return this->rep_;
    }

    virtual ~constUnstructuredVariable() {}

    operator const Variable&() const
    { return this->rep_; }
    virtual const VariableBase& getBaseRep() const
    { return this->rep_; }

    // It's ok for a constUnstructuredVariable to copyPointer of a const variable
    // (even though a non-const variable can't).
//    inline void copyPointer(const Variable& copy)
//    { this->rep_.copyPointer(const_cast<Variable&>(copy)); }
    virtual void copyPointer(const VariableBase& copy)
    { this->rep_.copyPointer(const_cast<VariableBase&>(copy)); }

    virtual const VariableBase* clone() const
      // need to cast it if it is a GridVariable
    { return dynamic_cast<const VariableBase*>(this->rep_.clone()); }

    virtual VariableBase* cloneType() const
      // need to cast it if it is a GridVariable
    { return dynamic_cast<VariableBase*>(this->rep_.cloneType()); }

    inline const T& operator[](Index idx) const
    { return this->rep_[idx]; }

#ifdef UINTAH_ENABLE_KOKKOS
      inline KokkosView3<const T> getKokkosView() const
      {
        auto v = this->rep_.getKokkosView();
        return KokkosView3<const T>( v.m_view, v.m_i, v.m_j, v.m_k );
      }
#endif

    inline const T& operator()(int i, int j, int k) const
    {
      return this->rep_(i,j,k);
    }


    virtual const UnstructuredTypeDescription* virtualGetUnstructuredTypeDescription() const
    { return this->rep_.virtualGetUnstructuredTypeDescription(); }

    // used to get size info of the underlying data; this is for host-->device variable copy
    virtual size_t getDataSize() const {
      return this->rep_.getDataSize();
    }

    // used to copy Variables to contiguous buffer prior to bulk host-->device copy
    virtual bool copyOut(void* dst) const {
      return this->rep_.copyOut(dst);
    }

  protected:
    Variable rep_;
  };

} // end namespace Uintah


#endif

