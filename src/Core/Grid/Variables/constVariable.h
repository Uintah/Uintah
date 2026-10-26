/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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
#ifndef UINTAH_HOMEBREW_CONSTVARIABLE_H
#define UINTAH_HOMEBREW_CONSTVARIABLE_H

#include <Core/Grid/Variables/constVariableBase.h>
#include <Core/Util/Assert.h>

#include <sci_defs/gpu_defs.h>

#if defined( KOKKOS_ENABLE_OPENMP ) // && defined( _OPENMP )
  #include <Core/Grid/Variables/Array3.h>
#endif

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
      : m_rep() {}

    constVariable(const Variable& copy)
      : m_rep(copy) {}

    constVariable<VariableBase, Variable, T, Index>&
    operator = (const constVariable<VariableBase, Variable, T, Index>& v){
      copyPointer(v.m_rep);
      return *this;
    }

    constVariable<VariableBase, Variable, T, Index>&
    operator = (const Variable& v) {
      copyPointer(v);
      return *this;
    }

    constVariableBase<VariableBase>&
    operator = (const constVariableBase<VariableBase>& v){

      const constVariable<VariableBase, Variable, T, Index>* cvp =
        dynamic_cast<const constVariable<VariableBase, Variable, T, Index>*>(&v);
      ASSERT(cvp != 0);
      copyPointer(cvp->m_rep);
      return *this;
    }

    constVariableBase<VariableBase>&
    operator = (const VariableBase& v){
      copyPointer(v);
      return *this;
    }

    // Steve writes: castOffConst() is evil.  It returns a CCVariable
    // from a constCCVariable that you can modify.  However, you
    // should NOT modify it or you will cause serious problems for
    // your simulation.  I used it in SimpleCFD as a hack to get
    // around some silliness, but if you feel tempted to use it please
    // let us know why and we will see if we can come up with a better
    // solution.  So the answer to how/where is never/nowhere.
    Variable& castOffConst() {
      return this->m_rep;
    }

    virtual ~constVariable() {}

    operator const Variable&() const{
      return this->m_rep;
    }

    virtual const VariableBase& getBaseRep() const {
      return this->m_rep;
    }

    // It's ok for a constVariable to copyPointer of a const variable
    // (even though a non-const variable can't).
//    inline void copyPointer(const Variable& copy)
//    { this->m_rep.copyPointer(const_cast<Variable&>(copy)); }
    virtual void copyPointer(const VariableBase& copy){
      this->m_rep.copyPointer(const_cast<VariableBase&>(copy));
    }

    // need to cast it if it is a GridVariable
    virtual const VariableBase* clone() const {
      return dynamic_cast<const VariableBase*>(this->m_rep.clone());
    }

    // need to cast it if it is a GridVariable
    virtual VariableBase* cloneType() const {
      return dynamic_cast<VariableBase*>(this->m_rep.cloneType());
    }

    inline const T& operator[](Index idx) const{
      return this->m_rep[idx];
    }

#if defined( KOKKOS_ENABLE_OPENMP ) // && defined( _OPENMP )
      inline KokkosView3<const T, Kokkos::HostSpace> getKokkosView() const{
        auto v = this->m_rep.getKokkosView();
        Array3Data<const T>* constA3Data = reinterpret_cast<Array3Data<const T>*>(v.m_A3Data);
        return KokkosView3<const T, Kokkos::HostSpace>( v.m_view, v.m_i, v.m_j, v.m_k, constA3Data );
      }
#endif

    inline const T& operator()(int i, int j, int k) const{
      return this->m_rep(i,j,k);
    }


    virtual const TypeDescription* virtualGetTypeDescription() const{
      return this->m_rep.virtualGetTypeDescription();
    }

    // used to get size info of the underlying data; this is for host-->device variable copy
    virtual size_t getDataSize() const {
      return this->m_rep.getDataSize();
    }

    // used to copy Variables to contiguous buffer prior to bulk host-->device copy
    virtual bool copyOut(void* dst) const {
      return this->m_rep.copyOut(dst);
    }

  protected:
    Variable m_rep;
  };

} // end namespace Uintah


#endif

