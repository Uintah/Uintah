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

#ifndef UINTAH_HOMEBREW_ReductionVARIABLE_H
#define UINTAH_HOMEBREW_ReductionVARIABLE_H

#include <Core/Grid/Variables/ReductionVariableBase.h>
#include <Core/Grid/Variables/DataItem.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/TypeUtils.h>
#include <Core/Exceptions/TypeMismatchException.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/Variables/Reductions.h>
#include <Core/Util/Endian.h>

#include <iosfwd>
#include <iostream>


namespace Uintah {

/**************************************

CLASS
   ReductionVariable
   
   Short description...

GENERAL INFORMATION

   ReductionVariable.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   Reduction_Variable

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  template<class T, class Op> class ReductionVariable : public ReductionVariableBase {
  public:
    inline ReductionVariable() {}
    inline ReductionVariable(T value) : value(value) {}
    inline ReductionVariable(const ReductionVariable<T, Op>& copy) :
      value(copy.value) {}
    virtual ~ReductionVariable();
      
    static const TypeDescription* getTypeDescription();
      
    inline operator T () const {
      return value;
    }
    virtual ReductionVariableBase* clone() const;
    virtual void copyPointer(Variable&);

    virtual void reduce(const ReductionVariableBase&);

    virtual void print(std::ostream& out) const { out << value; }

    virtual void emitNormal(std::ostream& out, const IntVector& /*l*/,
                            const IntVector& /*h*/, ProblemSpecP /*varnode*/, bool /*outputDoubleAsFloat*/)
    { out.write((char*)&value, sizeof(double)); }

    virtual void readNormal(std::istream& in, bool swapBytes)
    {
      in.read((char*)&value, sizeof(double));
      if (swapBytes) SCIRun::swapbytes(value);
    }
     
    virtual void allocate(const Patch*, const IntVector& /*boundary*/)
    {
      SCI_THROW(SCIRun::InternalError("Should not call ReductionVariable<T, Op>"
                                      "::allocate(const Patch*)", __FILE__, __LINE__)); 
    }

    virtual const TypeDescription* virtualGetTypeDescription() const;
    virtual void getMPIInfo(int& count, MPI_Datatype& datatype, MPI_Op& op);
    virtual void getMPIData(std::vector<char>& buf, int& index);
    virtual void putMPIData(std::vector<char>& buf, int& index);
    virtual void getSizeInfo(string& elems, unsigned long& totsize,
                             void*& ptr) const {
      elems="1";
      totsize = sizeof(T);
      ptr = 0;
    }

    //! Sets the value to a harmless value that will have no impact
    //! on a reduction.
    virtual void setBenignValue() {
      Op op;
      value = op.getBenignValue();
    }
  private:
    static TypeDescription* td;
    ReductionVariable<T, Op>& operator=(const ReductionVariable<T, Op>&copy);
    static Variable* maker();
    T value;
  };
   
  template<class T, class Op>
  TypeDescription* ReductionVariable<T, Op>::td = 0;
  
  template<class T, class Op>
  const TypeDescription*
  ReductionVariable<T, Op>::getTypeDescription()
  {
    if(!td){
      T* junk=0;
      td = scinew TypeDescription(TypeDescription::ReductionVariable,
                                  "ReductionVariable", &maker,
                                  fun_getTypeDescription(junk));
    }
    return td;
  }

  template<class T, class Op>
  Variable*
  ReductionVariable<T, Op>::maker()
  {
          
     return scinew ReductionVariable<T, Op>();
  }
   
  template<class T, class Op>
  const TypeDescription*
  ReductionVariable<T, Op>::virtualGetTypeDescription() const
  {
    return getTypeDescription();
  }
   
  template<class T, class Op>
  ReductionVariable<T, Op>::~ReductionVariable()
  {
  }

  template<class T, class Op>
  ReductionVariableBase*
  ReductionVariable<T, Op>::clone() const
  {
    return scinew ReductionVariable<T, Op>(*this);
  }

  template<class T, class Op>
  void
  ReductionVariable<T, Op>::copyPointer(Variable& copy)
  {
    const ReductionVariable<T, Op>* c = dynamic_cast<const ReductionVariable<T, Op>* >(&copy);
    if(!c)
      SCI_THROW(TypeMismatchException("Type mismatch in reduction variable", __FILE__, __LINE__));
    *this = *c;
  }
   
  template<class T, class Op>
  void
  ReductionVariable<T, Op>::reduce(const ReductionVariableBase& other)
  {
    const ReductionVariable<T, Op>* c = dynamic_cast<const ReductionVariable<T, Op>* >(&other);
    if(!c)
      SCI_THROW(TypeMismatchException("Type mismatch in reduction variable", __FILE__, __LINE__));
    Op op;
    value = op(value, c->value);
  }
   
  template<class T, class Op>
  ReductionVariable<T, Op>&
  ReductionVariable<T, Op>::operator=(const ReductionVariable<T, Op>& copy)
  {
    value = copy.value;
    return *this;
  }
  
} // End namespace Uintah

#endif
