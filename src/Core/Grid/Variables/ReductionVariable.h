/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/TypeUtils.h>
#include <Core/Exceptions/TypeMismatchException.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>
#include <Core/Exceptions/InternalError.h>

#include <cstring>
#include <iosfwd>
#include <memory>

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

  // Uses C++11's shared_ptr to handle memory management.
  template<class T, class Op> class ReductionVariable : public ReductionVariableBase {
  public:
    inline ReductionVariable() : value(std::make_shared<T>()) {}
    inline ReductionVariable(T value) : value(std::make_shared<T>(value)) {}
    inline ReductionVariable(const ReductionVariable<T, Op>& copy) : value(copy.value) {}

    virtual void copyPointer(Variable&);

    virtual ~ReductionVariable() {};
      
    virtual const TypeDescription* virtualGetTypeDescription() const {
      return getTypeDescription();
    };

    static const TypeDescription* getTypeDescription();
      
    inline operator T () const {
      return *value;
    }
    
    inline T& get() {
      return *value;
    }
     
    inline const T& get() const {
      return *value;
    }
     
    void setData(const T& val) {
      value = std::make_shared<T>(val);
    };

    virtual ReductionVariableBase* clone() const {
      return scinew ReductionVariable<T, Op>(*this);
    } ;
  private:
    ReductionVariable<T, Op>& operator=(const ReductionVariable<T, Op>&copy) {
      value = copy.value;
      return *this;
    };
  public:
    virtual void getSizeInfo(std::string& elems, unsigned long& totsize, void*& ptr) const {
      elems="1";
      totsize = sizeof(T);
      ptr = getBasePointer();
    }

    virtual size_t getDataSize() const {
      return sizeof(T);
    }

    virtual void* getBasePointer() const {
      return value.get();
      // return (void*)&value;
    }

    virtual bool copyOut(void* dst) const {
      void* src = (void*)(&value);
      size_t numBytes = getDataSize();
      void* retVal = std::memcpy(dst, src, numBytes);
      return (retVal == dst);
    }

    virtual void emitNormal(std::ostream& out, const IntVector& /*l*/, const IntVector& /*h*/,
                            ProblemSpecP /*varnode*/, bool /*outputDoubleAsFloat*/) {
      ssize_t linesize = (ssize_t)(sizeof(T));

      out.write((char*) (value.get()), linesize);
      // out.write((char*) &value, linesize);
    }

    virtual void readNormal(std::istream& in, bool swapBytes) {
      ssize_t linesize = (ssize_t)(sizeof(T));

      T val;

      in.read((char*) &val, linesize);
       
      if (swapBytes)
        Uintah::swapbytes(val);
       
      value = std::make_shared<T>(val);
    }

    virtual void print(std::ostream& out) const {
      out << *(value.get());
    }

    virtual void reduce(const ReductionVariableBase&);

    virtual void getMPIInfo(int& count, MPI_Datatype& datatype, MPI_Op& op);
    virtual void getMPIData(std::vector<char>& buf, int& index);
    virtual void putMPIData(std::vector<char>& buf, int& index);

    //! Sets the value to a harmless value that will have no impact
    //! on a reduction.
    virtual void setBenignValue() {
      Op op;
      value = std::make_shared<T>(op.getBenignValue());
    }

    // check if the value is benign value
    virtual bool isBenignValue() const {
      Op op;
      return (*(value.get()) == op.getBenignValue());
    }

    // Static variable whose entire purpose is to cause the
    // (instantiated) type of this class to be registered with the
    // Core/Disclosure/TypeDescription class when this class' object
    // code is originally loaded from the shared library.  The
    // 'registerMe' variable is not used for anything else in the
    // program.
    static TypeDescription::Register registerMe;

  private:
    static TypeDescription* td;
    static Variable* maker() {
      return scinew ReductionVariable<T, Op>();
    };

    std::shared_ptr<T> value;
    // T value;
  };  // end class ReductionVariable

  
  template<class T, class Op>
  TypeDescription* ReductionVariable<T, Op>::td = nullptr;
   
  // The following line is the initialization (creation) of the
  // 'registerMe' static variable (for each version of CCVariable
  // (double, int, etc)).  Note, the 'registerMe' variable is created
  // when the object code is initially loaded (usually during intial
  // program load by the operating system).
  template<class T, class Op>
  TypeDescription::Register
  ReductionVariable<T, Op>::registerMe( getTypeDescription() );

  template<class T, class Op>
  const TypeDescription*
  ReductionVariable<T, Op>::getTypeDescription()
  {
    if(!td) {

      // this is a hack to get a non-null ReductionVariable var for some
      // functions the ReductionVariables are used in (i.e., task->computes).
      // Since they're not fully-qualified variables, maker would fail
      // anyway.  And since most instances use Handle, it would be
      // difficult.
      T* tmp = nullptr;
      td = scinew TypeDescription(TypeDescription::ReductionVariable,
                                  "ReductionVariable", &maker,
                                  fun_getTypeDescription(tmp));
    }
    return td;
  }

  template<class T, class Op>
  void
  ReductionVariable<T, Op>::copyPointer(Variable& copy)
  {
    const ReductionVariable<T, Op>* c = dynamic_cast<const ReductionVariable<T, Op>* >(&copy);
    if(!c) {
      SCI_THROW(TypeMismatchException("Type mismatch in reduction variable", __FILE__, __LINE__));
    }
    *this = *c;
  }
   
  template<class T, class Op>
  void
  ReductionVariable<T, Op>::reduce(const ReductionVariableBase& other)
  {
    const ReductionVariable<T, Op>* c = dynamic_cast<const ReductionVariable<T, Op>* >(&other);
    if(!c) {
      SCI_THROW(TypeMismatchException("Type mismatch in reduction variable", __FILE__, __LINE__));
    }
    Op op;
    T val = op(*(value.get()), *(c->value.get()));
    value = std::make_shared<T>(val);
  }

} // End namespace Uintah

#endif
