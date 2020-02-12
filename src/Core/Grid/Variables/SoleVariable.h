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

#ifndef UINTAH_HOMEBREW_SoleVARIABLE_H
#define UINTAH_HOMEBREW_SoleVARIABLE_H

#include <Core/Grid/Variables/SoleVariableBase.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/TypeUtils.h>
#include <Core/Exceptions/TypeMismatchException.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>

#include <iosfwd>
#include <cstring>

namespace Uintah {

/**************************************

CLASS
   SoleVariable
   
   Short description...

GENERAL INFORMATION

   SoleVariable.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   Sole_Variable

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  // Uses C++11's shared_ptr to handle memory management.
  template<class T> class SoleVariable : public SoleVariableBase {
  public:
    inline SoleVariable() : value(std::make_shared<T>()) {}
    inline SoleVariable(T value) : value(std::make_shared<T>(value)) {}
    inline SoleVariable(const SoleVariable<T>& copy) : value(copy.value) {}

    virtual void copyPointer(Variable&);
    
    virtual ~SoleVariable() {};
      
    const TypeDescription* virtualGetTypeDescription() const { return getTypeDescription(); }

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

    virtual SoleVariableBase* clone() const {
      return scinew SoleVariable<T>(*this);
    };

    SoleVariable<T>& operator=(const SoleVariable<T>& copy) {
      value = copy.value;
      return *this;
    };

    virtual void getSizeInfo(std::string& elems, unsigned long& totsize, void*& ptr) const {
      elems = "1";
      totsize = getDataSize();
      ptr = getBasePointer();
    }

    virtual size_t getDataSize() const {
      return sizeof(T);
    }

    virtual void* getBasePointer() const {
      return value.get();
      //return (void*)&value;
    }
     
    virtual bool copyOut(void* dst) const {
      void* src = (void*)(&value);
      size_t numBytes = getDataSize();
      void* retVal = std::memcpy(dst, src, numBytes);
      return (retVal == dst) ? true : false;
    }

    virtual void emitNormal(std::ostream& out, const IntVector& l, const IntVector& h,
                            ProblemSpecP /*varnode*/, bool outputDoubleAsFloat) {
      ssize_t linesize = (ssize_t)(sizeof(T));
      
      out.write((char*) (value.get()), linesize);
    }
    
    virtual void readNormal(std::istream& in, bool swapBytes) {
      ssize_t linesize = (ssize_t)(sizeof(T));
       
      T val;
       
      in.read((char*) &val, linesize);
       
      if (swapBytes)
        Uintah::swapbytes(val);
       
      value = std::make_shared<T>(val);
    }

    void print(std::ostream& out) const {
      out << *(value.get());
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
      return scinew SoleVariable<T>();
    };

    std::shared_ptr<T> value;
  };  // end class SoleVariable

  
  template<class T>
  TypeDescription* SoleVariable<T>::td = nullptr;
   
  // The following line is the initialization (creation) of the
  // 'registerMe' static variable (for each version of CCVariable
  // (double, int, etc)).  Note, the 'registerMe' variable is created
  // when the object code is initially loaded (usually during intial
  // program load by the operating system).
  template<class T>
  TypeDescription::Register
  SoleVariable<T>::registerMe( getTypeDescription() );
  
  template<class T>
  const TypeDescription*
  SoleVariable<T>::getTypeDescription()
  {
    if(!td){

      // this is a hack to get a non-null SoleVariable var for some
      // functions the SoleVariables are used in (i.e., task->computes).
      // Since they're not fully-qualified variables, maker would fail
      // anyway.  And since most instances use Handle, it would be
      // difficult.
      int* tmp = nullptr;
      td = scinew TypeDescription(TypeDescription::SoleVariable,
                                  "SoleVariable", &maker,
                                  fun_getTypeDescription(tmp));
    }
    return td;
  }

  template<class T> void 
  SoleVariable<T>::copyPointer(Variable& copy)
  {
    SoleVariable<T>* c = dynamic_cast<SoleVariable<T>* >(&copy);
    if(!c) {
      SCI_THROW(TypeMismatchException("Type mismatch in sole variable", __FILE__, __LINE__));
    }
    *this = *c;
  }

} // End namespace Uintah

#endif
