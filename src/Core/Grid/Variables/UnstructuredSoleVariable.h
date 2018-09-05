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

#ifndef UINTAH_HOMEBREW_UnstructuredSoleVARIABLE_H
#define UINTAH_HOMEBREW_UnstructuredSoleVARIABLE_H

#include <Core/Grid/Variables/UnstructuredSoleVariableBase.h>
#include <Core/Grid/Variables/DataItem.h>
#include <Core/Disclosure/UnstructuredTypeDescription.h>
#include <Core/Disclosure/UnstructuredTypeUtils.h>
#include <Core/Exceptions/TypeMismatchException.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>

#include <iosfwd>
#include <iostream>
#include <cstring>


namespace Uintah {

/**************************************

CLASS
   UnstructuredSoleVariable
   
   Short description...

GENERAL INFORMATION

   UnstructuredSoleVariable.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   UnstructuredSole_Variable

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  template<class T> class UnstructuredSoleVariable : public UnstructuredSoleVariableBase {
  public:
    inline UnstructuredSoleVariable() {}
    inline UnstructuredSoleVariable(T value) : value(value) {}
    inline UnstructuredSoleVariable(const UnstructuredSoleVariable<T>& copy) :
      value(copy.value) {}
    virtual ~UnstructuredSoleVariable();
      
    static const UnstructuredTypeDescription* getUnstructuredTypeDescription();
      
    inline operator T () const {
      return value;
    }
    inline T& get() {
      return value;
    }
    inline const T& get() const {
      return value;
    }

    void setData(const T&);

    virtual UnstructuredSoleVariableBase* clone() const;
    virtual void copyPointer(Variable&);

    virtual void getSizeInfo(std::string& elems, unsigned long& totsize,
                             void*& ptr) const {
      elems="1";
      totsize = sizeof(T);
      ptr=(void*)&value;
    }

    virtual size_t getDataSize() const {
      return sizeof(T);
    }

    virtual bool copyOut(void* dst) const {
      void* src = (void*)(&value);
      size_t numBytes = getDataSize();
      void* retVal = std::memcpy(dst, src, numBytes);
      return (retVal == dst) ? true : false;
    }

  private:
    UnstructuredSoleVariable<T>& operator=(const UnstructuredSoleVariable<T>&copy);
    static Variable* maker();
    T value;
  };

  template<class T>  const UnstructuredTypeDescription* 
    UnstructuredSoleVariable<T>::getUnstructuredTypeDescription()
  {
    static UnstructuredTypeDescription* td;
    if(!td){
      td = scinew UnstructuredTypeDescription(UnstructuredTypeDescription::UnstructuredSoleVariable,
                                  "UnstructuredSoleVariable", &maker,
                                  fun_getUnstructuredTypeDescription((int*)0));
    }
    return td;
  }

  template<class T> Variable*  UnstructuredSoleVariable<T>::maker()
  {
    //    return scinew UnstructuredSoleVariable<T>();
    return 0;
  }
   
  template<class T> UnstructuredSoleVariable<T>::~UnstructuredSoleVariable()
  {
  }

  template<class T> UnstructuredSoleVariableBase*  UnstructuredSoleVariable<T>::clone() const
  {
    return scinew UnstructuredSoleVariable<T>(*this);
  }

  template<class T> void 
    UnstructuredSoleVariable<T>::copyPointer(Variable& copy)
  {
    UnstructuredSoleVariable<T>* c = dynamic_cast<UnstructuredSoleVariable<T>* >(&copy);
    if(!c)
      SCI_THROW(TypeMismatchException("Type mismatch in sole variable", __FILE__, __LINE__));
    *this = *c;
  }
   
  template<class T> UnstructuredSoleVariable<T>&
  UnstructuredSoleVariable<T>::operator=(const UnstructuredSoleVariable<T>& copy)
  {
    value = copy.value;
    return *this;
  }

  template<class T>
    void
    UnstructuredSoleVariable<T>::setData(const T& val)
    {
      value = val;
    }  
} // End namespace Uintah

#endif
