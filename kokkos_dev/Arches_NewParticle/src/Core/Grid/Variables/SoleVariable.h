/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef UINTAH_HOMEBREW_SoleVARIABLE_H
#define UINTAH_HOMEBREW_SoleVARIABLE_H

#include <Core/Grid/Variables/SoleVariableBase.h>
#include <Core/Grid/Variables/DataItem.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/TypeUtils.h>
#include <Core/Exceptions/TypeMismatchException.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>

#include <iosfwd>
#include <iostream>


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
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Sole_Variable

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  template<class T> class SoleVariable : public SoleVariableBase {
  public:
    inline SoleVariable() {}
    inline SoleVariable(T value) : value(value) {}
    inline SoleVariable(const SoleVariable<T>& copy) :
      value(copy.value) {}
    virtual ~SoleVariable();
      
    static const TypeDescription* getTypeDescription();
      
    inline operator T () const {
      return value;
    }
    virtual SoleVariableBase* clone() const;
    virtual void copyPointer(Variable&);
    virtual void print(std::ostream& out)
    { out << value; }
    virtual void emitNormal(std::ostream& out, const IntVector& /*l*/,
                            const IntVector& /*h*/, ProblemSpecP /*varnode*/, bool /*outputDoubleAsFloat*/)
    { out.write((char*)&value, sizeof(double)); }
    virtual void readNormal(std::istream& in, bool swapBytes)
    {
      in.read((char*)&value, sizeof(double));
      if (swapBytes) swapbytes(value);
    }
     
    virtual void allocate(const Patch*,const IntVector& boundary)
    {
      SCI_THROW(SCIRun::InternalError("Should not call SoleVariable<T>"
                          "::allocate(const Patch*)", __FILE__, __LINE__)); 
    }

    virtual const TypeDescription* virtualGetTypeDescription() const;
    virtual void getMPIInfo(int& count, MPI_Datatype& datatype);
    virtual void getMPIData(std::vector<char>& buf, int& index);
    virtual void putMPIData(std::vector<char>& buf, int& index);
    virtual void getSizeInfo(std::string& elems, unsigned long& totsize,
                             void*& ptr) const {
      elems="1";
      totsize = sizeof(T);
    }

  private:
    SoleVariable<T>& operator=(const SoleVariable<T>&copy);
    static Variable* maker();
    T value;
  };
   
  template<class T>  const TypeDescription* 
    SoleVariable<T>::getTypeDescription()
  {
    static TypeDescription* td;
    if(!td){
      T* junk=0;
      td = scinew TypeDescription(TypeDescription::SoleVariable,
                                  "SoleVariable", &maker,
                                  fun_getTypeDescription(junk));
    }
    return td;
  }

  template<class T> Variable*  SoleVariable<T>::maker()
  {
    return scinew SoleVariable<T>();
  }
   
  template<class T> const TypeDescription*
    SoleVariable<T>::virtualGetTypeDescription() const
  {
    return getTypeDescription();
  }
   
  template<class T> SoleVariable<T>::~SoleVariable()
  {
  }

  template<class T> SoleVariableBase*  SoleVariable<T>::clone() const
  {
    return scinew SoleVariable<T>(*this);
  }

  template<class T> void 
    SoleVariable<T>::copyPointer(Variable& copy)
  {
    SoleVariable<T>* c = dynamic_cast<SoleVariable<T>* >(&copy);
    if(!c)
      SCI_THROW(TypeMismatchException("Type mismatch in sole variable", __FILE__, __LINE__));
    *this = *c;
  }
   
  template<class T> SoleVariable<T>&
  SoleVariable<T>::operator=(const SoleVariable<T>& copy)
  {
    value = copy.value;
    return *this;
  }
  
} // End namespace Uintah

#endif
