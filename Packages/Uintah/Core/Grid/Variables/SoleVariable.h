#ifndef UINTAH_HOMEBREW_SoleVARIABLE_H
#define UINTAH_HOMEBREW_SoleVARIABLE_H

#include <Packages/Uintah/Core/Grid/Variables/SoleVariableBase.h>
#include <Packages/Uintah/Core/Grid/Variables/DataItem.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Packages/Uintah/Core/Exceptions/TypeMismatchException.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>

namespace SCIRun {
  void swapbytes( Uintah::Matrix3& m);

#if defined(_AIX)
  // Looks like AIX doesn't have swapbytes(bool) but others do!... sigh...

  // I assume a bool is one byte and swapping the bytes is really a no-op.
  // This is needed for the instantiation (xlC) of :
  // class Uintah::SoleVariable<bool,Uintah::Soles::And<bool> >".
  void swapbytes( bool ) {}
#endif
} //end namespace SCIRun

#include <sgi_stl_warnings_off.h>
#include <iosfwd>
#include <iostream>
#include <sgi_stl_warnings_on.h>


namespace Uintah {

using namespace std;

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
    virtual void copyPointer(const SoleVariableBase&);
    virtual void print(ostream& out)
    { out << value; }
    virtual void emitNormal(ostream& out, const IntVector& /*l*/,
			    const IntVector& /*h*/, ProblemSpecP /*varnode*/, bool /*outputDoubleAsFloat*/)
    { out.write((char*)&value, sizeof(double)); }
    virtual void readNormal(istream& in, bool swapBytes)
    {
      in.read((char*)&value, sizeof(double));
      if (swapBytes) swapbytes(value);
    }
     
    virtual void allocate(const Patch*,const IntVector& boundary)
    {
      SCI_THROW(SCIRun::InternalError("Should not call SoleVariable<T>"
			  "::allocate(const Patch*)")); 
    }

    virtual const TypeDescription* virtualGetTypeDescription() const;
    virtual void getMPIInfo(int& count, MPI_Datatype& datatype);
    virtual void getMPIData(vector<char>& buf, int& index);
    virtual void putMPIData(vector<char>& buf, int& index);
    virtual void getSizeInfo(string& elems, unsigned long& totsize) const {
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
    SoleVariable<T>::copyPointer(const SoleVariableBase& copy)
  {
    const SoleVariable<T>* c = dynamic_cast<const SoleVariable<T>* >(&copy);
    if(!c)
      SCI_THROW(TypeMismatchException("Type mismatch in sole variable"));
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
