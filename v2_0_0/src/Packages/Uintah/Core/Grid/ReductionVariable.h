#ifndef UINTAH_HOMEBREW_ReductionVARIABLE_H
#define UINTAH_HOMEBREW_ReductionVARIABLE_H

#include <Packages/Uintah/Core/Grid/ReductionVariableBase.h>
#include <Packages/Uintah/Core/Grid/DataItem.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Packages/Uintah/Core/Exceptions/TypeMismatchException.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Grid/Reductions.h>
#include <Core/Util/Endian.h>

namespace SCIRun {
  void swapbytes( Uintah::Matrix3& m);

#if defined(_AIX)
  // Looks like AIX doesn't have swapbytes(bool) but others do!... sigh...

  // I assume a bool is one byte and swapping the bytes is really a no-op.
  // This is needed for the instantiation (xlC) of :
  // class Uintah::ReductionVariable<bool,Uintah::Reductions::And<bool> >".
  void swapbytes( bool ) {}
#endif
} //end namespace SCIRun

#include <sgi_stl_warnings_off.h>
#include <iosfwd>
#include <sgi_stl_warnings_on.h>


namespace Uintah {

using namespace std;

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
  
   Copyright (C) 2000 SCI Group

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
    virtual void copyPointer(const ReductionVariableBase&);
    virtual void reduce(const ReductionVariableBase&);
    virtual void print(ostream& out)
    { out << value; }
    virtual void emitNormal(ostream& out, const IntVector& /*l*/,
			    const IntVector& /*h*/, ProblemSpecP /*varnode*/)
    { out.write((char*)&value, sizeof(double)); }
    virtual void readNormal(istream& in, bool swapBytes)
    {
      in.read((char*)&value, sizeof(double));
      if (swapBytes) swapbytes(value);
    }
     
    virtual void allocate(const Patch*)
    {
      SCI_THROW(InternalError("Should not call ReductionVariable<T, Op>"
			  "::allocate(const Patch*)")); 
    }

    virtual const TypeDescription* virtualGetTypeDescription() const;
    virtual void getMPIInfo(int& count, MPI_Datatype& datatype, MPI_Op& op);
    virtual void getMPIData(vector<char>& buf, int& index);
    virtual void putMPIData(vector<char>& buf, int& index);
    virtual void getSizeInfo(string& elems, unsigned long& totsize,
			     void*& ptr) const {
      elems="1";
      totsize = sizeof(T);
      ptr = 0;
    }
  private:
    ReductionVariable<T, Op>& operator=(const ReductionVariable<T, Op>&copy);
    static Variable* maker();
    T value;
  };
   
  template<class T, class Op>
  const TypeDescription*
  ReductionVariable<T, Op>::getTypeDescription()
  {
    static TypeDescription* td;
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
  ReductionVariable<T, Op>::copyPointer(const ReductionVariableBase& copy)
  {
    const ReductionVariable<T, Op>* c = dynamic_cast<const ReductionVariable<T, Op>* >(&copy);
    if(!c)
      SCI_THROW(TypeMismatchException("Type mismatch in reduction variable"));
    *this = *c;
  }
   
  template<class T, class Op>
  void
  ReductionVariable<T, Op>::reduce(const ReductionVariableBase& other)
  {
    const ReductionVariable<T, Op>* c = dynamic_cast<const ReductionVariable<T, Op>* >(&other);
    if(!c)
      SCI_THROW(TypeMismatchException("Type mismatch in reduction variable"));
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
