#ifndef UINTAH_HOMEBREW_ReductionVARIABLE_H
#define UINTAH_HOMEBREW_ReductionVARIABLE_H

#include <Packages/Uintah/Core/Grid/ReductionVariableBase.h>
#include <Packages/Uintah/Core/Grid/DataItem.h>
#include <Packages/Uintah/Core/Grid/TypeDescription.h>
#include <Packages/Uintah/Core/Grid/TypeUtils.h>
#include <Packages/Uintah/Core/Exceptions/TypeMismatchException.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Grid/Reductions.h>

#include <iosfwd>
#include <iostream>

using namespace std;

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
      virtual void emit(ostream&);
      virtual void allocate(const Patch*)
      {
	throw InternalError("Should not call ReductionVariable<T, Op>"
			    "::allocate(const Patch*)"); 
      }

      virtual const TypeDescription* virtualGetTypeDescription() const;
      virtual void getMPIBuffer(void*& buf, int& count,
				MPI_Datatype& datatype, MPI_Op& op);
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
	    throw TypeMismatchException("Type mismatch in reduction variable");
	 *this = *c;
      }
   
   template<class T, class Op>
      void
      ReductionVariable<T, Op>::reduce(const ReductionVariableBase& other)
      {
	 const ReductionVariable<T, Op>* c = dynamic_cast<const ReductionVariable<T, Op>* >(&other);
	 if(!c)
	    throw TypeMismatchException("Type mismatch in reduction variable");
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

   template<class T, class Op>
      void
      ReductionVariable<T, Op>::emit(ostream& intout)
      {
        intout << value;
      }

   template<class T, class Op>
      void
      ReductionVariable<T, Op>::emit(OutputContext& oc)
      {
	 ssize_t s = write(oc.fd, &value, sizeof(double));
	 if (s != sizeof(double))
	    throw ErrnoException("ReductionVariable::emit (write call)", errno);
	 oc.cur += s;
      }

   template<class T, class Op>
      void
      ReductionVariable<T, Op>::read(InputContext& ic)
      {
	 ssize_t s = ::read(ic.fd, &value, sizeof(double));
	 if (s != sizeof(double))
	    throw ErrnoException("ReductionVariable::read (read call)", errno);
	 ic.cur += s;
      }
   
} // End namespace Uintah

#endif
