#ifndef UINTAH_HOMEBREW_ReductionVARIABLE_H
#define UINTAH_HOMEBREW_ReductionVARIABLE_H

#include <Uintah/Grid/ReductionVariableBase.h>
#include <Uintah/Grid/DataItem.h>
#include <Uintah/Grid/TypeDescription.h>
#include <Uintah/Grid/TypeUtils.h>
#include <Uintah/Exceptions/TypeMismatchException.h>
#include <SCICore/Malloc/Allocator.h>

#include <iosfwd>
using namespace std;

namespace Uintah {
   class TypeDescription;

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
      inline ReductionVariable(const ReductionVariable<T, Op>& copy) : value(copy.value) {}
      virtual ~ReductionVariable();
      
      static const TypeDescription* getTypeDescription();
      
      inline operator T () const {
	 return value;
      }
      virtual ReductionVariable<T, Op>* clone() const;
      virtual void copyPointer(const ReductionVariableBase&);
      virtual void reduce(const ReductionVariableBase&);
      virtual void emit(ostream&);
   private:
      ReductionVariable<T, Op>& operator=(const ReductionVariable<T, Op>& copy);
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
				     "ReductionVariable",
				     fun_getTypeDescription(junk));
	 }
	 return td;
      }
   
   template<class T, class Op>
      ReductionVariable<T, Op>::~ReductionVariable()
      {
      }
   
   template<class T, class Op>
      ReductionVariable<T, Op>*
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
   
} // end namespace Uintah

//
// $Log$
// Revision 1.9  2000/06/03 05:29:44  sparker
// Changed reduction variable emit to require ostream instead of ofstream
// emit now only prints number without formatting
// Cleaned up a few extraneously included files
// Added task constructor for an non-patch-based action with 1 argument
// Allow for patches and actions to be null
// Removed back pointer to this from Task::Dependency
//
// Revision 1.8  2000/06/01 23:16:18  guilkey
// Added code to the ReductionVariable stuff to "emit" it's data.  Added
// NPAction tasks.  NP=NonPatch, this is for tasks that don't need the patch.
//
// Revision 1.7  2000/05/30 20:19:32  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.6  2000/05/20 08:09:26  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.5  2000/05/15 19:39:49  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.4  2000/05/07 06:02:12  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.3  2000/05/02 06:07:22  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.2  2000/04/26 06:48:53  sparker
// Streamlined namespaces
//
// Revision 1.1  2000/04/19 05:26:14  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
//

#endif
