
#ifndef UINTAH_HOMEBREW_SFCXVariableBase_H
#define UINTAH_HOMEBREW_SFCXVariableBase_H

#include <Uintah/Grid/Variable.h>
#include <mpi.h>

namespace SCICore {
   namespace Geometry {
      class IntVector;
   }
}

namespace Uintah {
   class OutputContext;
   class Patch;
   using SCICore::Geometry::IntVector;

/**************************************

CLASS
   SFCXVariableBase
   
   Short description...

GENERAL INFORMATION

   SFCXVariableBase.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SFCXVariableBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class SFCXVariableBase : public Variable {
   public:
      
      virtual ~SFCXVariableBase();
      
      virtual void copyPointer(const SFCXVariableBase&) = 0;
      
      virtual void rewindow(const IntVector& low, const IntVector& high) = 0;

      //////////
      // Insert Documentation Here:
      virtual SFCXVariableBase* clone() const = 0;

      virtual void allocate(const Patch*) = 0;
      virtual void allocate(const IntVector& lowIndex,
			    const IntVector& highIndex) = 0;
      virtual void copyPatch(SFCXVariableBase* src,
			      const IntVector& lowIndex,
			      const IntVector& highIndex) = 0;
      virtual void emit(OutputContext&) = 0;
      virtual void* getBasePointer() = 0;
      void getMPIBuffer(void*& buf, int& count,
			MPI_Datatype& datatype, bool& free_datatype,
			const IntVector& low, const IntVector& high);
      void getMPIBuffer(void*& buf, int& count, MPI_Datatype& datatype);
      virtual const TypeDescription* virtualGetTypeDescription() const = 0;
      virtual void getSizes(IntVector& low, IntVector& high, IntVector& siz) const = 0;
      virtual void getSizes(IntVector& low, IntVector& high,
			    IntVector& siz, IntVector& strides) const = 0;

   protected:
      SFCXVariableBase(const SFCXVariableBase&);
      SFCXVariableBase();
      
   private:
      SFCXVariableBase& operator=(const SFCXVariableBase&);
   };

} // end namespace Uintah

//
// $Log$
// Revision 1.6  2000/12/23 00:32:47  witzel
// Added emit(OutputContext), read(InputContext), and allocate(Patch*) as
// pure virtual methods to class Variable and did any needed implementations
// of these in sub-classes.
//
// Revision 1.5  2000/12/10 09:06:17  sparker
// Merge from csafe_risky1
//
// Revision 1.3.4.2  2000/10/20 02:06:37  rawat
// modified cell centered and staggered variables to optimize communication
//
// Revision 1.3.4.1  2000/10/19 05:18:03  sparker
// Merge changes from main branch into csafe_risky1
//
// Revision 1.4  2000/10/12 20:05:37  sparker
// Removed print statement from FCVariable
// Added rewindow to SFC{X,Y,Z}Variables
// Uncommented assertion in CCVariable
//
// Revision 1.3  2000/09/25 14:41:32  rawat
// added mpi support for cell centered and staggered cell variables
//
// Revision 1.2  2000/07/27 22:39:50  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.1  2000/06/27 23:18:17  rawat
// implemented Staggered cell variables. Modified Patch.cc to get ghostcell
// and staggered cell indexes.
//
//

#endif
