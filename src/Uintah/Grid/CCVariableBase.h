
#ifndef UINTAH_HOMEBREW_CCVariableBase_H
#define UINTAH_HOMEBREW_CCVariableBase_H

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
   CCVariableBase
   
   Short description...

GENERAL INFORMATION

   CCVariableBase.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   CCVariableBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class CCVariableBase : public Variable {
   public:
      
      virtual ~CCVariableBase();
      
      virtual void copyPointer(const CCVariableBase&) = 0;
      
      virtual void rewindow(const IntVector& low, const IntVector& high) = 0;

      //////////
      // Insert Documentation Here:
      virtual CCVariableBase* clone() const = 0;

      virtual void allocate(const IntVector& lowIndex,
			    const IntVector& highIndex) = 0;
      virtual void copyPatch(CCVariableBase* src,
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
      CCVariableBase(const CCVariableBase&);
      CCVariableBase();
      
   private:
      CCVariableBase& operator=(const CCVariableBase&);
   };

} // end namespace Uintah

//
// $Log$
// Revision 1.7  2000/12/10 09:06:16  sparker
// Merge from csafe_risky1
//
// Revision 1.5.4.2  2000/10/20 02:06:37  rawat
// modified cell centered and staggered variables to optimize communication
//
// Revision 1.5.4.1  2000/10/19 05:18:03  sparker
// Merge changes from main branch into csafe_risky1
//
// Revision 1.6  2000/10/11 21:39:59  sparker
// Added rewindow to CCVariable - just copies the array to a different window
//
// Revision 1.5  2000/09/25 14:41:32  rawat
// added mpi support for cell centered and staggered cell variables
//
// Revision 1.4  2000/07/27 22:39:50  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.3  2000/05/31 04:01:51  rawat
// partially completed CCVariable implementation
//
// Revision 1.2  2000/05/30 20:19:28  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.1  2000/05/11 20:12:11  dav
// Added CCVaraibleBase
//
//

#endif
