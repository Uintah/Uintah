
#ifndef UINTAH_HOMEBREW_SFCXVariableBase_H
#define UINTAH_HOMEBREW_SFCXVariableBase_H

#include <Packages/Uintah/Core/Grid/Variable.h>
#include <mpi.h>

namespace SCIRun {
  class IntVector;
}

namespace Uintah {

using namespace SCIRun;

class OutputContext;
class Patch;

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
} // End namespace Uintah

#endif
