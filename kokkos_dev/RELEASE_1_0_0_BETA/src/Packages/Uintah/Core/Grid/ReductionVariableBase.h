
#ifndef UINTAH_HOMEBREW_ReductionVariableBase_H
#define UINTAH_HOMEBREW_ReductionVariableBase_H

#include <Packages/Uintah/Core/Grid/Variable.h>
#include <iosfwd>
#include <mpi.h>

namespace Uintah {

using std::ostream;

/**************************************

CLASS
   ReductionVariableBase
   
   Short description...

GENERAL INFORMATION

   ReductionVariableBase.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   ReductionVariableBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class ReductionVariableBase : public Variable {
   public:
      
      virtual ~ReductionVariableBase();
      
      virtual void copyPointer(const ReductionVariableBase&) = 0;
      virtual ReductionVariableBase* clone() const = 0;
      virtual void reduce(const ReductionVariableBase&) = 0;
      virtual void print(ostream&) = 0;
      virtual void getMPIBuffer(void*& buf, int& count,
				MPI_Datatype& datatype, MPI_Op& op) = 0;
      virtual const TypeDescription* virtualGetTypeDescription() const = 0;
   protected:
      ReductionVariableBase(const ReductionVariableBase&);
      ReductionVariableBase();
      
   private:
      ReductionVariableBase& operator=(const ReductionVariableBase&);
   };
} // End namespace Uintah

#endif
