
#ifndef UINTAH_HOMEBREW_ReductionVariableBase_H
#define UINTAH_HOMEBREW_ReductionVariableBase_H

#include <Packages/Uintah/Core/Grid/Variable.h>
#include <sgi_stl_warnings_off.h>
#include <iosfwd>
#include <sgi_stl_warnings_on.h>
#include <sci_config.h> // For MPIPP_H on SGI
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
      virtual void getMPIInfo(int& count, MPI_Datatype& datatype, MPI_Op& op) = 0;
      virtual void getMPIData(vector<char>& buf, int& index) = 0;
      virtual void putMPIData(vector<char>& buf, int& index) = 0;
      virtual const TypeDescription* virtualGetTypeDescription() const = 0;
      virtual RefCounted* getRefCounted();
      virtual void getSizeInfo(string& elems, unsigned long& totsize,
			       void*& ptr) const = 0;
   protected:
      ReductionVariableBase(const ReductionVariableBase&);
      ReductionVariableBase();
      
   private:
      ReductionVariableBase& operator=(const ReductionVariableBase&);
   };
} // End namespace Uintah

#endif
