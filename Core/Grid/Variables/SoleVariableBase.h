
#ifndef UINTAH_HOMEBREW_SoleVariableBase_H
#define UINTAH_HOMEBREW_SoleVariableBase_H

#include <Packages/Uintah/Core/Grid/Variables/Variable.h>
#include <sgi_stl_warnings_off.h>
#include <iosfwd>
#include <sgi_stl_warnings_on.h>
#include <sci_defs/mpi_defs.h> // For MPIPP_H on SGI
#include <mpi.h>

namespace Uintah {

using std::ostream;

/**************************************

CLASS
   SoleVariableBase
   
   Short description...

GENERAL INFORMATION

   SoleVariableBase.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SoleVariableBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class SoleVariableBase : public Variable {
   public:
      
      virtual ~SoleVariableBase();
      
      virtual void copyPointer(Variable&) = 0;
      virtual SoleVariableBase* clone() const = 0;
      virtual void print(ostream&) = 0;
      virtual void getMPIInfo(int& count, MPI_Datatype& datatype) = 0;
      virtual void getMPIData(vector<char>& buf, int& index) = 0;
      virtual void putMPIData(vector<char>& buf, int& index) = 0;
      virtual const TypeDescription* virtualGetTypeDescription() const = 0;
      virtual RefCounted* getRefCounted();
      virtual void getSizeInfo(string& elems,unsigned long& totsize,
                               void*& ptr) const = 0;
   protected:
      SoleVariableBase(const SoleVariableBase&);
      SoleVariableBase();
      
   private:
      SoleVariableBase& operator=(const SoleVariableBase&);
   };
} // End namespace Uintah

#endif
