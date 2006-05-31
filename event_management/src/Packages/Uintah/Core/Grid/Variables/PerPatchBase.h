
#ifndef UINTAH_HOMEBREW_PerPatchBase_H
#define UINTAH_HOMEBREW_PerPatchBase_H

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Variables/Variable.h>

namespace Uintah {

  using SCIRun::IntVector;

  class Patch;
  class RefCounted;

/**************************************

CLASS
   PerPatchBase
   
   Short description...

GENERAL INFORMATION

   PerPatchBase.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   PerPatchBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  // inherits from Variable solely for the purpose of stuffing it in the DW
  class PerPatchBase : public Variable {
   public:
      
      virtual ~PerPatchBase();
      
      virtual const TypeDescription* virtualGetTypeDescription() const;
      virtual void copyPointer(Variable&) = 0;
      virtual PerPatchBase* clone() const = 0;
      virtual RefCounted* getRefCounted();
      virtual void getSizeInfo(std::string& elems, unsigned long& totsize,
			       void*& ptr) const = 0;

      // Only affects grid variables
      void offsetGrid(const IntVector& /*offset*/);
 
      virtual void emitNormal(std::ostream& out, const IntVector& l,
			      const IntVector& h, ProblemSpecP varnode, bool outputDoubleAsFloat );
      virtual void readNormal(std::istream& in, bool swapbytes);      
      virtual void allocate(const Patch* patch, const IntVector& boundary);

   protected:
      PerPatchBase(const PerPatchBase&);
      PerPatchBase();
      
   private:
      PerPatchBase& operator=(const PerPatchBase&);
   };
} // End namespace Uintah

#endif
