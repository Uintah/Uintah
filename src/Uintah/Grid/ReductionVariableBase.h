
#ifndef UINTAH_HOMEBREW_ReductionVariableBase_H
#define UINTAH_HOMEBREW_ReductionVariableBase_H

#include <iostream>
#include <fstream>
using namespace std;

namespace Uintah {
   
   class Patch;

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

   class ReductionVariableBase {
   public:
      
      virtual ~ReductionVariableBase();
      
      virtual void copyPointer(const ReductionVariableBase&) = 0;
      virtual ReductionVariableBase* clone() const = 0;
      virtual void reduce(const ReductionVariableBase&) = 0;
      virtual void emit(ofstream&) = 0;
   protected:
      ReductionVariableBase(const ReductionVariableBase&);
      ReductionVariableBase();
      
   private:
      ReductionVariableBase& operator=(const ReductionVariableBase&);
   };

} // end namespace Uintah

//
// $Log$
// Revision 1.5  2000/06/01 23:16:18  guilkey
// Added code to the ReductionVariable stuff to "emit" it's data.  Added
// NPAction tasks.  NP=NonPatch, this is for tasks that don't need the patch.
//
// Revision 1.4  2000/05/30 20:19:32  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.3  2000/05/02 06:07:22  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.2  2000/04/26 06:48:53  sparker
// Streamlined namespaces
//
// Revision 1.1  2000/04/19 05:26:15  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
//

#endif
