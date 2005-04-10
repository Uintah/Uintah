#ifndef UINTAH_HOMEBREW_Variable_H
#define UINTAH_HOMEBREW_Variable_H

#include <Uintah/Interface/InputContext.h>
#include <Uintah/Interface/OutputContext.h>
#include <Uintah/Grid/Patch.h>

namespace Uintah {
   class TypeDescription;

   /**************************************
     
     CLASS
       Variable
      
       Short Description...
      
     GENERAL INFORMATION
      
       Variable.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       Variable
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
    
   class Variable {
   public:
      virtual const TypeDescription* virtualGetTypeDescription() const = 0;
      void setForeign();
      bool isForeign() const {
	 return foreign;
      }
      
      virtual void emit(OutputContext&) = 0;
      virtual void read(InputContext&) = 0;
      virtual void allocate(const Patch* patch) = 0;
   protected:
      Variable();
      virtual ~Variable();
   private:
      Variable(const Variable&);
      Variable& operator=(const Variable&);

      bool foreign;
   };
} // end namespace Uintah

//
// $Log$
// Revision 1.3  2000/12/23 00:32:47  witzel
// Added emit(OutputContext), read(InputContext), and allocate(Patch*) as
// pure virtual methods to class Variable and did any needed implementations
// of these in sub-classes.
//
// Revision 1.2  2000/10/13 20:46:11  sparker
// Added the notion of a "foreign" variable, to assist in cleaning
//  them out of the data warehouse at the end of a timestep
//
// Revision 1.1  2000/07/27 22:39:51  sparker
// Implemented MPIScheduler
// Added associated support
//
//

#endif

