#ifndef UINTAH_HOMEBREW_VarTypes_H
#define UINTAH_HOMEBREW_VarTypes_H

#include <Uintah/Grid/VarTypes.h>
#include <Uintah/Grid/Reductions.h>
#include <Uintah/Grid/ReductionVariable.h>

namespace Uintah {
   
   /**************************************
     
     CLASS
       VarTypes
      
       Short Description...
      
     GENERAL INFORMATION
      
       VarTypes.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       VarTypes
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/

   typedef ReductionVariable<double, Reductions::Min<double> > delt_vartype;
    
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/05/02 06:07:24  sparker
// Implemented more of DataWarehouse and SerialMPM
//
//

#endif

