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

   typedef ReductionVariable<double, Reductions::Sum<double> > sum_vartype;
    
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/05/31 20:25:33  guilkey
// Added the beginnings of a Sum reduction, which would take data from
// multiple patches, materials, etc. and add them together.  The immediate
// application is for computing the strain energy and storing it.  I'm
// going to need some help with this.
//
// Revision 1.1  2000/05/02 06:07:24  sparker
// Implemented more of DataWarehouse and SerialMPM
//
//

#endif

