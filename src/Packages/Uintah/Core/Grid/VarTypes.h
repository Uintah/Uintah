#ifndef UINTAH_HOMEBREW_VarTypes_H
#define UINTAH_HOMEBREW_VarTypes_H


#include <Packages/Uintah/Core/Grid/Reductions.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>

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

   typedef ReductionVariable<double, Reductions::Max<double> > max_vartype;

   typedef ReductionVariable<double, Reductions::Sum<double> > sum_vartype;

   typedef ReductionVariable<bool, Reductions::And<bool> > bool_and_vartype;
    
   typedef ReductionVariable<Vector, Reductions::Sum<Vector> > sumvec_vartype;

//   typedef ReductionVariable<long, Reductions::Sum<long> > sumlong_vartype;
   typedef ReductionVariable<long64, Reductions::Sum<long64> > sumlong_vartype;
} // End namespace Uintah

#endif
