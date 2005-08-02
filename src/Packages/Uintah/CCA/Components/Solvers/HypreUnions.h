/*--------------------------------------------------------------------------
CLASS
   HypreUnions
   
   Union types of different Hypre system interfaces.

GENERAL INFORMATION

   File: HypreUnions.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
   HypreUnions, HypreGenericMatrix, HypreGenericVector.

DESCRIPTION 
   Class HypreUnions allows a generic matrix and vector types
   that work for multiple Hypre interfaces. All Hypre solvers are
   "generic" in the sense that they work with the proper parts of
   generic types defined here.  with
  
WARNING
   Any new Hypre interface that we use should be added to the unions here.
   --------------------------------------------------------------------------*/
#ifndef Packages_Uintah_CCA_Components_Solvers_HypreUnions_h
#define Packages_Uintah_CCA_Components_Solvers_HypreUnions_h

// hypre includes
#include <utilities.h>
#include <HYPRE_struct_ls.h>
#include <HYPRE_sstruct_ls.h>
#include <krylov.h>

namespace Uintah {

  union HypreGenericMatrix {
    Hypre_StructMatrix Struct;
    Hypre_StructMatrix SStruct;
    Hypre_ParCSRMatrix ParCSR;
  };

  union HypreGenericVector {
    Hypre_StructVector Struct;
    Hypre_StructVector SStruct;
    Hypre_ParCSRVector ParCSR;
  };

} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreUnions_h
