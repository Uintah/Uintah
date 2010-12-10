#ifndef Wasatch_BCHelperTools_h
#define Wasatch_BCHelperTools_h

//-- Uintah framework includes --//
#include <Core/Grid/Patch.h>
#include <Core/Grid/Material.h>

//-- Wasatch includes --//
#include "PatchInfo.h"
#include "FieldAdaptor.h"
#include "transport/ParseEquation.h"

namespace Wasatch {
 
  class GraphHelper; //forward declaration
  
  /*!
    @function   buildBoundaryConditions
    @abstract   Function that builds the boundary conditions associated with the
   dependent variable of a scalar transport equation.
    @param      theEqnAdaptors: a vector containing the equation adaptors stored
   in Wasatch.h. This is needed to get information about the trasnport equation
   associated with each adaptor.
   @param       theGraphHelper: a pointer to the graph helper. This is needed
   to extract the expression associated with a transport equation dependent var.
   @ param      theLocalPatches: a pointer to the uintah PatchSet.
   @param       thePatchInfoMap: a map containing the patch information. This 
   is needed to extract the operators database associated with a given patch.
   @param       theMaterials: a pointer to the uintah material subset.
  */
  void buildBoundaryConditions(std::vector<EqnTimestepAdaptorBase*>* theEqnAdaptors, 
                               const GraphHelper* theGraphHelper,
                               const Uintah::PatchSet* const theLocalPatches,
                               PatchInfoMap* const thePatchInfoMap,
                               const Uintah::MaterialSubset* const theMaterials);
}

#endif // Wasatch_BCHelperTools_h                            