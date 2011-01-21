#ifndef Wasatch_BCHelperTools_h
#define Wasatch_BCHelperTools_h
/* ----------------------------------------------------------------------------
 ########   ######     ########  #######   #######  ##        ######  
 ##     ## ##    ##       ##    ##     ## ##     ## ##       ##    ## 
 ##     ## ##             ##    ##     ## ##     ## ##       ##       
 ########  ##             ##    ##     ## ##     ## ##        ######  
 ##     ## ##             ##    ##     ## ##     ## ##             ## 
 ##     ## ##    ##       ##    ##     ## ##     ## ##       ##    ## 
 ########   ######        ##     #######   #######  ########  ######  
------------------------------------------------------------------------------*/

//-- Uintah framework includes --//
#include <Core/Grid/Patch.h>
#include <Core/Grid/Material.h>

//-- Wasatch includes --//
#include "PatchInfo.h"
#include "FieldAdaptor.h"
#include "transport/ParseEquation.h"

/**
 *  \file BCHelperTools.h
 *  \author    Tony Saad
 *  \date      December 2010
 *
 *  \brief Provides tools to apply boundary conditions to transport equations.
 */

namespace Wasatch {
 
  class GraphHelper; //forward declaration
  
  /**
   *  \function   buildBoundaryConditions
   *
   *  \brief Function that builds the boundary conditions associated
   *         with the dependent variable of a scalar transport equation.
   *
   *  \param eqnAdaptors a vector containing the equation adaptors
   *         stored in Wasatch.h. This is needed to get information
   *         about the transport equation associated with each
   *         adaptor.
   *
   *  \param graphHelper This is needed to extract the expression
   *         associated with a transport equation dependent var.
   *
   *  \param localPatches a pointer to the Uintah::PatchSet.
   *
   *  \param patchInfoMap This is needed to extract the operators
   *         database associated with a given patch.
   *
   *  \param materials a pointer to the Uintah::MaterialSubset.
   */
  void build_bcs( const std::vector<EqnTimestepAdaptorBase*>& eqnAdaptors, 
                  const GraphHelper& graphHelper,
                  const Uintah::PatchSet* const localPatches,
                  const PatchInfoMap& patchInfoMap,
                  const Uintah::MaterialSubset* const materials );
}

#endif // Wasatch_BCHelperTools_h
