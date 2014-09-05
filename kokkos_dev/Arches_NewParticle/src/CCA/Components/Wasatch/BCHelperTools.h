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
#include "GraphHelperTools.h"

/**
 *  \file BCHelperTools.h
 *  \author    Tony Saad
 *  \date      December 2010
 *
 *  \brief Provides tools to apply boundary conditions to transport equations.
 */

namespace Wasatch {
  
  //class GraphHelper; //forward declaration
  //class TransportEquation;
  
  /**
   *  @function   buildBoundaryConditions
   *
   *  @brief Function that builds the boundary conditions associated
   *         with the dependent variable of a scalar transport equation.
   *
   *  @param theEqnAdaptors a vector containing the equation adaptors
   *         stored in Wasatch.h. This is needed to get information
   *         about the transport equation associated with each
   *         adaptor.
   *
   *  @param theGraphHelper This is needed to extract the expression
   *         associated with a transport equation dependent var.
   *
   *  @param theLocalPatches a pointer to the Uintah::PatchSet.
   *
   *  @param thePatchInfoMap This is needed to extract the operators
   *         database associated with a given patch.
   *
   *  @param  theMaterials a pointer to the Uintah::MaterialSubset.
   */
  void buildBoundaryConditions( const std::vector<EqnTimestepAdaptorBase*>& theEqnAdaptors, 
                               const GraphHelper& theGraphHelper,
                               const Uintah::PatchSet* const theLocalPatches,
                               const PatchInfoMap& thePatchInfoMap,
                               const Uintah::MaterialSubset* const theMaterials);
}

#endif // Wasatch_BCHelperTools_h
