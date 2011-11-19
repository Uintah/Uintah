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

//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- Wasatch includes --//
#include "PatchInfo.h"
#include "FieldAdaptor.h"
#include "GraphHelperTools.h"

/**
 *  \file 	BCHelperTools.h
 *  \author 	Tony Saad
 *  \date  	December, 2010
 *
 *  \brief Provides tools to apply boundary conditions to transport equations.
 */

namespace Wasatch {
  
  //class GraphHelper; //forward declaration
  //class TransportEquation;
  
  /**
   *  \ingroup WasatchCore
   *
   *  \brief Function that builds the boundary conditions associated
   *         with the dependent variable of a scalar transport equation.
   *
   *  \param phiTag the tag for the variable to set BCs on
   *
   *  \param staggeredLocation the direction for staggering for this variable
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
  template < typename FieldT >
  void process_boundary_conditions( const Expr::Tag phiTag,
                                   const std::string fieldName,
                     const Direction staggeredLocation,
                     const GraphHelper& graphHelper,
                     const Uintah::PatchSet* const localPatches,
                     const PatchInfoMap& patchInfoMap,
                     const Uintah::MaterialSubset* const materials);  
   
//  void set_pressure_matrix_bc( const Expr::Tag pressureTag, 
//                              Uintah::CCVariable<Uintah::Stencil7>& pressureMatrix,
//                              const Uintah::Patch* patch);

  void update_pressure_rhs( const Expr::Tag pressureTag, 
                       Uintah::CCVariable<Uintah::Stencil7>& pressureMatrix,
                        SVolField& pressureField,
                        SVolField& pressureRHS,
                        const Uintah::Patch* patch);
  
  
}

#endif // Wasatch_BCHelperTools_h
