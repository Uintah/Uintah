/*
 * Copyright (c) 2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

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
  void process_boundary_conditions( const Expr::Tag& phiTag,
                                    const std::string& fieldName,
                                    const Direction staggeredLocation,
                                    const GraphHelper& graphHelper,
                                    const Uintah::PatchSet* const localPatches,
                                    const PatchInfoMap& patchInfoMap,
                                    const Uintah::MaterialSubset* const materials );

  void update_pressure_rhs( const Expr::Tag& pressureTag,
                            Uintah::CCVariable<Uintah::Stencil4>& pressureMatrix,
                            SVolField& pressureField,
                            SVolField& pressureRHS,
                            const Uintah::Patch* patch);

  void update_pressure_matrix( const Expr::Tag& pressureTag,
                               Uintah::CCVariable<Uintah::Stencil4>& pressureMatrix,
                               const Uintah::Patch* patch,
                               const int material);
  
}

#endif // Wasatch_BCHelperTools_h
