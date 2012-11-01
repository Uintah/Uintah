/*
 * The MIT License
 *
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
                                    const Uintah::MaterialSubset* const materials,
                                    const std::set<std::string>& functorSet);
  /**
   *  \ingroup WasatchCore
   *
   *  \brief Function that updates poisson rhs when boundaries are present.
   *
   *  \param poissonTag The Expr::Tag of the poisson variable (e.g. pressure).
   This Tag is needed to extract the boundary iterators from Uintah.
   *
   *  \param poissonMatrix A reference to the poisson coefficient matrix which
   we intend to modify.
   *
   *  \param poissonField A reference to the poisson field. This contains the
   values of the poisson variable, e.g. pressure.
   *
   *  \param poissonRHS A reference to the poisson RHS field. This should be
   a MODIFIABLE field since it will be updated using bcs on the poisson field.
   *   
   *  \param patch A pointer to the current patch. If the patch does NOT contain
   the reference cells, then nothing is set.
   *
   *  \param material The Uintah material ID (an integer).
   */      
  void update_poisson_rhs( const Expr::Tag& poissonTag,
                            Uintah::CCVariable<Uintah::Stencil4>& poissonMatrix,
                            SVolField& poissonField,
                            SVolField& poissonRHS,
                            const Uintah::Patch* patch,
                            const int material);

  /**
   *  \ingroup WasatchCore
   *
   *  \brief Function that updates pressure matrix coefficients when boundaries
   are present.
   *
   *  \param poissonTag The Expr::Tag of the poisson variable (e.g. pressure).
   This Tag is needed to extract the boundary iterators from Uintah.
   *
   *  \param poissonMatrix A reference to the poisson coefficient matrix which
   we intend to modify.
   *
   *  \param patch A pointer to the current patch. If the patch does NOT contain
   the reference cells, then nothing is set.
   *
   *  \param material The Uintah material ID (an integer).
   */    
  void update_poisson_matrix( const Expr::Tag& poissonTag,
                               Uintah::CCVariable<Uintah::Stencil4>& poissonMatrix,
                               const Uintah::Patch* patch,
                               const int material);
  /**
   *  \ingroup WasatchCore
   *
   *  \brief Function that updates pressure matrix coefficients when setting a
             reference pressure. The reference cell defaults to [1,1,1] but the
             user/developer can specify other reference locations. Caution must
             be exercised when setting reference cell location so as to avoid
             cells at patch boundaries.
   *
   *  \param pressureMatrix A reference to the pressure coefficient matrix which
                            we intend to modify.
   *
   *  \param patch A pointer to the current patch. If the patch does NOT contain
                   the reference cells, then nothing is set.
   *
   *  \param refCell A SCIRun::IntVector that designates the reference cell. 
                     This defaults to [1,1,1].
   */  
  void set_ref_poisson_coefs( Uintah::CCVariable<Uintah::Stencil4>& pressureMatrix,
                               const Uintah::Patch* patch,
                               const SCIRun::IntVector refCell );

  /**
   *  \ingroup WasatchCore
   *
   *  \brief Function that updates pressure RHS when setting a
             reference pressure. The reference pressure value defaults to 0.0 
             for convenience.   
   *
   *  \param pressureMatrix A reference to the pressure coefficient matrix which
                            we intend to modify.
   *
   *  \param patch A pointer to the current patch. If the patch does NOT contain
                   the reference cells, then nothing is set.
   *
   *  \param refCell A SCIRun::IntVector that designates the reference cell. 
                     This defaults to [1,1,1].
   */    
  void set_ref_poisson_rhs  ( SVolField& pressureRHS,
                               const Uintah::Patch* patch, 
                               const double referencePressureValue,
                               const SCIRun::IntVector refCell );
  /**
   *  \ingroup WasatchCore
   *
   *  \brief Special function to handle pressure boundary conditions. This is
      needed because process_after_evaluate() will NOT work on the newly solved
      pressure.
   *
   *  \param pressureTag Pressure tag.
   *
   *  \param pressureField A reference to the pressure field.
   *
   *  \param patch The patch on which we wish to apply the BCs.
   *
   *  \param patch The material on which we wish to apply the BCs.   
   */    
  void process_poisson_bcs( const Expr::Tag& pressureTag,
                            SVolField& pressureField,
                            const Uintah::Patch* patch,
                            const int material);  
}

#endif // Wasatch_BCHelperTools_h
