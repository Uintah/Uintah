/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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

#ifndef RHSTerms_h
#define RHSTerms_h

#include <map>

//-- ExprLib includes --//
#include <expression/Tag.h>

/**
 *  \enum FieldSelector
 *  \brief Use this enum to populate information in the FieldTagInfo type.
 */
enum FieldSelector{
  PRIMITIVE_VARIABLE,  ///< when solving a PDE for rho*phi, this is phi.
  CONVECTIVE_FLUX_X,   ///< convective flux in the x-direction
  CONVECTIVE_FLUX_Y,   ///< convective flux in the y-direction
  CONVECTIVE_FLUX_Z,   ///< convective flux in the z-direction
  DIFFUSIVE_FLUX_X,    ///< diffusive flux in the x-direction
  DIFFUSIVE_FLUX_Y,    ///< diffusive flux in the y-direction
  DIFFUSIVE_FLUX_Z,    ///< diffusive flux in the z-direction
  SOURCE_TERM,         ///< the source term
  VOLUME_FRAC,         ///< for cases with embedded boundaries, this supplies the volume fraction field
  AREA_FRAC_X,         ///< for cases with embedded boundaries, this supplies the x-area-fraction field
  AREA_FRAC_Y,         ///< for cases with embedded boundaries, this supplies the y-area-fraction field
  AREA_FRAC_Z          ///< for cases with embedded boundaries, this supplies the z-area-fraction field
};

/**
 * \todo currently we only allow one of each info type.  But there
 *       are cases where we may want multiple ones.  Example:
 *       diffusive terms in energy equation.  Expand this
 *       capability.
 */
typedef std::map< FieldSelector, Expr::Tag > FieldTagInfo; //< Defines a map to hold information on ExpressionIDs for the RHS.

#endif // RHSTerms_h
