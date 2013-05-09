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

#ifndef Extrapolant_h
#define Extrapolant_h

/* ------------------------------------------------------------------------------------------------------------
 ######## ##     ## ######## ########     ###    ########   #######  ##          ###    ##    ## ########
 ##        ##   ##     ##    ##     ##   ## ##   ##     ## ##     ## ##         ## ##   ###   ##    ##
 ##         ## ##      ##    ##     ##  ##   ##  ##     ## ##     ## ##        ##   ##  ####  ##    ##
 ######      ###       ##    ########  ##     ## ########  ##     ## ##       ##     ## ## ## ##    ##
 ##         ## ##      ##    ##   ##   ######### ##        ##     ## ##       ######### ##  ####    ##
 ##        ##   ##     ##    ##    ##  ##     ## ##        ##     ## ##       ##     ## ##   ###    ##
 ######## ##     ##    ##    ##     ## ##     ## ##         #######  ######## ##     ## ##    ##    ##
 --------------------------------------------------------------------------------------------------------------*/

#include <vector>
#include <spatialops/SpatialOpsDefs.h>
#include <spatialops/structured/stencil/Stencil2.h>

/**
 *  \class     Extrapolant
 *  \author    Tony Saad
 *  \date      April, 2013
 *  \ingroup   WasatchOperators
 *
 *  \brief     Extrapolates interior data to patch boundaries using second order
 extrapolation.
 *
 */

template < typename FieldT >
class Extrapolant {
  const std::vector<bool> bcMinus_;
  const std::vector<bool> bcPlus_;
  std::vector<SpatialOps::structured::IntVec> unitNormal_;

public:

  /**
   *  \brief Constructor for Extrapolant.
   */
  Extrapolant( const std::vector<bool>& bcMinus,
               const std::vector<bool>& bcPlus );

  /**
   *  \brief Destructor for Extrapolant.
   */
  ~Extrapolant();

  /**
   *  \brief Applies extrapolation to a field.
   *
   *  \param src: A reference to the field on which extrapolation is needed.
   Extrapolated data is stored back in the field itself - no need for source
   and destination fields.
   *  \param skipBCs: A boolean flag that allows one to skip extrapolation at 
   physical boundaries.
   *
   */
  void apply_to_field( FieldT& src,
                       const bool skipBCs=false);

};

#endif // Extrapolant_h
