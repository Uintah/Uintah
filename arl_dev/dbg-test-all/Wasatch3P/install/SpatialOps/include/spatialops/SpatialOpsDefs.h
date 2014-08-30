/*
 * Copyright (c) 2014 The University of Utah
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

#ifndef UT_SpatialOpsDefs_h
#define UT_SpatialOpsDefs_h

#include <spatialops/SpatialOpsConfigure.h>

/** \file SpatialOpsDefs.h */

namespace SpatialOps{

  //==================================================================

  /**
   * @defgroup DirectionDefinitions Direction Definitions
   * @brief Specification of the directions in Cartesian coordinates
   * @{
   */

  /**
   *  @struct XDIR
   *  @brief Defines a type for the x-direction.
   */
  struct XDIR{ enum{value=0}; };

  /**
   *  @struct YDIR
   *  @brief Defines a type for the y-direction.
   */
  struct YDIR{ enum{value=1}; };

  /**
   *  @struct ZDIR
   *  @brief Defines a type for the z-direction.
   */
  struct ZDIR{ enum{value=2}; };

  /**
   *  @struct NODIR
   *  @brief Defines a type to represent no direction
   */
  struct NODIR{ enum{value=-10}; };

  /** @} */  // end of Direction group.




  /**
   *  @addtogroup optypes
   *  @{
   */

  /**
   *  @struct Interpolant
   *  @brief  Defines a type for Interpolant operators.
   */
  struct Interpolant{};

  /**
   *  @struct Gradient
   *  @brief  Defines a type for Gradient operators.
   */
  struct Gradient{};

  /**
   *  @struct Divergence
   *  @brief  Defines a type for Divergence operators.
   */
  struct Divergence{};

  /**
   *  @struct Filter
   *  @brief  Defines a type for Filter operators.
   */
  struct Filter{};

  /**
   *  @struct Restriction
   *  @brief  Defines a type for Restriction operators.
   */
  struct Restriction{};


  /**
   * @struct InterpolantX
   * @brief X-interpolant for use with FD operations whose src and dest fields are the same type
   */
  struct InterpolantX{ typedef XDIR DirT; };
  /**
   * @struct InterpolantY
   * @brief Y-interpolant for use with FD operations whose src and dest fields are the same type
   */
  struct InterpolantY{ typedef YDIR DirT; };

  /**
   * @struct InterpolantZ
   * @brief Z-interpolant for use with FD operations whose src and dest fields are the same type
   */
  struct InterpolantZ{ typedef ZDIR DirT; };

  /**
   * @struct GradientX
   * @brief X-interpolant for use with FD operations whose src and dest fields are the same type
   */
  struct GradientX{ typedef XDIR DirT; };
  /**
   * @struct GradientY
   * @brief Y-interpolant for use with FD operations whose src and dest fields are the same type
   */
  struct GradientY{ typedef YDIR DirT; };

  /**
   * @struct GradientZ
   * @brief Z-interpolant for use with FD operations whose src and dest fields are the same type
   */
  struct GradientZ{ typedef ZDIR DirT; };


/** @} */  // end of Operator Types group

  //==================================================================

}

#endif
