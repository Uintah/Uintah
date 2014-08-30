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

#ifndef ParticleFieldTypes_h
#define ParticleFieldTypes_h

#include <spatialops/SpatialOpsConfigure.h>
#include <spatialops/SpatialOpsDefs.h>
#include <spatialops/structured/SpatialField.h>
#include <spatialops/structured/IndexTriplet.h>

namespace SpatialOps{
namespace Particle{


  /**
   *  @file ParticleFieldTypes.h
   *
   *  Particle fields are dimensioned by the number of particles.  They
   *  are distinctly different types than fields on the underlying mesh,
   *  and we must define operators to move between particle fields and
   *  mesh fields.
   */

  /**
   * \struct ParticleFieldTraits
   * \brief defines type traits for particle fields
   * \ingroup fieldtypes
   */
  struct ParticleFieldTraits{
    typedef NODIR FaceDir;
    typedef IndexTriplet<0,0,0>  Offset;
    typedef IndexTriplet<0,0,0>  BCExtra;
  };

  /**
   * \typedef ParticleField
   * \brief defines a ParticleField
   * \ingroup fieldtypes
   *
   * Note that ParticleField objects should not have any structure associated
   * with them.  They are simply treated as an array of values.
   */
  typedef SpatialField< ParticleFieldTraits > ParticleField;


} // namespace Particle
} // namespace SpatialOps

#endif // ParticleFieldTypes_h
