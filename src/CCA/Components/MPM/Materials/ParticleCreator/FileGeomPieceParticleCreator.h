/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
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

#ifndef __FILE_GEOM_PIECE_PARTICLE_CREATOR_H__
#define __FILE_GEOM_PIECE_PARTICLE_CREATOR_H__

#include "ParticleCreator.h"

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/MaterialManager.h>
#include <vector>
#include <map>

namespace Uintah {

  class FileGeomPieceParticleCreator : public ParticleCreator {
  public:
    
    FileGeomPieceParticleCreator(MPMMaterial* matl, MPMFlags* flags);


    virtual ~FileGeomPieceParticleCreator();


    virtual particleIndex createParticles(MPMMaterial* matl,
                                          CCVariable<int>& cellNAPID,
                                          const Patch*,DataWarehouse* new_dw,
                                          std::vector<GeometryObject*>&);

  protected:

    virtual particleIndex countAndCreateParticles(const Patch*,
                                                  GeometryObject* obj,
                                                  ObjectVars& vars);

    void createPoints(const Patch* patch, GeometryObject* obj,ObjectVars& vars);

    virtual void initializeParticle(const Patch* patch,
                                    std::vector<GeometryObject*>::const_iterator obj,
                                    MPMMaterial* matl,
                                    Point p, IntVector cell_idx,
                                    particleIndex i,
                                    CCVariable<int>& cellNAPI,
                                    ParticleVars& pvars);
  };

} // End of namespace Uintah

#endif // __FILE_GEOM_PIECE_PARTICLE_CREATOR_H__
