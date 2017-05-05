/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#ifndef UINTAH_SINGLELEVELREGRIDDER_H
#define UINTAH_SINGLELEVELREGRIDDER_H
#include <CCA/Components/Regridder/RegridderCommon.h>
#include <CCA/Components/Regridder/TiledRegridder.h>

#include <vector> 

namespace Uintah {

/**************************************

CLASS
   SingleLevelRegridder
  
DESCRIPTION 
    This regridder was designed to allow the user to alter a single level's
    patch configuration.  This is useful when restarting a single level uda
    and use more processors than the original patch configuration
    would allow.
     
****************************************/
  class SingleLevelRegridder : public TiledRegridder {
  public:
    SingleLevelRegridder(const ProcessorGroup* pg);
    virtual ~SingleLevelRegridder();
    
    virtual Grid* regrid( Grid* oldGrid );
		
    virtual void problemSetup(const ProblemSpecP& params,
			         const GridP& grid,
			         const SimulationStateP& state);

    std::vector<IntVector> getMinPatchSize() {return d_tileSize;}

  protected:
    void problemSetup_BulletProofing(const int l);
    int d_level_index;              // perform regrid on this level index

  };

} // End namespace Uintah

#endif
