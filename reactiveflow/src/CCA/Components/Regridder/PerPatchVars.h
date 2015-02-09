/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#ifndef UINTAH_CCA_COMPONENTS_REGRIDDERS_PERPATCHVARS_H
#define UINTAH_CCA_COMPONENTS_REGRIDDERS_PERPATCHVARS_H

//-- Uintah framework includes --//
#include <Core/Util/RefCounted.h>
#include <Core/Util/Handle.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Exceptions/InternalError.h>

namespace Uintah {

  /**
   *  @ingroup Regridders
   *  @struct  PatchFlag
   *  @author  Bryan Worthen
   *           Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
   *           University of Utah
   *  @date    CSAFE days - circa 08/04 (updated 08/14)
   *  @brief   A reference counted struct to help the regridder on a per-patch basis
   *           when doing AMR.
   */
  struct PatchFlag : public RefCounted {

    /**
     * @brief PatchFlag constructor. By default, flag is set to false.
     * @param None
     */
    inline PatchFlag()
    {
      flag = false;
    }

    /**
     * @brief Sets this PatchFlag to true.
     * @param None
     */
    inline void set()
    {
      flag = true;
    }

    bool flag; ///< The AMR refinement flag. Used on a PerPatch basis.

  };
  // End PatchFlag

  typedef Handle<PatchFlag> PatchFlagP;

}

#endif // End UINTAH_CCA_COMPONENTS_REGRIDDERS_PERPATCHVARS_H
