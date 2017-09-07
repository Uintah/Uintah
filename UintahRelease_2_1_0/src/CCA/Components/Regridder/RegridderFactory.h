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

#ifndef UINTAH_CCA_COMPONENTS_REGRIDDERS_REGRIDDERFACTORY_H
#define UINTAH_CCA_COMPONENTS_REGRIDDERS_REGRIDDERFACTORY_H

//-- Uintah component includes --//
#include <CCA/Components/Regridder/RegridderCommon.h>

//-- Uintah framework includes --//
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  class ProcessorGroup;

  /**
   *  @ingroup Regridders
   *  @class   RegridderFactory
   *  @author  Steve Parker
   *           Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
   *           University of Utah
   *  @date    Long ago - CSAFE early days
   *  @brief   Class to create one of all known regridders via factory method
   */
  class RegridderFactory {

  public:

    static RegridderCommon* ///< Pointer to the single simulation Regridder instance

    /**
     * @brief Create a simulation Regridder . Type is based on what the ProblemSpec is provided from the input file.
     * @param ps Uintah::ProblemSpec used to parse the XML Uintah problem specification input file.
     * @param world the MPI communicator used. To date this is normally MPI_COMM_WORLD.
     */
    create(ProblemSpecP& ps, const ProcessorGroup* world);

  };

}  // End namespace Uintah

#endif // END UINTAH_CCA_COMPONENTS_REGRIDDERS_REGRIDDERFACTORY_H
