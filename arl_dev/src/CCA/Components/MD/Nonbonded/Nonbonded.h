/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#ifndef UINTAH_MD_NONBONDED_H
#define UINTAH_MD_NONBONDED_H


#include <Core/Grid/Variables/ComputeSet.h>

#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>

#include <CCA/Components/MD/MDSystem.h>

#include <CCA/Components/MD/CoordinateSystems/coordinateSystem.h>

namespace Uintah {

  /**
   *  @class Nonbonded
   *  @ingroup MD
   *  @author Alan Humphrey and Justin Hooper
   *  @date   June, 2013
   *
   *  @brief Interface for Non-bonded interaction types (both analytic and numeric)
   *
   *  @param
   */
  class Nonbonded {

    public:

      /**
       * @brief Default constructor.
       * @param None.
       */
      Nonbonded();

      /**
       * @brief Default destructor
       * @param None
       */
      virtual ~Nonbonded();

      /**
       * @brief
       * @param
       */
      virtual void initialize(const ProcessorGroup* pg,
                              const PatchSubset*    patches,
                              const MaterialSubset* materials,
                              DataWarehouse*        oldDW,
                              DataWarehouse*        newDW,
                              SimulationStateP&     simState,
                              MDSystem*             systemInfo,
                              const MDLabel*        label,
                              coordinateSystem*     coordSys) = 0;

      /**
       * @brief
       * @param
       */
      virtual void setup(const ProcessorGroup*      pg,
                         const PatchSubset*         patches,
                         const MaterialSubset*      materials,
                         DataWarehouse*             oldDW,
                         DataWarehouse*             newDW,
                         SimulationStateP&          simState,
                         MDSystem*                  systemInfo,
                         const MDLabel*             label,
                         coordinateSystem*          coordSys) = 0;

      /**
       * @brief
       * @param
       */
      virtual void calculate(const ProcessorGroup*  pg,
                             const PatchSubset*     patches,
                             const MaterialSubset*  materials,
                             DataWarehouse*         oldDW,
                             DataWarehouse*         newDW,
                             SimulationStateP&      simState,
                             MDSystem*              systemInfo,
                             const MDLabel*         label,
                             coordinateSystem*      coordSys) = 0;

      /**
       * @brief
       * @param
       */
      virtual void finalize(const ProcessorGroup*   pg,
                            const PatchSubset*      patches,
                            const MaterialSubset*   materials,
                            DataWarehouse*          oldDW,
                            DataWarehouse*          newDW,
                            SimulationStateP&       simState,
                            MDSystem*               systemInfo,
                            const MDLabel*          label,
                            coordinateSystem*       coordSys) = 0;

      /**
       * @brief
       * @param
       * @return
       */
      virtual std::string getNonbondedType() const = 0;
      virtual int requiredGhostCells() const = 0;

    private:

      /**
       * @brief
       * @param
       * @return
       */
      Nonbonded(const Nonbonded&);

      /**
       * @brief
       * @param
       * @return
       */
      Nonbonded& operator=(const Nonbonded&);

  };

}  // End namespace Uintah

#endif
