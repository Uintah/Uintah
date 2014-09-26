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

#ifndef UINTAHMD_ELECTROSTATICS_H
#define UINTAHMD_ELECTROSTATICS_H

#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/VarTypes.h>

#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>

#include <CCA/Components/MD/MDSystem.h>
#include <CCA/Components/MD/MDUtil.h>
#include <CCA/Components/MD/MDLabel.h>

#include <CCA/Components/MD/CoordinateSystems/CoordinateSystem.h>

namespace Uintah {

  /**
   *  @class Electrostatics
   *  @ingroup MD
   *  @author Alan Humphrey and Justin Hooper
   *  @date   January, 2013
   *
   *  @brief Interface for Electrostatics calculation types
   *
   *  @param
   */
  class Electrostatics {

    public:

      /**
       * @brief Enumeration of all supported ElectroStatics types.
       */
      enum ElectrostaticsType {
        EWALD, SPME, FMM, NONE
      };

      /**
       * @brief
       * @param
       */
      Electrostatics();

      /**
       * @brief
       * @param
       */
      virtual ~Electrostatics();

      /**
       * @brief
       * @param
       */
      virtual void initialize(const ProcessorGroup* pg,
                              const PatchSubset*    patches,
                              const MaterialSubset* materials,
                              DataWarehouse*      /*old_dw*/,
                              DataWarehouse*        new_dw,
                              const SimulationStateP*     simState,
                              MDSystem*             systemInfo,
                              const MDLabel*        label,
                              CoordinateSystem*     coordinateSys) = 0;

      /**
       * @brief
       * @param
       */
      virtual void setup(const ProcessorGroup*  pg,
                         const PatchSubset*     patches,
                         const MaterialSubset*  materials,
                         DataWarehouse*         old_dw,
                         DataWarehouse*         new_dw,
                         const SimulationStateP*      simState,
                         MDSystem*              systemInfo,
                         const MDLabel*         label,
                         CoordinateSystem*      coordSys) = 0;

      /**
       * @brief
       * @param
       */
      virtual void calculate(const ProcessorGroup*  pg,
                             const PatchSubset*     patches,
                             const MaterialSubset*  materials,
                             DataWarehouse*         old_dw,
                             DataWarehouse*         new_dw,
                             const SimulationStateP*      simState,
                             MDSystem*              systemInfo,
                             const MDLabel*         label,
                             CoordinateSystem*      coordinateSys,
                             SchedulerP&            subscheduler,
                             const LevelP&          level) = 0;

      /**
       * @brief
       * @param
       */
      virtual void finalize(const ProcessorGroup*   pg,
                            const PatchSubset*      patches,
                            const MaterialSubset*   materials,
                            DataWarehouse*          old_dw,
                            DataWarehouse*          new_dw,
                            const SimulationStateP*       simState,
                            MDSystem*               systemInfo,
                            const MDLabel*          label,
                            CoordinateSystem*       coordinateSys) = 0;

      /**
       * @brief
       * @param
       * @return
       */
      virtual ElectrostaticsType getType() const = 0;
      virtual int requiredGhostCells() const = 0;
      virtual bool isPolarizable() const = 0;

    private:

      /**
       * @brief
       * @param
       * @return
       */
      Electrostatics(const Electrostatics&);

      /**
       * @brief
       * @param
       * @return
       */
      Electrostatics& operator=(const Electrostatics&);

  };

}  // End namespace Uintah

#endif
