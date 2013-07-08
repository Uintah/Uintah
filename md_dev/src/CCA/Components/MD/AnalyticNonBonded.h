/*
 * The MIT License
 *
 * Copyright (c) 1997-2013 The University of Utah
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

#ifndef UINTAH_MD_NONBONDED_ANALYTIC_H
#define UINTAH_MD_NONBONDED_ANALYTIC_H

#include <CCA/Components/MD/NonBonded.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/ParticleVariable.h>

#include <vector>
#include <map>

namespace Uintah {

  using namespace SCIRun;

  typedef std::vector<std::vector<int> > neighborlist;

  class MDSystem;
  class ParticleSubset;
  class MDLabel;
  class PatchMaterialKey;

  /**
   *  @class AnalyticNonBonded
   *  @ingroup MD
   *  @author Alan Humphrey and Justin Hooper
   *  @date   June, 2013
   *
   *  @brief
   *
   *  @param
   */
  class AnalyticNonBonded : public NonBonded {

    public:

      /**
       * @brief
       * @param
       */
      AnalyticNonBonded();

      /**
       * @brief
       * @param
       */
      ~AnalyticNonBonded();

      /**
       * @brief
       * @param
       * @param
       * @param
       * @param
       */
      AnalyticNonBonded(MDSystem* system,
                        const double r12,
                        const double r6,
                        const double cutoffRadius);

      /**
       * @brief
       * @param
       * @param
       * @param
       * @return
       */
      void initialize(const ProcessorGroup* pg,
                      const PatchSubset* patches,
                      const MaterialSubset* materials,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw);

      /**
       * @brief
       * @param None
       * @return None
       */
      void setup(const ProcessorGroup* pg,
                 const PatchSubset* patches,
                 const MaterialSubset* materials,
                 DataWarehouse* old_dw,
                 DataWarehouse* new_dw);

      /**
       * @brief
       * @param
       * @return None
       */
      void calculate(const ProcessorGroup* pg,
                     const PatchSubset* patches,
                     const MaterialSubset* materials,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw,
                     SchedulerP& subscheduler,
                     const LevelP& level);

      /**
       * @brief
       * @param None
       * @return None
       */
      void finalize(const ProcessorGroup* pg,
                    const PatchSubset* patches,
                    const MaterialSubset* materials,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw);

      /**
       * @brief
       * @param None
       * @return
       */
      inline NonBondedType getType() const
      {
        return d_nonBondedInteractionType;
      }

      /**
       * @brief
       * @param
       * @return
       */
      inline void setMDLabel(MDLabel* lb)
      {
        d_lb = lb;
      }

      /**
       * @brief
       * @param None
       * @return
       */
      void generateNeighborList(ParticleSubset* local_pset,
                                ParticleSubset* neighbor_pset,
                                constParticleVariable<Point>& px_local,
                                constParticleVariable<Point>& px_neighbors,
                                std::vector<std::vector<int> >& neighbors);

    private:

      /**
       * @brief
       * @param None
       * @return
       */
      bool isNeighbor(const Point* atom1,
                      const Point* atom2);

      NonBondedType d_nonBondedInteractionType;  //!< Implementation type for the non-bonded interactions
      MDSystem* d_system;                        //!< A handle to the MD simulation system object
      MDLabel* d_lb;                             //!< A handle on the set of MD specific labels
      double d_r12;						                   //!< The van der Waals repulsive parameter
      double d_r6;				                       //!< The van der Waals dispersion parameter
      double d_cutoffRadius;                     //!< The short-range cut, in Angstroms
  };

}  // End namespace Uintah

#endif
