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

#ifndef UINTAH_MD_MDSYSTEM_H
#define UINTAH_MD_MDSYSTEM_H

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Math/Matrix3.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Containers/Array2.h>
#include <Core/Exceptions/InternalError.h>
#include <CCA/Components/MD/Forcefields/Forcefield.h>
#include <CCA/Components/MD/Potentials/TwoBody/NonbondedTwoBodyPotential.h>
#include <CCA/Components/MD/MDMaterial.h>
#include <CCA/Components/MD/atomMap.h>

namespace Uintah {


  enum ensembleType { NVE, NVT, NPT, ISOKINETIC };



  using namespace Uintah;
  using namespace SCIRun;

  /**
   *  @class MDSystem
   *  @ingroup MD
   *  @author Alan Humphrey and Justin Hooper
   *  @date   December, 2012
   *
   *  @brief
   *
   *  @param
   */
  class MDSystem {


    public:
     ~MDSystem ();

      /**
       * @brief
       * @param
       */
      MDSystem(const ProblemSpecP&,
                     GridP&,
                     Forcefield*);

      /**
       * @brief
       * @param  None
       * @return
       */
      inline Vector getPressure() const {
        return d_pressure;
      }

      /**
       * @brief
       * @param
       * @return
       */
      inline double getTemperature() const {
        return d_temperature;
      }

      /**
       * @brief
       * @param
       * @return
       */
      inline size_t getNumAtoms() const {
        return d_numAtoms;
      }

      /*
       * @brief
       * @param
       * @return
       */
      inline size_t getNumAtomTypes() const {
        return d_numAtomsOfType.size();
      }

      /*
       * @brief
       * @param
       * @return
       */
      inline size_t getNumAtomsOfType(size_t TypeIndex) const {
        return d_numAtomsOfType[TypeIndex];
      }

      inline size_t getNumMolecules() const {
        return d_numMolecules;
      }

      inline size_t getNumMoleculeTypes() const {
        return d_numMoleculesOfType.size();
      }

      inline size_t getNumMoleculesOfType(size_t TypeIndex) const {
        return d_numMoleculesOfType[TypeIndex];
      }

      inline Forcefield* getForcefieldPointer() const {
        return d_forcefield;
      };

      size_t registerAtomTypes(const atomMap*            incomingMap,
                               const SimulationStateP&   simState);

      size_t registerAtomCount(const size_t count,
                               const size_t matlIndex);

      inline ensembleType getEnsemble() const {
        return d_ensemble;
      }

//      inline void attachForcefield(Forcefield* ff) {
//        if (!d_forcefield) {
//          d_forcefield = ff;
//        }
//        else
//        {
//          InternalError("Error:  Attempted to attach a forcefield to an already populated system", __FILE__, __LINE__);
//        }
//      }

    private:

      ensembleType          d_ensemble;             //!< Type of the simulation ensemble
      unsigned long         d_numAtoms;             //!< Total number of atoms in the simulation
      std::vector<size_t>   d_numAtomsOfType;       //!< List of total number of each atom type
      unsigned long         d_numMolecules;         //!< Number of molecules in the simulation
      std::vector<size_t>   d_numMoleculesOfType;   //!< List of number of each molecule type

      Vector                d_pressure;             //!< Total MD System pressure
      double                d_temperature;          //!< Total MD system temperature

      bool                  f_atomsRegistered;      //!< Flag to ensure atoms only registered once

      // Methods
//      void calcCellVolume();
      inline size_t max(int a, int b){ return (a > b ? a : b); }
      inline double max(int a, int b, int c) { return (max(max(a,b),c)); }

// Maybe these should be passed around seperately?
      Forcefield*           d_forcefield;           //! Pointer to the established forcefield
//      SimulationStateP  d_simState;            //! Pointer to the simulation state (for material access)

      // Total cell variables



      // disable copy and assignment and default construction
      MDSystem();
      MDSystem(const MDSystem& system);
      MDSystem& operator=(const MDSystem& system);

  };

}  // End namespace Uintah

#endif
