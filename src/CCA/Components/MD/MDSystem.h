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
#include <CCA/Components/MD/Forcefields/Forcefield.h>
#include <CCA/Components/MD/Potentials/TwoBody/NonbondedTwoBodyPotential.h>
#include <CCA/Components/MD/MDMaterial.h>

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

      /**
       * @brief Default constructor
       * @param
       */
      MDSystem();

      /**
       * @brief Default destructor
       * @param
       */
      ~MDSystem();

      /**
       * @brief
       * @param
       */
      MDSystem(const ProblemSpecP&,
               GridP&,
               SimulationStateP&,
               Forcefield*);

      /**
       * @brief
       * @param  None
       * @return
       */
      inline Vector getPressure() const { return d_pressure; }

      /**
       * @brief
       * @param
       * @return
       */
      inline double getTemperature() const { return d_temperature; }

      /**
       * @brief
       * @param None
       * @return
       */
      inline bool isOrthorhombic() const { return d_orthorhombic; }

      /**
       * @brief
       * @param
       * @return
       */
      inline bool queryBoxChanged() const { return d_boxChanged; }

      /*
       * @brief
       * @param
       * @return
       */
      inline void clearBoxChanged() { d_boxChanged = false; }

      /**
       * @brief
       * @param
       * @return
       */
      inline void markBoxChanged() { d_boxChanged = true; }

      /**
       * @brief
       * @param
       * @return
       */
      inline Matrix3 getUnitCell() const { return d_unitCell; }

      /**
       * @brief
       * @param
       * @return
       */
      inline Matrix3 getInverseCell() const { return d_inverseCell; }

      /**
       * @brief
       * @param
       * @return
       */
      inline double getCellVolume() const { return d_cellVolume; }

      /**
       * @brief
       * @param
       * @return
       */
      inline IntVector getCellExtent() const { return d_totalCellExtent; }

      /*
       *
       */
      inline IntVector getPeriodic() const { return d_periodicVector; }

      /**
       * @brief
       * @param
       * @return
       */
      inline unsigned int getNumAtoms() const { return d_numAtoms; }

      /**
       * @brief
       * @param
       * @return
       */
      inline Vector getBox() const { return d_box; }

      /*
       * @brief
       * @param
       * @return
       */
      inline size_t getNumAtomTypes() const { return d_atomTypeList.size(); }

//      /*
//       * @brief
//       * @param
//       * @return
//       */
//      inline size_t getNumMoleculeType() const { return d_moleculeTypeList.size(); }

      /*
       * @brief
       * @param
       * @return
       */
      inline size_t getNumAtomsOfType(size_t TypeIndex) const { return d_atomTypeList[TypeIndex]; }

//      /*
//       * @brief
//       * @param
//       * @return
//       */
//      inline size_t getNumMoleculesOfType(size_t TypeIndex) const { return d_moleculeTypeList[TypeIndex]; }

      /*
       * @brief
       * @param
       * @return
       */
      inline size_t getNonbondedGhostCells() const { return d_nonbondedGhostCells; }

      /*
       * @brief Return number of ghost cells required for electrostatic realspace calculation
       * @param None
       * @return size_t: Number of ghost cells required in all directions
       */
      inline size_t getElectrostaticGhostCells() const { return d_electrostaticGhostCells; }

      inline double getAtomicCharge(size_t MaterialIndex) const {
    	  return d_simState->getMDMaterial(MaterialIndex)->getCharge();
      }

      inline SimulationStateP getStatePointer() { return d_simState; };

      inline Forcefield* getForcefieldPointer() { return d_forcefield; };

    private:

      ensembleType d_ensemble;                // Variable holding the type of the simulation ensemble

      Forcefield* d_forcefield;               //! Pointer to the established forcefield
      SimulationStateP d_simState;            //! Pointer to the simulation state (for material access)

      unsigned int d_numAtoms;                //!< Total number of atoms in the simulation
      std::vector<size_t> d_atomTypeList;     //!< List of total number of each atom type in the simulation

//      unsigned int d_numMolecules;            //!< Total number of molecules in the simulation
//      std::vector<size_t> d_moleculeTypeList; //!< List of total number of each molecule type in the simulation
      
      Vector d_pressure;                //!< Total MD system pressure
      double d_temperature;             //!< Total MD system temperature
      bool d_orthorhombic;              //!< Whether or not the MD system is using orthorhombic coordinates

      // Unit cell variables
      Matrix3 d_unitCell;               //!< MD system unit cell
      Matrix3 d_inverseCell;            //!< MD system inverse unit cell
      double  d_cellVolume;             //!< Cell volume; calculate internally, return at request for efficiency

      // Total cell variables
      Vector d_box;                     //!< The MD system input box size
      bool d_boxChanged;                //!< Whether or not the system size has changed... create a new box

      IntVector d_totalCellExtent;      //!< Number of sub-cells in the global unit cell
      size_t d_nonbondedGhostCells;     //!< Number of ghost cells for nonbonded realspace neighbor calculations
      size_t d_electrostaticGhostCells; //!< Number of ghost cells for electrostatic realspace neighbor calculations
      IntVector d_periodicVector;       //!< Grid's periodic vector

      inline size_t max(int a, int b){ return (a > b ? a : b); }
      inline double max(int a, int b, int c) { return (max(max(a,b),c)); }

      // disable copy and assignment
      MDSystem(const MDSystem& system);
      MDSystem& operator=(const MDSystem& system);

      void calcCellVolume();
  };

}  // End namespace Uintah

#endif
