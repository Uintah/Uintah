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

#include <CCA/Components/MD/MDSystem.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>

#include <iostream>
#include <cmath>

#include <sci_values.h>

using namespace Uintah;

namespace Uintah {
  static const double PI = acos(-1.0);
  static const double degToRad = PI/180.0;

  MDSystem::MDSystem()
  {

  }

  MDSystem::~MDSystem()
  {

  }

  MDSystem::MDSystem(ProblemSpecP& ps,
                     GridP& grid,
                     SimulationStateP& shared_state)
  {
    //std::vector<int> d_atomTypeList;
    ProblemSpecP mdsystem_ps = ps->findBlock("MDSystem");
    mdsystem_ps->require("numAtoms", d_numAtoms);
    mdsystem_ps->require("pressure", d_pressure);
    mdsystem_ps->require("temperature", d_temperature);
    mdsystem_ps->require("orthorhombic", d_orthorhombic);

    if (d_orthorhombic) {
      mdsystem_ps->get("boxSize", d_box);
      d_unitCell *= 0.0; // Initialize
      d_unitCell(0, 0) = d_box[0];
      d_unitCell(1, 1) = d_box[1];
      d_unitCell(2, 2) = d_box[2];
    }
    else {
      d_unitCell *= 0.0; // Initialize
      //  Assume one of two forms:
      //  Coordinate Magnitude/Angle:
      //  a, b, c:  Vector magnitudes
      //  alpha_deg, beta_deg, gamma_deg:  Cell angles
      std::string generalFormat;
      mdsystem_ps->require("generalFormat",generalFormat);
      if (generalFormat == "Axis-Angle") {
        SCIRun::Vector abc;
        SCIRun::Vector alphabetagamma;
        mdsystem_ps->require("magnitudes", abc);
        mdsystem_ps->require("angles", alphabetagamma);
        double a = abc[0];
        double b = abc[1];
        double c = abc[2];
        double alphaRad = alphabetagamma[0]*degToRad;
        double betaRad = alphabetagamma[1]*degToRad;
        double gammaRad = alphabetagamma[2]*degToRad;
        d_unitCell(0, 0) = a;
        d_unitCell(0, 1) = b * cos(gammaRad);
        d_unitCell(1, 1) = b * sin(gammaRad);
        d_unitCell(0, 2) = c * cos(betaRad);
        d_unitCell(1, 2) = c * (cos(alphaRad) - cos(betaRad) * cos(gammaRad))/sin(gammaRad);
        d_unitCell(2, 2) = sqrt(c * c - d_unitCell(0, 2) * d_unitCell(0, 2) - d_unitCell(1, 2) * d_unitCell(1, 2));
      }
      else { // Assume matrix representation
        SCIRun::Vector row;
        mdsystem_ps->require("row 1", row);
        d_unitCell(0, 0) = row[0];
        d_unitCell(0, 1) = row[1];
        d_unitCell(0, 2) = row[2];
        mdsystem_ps->require("row 2", row);
        d_unitCell(1, 0) = row[0];
        d_unitCell(1, 1) = row[1];
        d_unitCell(1, 2) = row[2];
        mdsystem_ps->require("row 3", row);
        d_unitCell(2, 0) = row[0];
        d_unitCell(2, 1) = row[1];
        d_unitCell(2, 2) = row[2];
      }
      // Read in non orthorhombic unit cell
    }
    calcCellVolume();
    d_inverseCell = d_unitCell.Inverse();

    // Determine the total number of cells in the system so we can map dimensions
    IntVector lowIndex, highIndex;
    grid->getLevel(0)->findCellIndexRange(lowIndex, highIndex);
    d_totalCellExtent = highIndex - lowIndex;

    // Determine number of ghost cells tasks should request for neighbor calculations
    IntVector resolution;
    ProblemSpecP root_ps = ps->getRootNode();
    root_ps->findBlock("Grid")->findBlock("Level")->findBlock("Box")->require("resolution", resolution);
    Vector resInverse = resolution.asVector() * d_inverseCell;

    double nonbondedRadius = -1.0;
    double electrostaticRadius = -1.0;
    ProblemSpecP universalCutoff = root_ps->findBlock("MD")->findBlock("MDSystem")->get("cutoffRadius", nonbondedRadius);
    if (universalCutoff) {  // Same cutoff for nonbonded and electrostatics
      Vector normalized = Vector(nonbondedRadius) * resInverse;
      IntVector maxDimValues(ceil(normalized.x()), ceil(normalized.y()), ceil(normalized.z()));
      d_nonbondedGhostCells = max(maxDimValues.x(), maxDimValues.y(), maxDimValues.z());
      d_electrostaticGhostCells = d_nonbondedGhostCells;
    }
    else // Cutoff for each component
    {
      // Find ghost cells for nonbonded
      root_ps->findBlock("MD")->findBlock("Nonbonded")->require("cutoffRadius", nonbondedRadius);
      Vector normalized = Vector(nonbondedRadius) * resInverse;
      IntVector maxDimValues(ceil(normalized.x()), ceil(normalized.y()), ceil(normalized.z()));
      d_nonbondedGhostCells = max(maxDimValues.x(), maxDimValues.y(), maxDimValues.z());

      // Find ghost cells for electrostatic
      root_ps->findBlock("MD")->findBlock("Electrostatics")->require("cutoffRadius", electrostaticRadius);
      normalized = Vector(electrostaticRadius, electrostaticRadius, electrostaticRadius) * resInverse;
      maxDimValues = IntVector(ceil(normalized.x()), ceil(normalized.y()), ceil(normalized.z()));
      d_electrostaticGhostCells = max(maxDimValues.x(), maxDimValues.y(), maxDimValues.z());
    }

//  int numAtomTypes = 1; //shared_state->getNumMatls();
//  std::vector<size_t> tempAtomTypeList(numAtomTypes);
//  d_atomTypeList = tempAtomTypeList;
    //d_atomTypeList.resize(shared_state->getNumMatls());
    d_atomTypeList.resize(1);  // Hard coded for our simple Material case
    // Not so easy to do.
//  const MaterialSet* materialList = shared_state->allMaterials();
//  size_t numberMaterials = materialList->size();
//  for (size_t matlIndex=0; matlIndex < numberMaterials; ++matlIndex) {
//    d_atomTypeList.push_back()
//  }
//  // Determine the total number of atom types (from the system material list)
//  d_numAtomType = shared_state->getNumMatls();

    d_numMolecules = 0;
    d_moleculeTypeList.resize(d_numMolecules);
    // Determine total number of molecule types (looking ahead)
    //d_numMoleculeType = shared_state->getNumMolecules();  ???

    d_boxChanged = true;
    d_cellVolume = 0.0;
    calcCellVolume();
  }

  void MDSystem::calcCellVolume()
  {
    if (d_orthorhombic) {
      d_cellVolume = d_unitCell(0, 0) * d_unitCell(1, 1) * d_unitCell(2, 2);
      return;
    }

    Vector A, B, C;
    A[0] = d_unitCell(0, 0);
    A[1] = d_unitCell(0, 1);
    A[2] = d_unitCell(0, 2);
    B[0] = d_unitCell(1, 0);
    B[1] = d_unitCell(1, 1);
    B[2] = d_unitCell(1, 2);
    C[0] = d_unitCell(2, 0);
    C[1] = d_unitCell(2, 1);
    C[2] = d_unitCell(2, 2);

    d_cellVolume = Dot(Cross(A, B), C);

    return;
  }
} // namespace Uintah_MD
