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

#include <Core/Exceptions/ProblemSetupException.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimulationStateP.h>

#include <CCA/Components/MD/Forcefields/Forcefield.h>
#include <CCA/Components/MD/MDSystem.h>

#include <iostream>
#include <cmath>

#include <sci_values.h>

//using namespace Uintah;

namespace Uintah {

  MDSystem::MDSystem()
  {

  }

  MDSystem::~MDSystem()
  {

  }

  MDSystem::MDSystem(const ProblemSpecP& ps,
                     GridP& grid,
                     SimulationStateP& _state)
                    :d_simState(_state)
  {
    // Forcefield must be attached externally
    d_forcefield = 0;

    //std::vector<int> d_atomTypeList;
    ProblemSpecP mdsystem_ps = ps->findBlock("MD")->findBlock("System");
    std::string ensembleLabel;

    if (!mdsystem_ps) {
      throw ProblemSetupException("Could not find \"System\" subsection of MD block.", __FILE__, __LINE__);
    }
    mdsystem_ps->getAttribute("ensemble",ensembleLabel);
    if ("NVE" == ensembleLabel) {
      d_ensemble = NVE;
    }
    else if ("NVT" == ensembleLabel) {
      d_ensemble = NVT;
      mdsystem_ps->require("temperature", d_temperature);
    }
    else if ("NPT" == ensembleLabel) {
      d_ensemble = NPT;
      mdsystem_ps->require("temperature", d_temperature);
      mdsystem_ps->require("pressure", d_pressure);
    }
    else if ("Isokinetic" == ensembleLabel) {
      d_ensemble = ISOKINETIC;
      mdsystem_ps->require("temperature", d_temperature);
    }
    else { // Unknown ensemble listed
      std::stringstream errorOut;
      errorOut << "ERROR in the System section of the MD specification block!" << std::endl
               << "  Unknown ensemble requested: " << ensembleLabel << std::endl
               << "  Available ensembles are:  \"NVE  NVT   NPT   Isokinetic\"" << std::endl;
      throw ProblemSetupException(errorOut.str(), __FILE__, __LINE__);
    }

    std::cerr << "MDSystem::MDSystem --> Parsed ensemble information" << std::endl;

//    ProblemSpecP unitcell_ps = mdsystem_ps->findBlock("unitCell");
//    if (!unitcell_ps) {
//      throw ProblemSetupException("Could not find \"unitCell\" subsection of MD block.", __FILE__, __LINE__);
//    }
//
//
//
//    std::string unitCellType;
//    unitcell_ps->getAttribute("format",unitCellType);
//    bool orthorhombic = false;
//    Uintah::Matrix3 unitCell = Matrix3(0.0);
//    if ("Isotropic" == unitCellType) {  // Cubic unit cell is orthorhombic
//      double boxLength;
//      unitcell_ps->require("boxSize",boxLength);
//
//      unitCell(0,0) = boxLength;
//      unitCell(1,1) = boxLength;
//      unitCell(2,2) = boxLength;
//      orthorhombic = true;
//    }
//    else if ("Length-Angle" == unitCellType) { // Enter unit cell as three basis vector lengths and angles between them
//      double zeroTol = 1e-10;  // Tolerance for angles being "close enough" to zero
//      SCIRun::Vector lengths, angles;
//      unitcell_ps->require("abc",lengths);
//      unitcell_ps->require("AlphaBetaGamma",angles);
//      SCIRun::Vector angleOrthoDeviation = angles - SCIRun::Vector(90.0);
//      if ( abs(angleOrthoDeviation[0]) <= zeroTol    // Orthorhombic
//        && abs(angleOrthoDeviation[1]) <= zeroTol
//        && abs(angleOrthoDeviation[2]) <= zeroTol) { // All angles are 90 degrees
//        orthorhombic = true;
//        unitCell(0,0) = lengths[0];
//        unitCell(1,1) = lengths[1];
//        unitCell(2,2) = lengths[2];
//      }
//      else { // General
//        double a=lengths[0];
//        double b=lengths[1];
//        double c=lengths[2];
//        double alpha=angles[0];
//        double beta=angles[1];
//        double gamma=angles[2];
//
//        unitCell(0,0) = a;
//        unitCell(0,1) = b*cos(gamma);
//        unitCell(1,1) = b*sin(gamma);
//        unitCell(0,2) = c*cos(beta);
//        unitCell(1,2) = c*(cos(alpha) - cos(beta)*cos(gamma))/sin(gamma);
//        unitCell(2,2) = sqrt(c*c - unitCell(0,2)*unitCell(0,2) - unitCell(1,2)*unitCell(1,2));
//      }
//    }
//    else if ("basisVectors") { // Explicit basis vectors
//      // Need to write a std::vector<SCIRun::Vector> parser in ProblemSpec.cc before this will work
//      throw ProblemSetupException("Entering a unit cell via basis_vectors is not yet supported", __FILE__, __LINE__);
//    }
//    else { // Unrecognized unit cell option
//      throw ProblemSetupException("Unrecognized option for specifying the unit cell.", __FILE__, __LINE__);
//    }
//
//    std::cerr << "MDSystem::MDSystem --> Parsed unit cell information" << std::endl;


//    calcCellVolume();
//    d_inverseCell = d_unitCell.Inverse();
    // FIXME JBH We should set d_Cell somewhere around here

//    // Determine the total number of cells in the system so we can map dimensions
//    IntVector lowIndex, highIndex;
//    LevelP gridLevel0 = grid->getLevel(0);
//    gridLevel0->findCellIndexRange(lowIndex, highIndex);
//    d_totalCellExtent = highIndex - lowIndex;
//    d_periodicVector = gridLevel0->getPeriodicBoundaries();
//
//    // Determine number of ghost cells tasks should request for neighbor calculations
//    IntVector resolution;
//    ps->findBlock("Grid")->findBlock("Level")->findBlock("Box")->require("resolution", resolution);
//    Vector resInverse = resolution.asVector() * d_inverseCell;
//
////    std::cerr << " 1..." << std::endl;
//    // We require a cutoff radius somewhere.  Initially we bind the cutoff to both the nonbonded and the electrostatic cutoff
//    double nonbondedRadius = -1.0;
//    double electrostaticRadius = -1.0;
//    mdsystem_ps->require("cutoffRadius",nonbondedRadius);
//    Vector normalized = Vector(nonbondedRadius) * resInverse;
//    IntVector maxDimValues(ceil(normalized.x()), ceil(normalized.y()), ceil(normalized.z()));
////    d_nonbondedGhostCells = max(maxDimValues.x(), maxDimValues.y(), maxDimValues.z());
////    d_electrostaticGhostCells = d_nonbondedGhostCells;
////    std::cerr << " 2..." << std::endl;
//
//    ProblemSpecP electrostatics_ps = ps->findBlock("MD")->findBlock("Electrostatics")->get("cutoffRadius",electrostaticRadius);
//
//    if (electrostatics_ps) {  // cutoffRadius specification found in Electrostatics block so use it for electrostatic cutoff
////      std::cerr << "Found an electrostatic only cutoff Radius of " << electrostaticRadius << std::endl;
//      normalized = Vector(electrostaticRadius) * resInverse;
//      maxDimValues = IntVector(ceil(normalized.x()), ceil(normalized.y()), ceil(normalized.z()));
////      d_electrostaticGhostCells = max(maxDimValues.x(), maxDimValues.y(), maxDimValues.z());
//    }
//
//    std::cerr << "MDSystem::MDSystem --> Parsed cutoff neighbor cell information" << std::endl;


    int numAtomTypes = d_simState->getNumMatls();
    d_numAtomsOfType = std::vector<size_t> (numAtomTypes,0);
    d_numAtoms=0;

    std::cerr << "MDSystem::MDSystem --> Parsed atom type list information" << std::endl;

    // Reference the atomMap here to get number of atoms of Type into system

//    d_numMolecules = 0;
//    d_moleculeTypeList = std::vector<size_t> (d_numMolecules,0);
//    d_moleculeTypeList.resize(d_numMolecules);

//    d_cellChanged = true;
//    d_cellVolume = 0.0;
//    calcCellVolume();
  }

//  void MDSystem::calcCellVolume()
//  {
//    if (d_orthorhombic) {
//      d_cellVolume = d_unitCell(0, 0) * d_unitCell(1, 1) * d_unitCell(2, 2);
//      return;
//    }
//
//    Vector A, B, C;
//    A[0] = d_unitCell(0, 0);
//    A[1] = d_unitCell(0, 1);
//    A[2] = d_unitCell(0, 2);
//    B[0] = d_unitCell(1, 0);
//    B[1] = d_unitCell(1, 1);
//    B[2] = d_unitCell(1, 2);
//    C[0] = d_unitCell(2, 0);
//    C[1] = d_unitCell(2, 1);
//    C[2] = d_unitCell(2, 2);
//
//    d_cellVolume = Dot(Cross(A, B), C);
//
//    return;
//  }
} // namespace Uintah_MD
