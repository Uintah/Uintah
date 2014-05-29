/*
 *
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
 *
 * ----------------------------------------------------------
 * coordinateSystemFactory.cc
 *
 *  Created on: May 13, 2014
 *      Author: jbhooper
 */

#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>

#include <Core/Exceptions/ProblemSetupException.h>

#include <Core/ProblemSpec/ProblemSpec.h>

#include <CCA/Components/MD/MDUtil.h>

#include <CCA/Components/MD/CoordinateSystems/coordinateSystemFactory.h>
#include <CCA/Components/MD/CoordinateSystems/orthorhombicCoordinates.h>
#include <CCA/Components/MD/CoordinateSystems/genericCoordinates.h>

using namespace Uintah;

coordinateSystem* coordinateSystemFactory::create(const ProblemSpecP&     ps,
                                                 const SimulationStateP& shared_state,
                                                 const GridP&            grid) {
  coordinateSystem* coordinates = 0;

  ProblemSpecP unitcell_ps = ps->findBlock("MD")->findBlock("System")->findBlock("unitCell");
  if (!unitcell_ps) {
    throw ProblemSetupException("Could not find \"unitCell\" subsection of MD block.", __FILE__, __LINE__);
  }

  std::string unitCellType;
  unitcell_ps->getAttribute("format",unitCellType);


  SCIRun::IntVector lowIndex, highIndex;
  LevelP gridLevel0 = grid->getLevel(0);
  gridLevel0->findCellIndexRange(lowIndex, highIndex);

  SCIRun::IntVector gridExtent, periodicVector;
  gridExtent = highIndex - lowIndex;
  periodicVector = gridLevel0->getPeriodicBoundaries();

  bool orthorhombic = false;
  Uintah::Matrix3 unitCell = Matrix3(0.0);

  if ("Isotropic" == unitCellType) { // Cubic unit cell is orthorhombic)
    double boxLength;
    unitcell_ps->require("boxSize", boxLength);
    SCIRun::Vector length(boxLength);
    SCIRun::Vector angles(MDConstants::orthogonalAngle);
    orthorhombic = true;
    coordinates = scinew orthorhombicCoordinates(gridExtent, periodicVector, length, angles);
  }
  else if ("Length-Angle" == unitCellType) {
    // Unit cell is entered as a vector of three basis vector lengths and a vector of three angles between basis vectors
    SCIRun::Vector lengths, angles;
    unitcell_ps->require("abc",lengths);
    unitcell_ps->require("AlphaBetaGamma",angles);
    SCIRun::Vector angleOrthoDeviation = angles*MDConstants::degToRad - SCIRun::Vector(MDConstants::orthogonalAngle);
    if ( angleOrthoDeviation.maxComponent() < MDConstants::zeroTol) {
      orthorhombic = true;
      coordinates = scinew orthorhombicCoordinates(gridExtent, periodicVector, lengths, angles);
    }
    else { // Not orthorhombic
      coordinates = scinew genericCoordinates(gridExtent, periodicVector, lengths, angles);
    }
  }
  else if ("basisVectors" == unitCellType) { // Explicit vasis vectors
    throw ProblemSetupException("Entering a unit cell via basis vectors is not yet supported",
                                __FILE__,
                                __LINE__);
  }
  else { // Unrecognized unit cell type
    throw ProblemSetupException("Unrecognized option for specifying the unit cell.",
                                __FILE__,
                                __LINE__);

  }
  return coordinates;

}


