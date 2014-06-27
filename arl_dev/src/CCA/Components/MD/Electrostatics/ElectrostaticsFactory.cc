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

#include <CCA/Components/MD/Electrostatics/ElectrostaticsFactory.h>
#include <CCA/Components/MD/Electrostatics/Electrostatics.h>
#include <CCA/Components/MD/Electrostatics/DefinedElectrostaticsTypes.h>
#include <CCA/Components/MD/CoordinateSystems/CoordinateSystem.h>

#include <CCA/Components/MD/MDSystem.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/DebugStream.h>

#include <iostream>
#include <sstream>
#include <string>
#include <cstring>

using namespace std;
using namespace Uintah;

static DebugStream spme("SPME", false);

Electrostatics* ElectrostaticsFactory::create(const ProblemSpecP& ps,
                                                    CoordinateSystem* coords)
{
  Electrostatics* electrostatics = 0;
  string type = "";

  ProblemSpecP electrostatics_ps;
  electrostatics_ps = ps->findBlock("MD")->findBlock("Electrostatics");
  if (electrostatics_ps) {
    electrostatics_ps->getAttribute("type", type);
  }
  else {
    std::stringstream errorMsg;
    errorMsg << "Error:  Could not find an Electrostatics sub-block in the "
             << "MD input section." << std::endl;
    throw ProblemSetupException(errorMsg.str(), __FILE__, __LINE__);
  }
  // Find the resolution to determine the cutoff cell information
  Vector resInverse = coords->getCellExtent().asVector() * coords->getInverseCell();

  // Default settings
  if (type == "") {
    if (spme.active()) {
      type = "SPME";
    } else {
      throw ProblemSetupException("Must specify Electrostatics type in input file ", __FILE__, __LINE__);
    }
  }

  // Pull cutoff radius from upstream in case there's not a specific one in the electrostatics specification
  double cutoffRadius = -1.0;
  ps->findBlock("MD")->findBlock("System")->require("cutoffRadius",cutoffRadius);
  // Check for specific electrostatics request
  if (type == "SPME" || type == "spme") {
    ProblemSpecP spme_ps = ps->findBlock("MD")->findBlock("Electrostatics");
    double ewaldBeta;
    spme_ps->require("ewaldBeta", ewaldBeta);
    IntVector kGrids;
    spme_ps->require("kGrids",kGrids);
    int splineOrder;
    spme_ps->require("splineOrder",splineOrder);
    std::string polarizableEnabled;
    ProblemSpecP pol_ps = spme_ps->findBlock("polarizable");
    double polTolerance = MDConstants::defaultPolarizationTolerance;
    int polMaxIterations = 0;
    bool polarizable = false;
    if (pol_ps) {
      pol_ps->getAttribute("enabled",polarizableEnabled);
      std::transform(polarizableEnabled.begin(),polarizableEnabled.end(),polarizableEnabled.begin(),::toupper);
      if ( "TRUE" == polarizableEnabled) {
        polarizable = true;
        pol_ps->require("polarizationTolerance",polTolerance);
        pol_ps->require("maxIterations",polMaxIterations);
      }
    }
    else {
      throw ProblemSetupException("ERROR:  No polarizable block found in the Electrostatics block", __FILE__, __LINE__);
    }
    double tempCutoff;
    if (spme_ps->get("cutoffRadius",tempCutoff)) {
      cutoffRadius = tempCutoff;
    }

    Vector fractionalCutoffCells = Vector(cutoffRadius) * resInverse;
    int electrostaticGhostCells = ceil(fractionalCutoffCells.x());
    int yCells = ceil(fractionalCutoffCells.y());
    int zCells = ceil(fractionalCutoffCells.z());
    electrostaticGhostCells = max(electrostaticGhostCells,max(yCells, zCells));

    electrostatics = scinew SPME(ewaldBeta,
                                 cutoffRadius,
                                 electrostaticGhostCells,
                                 kGrids,
                                 splineOrder,
                                 polarizable,
                                 polMaxIterations,
                                 polTolerance);

  } else {
    throw ProblemSetupException("Unknown Electrostatics type", __FILE__, __LINE__);
  }

  // Output which electrostatics type that will be used
  const ProcessorGroup* world = Uintah::Parallel::getRootProcessorGroup();
  if (world->myrank() == 0) {
    cout << "Electrostatics Method: \t\t" << type << endl;

  }

  return electrostatics;

}
