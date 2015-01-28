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

#include <CCA/Components/MD/MDLabel.h>
#include <CCA/Components/MD/Forcefields/Forcefield.h>
#include <CCA/Components/MD/Integrators/Integrator.h>
#include <CCA/Components/MD/Nonbonded/NonbondedFactory.h>
#include <CCA/Components/MD/Nonbonded/Nonbonded.h>

#include <CCA/Components/MD/MDSystem.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <CCA/Components/MD/Nonbonded/TwoBodyDeterministic.h>


#include <Core/Util/DebugStream.h>

#include <iostream>
#include <algorithm>

using namespace Uintah;


static DebugStream nbFactory("nonbondedFactory", false);

Nonbonded* NonbondedFactory::create(const ProblemSpecP& ps,
                                           CoordinateSystem* coords,
                                           MDLabel* label,
                                           forcefieldInteractionClass ffType,
                                           interactionModel imType) {
  Nonbonded* nonbonded = 0;
  std::string type = "";

  double nonbondedCutoff;
  ps->findBlock("MD")->findBlock("System")->require("cutoffRadius",
                                                    nonbondedCutoff);

//  // Find the resolution to determine the cutoff cell information
//  SCIRun::Vector realExtents= coords->getCellExtent().asVector();
//  SCIRun::Vector resInverse;
//  coords->toReduced(realExtents, resInverse);
//  Vector resInverse = coords->getCellExtent().asVector() * coords->getInverseCell();
//  Vector fractionalCutoffCells = Vector(nonbondedCutoff) * resInverse;

  // Calculate the fractional number of cells necessary to accomodate the cutoff
  // radius.
  SCIRun::Vector resInverse;
  coords->toReduced(nonbondedCutoff,resInverse);
  proc0cout << "Cutoff as fractional cell: " << resInverse << std::endl;
  SCIRun::Vector fractionalCutoffCells;
  fractionalCutoffCells = coords->getCellExtent().asVector() * resInverse;
  proc0cout << "Double vector with cutoff cell reqiurements: " << fractionalCutoffCells << std::endl;

  int xCells = ceil(fractionalCutoffCells.x());
  int yCells = ceil(fractionalCutoffCells.y());
  int zCells = ceil(fractionalCutoffCells.z());
  int nonbondedGhostCells = std::max(xCells,std::max(yCells, zCells));

  proc0cout << "Integral number of cutoff cells: " << nonbondedGhostCells << std::endl;

  switch (ffType) {
    case(TwoBody):
      switch(imType) {
        case(Deterministic):
          nonbonded = scinew TwoBodyDeterministic(nonbondedCutoff,
                                                  nonbondedGhostCells);
          break;
        default:
          throw ProblemSetupException("Only the two-body deterministic integrator is defined at this time.", __FILE__, __LINE__);
      }
      break;
    default:
      throw ProblemSetupException("Only the two-body deterministic integrator is defined at this time.", __FILE__, __LINE__);
  }

  return nonbonded;
}

//static DebugStream analytic_lj12_6("LJ12_6", false);
//
//Nonbonded* NonbondedFactory::create(const ProblemSpecP& ps,
//                                    MDSystem* system)
//{
//  Nonbonded* nonbonded = 0;
//  string type = "";
//
//  ProblemSpecP electrostatics_ps = ps->findBlock("MD")->findBlock("Nonbonded");
//  if (electrostatics_ps) {
//    electrostatics_ps->getAttribute("type", type);
//  }
//
//  // Default settings
//  if (type == "") {
//    if (analytic_lj12_6.active()) {
//      type = "LJ12_6";
//    } else {
//      throw ProblemSetupException("Must specify Non-bonded type in input file ", __FILE__, __LINE__);
//    }
//  }
//  double cutoffRadius = -1.0;
//  ProblemSpecP universalCutoff = ps->findBlock("MD")->findBlock("MDSystem")->get("cutoffRadius",cutoffRadius);
//
//  // Check for specific non-bonded interaction request
//  if (type == "LJ12_6" || type == "lj12_6") {
//    ProblemSpecP lj12_6_ps = ps->findBlock("MD")->findBlock("Nonbonded");
//    double r12;
//    double r6;
//
//    lj12_6_ps->require("r12", r12);
//    lj12_6_ps->require("r6", r6);
//    if (!universalCutoff) {
//    	lj12_6_ps->require("cutoffRadius", cutoffRadius);
//    }
//
//    nonbonded = scinew AnalyticNonbonded(system, r12, r6, cutoffRadius);
//  } else {
//    throw ProblemSetupException("Unknown Nonbonded type", __FILE__, __LINE__);
//  }
//
//  // Output which non-bonded interactions will be used
//  const ProcessorGroup* world = Uintah::Parallel::getRootProcessorGroup();
//  if (world->myrank() == 0) {
//    cout << "Non-bonded Interactions: \t" << type << endl;
//  }
//
//  return nonbonded;
//
//}
