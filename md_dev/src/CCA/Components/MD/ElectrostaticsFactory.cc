/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include <CCA/Components/MD/ElectrostaticsFactory.h>
#include <CCA/Components/MD/Electrostatics.h>
#include <CCA/Components/MD/MDSystem.h>
#include <CCA/Components/MD/SPME.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DebugStream.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>

#include <iostream>

using namespace std;
using namespace Uintah;

static DebugStream SPMEDBG("SPME", false);

Electrostatics* ElectrostaticsFactory::create(const ProblemSpecP& ps,
                                              MDSystem* system)
{
  Electrostatics* electrostatics = 0;
  string type = "";

  ProblemSpecP electrostatics_ps = ps->findBlock("Electrostatics");
  if (electrostatics_ps) {
    electrostatics_ps->getAttribute("type", type);
  }

  // Default settings
  if (type == "") {
    if (SPMEDBG.active()) {
      type = "SPME";
    } else {
      throw ProblemSetupException("Must specify Electrostatics type in input file.", __FILE__, __LINE__);
    }
  }

  // Output which electrostatics type that will be used
  const ProcessorGroup* world = Uintah::Parallel::getRootProcessorGroup();
  if (world->myrank() == 0) {
    cout << "Electrostatics: \t\t" << type << endl;
  }

  // Check for specific electrostatics request
  if (type == "SPME" || type == "spme") {
    ProblemSpecP spme_ps = ps->findBlock("Electrostatics");
    double ewaldBeta;
    bool polarizable;
    double polTolerance;
    IntVector kLimits;
    int splineOrder;

    spme_ps->require("ewaldBeta", ewaldBeta);
    spme_ps->require("polarizable", polarizable);
    spme_ps->require("polarizationTolerance", polTolerance);
    spme_ps->require("kLimits", kLimits);
    spme_ps->require("splineOrder", splineOrder);

    electrostatics = scinew SPME(system, ewaldBeta, polarizable, polTolerance, kLimits, splineOrder);
  } else {
    throw ProblemSetupException("Unknown Electrostatics type", __FILE__, __LINE__);
  }

  return electrostatics;

}
