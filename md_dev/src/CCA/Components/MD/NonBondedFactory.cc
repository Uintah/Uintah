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

#include <CCA/Components/MD/NonBondedFactory.h>
#include <CCA/Components/MD/NonBonded.h>
#include <CCA/Components/MD/MDSystem.h>
#include <CCA/Components/MD/LJTwelveSix.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/DebugStream.h>

#include <iostream>

using namespace std;
using namespace Uintah;

static DebugStream lj12_6("LJ12_6", false);

NonBonded* NonBondedFactory::create(const ProblemSpecP& ps,
                                    MDSystem* system)
{
  NonBonded* nonbonded = 0;
  string type = "";

  ProblemSpecP electrostatics_ps = ps->findBlock("MD")->findBlock("Nonbonded");
  if (electrostatics_ps) {
    electrostatics_ps->getAttribute("type", type);
  }

  // Default settings
  if (type == "") {
    if (lj12_6.active()) {
      type = "LJ12_6";
    } else {
      throw ProblemSetupException("Must specify Non-bonded type in input file ", __FILE__, __LINE__);
    }
  }

  // Check for specific non-bonded interaction request
  if (type == "LJ12_6" || type == "lj12_6") {
    ProblemSpecP lj12_6_ps = ps->findBlock("MD")->findBlock("Nonbonded");
    double r12;
    double r6;
    double cutoffRadius;

    lj12_6_ps->require("r12", r12);
    lj12_6_ps->require("r6", r6);
    lj12_6_ps->require("cutoffRadius", cutoffRadius);

    nonbonded = scinew LJTwelveSix(system, r12, r6, cutoffRadius);
  } else {
    throw ProblemSetupException("Unknown NonBonded type", __FILE__, __LINE__);
  }

  // Output which non-bonded interactions will be used
  const ProcessorGroup* world = Uintah::Parallel::getRootProcessorGroup();
  if (world->myrank() == 0) {
    cout << "Non-bonded Interactions: \t" << type << endl;
  }

  return nonbonded;

}
