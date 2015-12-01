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
 * IntegratorFactory.cc
 *
 *  Created on: Oct 17, 2014
 *      Author: jbhooper
 */

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/DebugStream.h>

#include <CCA/Components/MD/MDSystem.h>

#include <CCA/Components/MD/Integrators/Integrator.h>
#include <CCA/Components/MD/Integrators/IntegratorFactory.h>
#include <CCA/Components/MD/Integrators/velocityVerlet/velocityVerlet.h>

#include <sstream>

using namespace Uintah;

Integrator* IntegratorFactory::create(const ProblemSpecP&   ps,
                                            MDSystem*       system,
                                      const VarLabel*       dt_label)
{
  Integrator*   integrator      = 0;
  std::string   integratorType  = "";

  ProblemSpecP integrator_ps;
  integrator_ps = ps->findBlock("MD")->findBlock("Integrator");
  if (!integrator_ps)
  {
    std::stringstream errorMsg;
    errorMsg << "Error:  Could not find an Integrator sub-block in the MD input "
             << "section." << std::endl;
    throw ProblemSetupException(errorMsg.str(), __FILE__, __LINE__);
  }
  integrator_ps->getAttribute("type",integratorType);

  if (integratorType == "velocity") {
    double timestep;
    integrator_ps->require("timestep",timestep);
    long numSteps;
    integrator_ps->require("numberSteps",numSteps);
    integrator = scinew velocityVerlet(dt_label);
  }
  else
  {
    std::ostringstream errorMsg;
    errorMsg << "Unknown integrator type: " << integratorType << std::endl;
    throw ProblemSetupException(errorMsg.str(), __FILE__, __LINE__ );
  }

  return integrator;

}

