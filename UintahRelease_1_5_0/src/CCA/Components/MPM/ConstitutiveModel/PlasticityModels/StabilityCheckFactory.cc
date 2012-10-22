/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#include "StabilityCheckFactory.h"
#include "DruckerCheck.h"
#include "BeckerCheck.h"
#include "NoneCheck.h"
#include "DruckerBeckerCheck.h"
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <string>
#include <Core/Parallel/Parallel.h>

using namespace std;
using namespace Uintah;

/// Create an instance of a stabilty check method
/*! Available checks are : loss of ellipticity of the acoustic tensor */
StabilityCheck* StabilityCheckFactory::create(ProblemSpecP& ps)
{
  ProblemSpecP child = ps->findBlock("stability_check");
  if(!child) {
    proc0cout << "**WARNING** Creating default action (no stability check)" << endl;
    return(scinew NoneCheck());
    throw ProblemSetupException("Cannot find stability check criterion.", __FILE__, __LINE__);
  }

  string mat_type;
  if(!child->getAttribute("type", mat_type))
    throw ProblemSetupException("No type for stability check criterion.", __FILE__, __LINE__);
   
  if (mat_type == "drucker")
    return(scinew DruckerCheck(child));
  else if (mat_type == "becker")
    return(scinew BeckerCheck(child));
  else if (mat_type == "drucker_becker")
    return(scinew DruckerBeckerCheck(child));
  else if (mat_type == "none")
    return(scinew NoneCheck(child));
  else {
    proc0cout << "**WARNING** Creating default action (no stability check)" << endl;
    return(scinew NoneCheck(child));
    // throw ProblemSetupException("Unknown Stability Check ("+mat_type+")", __FILE__, __LINE__);
  }
}

StabilityCheck* 
StabilityCheckFactory::createCopy(const StabilityCheck* sc)
{
  if (dynamic_cast<const DruckerCheck*>(sc))
    return(scinew DruckerCheck(dynamic_cast<const DruckerCheck*>(sc)));

  else if (dynamic_cast<const BeckerCheck*>(sc))
    return(scinew BeckerCheck(dynamic_cast<const BeckerCheck*>(sc)));

  else if (dynamic_cast<const DruckerBeckerCheck*>(sc))
    return(scinew DruckerBeckerCheck(dynamic_cast<const DruckerBeckerCheck*>(sc)));
  else if (dynamic_cast<const NoneCheck*>(sc))
    return(scinew NoneCheck(dynamic_cast<const NoneCheck*>(sc)));

  else {
    proc0cout << "**WARNING** Creating copy of default action (no stability check)" << endl;
    return(scinew NoneCheck(dynamic_cast<const NoneCheck*>(sc)));
    //  return 0;
  }
}
