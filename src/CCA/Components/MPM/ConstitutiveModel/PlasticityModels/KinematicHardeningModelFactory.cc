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

#include "KinematicHardeningModelFactory.h"                                             
#include "NoKinematicHardening.h"
#include "PragerKinematicHardening.h"
#include "ArmstrongFrederickKinematicHardening.h"
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>
#include <string>
using std::cerr;
using std::ifstream;
using std::ofstream;

using namespace Uintah;

KinematicHardeningModel* KinematicHardeningModelFactory::create(ProblemSpecP& ps)
{
   ProblemSpecP child = ps->findBlock("kinematic_hardening_model");
   if(!child) {
      cerr << "**WARNING** Creating default (no kinematic hardening) model" << endl;
      return(scinew NoKinematicHardening());
      //throw ProblemSetupException("Cannot find kinematic hardening model tag", __FILE__, __LINE__);
   }

   string mat_type;
   if(!child->getAttribute("type", mat_type))
      throw ProblemSetupException("No type for kinematic hardening model", __FILE__, __LINE__);

   if (mat_type == "none")
      return(scinew NoKinematicHardening(child));
   else if (mat_type == "prager_hardening")
      return(scinew PragerKinematicHardening(child));
   else if (mat_type == "armstrong_frederick_hardening")
      return(scinew ArmstrongFrederickKinematicHardening(child));
   else {
      cerr << "**WARNING** Creating default (no kinematic hardening) model" << endl;
      return(scinew NoKinematicHardening(child));
      //throw ProblemSetupException("Unknown KinematicHardening Model ("+mat_type+")", __FILE__, __LINE__);
   }
}

KinematicHardeningModel* 
KinematicHardeningModelFactory::createCopy(const KinematicHardeningModel* pm)
{
   if (dynamic_cast<const NoKinematicHardening*>(pm))
      return(scinew NoKinematicHardening(dynamic_cast<const 
                                        NoKinematicHardening*>(pm)));

   else if (dynamic_cast<const PragerKinematicHardening*>(pm))
      return(scinew PragerKinematicHardening(dynamic_cast<const 
                                       PragerKinematicHardening*>(pm)));

   else if (dynamic_cast<const ArmstrongFrederickKinematicHardening*>(pm))
      return(scinew ArmstrongFrederickKinematicHardening(dynamic_cast<const ArmstrongFrederickKinematicHardening*>(pm)));

   else {
      cerr << "**WARNING** Creating copy of default (no kinematic hardening) model" << endl;
      return(scinew NoKinematicHardening(dynamic_cast<const 
                                        NoKinematicHardening*>(pm)));
      //throw ProblemSetupException("Cannot create copy of unknown kinematic_hardening model", __FILE__, __LINE__);
   }
}

