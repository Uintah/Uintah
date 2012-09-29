/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "SpecificHeatModelFactory.h"
#include "ConstantCp.h"
#include "CopperCp.h"
#include "SteelCp.h"
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <string>
#include <iostream>
#include <sstream>
#include <Core/Parallel/Parallel.h>

using namespace Uintah;
using namespace std;

/// Create an instance of a specific heat model
SpecificHeatModel* SpecificHeatModelFactory::create(ProblemSpecP& ps)
{
  ProblemSpecP child = ps->findBlock("specific_heat_model");
  if(!child) {
    proc0cout << "** WARNING ** Creating default (constant specific heat) model" << endl;
    return(scinew ConstantCp());
    //ostringstream desc;
    //desc << "**Error in Input UPS File: " 
        // << "MPM:SpecificHeatModel:  "
        // << "No specific_heat_model tag found in input file." << endl;
    //throw ProblemSetupException(desc.str(), __FILE__, __LINE__);
  }
  string mat_type;
  if(!child->getAttribute("type", mat_type)) {
    ostringstream desc;
    desc << "**Error in Input UPS File: " 
         << "MPM:SpecificHeatModel:  "
         << "No specific_heat_model type tag found in input file. " << endl 
         << "Types include constant_Cp, copper_Cp, and steel_Cp." << endl;
    throw ProblemSetupException(desc.str(), __FILE__, __LINE__);
  }
   
  if (mat_type == "constant_Cp")
    return(scinew ConstantCp(child));
  else if (mat_type == "copper_Cp")
    return(scinew CopperCp(child));
  else if (mat_type == "steel_Cp")
    return(scinew SteelCp(child));
  else {
    proc0cout << "** WARNING ** Creating default (constant specific heat) model" << endl;
    return(scinew ConstantCp(child));
    //ostringstream desc;
    //desc << "**Error in Input UPS File: " 
         //<< "MPM:SpecificHeatModel:  "
         //<< "Incorrect specific_heat_model type (" << mat_type 
         //<< ") found in input file. " << endl 
         //<< "Correct type tags include constant_Cp, copper_Cp, and steel_Cp." 
         //<< endl;
    //throw ProblemSetupException(desc.str(), __FILE__, __LINE__);
  }
}

SpecificHeatModel* 
SpecificHeatModelFactory::createCopy(const SpecificHeatModel* smm)
{
  if (dynamic_cast<const ConstantCp*>(smm))
    return(scinew ConstantCp(dynamic_cast<const ConstantCp*>(smm)));
  else if (dynamic_cast<const CopperCp*>(smm))
    return(scinew CopperCp(dynamic_cast<const CopperCp*>(smm)));
  else if (dynamic_cast<const SteelCp*>(smm))
    return(scinew SteelCp(dynamic_cast<const SteelCp*>(smm)));
  else {
    proc0cout << "** WARNING ** Creating copy of default (constant specific heat) model" << endl;
    return(scinew ConstantCp(dynamic_cast<const ConstantCp*>(smm)));
    //ostringstream desc;
    //desc << "**Error in Material Copying: " 
        // << "MPM:SpecificHeatModel:  "
        // << "Cannot create copy of unknown specific heat model"
        // << endl;
    //throw ProblemSetupException(desc.str(), __FILE__, __LINE__);
  }
}
