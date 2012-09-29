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

#include "InternalVariableModelFactory.h"                                             
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>
#include <string>
#include "InternalVar_BorjaPressure.h"
#include "InternalVar_ArenaKappa.h"

using namespace UintahBB;
using Uintah::ProblemSpecP;
using Uintah::ProblemSetupException;
using std::cerr;
using std::ifstream;
using std::ofstream;

InternalVariableModel* InternalVariableModelFactory::create(ProblemSpecP& ps)
{
   ProblemSpecP child = ps->findBlock("internal_variable_model");
   if(!child)
      throw ProblemSetupException("Cannot find internal_var_model tag", __FILE__, __LINE__);
   std::string mat_type;
   if(!child->getAttribute("type", mat_type))
      throw ProblemSetupException("No type for internal_var_model", __FILE__, __LINE__);
   if (mat_type == "borja_consolidation_pressure")
      return(scinew InternalVar_BorjaPressure(child));
   else if (mat_type == "arena_kappa")
      return(scinew InternalVar_ArenaKappa(child));
   else {
      throw ProblemSetupException("Unknown InternalVariable Model ("+mat_type+")", __FILE__, __LINE__);
   }
}

InternalVariableModel* 
InternalVariableModelFactory::createCopy(const InternalVariableModel* pm)
{
   if (dynamic_cast<const InternalVar_BorjaPressure*>(pm))
      return(scinew InternalVar_BorjaPressure(dynamic_cast<const InternalVar_BorjaPressure*>(pm)));

   else if (dynamic_cast<const InternalVar_ArenaKappa*>(pm))
      return(scinew InternalVar_ArenaKappa(dynamic_cast<const InternalVar_ArenaKappa*>(pm)));

   else {
      throw Uintah::ProblemSetupException("Cannot create copy of unknown internal var model", __FILE__, __LINE__);
   }
}

