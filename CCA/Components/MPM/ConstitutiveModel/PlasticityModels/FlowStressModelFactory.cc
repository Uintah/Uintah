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

#include "FlowStressModelFactory.h"                                             
#include "IsoHardeningFlow.h"
#include "JohnsonCookFlow.h"
#include "ZAFlow.h"
#include "ZAPolymerFlow.h"
#include "MTSFlow.h"
#include "SCGFlow.h"
#include "PTWFlow.h"
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

FlowModel* FlowStressModelFactory::create(ProblemSpecP& ps)
{
   ProblemSpecP child = ps->findBlock("flow_model");
   if(!child)
      throw ProblemSetupException("Cannot find flow_model tag", __FILE__, __LINE__);
   string mat_type;
   if(!child->getAttribute("type", mat_type))
      throw ProblemSetupException("No type for flow_model", __FILE__, __LINE__);
   if (mat_type == "isotropic_hardening")
      return(scinew IsoHardeningFlow(child));
   else if (mat_type == "johnson_cook")
      return(scinew JohnsonCookFlow(child));
   else if (mat_type == "zerilli_armstrong")
      return(scinew ZAFlow(child));
   else if (mat_type == "zerilli_armstrong_polymer")
      return(scinew ZAPolymerFlow(child));
   else if (mat_type == "mts_model")
      return(scinew MTSFlow(child));
   else if (mat_type == "steinberg_cochran_guinan")
      return(scinew SCGFlow(child));
   else if (mat_type == "preston_tonks_wallace")
      return(scinew PTWFlow(child));
   else {
      //cerr << "**WARNING** Creating default isotropic hardening flow model" << endl;
      //return(scinew IsoHardeningFlow(child));
      throw ProblemSetupException("Unknown flow Model ("+mat_type+")", __FILE__, __LINE__);
   }
}

FlowModel* 
FlowStressModelFactory::createCopy(const FlowModel* pm)
{
   if (dynamic_cast<const IsoHardeningFlow*>(pm))
      return(scinew IsoHardeningFlow(dynamic_cast<const 
                                        IsoHardeningFlow*>(pm)));

   else if (dynamic_cast<const JohnsonCookFlow*>(pm))
      return(scinew JohnsonCookFlow(dynamic_cast<const 
                                       JohnsonCookFlow*>(pm)));

   else if (dynamic_cast<const ZAFlow*>(pm))
      return(scinew ZAFlow(dynamic_cast<const ZAFlow*>(pm)));
      
   else if (dynamic_cast<const ZAPolymerFlow*>(pm))
      return(scinew ZAPolymerFlow(dynamic_cast<const ZAPolymerFlow*>(pm)));

   else if (dynamic_cast<const MTSFlow*>(pm))
      return(scinew MTSFlow(dynamic_cast<const MTSFlow*>(pm)));
   
   else if (dynamic_cast<const SCGFlow*>(pm))
      return(scinew SCGFlow(dynamic_cast<const SCGFlow*>(pm)));

   else if (dynamic_cast<const PTWFlow*>(pm))
      return(scinew PTWFlow(dynamic_cast<const PTWFlow*>(pm)));
   
   else {
      //cerr << "**WARNING** Creating copy of default isotropic hardening flow model" << endl;
      //return(scinew IsoHardeningFlow(dynamic_cast<const 
      //                                  IsoHardeningFlow*>(pm)));
      throw ProblemSetupException("Cannot create copy of unknown flow model", __FILE__, __LINE__);
   }
}

