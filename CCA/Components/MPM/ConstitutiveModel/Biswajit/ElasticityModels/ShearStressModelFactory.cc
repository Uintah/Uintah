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

#include "ShearStressModelFactory.h"
#include "LinearElasticShear.h"
#include "NeoHookean.h"
#include "MooneyRivlin.h"
#include "GentHyperelastic.h"
#include "BorjaHyperelasticShear.h"
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <string>
#include <iostream>

using namespace std;
using namespace Uintah;

ShearStressModel* ShearStressModelFactory::create(ProblemSpecP& ps)
{
   ProblemSpecP child = ps->findBlock("shear_stress_model");
   if(!child) {
      ostringstream msg;
      msg << "No <shear_stress_model> tag in input file." << endl;
      throw ProblemSetupException(msg.str(), _FILE__, __LINE__);
   }
   string model_type;
   if(!child->getAttribute("type", model_type)) {
      ostringstream msg;
      msg << "No type has been specified for <shear_stress_model type=?> in input file." << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
   }
   
   if (model_type == "linear_elastic")
      return(scinew LinearElasticShear(child));
   else if (model_type == "neo_hookean")
      return(scinew NeoHookean(child));
   else if (model_type == "mooney_rivlin")
      return(scinew MooneyRivlin(child));
   else if (model_type == "gent")
      return(scinew GentHyperelastic(child));
   else if (model_type == "borja")
      return(scinew BorjaHyperelasticShear(child));
   else {
      ostringstream msg;
      msg << "Unknown type in <shear_stress_model type=" << model_type << "> in input file." << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
   }
}

ShearStressModel* 
ShearStressModelFactory::createCopy(const ShearStressModel* smm)
{
   if (dynamic_cast<const LinearElasticShear*>(smm))
      return(scinew LinearElasticShear(dynamic_cast<const LinearElasticShear*>(smm)));
   else if (dynamic_cast<const NeoHookean*>(smm))
      return(scinew NeoHookean(dynamic_cast<const MTSShear*>(smm)));
   else if (dynamic_cast<const MooneyRivlin*>(smm))
      return(scinew MooneyRivlin(dynamic_cast<const MooneyRivlin*>(smm)));
   else if (dynamic_cast<const GentHyperelastic*>(smm))
      return(scinew GentHyperelastic(dynamic_cast<const GentHyperelastic*>(smm)));
   else if (dynamic_cast<const BorjaHyperelasticShear*>(smm))
      return(scinew BorjaHyperelasticShear(dynamic_cast<const BorjaHyperelasticShear*>(smm)));
   else {
      ostringstream msg;
      msg << "The type in <shear_stress_model type=" << model_type << "> does not exist." << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
   }
}
