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

#include "PressureModelFactory.h"
#include "LinearElasticPressure.h"
#include "DefaultHyperelasticPressure.h"
#include "BorjaHyperelasticPressure.h"
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <string>
#include <iostream>

using namespace std;
using namespace Uintah;

PressureModel* PressureModelFactory::create(ProblemSpecP& ps)
{
   ProblemSpecP child = ps->findBlock("pressure_model");
   if(!child) {
      ostringstream msg;
      msg << "No <pressure_model> tag in input file." << endl;
      throw ProblemSetupException(msg.str(), _FILE__, __LINE__);
   }
   string model_type;
   if(!child->getAttribute("type", model_type)) {
      ostringstream msg;
      msg << "No type has been specified for <pressure_model type=?> in input file." << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
   }
   
   if (model_type == "linear_elastic")
      return(scinew LinearElasticPressure(child));
   else if (model_type == "hyperelastic")
      return(scinew DefaultHyperelasticPressure(child));
   else if (model_type == "borja")
      return(scinew BorjaHyperelasticPressure(child));
   else {
      ostringstream msg;
      msg << "Unknown type in <pressure_model type=" << model_type << "> in input file." << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
   }
}

PressureModel* 
PressureModelFactory::createCopy(const PressureModel* smm)
{
   if (dynamic_cast<const LinearElasticPressure*>(smm))
      return(scinew LinearElasticPressure(dynamic_cast<const LinearElasticPressure*>(smm)));
   else if (dynamic_cast<const DefaultHyperelasticPressure*>(smm))
      return(scinew DefaultHyperelasticPressure(dynamic_cast<const DefaultHyperelasticPressure*>(smm)));
   else if (dynamic_cast<const BorjaHyperelasticPressure*>(smm))
      return(scinew BorjaHyperelasticPressure(dynamic_cast<const BorjaHyperelasticPressure*>(smm)));
   else {
      ostringstream msg;
      msg << "The type in <pressure_model type=" << model_type << "> does not exist." << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
   }
}
