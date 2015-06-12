/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#include "ElasticityModelFactory.h"                                             
#include "IsotropicLinearElastic.h"
#include "AnisotropicLinearElastic.h"
#include "NeoHookean.h"
#include "MooneyRivlin.h"
#include "GentHyperelastic.h"
#include "BorjaHyperelastic.h"
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <fstream>
#include <iostream>
#include <string>
using std::cerr;
using std::ifstream;
using std::ofstream;

using namespace Uintah;

ElasticityModel* ElasticityModelFactory::create(ProblemSpecP& ps)
{
   ProblemSpecP child = ps->findBlock("elasticity_model");
   if(!child)
      throw ProblemSetupException("Cannot find <elasticity_model type=model_name> tag", __FILE__, __LINE__);
   string mat_type;
   if(!child->getAttribute("type", mat_type))
      throw ProblemSetupException("No type specified for <elasticity_model type=?>", __FILE__, __LINE__);
   if (mat_type == "isotropic_linear_elastic")
      return(new IsotropicLinearElastic(child));
   else if (mat_type == "anisotropic_linear_elastic")
      return(new AnisotropicLinearElastic(child));
   else if (mat_type == "neo_hookean")
      return(new NeoHookean(child));
   else if (mat_type == "mooney_rivlin")
      return(new MooneyRivlin(child));
   else if (mat_type == "gent_hyperelastic")
      return(new GentHyperelastic(child));
   else if (mat_type == "borja_hyperelastic")
      return(new BorjaHyperelastic(child));
   else {
      throw ProblemSetupException("Unknown Elasticity Model ("+mat_type+")", __FILE__, __LINE__);
   }
}

ElasticityModel* 
ElasticityModelFactory::createCopy(const ElasticityModel* pm)
{
   if (dynamic_cast<const IsotropicLinearElastic*>(pm))
      return(new IsotropicLinearElastic(dynamic_cast<const IsotropicLinearElastic*>(pm)));

   else if (dynamic_cast<const AnisotropicLinearElastic*>(pm))
      return(new AnisotropicLinearElastic(dynamic_cast<const 
                                       AnisotropicLinearElastic*>(pm)));

   else if (dynamic_cast<const NeoHookean*>(pm))
      return(new NeoHookean(dynamic_cast<const NeoHookean*>(pm)));
      
   else if (dynamic_cast<const MooneyRivlin*>(pm))
      return(new MooneyRivlin(dynamic_cast<const MooneyRivlin*>(pm)));

   else if (dynamic_cast<const GentHyperelastic*>(pm))
      return(new GentHyperelastic(dynamic_cast<const GentHyperelastic*>(pm)));

   else if (dynamic_cast<const BorjaHyperelastic*>(pm))
      return(new BorjaHyperelastic(dynamic_cast<const BorjaHyperelastic*>(pm)));

   else {
      throw ProblemSetupException("Cannot create copy of unknown elasticity model", __FILE__, __LINE__);
   }
}

