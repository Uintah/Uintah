/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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
#include "DevStressModelFactory.h"
#include "HypoElasticDevStress.h"
#include "HypoViscoElasticDevStress.h" 
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <string>

 using namespace std;
using namespace Uintah;

DevStressModel* DevStressModelFactory::create(ProblemSpecP& ps)
{
   ProblemSpecP dsm_ps = ps->findBlock("deviatoric_stress_model");

  if(dsm_ps){

    string type="nullptr";

    if( !dsm_ps->getAttribute("type", type) ){
      throw ProblemSetupException("No type specified for DeviatoricStress", __FILE__, __LINE__);
    }
    if (type == "hypoElastic"){
      return( scinew HypoElasticDevStress() );

    } else if (type == "hypoViscoElastic"){
      return( scinew HypoViscoElasticDevStress(dsm_ps) );

    } else {
      throw ProblemSetupException("Unknown DeviatoricStress type ("+type+")", __FILE__, __LINE__);
    }
  } else{
    return( scinew HypoElasticDevStress() );  // DEFAULT  Deviatoric Stress Model
  }
}

