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

#include <CCA/Components/ICE/SpecificHeatModel/SpecificHeatFactory.h>
#include <CCA/Components/ICE/SpecificHeatModel/SpecificHeat.h>
#include <CCA/Components/ICE/SpecificHeatModel/Debye.h>
#include <CCA/Components/ICE/SpecificHeatModel/Component.h>
#include <CCA/Components/ICE/SpecificHeatModel/Polynomial.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <sstream>
#include <iostream>
#include <fstream>

using namespace Uintah;
using namespace std;

SpecificHeatFactory::SpecificHeatFactory()
{
}

SpecificHeatFactory::~SpecificHeatFactory()
{
}

SpecificHeat* SpecificHeatFactory::create(ProblemSpecP& ps)
{
  ProblemSpecP cv_ps = ps->findBlock("SpecificHeatModel");

  if(cv_ps){
    std::string cv_model;
    if(!cv_ps->getAttribute("type",cv_model)){
      throw ProblemSetupException("No model for specific_heat", __FILE__, __LINE__);
    }
    if (cv_model == "Debye"){
      return(scinew DebyeCv(cv_ps));
    }else if (cv_model == "Component"){
      return(scinew ComponentCv(cv_ps));
    }else if (cv_model == "Polynomial"){
      return(scinew PolynomialCv(cv_ps));
    }else{
      ostringstream warn;
      warn << "ERROR ICE: Unknown specific heat model ("<< cv_model << " )\n"
         << "Valid models are:\n"
         << " Debye\n"
         << " Component\n"
         << " Polynomial\n" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
  }
  return 0;
}
        
