/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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


#include <CCA/Components/ICE/WallShearStressModel/WallShearStressFactory.h>
#include <CCA/Components/ICE/WallShearStressModel/logLawModel.h>
#include <CCA/Components/ICE/WallShearStressModel/smoothWall.h>


#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;
using namespace std;

WallShearStressFactory::WallShearStressFactory()
{
}

WallShearStressFactory::~WallShearStressFactory()
{
}
//______________________________________________________________________
//
WallShearStress* WallShearStressFactory::create( ProblemSpecP& ps, 
                                                 MaterialManagerP& materialManager )
{
  ProblemSpecP wss_ps = ps->findBlock("wallShearStress");
  
  if(wss_ps){
    std::string WSS_model;
    
    if(!wss_ps->getAttribute("model",WSS_model)){
      throw ProblemSetupException("No model specified for WallShearStress", __FILE__, __LINE__); 
    }
    
    if (WSS_model == "logLawModel"){
      return( scinew logLawModel( wss_ps, materialManager ) );
    }
    else if (WSS_model == "smoothWall"){
      return( scinew smoothwall( wss_ps, materialManager) );
    }
else{
      ostringstream warn;
      warn << "ERROR ICE: Unknown WallShearStress model ("<< WSS_model << " )\n"
         << "Valid models are:\n" 
         << "Newtonian\n"<< endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
  }
  return 0;
}
