/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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

#include <CCA/Components/ICE/ViscosityModel/ViscosityFactory.h>
#include <CCA/Components/ICE/ViscosityModel/Viscosity.h>
#include <CCA/Components/ICE/ViscosityModel/Sutherland.h>
#include <CCA/Components/ICE/ViscosityModel/SpongeLayer.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>


using namespace Uintah;
using namespace std;

ViscosityFactory::ViscosityFactory()
{
}

ViscosityFactory::~ViscosityFactory()
{
}
//______________________________________________________________________
//
std::vector<Viscosity*> ViscosityFactory::create( ProblemSpecP & ps,
                                                  const GridP  & grid )
{
  ProblemSpecP dvm_ps = ps->findBlock("dynamicViscosityModels");

  std::vector<Viscosity*> models;

  if(dvm_ps){

    for( ProblemSpecP   model_ps = dvm_ps->findBlock( "Model" );
                        model_ps != nullptr;
                        model_ps = model_ps->findNextBlock( "Model" ) ) {

      std::string vm_model;
      if(!model_ps->getAttribute("name",vm_model)){
        throw ProblemSetupException("No model for dynamic viscosity", __FILE__, __LINE__);
      }

      proc0cout << "Creating dynamic viscosity model (" << vm_model << ")"<< endl;

      if (vm_model == "Sutherland"){
        models.push_back ( scinew Sutherland(model_ps) );
      }
      else if ( vm_model == "SpongeLayer" ) {
        models.push_back ( scinew SpongeLayer( model_ps, grid) );
      }

      else{
        ostringstream warn;
        warn << "ERROR ICE: Unknown viscosity model ("<< vm_model << " )\n"
           << "Valid models are:\n"
           << " Sutherland\n"
           << " SpongeLayer\n" << endl;
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
    }
    
    //__________________________________
    //  Bulletproofing  
    //    The order in which the models are listed matters.
    bool ans = Viscosity::inCorrectOrder( models );
    
    if( !ans ){
      ostringstream warn;
      warn << "ERROR ICE: The dynamicViscosity models are not specified in the correct order. \n"
           << "           The SpongeLayer model must be specified last\n";
      
      for( auto iter  = models.begin(); iter != models.end(); iter++){
        Viscosity* viscModel = *iter;
        warn << "    " << viscModel->getName() << "\n";
      }

      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    
  }

  return models;
}

