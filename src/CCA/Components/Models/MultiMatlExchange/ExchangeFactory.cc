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

#include <CCA/Components/Models/MultiMatlExchange/ExchangeFactory.h>
#include <CCA/Components/Models/MultiMatlExchange/Scalar.h>
#include <CCA/Components/Models/MultiMatlExchange/Slip.h>
#include <Core/Exceptions/ProblemSetupException.h>


using namespace std;
using namespace Uintah;
using namespace ExchangeModels;

ExchangeFactory::ExchangeFactory()
{
}

ExchangeFactory::~ExchangeFactory()
{
}

//______________________________________________________________________
//
ExchangeModel*
ExchangeFactory::create(const ProblemSpecP     & matl_ps,
                        const MaterialManagerP & materialManager,
                        const bool with_mpm)
{
  int numMatls = materialManager->getNumMatls();
  
  //__________________________________
  //    single matl 
  if( numMatls == 1){
    return ( scinew ExchangeModels::ScalarExch( matl_ps, materialManager, with_mpm) );
  }
  
  ProblemSpecP exchg_ps = matl_ps->findBlock("exchange_properties");
  ProblemSpecP model_ps = exchg_ps->findBlock( "Model" );

  //__________________________________
  //    default model
  if( model_ps == nullptr ) {
    return ( scinew ExchangeModels::ScalarExch( matl_ps, materialManager, with_mpm ));
  }
  
  //__________________________________
  //    Other models
  map<string,string> attributes;
  model_ps->getAttributes(attributes);
  std::string model = attributes["type"];

  if ( model == "slip" ) {
    return ( scinew ExchangeModels::SlipExch( exchg_ps, materialManager, with_mpm ));
  }      
  else {
    throw ProblemSetupException("\nERROR: Unknown exchange model.  "+model,__FILE__, __LINE__);
  }

  return nullptr;
}
