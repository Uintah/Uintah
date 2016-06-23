/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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


#include <CCA/Components/Models/ModelFactory.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Ports/ModelInterface.h>
#include <CCA/Components/Models/FluidsBased/Mixing.h>
#include <CCA/Components/Models/FluidsBased/AdiabaticTable.h>
#include <CCA/Components/Models/FluidsBased/PassiveScalar.h>
#include <CCA/Components/Models/FluidsBased/SimpleRxn.h>
#include <CCA/Components/Models/FluidsBased/TestModel.h>
#include <CCA/Components/Models/FluidsBased/flameSheet_rxn.h>
#include <CCA/Components/Models/FluidsBased/MassMomEng_src.h>
#if !defined( NO_ICE )
#  include <CCA/Components/Models/HEChem/Simple_Burn.h>
#  include <CCA/Components/Models/HEChem/Steady_Burn.h>
#  include <CCA/Components/Models/HEChem/Unsteady_Burn.h>
#  include <CCA/Components/Models/HEChem/MesoBurn.h>
#  include <CCA/Components/Models/HEChem/IandG.h>
#  include <CCA/Components/Models/HEChem/JWLpp.h>
#  include <CCA/Components/Models/HEChem/ZeroOrder.h>
#  include <CCA/Components/Models/HEChem/LightTime.h>
#  include <CCA/Components/Models/HEChem/DDT0.h>
#  include <CCA/Components/Models/HEChem/DDT1.h>
#  include <CCA/Components/Models/SolidReactionModel/SolidReactionModel.h>
#endif

#include <sci_defs/uintah_defs.h>

#include <iostream>

using namespace Uintah;
using namespace std;

ModelFactory::ModelFactory(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
}

ModelFactory::~ModelFactory()
{
  for (vector<ModelInterface*>::const_iterator it = d_models.begin();
       it != d_models.end(); it++) {
    delete *it;
  }
}

vector<ModelInterface*> ModelFactory::getModels()
{
  //cout << "calling getModels: " << d_models.size() << " models returned" << endl;
  return d_models;
}

void ModelFactory::clearModels()
{
  //cout << "clean up " << d_models.size() << " models" << endl;
  d_models.clear();
}

void
ModelFactory::makeModels( const ProblemSpecP& restart_prob_spec,
                          const ProblemSpecP& prob_spec,
                          GridP&,
                          SimulationStateP&,
                          const bool doAMR )
{
  ProblemSpecP m = restart_prob_spec->findBlock("Models");
  if(!m)
    return;
  for(ProblemSpecP model_ps = m->findBlock("Model"); model_ps != 0;
      model_ps = model_ps->findNextBlock("Model")){
    string type;
    if(!model_ps->getAttribute("type", type)){
      throw ProblemSetupException("Model does not specify type=\"name\"", __FILE__, __LINE__);
    }
    
#if !defined( NO_ICE ) && !defined( NO_MPM )
    // ICE and MPM turned on
    if(type == "SimpleRxn")
      d_models.push_back(new SimpleRxn(d_myworld, model_ps));
    else if(type == "AdiabaticTable")
      d_models.push_back(new AdiabaticTable(d_myworld, model_ps,doAMR));
    else if(type == "Mixing")
      d_models.push_back(new Mixing(d_myworld, model_ps));
    else if(type == "Test")
      d_models.push_back(new TestModel(d_myworld, model_ps));
    else if(type == "flameSheet_rxn")
      d_models.push_back(new flameSheet_rxn(d_myworld, model_ps));
    else if(type == "mass_momentum_energy_src")
      d_models.push_back(new MassMomEng_src(d_myworld, model_ps));
    else if(type == "PassiveScalar")
      d_models.push_back(new PassiveScalar(d_myworld, model_ps, doAMR));
    else if(type == "Simple_Burn")
      d_models.push_back(new Simple_Burn(d_myworld, model_ps, prob_spec));
    else if(type == "Steady_Burn")
      d_models.push_back(new Steady_Burn(d_myworld, model_ps, prob_spec));
    else if(type == "Unsteady_Burn")
      d_models.push_back(new Unsteady_Burn(d_myworld, model_ps, prob_spec));
    else if(type == "MesoBurn")
      d_models.push_back(new MesoBurn(d_myworld, model_ps, prob_spec));
    else if(type == "IandG")
      d_models.push_back(new IandG(d_myworld, model_ps));
    else if(type == "JWLpp")
      d_models.push_back(new JWLpp(d_myworld, model_ps, prob_spec));
    else if(type == "ZeroOrder")
      d_models.push_back(new ZeroOrder(d_myworld, model_ps, prob_spec));
    else if(type == "LightTime")
      d_models.push_back(new LightTime(d_myworld, model_ps));
    else if(type == "DDT0")
      d_models.push_back(new DDT0(d_myworld, model_ps, prob_spec));
    else if(type == "DDT1")
      d_models.push_back(new DDT1(d_myworld, model_ps, prob_spec));
    else if(type == "SolidReactionModel")
      d_models.push_back(new SolidReactionModel(d_myworld, model_ps, prob_spec));
    else
      throw ProblemSetupException( "Unknown model: " + type, __FILE__, __LINE__ );
#else
    // ICE and/or MPM turned off.
    throw ProblemSetupException( type + " not supported in this build", __FILE__, __LINE__ );
#endif
  }
}

void
ModelFactory::outputProblemSpec(ProblemSpecP& models_ps)
{
  for (vector<ModelInterface*>::const_iterator it = d_models.begin(); it != d_models.end(); it++) {
    (*it)->outputProblemSpec(models_ps);
  }
}
