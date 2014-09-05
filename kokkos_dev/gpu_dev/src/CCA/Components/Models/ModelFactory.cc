/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#include <CCA/Components/Models/ModelFactory.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/SimulationState.h>
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
#  include <CCA/Components/Models/HEChem/IandG.h>
#  include <CCA/Components/Models/HEChem/JWLpp.h>
#  include <CCA/Components/Models/HEChem/DDT0.h>
#  include <CCA/Components/Models/HEChem/LightTime.h>
#  include <CCA/Components/Models/HEChem/DDT0.h>
#  include <CCA/Components/Models/HEChem/DDT1.h>
#endif

#include <CCA/Components/Models/Radiation/RadiationDriver.h>
#include <Core/Malloc/Allocator.h>
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
      d_models.push_back(scinew SimpleRxn(d_myworld, model_ps));
    else if(type == "AdiabaticTable")
      d_models.push_back(scinew AdiabaticTable(d_myworld, model_ps,doAMR));
    else if(type == "Mixing")
      d_models.push_back(scinew Mixing(d_myworld, model_ps));
    else if(type == "Test")
      d_models.push_back(scinew TestModel(d_myworld, model_ps));
    else if(type == "flameSheet_rxn")
      d_models.push_back(scinew flameSheet_rxn(d_myworld, model_ps));
    else if(type == "mass_momentum_energy_src")
      d_models.push_back(scinew MassMomEng_src(d_myworld, model_ps));
    else if(type == "PassiveScalar")
      d_models.push_back(scinew PassiveScalar(d_myworld, model_ps, doAMR));
    else if(type == "Simple_Burn")
      d_models.push_back(scinew Simple_Burn(d_myworld, model_ps, prob_spec));
    else if(type == "Steady_Burn")
      d_models.push_back(scinew Steady_Burn(d_myworld, model_ps, prob_spec));
    else if(type == "Unsteady_Burn")
      d_models.push_back(scinew Unsteady_Burn(d_myworld, model_ps, prob_spec));
    else if(type == "DDT0")
      d_models.push_back(scinew DDT0(d_myworld, model_ps, prob_spec));
    else if(type == "IandG")
      d_models.push_back(scinew IandG(d_myworld, model_ps));
    else if(type == "JWLpp")
      d_models.push_back(scinew JWLpp(d_myworld, model_ps, prob_spec));
    else if(type == "LightTime")
      d_models.push_back(scinew LightTime(d_myworld, model_ps));
    else if(type == "DDT0")
      d_models.push_back(scinew DDT0(d_myworld, model_ps, prob_spec));
    else if(type == "DDT1")
      d_models.push_back(scinew DDT1(d_myworld, model_ps, prob_spec));
    else if(type == "Radiation")
#  if !defined( NO_FORTRAN )
      d_models.push_back(scinew RadiationDriver(d_myworld, model_ps));
#  else
      throw ProblemSetupException("Radiation not supported in this build", __FILE__, __LINE__);
#  endif
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
