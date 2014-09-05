
#include <Packages/Uintah/CCA/Components/Models/ModelFactory.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/CCA/Ports/ModelInterface.h>
#include <Packages/Uintah/CCA/Components/Models/test/Mixing.h>
#include <Packages/Uintah/CCA/Components/Models/test/AdiabaticTable.h>
#include <Packages/Uintah/CCA/Components/Models/test/PassiveScalar.h>
#include <Packages/Uintah/CCA/Components/Models/test/SimpleRxn.h>
#include <Packages/Uintah/CCA/Components/Models/test/TestModel.h>
#include <Packages/Uintah/CCA/Components/Models/test/flameSheet_rxn.h>
#include <Packages/Uintah/CCA/Components/Models/test/VorticityConfinement.h>
#include <Packages/Uintah/CCA/Components/Models/HEChem/Simple_Burn.h>
#include <Packages/Uintah/CCA/Components/Models/HEChem/Steady_Burn.h>
#include <Packages/Uintah/CCA/Components/Models/HEChem/Unsteady_Burn.h>
#include <Packages/Uintah/CCA/Components/Models/HEChem/IandG.h>
#include <Packages/Uintah/CCA/Components/Models/HEChem/JWLpp.h>
#include <Packages/Uintah/CCA/Components/Models/HEChem/LightTime.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/RadiationDriver.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using namespace Uintah;


ModelFactory::ModelFactory(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
}

ModelFactory::~ModelFactory()
{
  for (std::vector<ModelInterface*>::const_iterator it = d_models.begin();
       it != d_models.end(); it++) 
    delete *it;

}

std::vector<ModelInterface*> ModelFactory::getModels()
{
  return d_models;
}

void ModelFactory::makeModels(const ProblemSpecP& restart_prob_spec,
                              const ProblemSpecP& prob_spec,
                              GridP&,
                              SimulationStateP&,
                              const bool doAMR)
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
    
    if(type == "SimpleRxn")
      d_models.push_back(scinew SimpleRxn(d_myworld, model_ps));
    else if(type == "AdiabaticTable")
      d_models.push_back(scinew AdiabaticTable(d_myworld, model_ps,doAMR));
    else if(type == "Test")
      d_models.push_back(scinew TestModel(d_myworld, model_ps));
    else if(type == "Mixing")
      d_models.push_back(scinew Mixing(d_myworld, model_ps));
    else if(type == "Simple_Burn")
      d_models.push_back(scinew Simple_Burn(d_myworld, model_ps, prob_spec));
    else if(type == "Steady_Burn")
      d_models.push_back(scinew Steady_Burn(d_myworld, model_ps, prob_spec));
    else if(type == "Unsteady_Burn")
      d_models.push_back(scinew Unsteady_Burn(d_myworld, model_ps, prob_spec));
    else if(type == "IandG")
      d_models.push_back(scinew IandG(d_myworld, model_ps));
    else if(type == "JWLpp")
      d_models.push_back(scinew JWLpp(d_myworld, model_ps));
    else if(type == "LightTime")
      d_models.push_back(scinew LightTime(d_myworld, model_ps));
    else if(type == "flameSheet_rxn")
      d_models.push_back(scinew flameSheet_rxn(d_myworld, model_ps));
    else if(type == "PassiveScalar")
      d_models.push_back(scinew PassiveScalar(d_myworld, model_ps, doAMR));
    else if(type == "VorticityConfinement")
      d_models.push_back(scinew VorticityConfinement(d_myworld, model_ps));
    else if(type == "Radiation")
      d_models.push_back(scinew RadiationDriver(d_myworld, model_ps));
    else
      throw ProblemSetupException("Unknown model: "+type, __FILE__, __LINE__);
  }
}


void ModelFactory::outputProblemSpec(ProblemSpecP& models_ps)
{
  for (std::vector<ModelInterface*>::const_iterator it = d_models.begin();
       it != d_models.end(); it++) {
    (*it)->outputProblemSpec(models_ps);
  }

}
