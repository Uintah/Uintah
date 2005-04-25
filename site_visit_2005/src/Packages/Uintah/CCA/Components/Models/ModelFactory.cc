
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
#include <Packages/Uintah/CCA/Components/Models/HEChem/IandG.h>
#include <Packages/Uintah/CCA/Components/Models/HEChem/JWLpp.h>
#include <Packages/Uintah/CCA/Components/Models/HEChem/LightTime.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using namespace Uintah;
using namespace std;

ModelFactory::ModelFactory(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
}

ModelFactory::~ModelFactory()
{
}

void ModelFactory::makeModels(const ProblemSpecP& params, GridP&,
			      SimulationStateP&,
			      vector<ModelInterface*>& models)
{
  ProblemSpecP m = params->findBlock("Models");
  if(!m)
    return;
  for(ProblemSpecP model = m->findBlock("Model"); model != 0;
      model = model->findNextBlock("Model")){
    string type;
    if(!model->getAttribute("type", type)){
      throw ProblemSetupException("Model does not specify type=\"name\"");
    }
    
    if(type == "SimpleRxn")
      models.push_back(scinew SimpleRxn(d_myworld, model));
    else if(type == "AdiabaticTable")
      models.push_back(scinew AdiabaticTable(d_myworld, model));
    else if(type == "Test")
      models.push_back(scinew TestModel(d_myworld, model));
    else if(type == "Mixing")
      models.push_back(scinew Mixing(d_myworld, model));
    else if(type == "Simple_Burn")
      models.push_back(scinew Simple_Burn(d_myworld, model));
    else if(type == "Steady_Burn")
      models.push_back(scinew Steady_Burn(d_myworld, model));
    else if(type == "IandG")
      models.push_back(scinew IandG(d_myworld, model));
    else if(type == "JWLpp")
      models.push_back(scinew JWLpp(d_myworld, model));
    else if(type == "LightTime")
      models.push_back(scinew LightTime(d_myworld, model));
    else if(type == "flameSheet_rxn")
      models.push_back(scinew flameSheet_rxn(d_myworld, model));
    else if(type == "PassiveScalar")
      models.push_back(scinew PassiveScalar(d_myworld, model));
    else if(type == "VorticityConfinement")
      models.push_back(scinew VorticityConfinement(d_myworld, model));
    else
      throw ProblemSetupException("Unknown model: "+type);
  }
}
