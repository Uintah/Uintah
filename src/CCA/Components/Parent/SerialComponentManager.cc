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
 *
 *
 * SerialComponentManager.cc
 *
 *  Created on: Dec 2, 2015
 *      Author: j.b.hooper@utah.edu
 */

#include <CCA/Components/Parent/SerialComponentManager.h>

using namespace Uintah;
static DebugStream serialManagerDbg("SERIAL_COMPONENT_MANAGER", false);

SerialComponentManager::SerialComponentManager(
                                                const ProcessorGroup * myWorld
                                              ,       ProblemSpecP   & managerPS
                                              ,       bool             doAMR
                                              , const std::string    & udaName
                                              )
  : UintahParallelComponent(myWorld)
  , d_numPrincipalComponents(0)
  , d_numSubcomponents(0)
  , d_restarting(false)
  , d_totalWorld(myWorld)
{

  if (serialManagerDbg.active()) {
    serialManagerDbg << "Constructing SerialComponentManager object.\n"
  }
  ProblemSpecP sim_block = managerPS->findBlock("SimulationCompnent");
  ProblemSpecP principalSpec = sim_block->findBlock("principal");

  // Parse principal component...  we must have at least one!
  if (!principalSpec) {
    throw ProblemSetupException("ERROR:  Component manager cannot find a principal component to run.", __FILE__, __LINE__);
  }
  ProblemSpecP principalComponent = principalSpec->findBlock("Component");
  while (principalComponent) {

    // Parse component type...
    std::string componentType;
    if (!principalComponent->getAttribute("type",componentType)) {
      throw ProblemSetupException("ERROR:  Principal component type not specified!", __FILE__, __LINE__);
    }
    // .. reference label ...
    std::string label;
    if (!principalComponent->get("label", label)) {
      throw ProblemSetupException("ERROR:  Every principal component must be labeled!", __FILE__, __LINE__);
    }
    // ... and input file.
    std::string inputFile;
    if (!principalComponent->get("inputFile", inputFile)) {
      throw ProblemSetupException("ERROR:  A principal component did not have a specified input file!", __FILE__, __LINE__);
    }

    ProblemSpecP componentPS = ProblemSpecReader().readInputFile(inputFile);

    std::string inputFileComponentType;
    ProblemSpecP componentSimPS = componentPS->findBlock("SimulationComponent");
    componentSimPS->getAttribute("type",inputFileComponentType);

    if (componentType != inputFileComponentType) {
      std::ostringstream msg;
      msg << "ERROR: Component manager expected a component of type \"" << componentType
          << "\" but parsed one of type \"" << inputFileComponentType << "\" instead!\n";
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
    }

    // We probably should instantiate components elsewhere, but just to get up and going...
    bool componentIsAMR = Grid::specIsAMR(componentPS);
    UintahParallelComponent *comp           = ComponentFactory::create(componentPS, myWorld, componentIsAMR, "");
    SolverInterface         *solver         = SolverFactory::create(componentPS, myWorld);
    SwitchingCriteria       *switchCriteria = SwitchingCriteriaFactory::create(principalComponent, myWorld);
    comp->attachPort("solver", solver);
    if (switchCriteria) {
      comp->attachPort("switchCriteria", switchCriteria);
    }

    d_principalComponentArray.push_back(comp);
    // TODO If we're going to map things, we should do it here.  This is also where interfaces should be attached.
    d_PSArray.push_back(componentPS); // Push problemSpec into array so we can load the level sets from the original files.


    // Look for another principal component.
    principalComponent = principalComponent->findNextBlock("Component");
  }

  proc0cout << "Parsed " << d_principalComponentArray.size() << " principal simulation components." << std::endl;
  // Now parse subcomponents
  ProblemSpecP subgridSpec = sim_block->findBlock("subcomponent");
  if (!subgridSpec) {
    proc0cout << "WARNING:  No subcomponents have been found!" << std::endl;
  }
  else {
    ProblemSpecP subComponent = subgridSpec->findBlock("Component");
    while (subComponent) {

      std::string componentType;
      if (!subComponent->getAttribute("type",componentType)) {
        throw ProblemSetupException("ERROR:  Subcomponent specified but not type specified!", __FILE__, __LINE__);
      }
      std::string label;
      if (!subComponent->get("label",label)) {
        throw ProblemSetupException("ERROR:  Label value not set for subcomponent!", __FILE__, __LINE__);
      }
      std::string inputFile;
      if (!subComponent->get("inputFile", inputFile)) {
        throw ProblemSetupException("ERROR:  Subcomponent input file not specified.", __FILE__, __LINE__);
      }

      ProblemSpecP componentPS = ProblemSpecReader().readInputFile(inputFile);

      std::string inputFileComponentType;
      ProblemSpecP componentSimPS = componentPS->findBlock("SimulationComponent");
      componentSimPS->getAttribute("type",inputFileComponentType);

      if (componentType != inputFileComponentType) {
        std::ostringstream msg;
        msg << "ERROR: Component manager expected a component of type \"" << componentType
            << "\" but parsed one of type \"" << inputFileComponentType << "\" instead!\n";
        throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
      }

      bool componentIsAMR = Grid::specIsAMR(componentPS);
      UintahParallelComponent *comp           = ComponentFactory::create(componentPS, myWorld, componentIsAMR, "");
      SolverInterface         *solver         = SolverFactory::create(componentPS, myWorld);
      SwitchingCriteria       *switchCriteria = SwitchingCriteriaFactory::create(subComponent, myWorld);
      comp->attachPort("solver",solver);
      if (switchCriteria) {
        comp->attachPort("switchCriteria", switchCriteria);
      }

      d_subcomponentArray.push_back(comp);
      // TODO If we're going to map things we should do it here.  This is also where interfaces should be attached.
      d_PSArray.push_back(componentPS);

      subComponent = subComponent->findNextBlock("Component");
    }
  } // End subcomponent else
  proc0cout << "Parsed " << d_subcomponentArray.size() << " subgrid simulation components." << std::endl;
} // end SerialComponentManager()

SerialComponentManager::PostGridProblemSetup()
{
  // Delete problem specs we don't need here
}

void SerialComponentManager::problemSetup(
                                            const ProblemSpecP       & managerPS
                                          , const ProblemSpecP       & managerRestartPS
                                          ,       GridP              & grid
                                          ,       SimulationStateP   & managerState
                                         )
{
  if (serialManagerDbg.active()) {
    serialManagerDbg << "Doing SerialComponentManager::problemSetup() ." << std::endl;
  }

  if (managerRestartPS) {
    throw ProblemSetupException("ERROR:  Restarting not currently supported for the SerialComponentManager.", __FILE__, __LINE__);
  }

  d_managerState = managerState;

}

