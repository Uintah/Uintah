/*
 *
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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
 * ----------------------------------------------------------
 * HeirarchicalIndirectSiwtcher.cc
 *
 *  Created on: Nov 17, 2015
 *      Author: jbhooper
 */

#include <CCA/Components/Parent/HierarchicalIndirectSwitcher.h>

#include <sstream>

using namespace Uintah;

static DebugStream his_debug("HIERARCHICAL_INDIRECT_SWITCHER",false);

HierarchicalIndirectSwitcher::HierarchicalIndirectSwitcher(  const ProcessorGroup   * myWorld
                                                           , const ProblemSpecP     & switcher_ps
                                                           ,       bool               doAMR
                                                           , const std::string      & udaName
                                                          )
  :UintahParallelComponent(myWorld)
{
  if (his_debug.active()) {
    proc0cout << "HierarchicalIndirectSwitcher::HeirarchicalIndirectSwitcher" << std::endl;
  }

  ProblemSpecP switcherBlock = switcher_ps->findBlock("SimulationComponent");
  ProblemSpecP headBlock     = switcherBlock->findBlock("head");
  if (!headBlock) {
    std::ostringstream errorMsg;
    errorMsg << "ERROR: Hierarchical Indirect Switcher must specify a component to act as the head "
             << " component (Block: <head>)." << std::endl;
    throw ProblemSetupException(errorMsg.str(), __FILE__, __LINE__);
  }
  else {
    std::string headInput;
    if (!headBlock->get("input_file", headInput)) {
      throw ProblemSetupException("The <head> component must specify an input file.",
                                  __FILE__, __LINE__);
    }

    // Parse head-component's input file.
    ProblemSpecP head_ps = ProblemSpecReader().readInputFile(headInput);
    ProblemSpecP headSim_ps = head_ps->findBlock("SimulationComponent");
    headSim_ps->getAttribute("type",d_headComponentName);

    d_headComponent = ComponentFactory::create(head_ps, myWorld, doAMR,"");
    d_headComponentInterface = dynamic_cast<SimulationInterface*> (d_headComponent);

    SolverInterface* solver = SolverFactory::create(head_ps, myWorld);

    d_headComponent->attachPort("solver", solver);
    SwitchingCriteria* headSwitchCriteria = SwitchingCriteriaFactory::create(headBlock, myWorld);

    if (!headSwitchCriteria) {
      std::ostringstream warningMsg;
      warningMsg << "WARNING:  Hierarchical Indirect Switcher :: No switch criteria is defined for"
                 << " the head subcomponent." << std::endl
                 << "\tThe head component will not switch out with anything." << std::endl;
      proc0cerr << warningMsg.str() << std::endl;
    }
    else { // We have a switcher for the head component.
    // Make link between head component and this switcher
      attachPort("head_switcher", headSwitchCriteria);
      d_headComponent->attachPort("switch_criteria",headSwitchCriteria);
    }

    // FIXME TODO JBH - 11-19-2015 For future use
    // attach interface between head and switcher here (only between head and switcher because this
    // is an 'indirect' switcher).

  }


  ProblemSpecP subComponentBlock = switcherBlock->findBlock("subcomponent");
  if (!subComponentBlock) {
    std::ostringstream errorMsg;
    errorMsg << "ERROR: Hierarchical Indirect Switcher must specify one or more components to act "
             << " as the subgrid components. (Block: <subcomponent>)." << std::endl;
    throw ProblemSetupException(errorMsg.str(), __FILE__, __LINE__);
    d_numSubComponents = 0;
    while(subComponentBlock) {
      std::string subInput("");
      if (!subComponentBlock->get("input_file",subInput)) {
        std::ostringstream subErrorMsg;
        subErrorMsg << "Subcomponent number " << std::left << d_numSubComponents << " has no input "
                    << " file specified.  All subcomponents must specify an input file."
                    << std::endl;
        throw ProblemSetupException(subErrorMsg, __FILE__, __LINE__);
      }


      // Parse head-component's input file.
      ProblemSpecP sub_ps = ProblemSpecReader().readInputFile(subInput);
      ProblemSpecP subSim_ps = sub_ps->findBlock("SimulationComponent");
      std::string subComponentName;
      subSim_ps->getAttribute("type",subComponentName);
      std::string subComponentLabel = "";
      subSim_ps->getAttribute("label",subComponentLabel);
      subComponentKey subKey;
      if (subComponentLabel != "") {
        subKey = subComponentKey (subComponentName,subComponentLabel);
      }
      else {
        std::ostringstream index;
        index << std::left << d_numSubComponents;
        subKey = subComponentKey(subComponentName,index.str());
      }
      d_subComponentIndexMap.insert(std::pair<subComponentKey, int> (subKey,d_numSubComponents));
      d_subComponentNames.push_back(subComponentName);
      d_subComponentLabels.push_back(subComponentLabel);

      UintahParallelComponent* subComponent = ComponentFactory::create(sub_ps, myWorld, doAMR, "");
      d_subComponents.push_back(subComponent);
      d_subComponentInterfaces.push_back(dynamic_cast<SimulationInterface*> (d_subComponents[d_numSubComponents]));

      SolverInterface* solver = SolverFactory::create(sub_ps, myWorld);
      d_subComponents[d_numSubComponents]->attachPort("solver", solver);

      SwitchingCriteria* subSwitchCriteria = SwitchingCriteriaFactory::create(subComponentBlock, myWorld);
      if (!subSwitchCriteria) {
        std::ostringstream errorMsg;
        errorMsg << "ERROR:  Hierarchical Indirect Switcher :: No switch criteria is defined for"
                 << " subcomponent number " << std::left << d_numSubComponents << std::endl
                 << "\tIt must be possible to switch out of an attached subcomponent!" << std::endl;
        throw ProblemSetupException(errorMsg.str(), __FILE__, __LINE__);
      }
      else { // We have a switcher for the subcomponent.
      // Make link between subcomponent and this switcher
        attachPort("sub_switcher", subSwitchCriteria);
        d_subComponents[d_numSubComponents]->attachPort("switch_criteria",subSwitchCriteria);
      }

      // FIXME TODO JBH - 11-19-2015 For future use
      // attach interface between head and switcher here (only between head and switcher because this
      // is an 'indirect' switcher).


      ++d_numSubComponents;
      subComponentBlock = subComponentBlock->findBlock("subcomponent");
    }
  }

}

