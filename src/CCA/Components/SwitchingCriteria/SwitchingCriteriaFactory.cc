/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <CCA/Components/SwitchingCriteria/SwitchingCriteriaFactory.h>
#include <CCA/Components/SwitchingCriteria/None.h>
#include <CCA/Components/SwitchingCriteria/TimestepNumber.h>
#include <CCA/Components/SwitchingCriteria/SteadyState.h>

#include <sci_defs/uintah_defs.h>

#if !defined( NO_ICE ) && !defined( NO_MPM )
#  include <CCA/Components/SwitchingCriteria/SimpleBurn.h>
#  include <CCA/Components/SwitchingCriteria/SteadyBurn.h>
#  include <CCA/Components/SwitchingCriteria/DDT1.h>
#endif

#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <iostream>
#include <string>

using namespace Uintah;

SwitchingCriteria* SwitchingCriteriaFactory::create(ProblemSpecP& ps,
                                                    const ProcessorGroup* world)
{
  std::string criteria("");

  ProblemSpecP switch_ps = ps->findBlock("SwitchCriteria");

  if (switch_ps) {
    std::map<std::string, std::string> attributes;
    switch_ps->getAttributes(attributes);
    criteria = attributes["type"];
  }
  else {
    return nullptr;
  }

  SwitchingCriteria* switch_criteria = nullptr;
  
  if (criteria == "none" || criteria == "None" || criteria == "NONE") {
    switch_criteria = scinew None();
  }
  else if (criteria == "timestep" || criteria == "Timestep" || 
           criteria == "TIMESTEP")  {
    switch_criteria = scinew TimestepNumber(switch_ps);
  }
  else if (criteria == "SteadyState" || criteria == "steadystate")  {
    switch_criteria = scinew SteadyState(switch_ps);
  }
  
#if !defined( NO_ICE ) && !defined( NO_ICE )
  else if (criteria == "SimpleBurn" || criteria == "Simple_Burn" || 
           criteria == "simpleBurn" || criteria == "simple_Burn")  {
    switch_criteria = scinew SimpleBurnCriteria(switch_ps);
  }
  else if (criteria == "SteadyBurn" || criteria == "Steady_Burn" || 
           criteria == "steadyBurn" || criteria == "steady_Burn")  {
    switch_criteria = scinew SteadyBurnCriteria(switch_ps);
  }
  else if (criteria == "DDT1")  {
    switch_criteria = scinew DDT1Criteria(switch_ps);
  }
#endif  

  else {
    std::ostringstream msg;
    msg << "\nERROR<SwitchCriteria>: Unknown switching criteria : " << criteria << ".\n";
    throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
  }

  return switch_criteria;
}
