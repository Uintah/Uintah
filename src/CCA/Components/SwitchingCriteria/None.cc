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
 */

#include <CCA/Components/SwitchingCriteria/None.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Ports/Scheduler.h>


using namespace std;
using namespace Uintah;
static DebugStream dbg("SWITCHER", false);

None::None()
{
}

None::~None()
{
}

void None::problemSetup(const ProblemSpecP& ps, 
                        const ProblemSpecP& restart_prob_spec, 
                        SimulationStateP& state)
{
  d_sharedState = state;
}

void None::scheduleSwitchTest(const LevelP& level, SchedulerP& sched)
{
  printSchedule(level,dbg,"Switching Criteria:None::scheduleSwitchTest");
  
  Task* t = scinew Task("switchTest", this, &None::switchTest);

  t->computes(d_sharedState->get_switch_label());
  sched->addTask(t, level->eachPatch(),d_sharedState->allMaterials());
}


void None::switchTest(const ProcessorGroup* group,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw)
{
  dbg << "  Doing Switching Criteria:None::switchTest" <<  endl;
  double sw = 0;
  max_vartype switch_condition(sw);
  const Level* allLevels = 0;
  new_dw->put(switch_condition,d_sharedState->get_switch_label(),allLevels);
}
