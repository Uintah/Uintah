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

#include <CCA/Components/SwitchingCriteria/TimestepNumber.h>

#include <CCA/Ports/Scheduler.h>

#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;

extern DebugStream switching_dbg;

TimestepNumber::TimestepNumber(ProblemSpecP& ps)
{
  ps->require("timestep", m_timeStep);
  
  proc0cout << "Switching criteria: \tTimestep Number: switch components on timestep " << m_timeStep << std::endl;

  // Time Step
  m_timeStepLabel =
    VarLabel::create(timeStep_name, timeStep_vartype::getTypeDescription() );
}

TimestepNumber::~TimestepNumber()
{
  VarLabel::destroy(m_timeStepLabel);
}

void TimestepNumber::problemSetup(const ProblemSpecP& ps, 
                                  const ProblemSpecP& restart_prob_spec, 
                                  MaterialManagerP& materialManager)
{
  m_materialManager = materialManager;
}

void TimestepNumber::scheduleSwitchTest(const LevelP& level, SchedulerP& sched)
{
  printSchedule(level,switching_dbg,"Switchinng Criteria:TimestepNumber::scheduleSwitchTest");
  
  Task* t = scinew Task("switchTest", this, &TimestepNumber::switchTest);

  t->requires(Task::OldDW, m_timeStepLabel);
  t->computes(d_switch_label);
  sched->addTask(t, level->eachPatch(), m_materialManager->allMaterials());
}

void TimestepNumber::switchTest(const ProcessorGroup* group,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  switching_dbg << "Doing Switch Criteria:TimeStepNumber";

  timeStep_vartype timeStep_var(0);
  if( old_dw->exists(m_timeStepLabel))
    old_dw->get(timeStep_var, m_timeStepLabel);
  else if( new_dw->exists(m_timeStepLabel))
      new_dw->get(timeStep_var, m_timeStepLabel);
  int timeStep = timeStep_var;

  double sw = (timeStep == m_timeStep);

  switching_dbg  << " is it time to switch components: " << sw << std::endl;

  max_vartype switch_condition(sw);
  
  const Level* allLevels = nullptr;
  new_dw->put(switch_condition, d_switch_label, allLevels);
}
