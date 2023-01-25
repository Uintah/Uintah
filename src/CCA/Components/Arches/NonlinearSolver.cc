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

//----- NonlinearSolver.cc ----------------------------------------------

#include <CCA/Components/Arches/NonlinearSolver.h>

using namespace Uintah;

Uintah::Dout dbg_vartask_dep{"Arches_Variable_Task_Dep", "Arches::NonlinearSolver",
"Prints variable ghost req. across all tasks.", false };

NonlinearSolver::NonlinearSolver( const ProcessorGroup* myworld,
                                  ApplicationCommon* arches )
   : d_myworld(myworld), m_arches(arches)
{}

NonlinearSolver::~NonlinearSolver()
{}

void
NonlinearSolver::commonProblemSetup( ProblemSpecP db ){

  //The underflow uses a different method to compute the CFL
  // dt = dx * rho / (div(rhou)) about the CC.
  // otherwise, it is the standard conv/diff CFL.
  d_underflow = false;
  if ( db->findBlock("scalarUnderflowCheck") ){
    d_underflow = true;
  }

  db->getWithDefault("initial_dt",d_initial_dt,1.0);

  m_arches_spec = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES");
}

void NonlinearSolver::print_variable_max_ghost(){

  std::stringstream msg;
  msg << " :: Reporting max ghost cells :: " << std::endl;

  for ( auto i = m_total_variable_ghost_info.begin(); i != m_total_variable_ghost_info.end(); i++ ){
    msg << "   Variable: " << i->first << std::endl;
    if ( i->second.numTasksNewDW > 0 ){
      msg << "        Min NewDW Ghost: " << i->second.min_newdw_ghost << " Max NewDW Ghost: " << i->second.max_newdw_ghost <<
      " across " << i->second.numTasksNewDW << " tasks. " << std::endl;
      msg << "        In the following tasks: " << std::endl;
      for (auto niter = i->second.taskNamesNewDW.begin();
           niter != i->second.taskNamesNewDW.end(); niter++ ){
        msg << "         " << *niter << std::endl;
      }
    }
    if ( i->second.numTasksOldDW > 0 ){
      msg << "        Min OldDW Ghost: " << i->second.min_olddw_ghost << " Max OldDW Ghost: " << i->second.max_olddw_ghost <<
      " across " << i->second.numTasksOldDW << " tasks. " << std::endl;
      msg << "        In the following tasks: " << std::endl;
      for (auto niter = i->second.taskNamesOldDW.begin();
         niter != i->second.taskNamesOldDW.end(); niter++ ){
           msg << "         " << *niter << std::endl;
      }
    }
  }
  msg << " :: End report of max ghost cells :: " << std::endl;
  DOUT( dbg_vartask_dep, msg.str());
}
