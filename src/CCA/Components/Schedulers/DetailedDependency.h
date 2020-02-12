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

#ifndef CCA_COMPONENTS_SCHEDULERS_DETAILED_DEP_H
#define CCA_COMPONENTS_SCHEDULERS_DETAILED_DEP_H

#include <Core/Grid/Patch.h>
#include <Core/Grid/Task.h>

#include <list>

namespace Uintah {

class DetailedTask;

class DetailedDep {

public:

  enum CommCondition {
      Always
    , FirstIteration
    , SubsequentIterations
  };

  DetailedDep(       DetailedDep      * next
             ,       Task::Dependency * comp
             ,       Task::Dependency * req
             ,       DetailedTask     * toTask
             , const Patch            * fromPatch
             ,       int                matl
             , const IntVector        & low
             , const IntVector        & high
             ,       CommCondition      cond
             )
    : m_next(next)
    , m_comp(comp)
    , m_req(req)
    , m_from_patch(fromPatch)
    , m_low(low)
    , m_high(high)
    , m_matl(matl)
    , m_comm_condition(cond)
    , m_patch_low(low)
    , m_patch_high(high)
  {
    ASSERT(Min(high - low, IntVector(1, 1, 1)) == IntVector(1, 1, 1));

    USE_IF_ASSERTS_ON( Patch::VariableBasis basis = Patch::translateTypeToBasis(req->m_var->typeDescription()->getType(), true); )

    ASSERT(fromPatch == 0 || (Min(low, fromPatch->getExtraLowIndex(basis, req->m_var->getBoundaryLayer())) ==
      fromPatch->getExtraLowIndex(basis, req->m_var->getBoundaryLayer())));

    ASSERT(fromPatch == 0 || (Max(high, fromPatch->getExtraHighIndex(basis, req->m_var->getBoundaryLayer())) ==
      fromPatch->getExtraHighIndex(basis, req->m_var->getBoundaryLayer())));

    m_to_tasks.push_back(toTask);
  }


  // As an arbitrary convention, non-data dependency have a nullptr fromPatch.
  // These types of dependencies exist between a modifying task and any task
  // that requires the data (from ghost cells in particular) before it is modified,
  // preventing the possibility of modifying data while it is being used.
  bool isNonDataDependency() const { return (m_from_patch == nullptr); }

  DetailedDep              * m_next;
  Task::Dependency         * m_comp;
  Task::Dependency         * m_req;
  std::list<DetailedTask*>   m_to_tasks;
  const Patch              * m_from_patch;
  IntVector                  m_low;
  IntVector                  m_high;
  int                        m_matl;

  // this is to satisfy a need created by the DynamicLoadBalancer.  To keep it unrestricted on when it can perform, and
  // to avoid a costly second recompile on the next timestep, we add a comm condition which will send/recv data based
  // on whether some condition is met at run time - in this case whether it is the first execution or not.
  CommCondition m_comm_condition;

  // for SmallMessages - if we don't copy the complete patch, we need to know the range so we can store all segments properly
  IntVector m_patch_low;
  IntVector m_patch_high;


private:

  // eliminate copy, assignment and move
  DetailedDep( const DetailedDep & )            = delete;
  DetailedDep& operator=( const DetailedDep & ) = delete;
  DetailedDep( DetailedDep && )                 = delete;
  DetailedDep& operator=( DetailedDep && )      = delete;

};

std::ostream& operator<<( std::ostream & out, const Uintah::DetailedDep & task );

} // namespace Uintah

#endif // CCA_COMPONENTS_SCHEDULERS_DETAILED_DEP_H


