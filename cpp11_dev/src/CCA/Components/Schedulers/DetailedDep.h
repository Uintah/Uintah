/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

class DetailedDep
{

public:

  enum CommCondition {
    Always
  , FirstIteration
  , SubsequentIterations
  };

  DetailedDep(       DetailedDep * next
             ,       Task::Dependency   * comp
             ,       Task::Dependency   * req
             ,       DetailedTask       * toTask
             , const Patch              * fromPatch
             ,       int                  matl
             , const IntVector          & low
             , const IntVector          & high
             , CommCondition              cond
             )
    : next(next)
    , comp(comp)
    , req(req)
    , fromPatch(fromPatch)
    , low(low), high(high)
    , matl(matl)
    , condition(cond)
    , patchLow(low)
    , patchHigh(high)
  {
    ASSERT(Min(high - low, IntVector(1, 1, 1)) == IntVector(1, 1, 1));

    USE_IF_ASSERTS_ON( Patch::VariableBasis basis = Patch::translateTypeToBasis(req->var->typeDescription()->getType(), true); )

      ASSERT(fromPatch == 0 || (Uintah::Min(low, fromPatch->getExtraLowIndex(basis, req->var->getBoundaryLayer())) ==
            fromPatch->getExtraLowIndex(basis, req->var->getBoundaryLayer())));

    ASSERT(fromPatch == 0 || (Uintah::Max(high, fromPatch->getExtraHighIndex(basis, req->var->getBoundaryLayer())) ==
          fromPatch->getExtraHighIndex(basis, req->var->getBoundaryLayer())));

    toTasks.push_back(toTask);
  }


  // As an arbitrary convention, non-data dependency have a NULL fromPatch.
  // These types of dependency exist between a modifying task and any task
  // that requires the data (from ghost cells in particular) before it is
  // modified preventing the possibility of modifying data while it is being
  // used.
  bool isNonDataDependency() const { return (fromPatch == nullptr); }

  DetailedDep              * next;
  Task::Dependency         * comp;
  Task::Dependency         * req;
  std::list<DetailedTask*>   toTasks;
  const Patch              * fromPatch;
  IntVector                  low;
  IntVector                  high;
  int                        matl;

  // this is to satisfy a need created by the DynamicLoadBalancer.  To keep it unrestricted on when it can perform, and
  // to avoid a costly second recompile on the next timestep, we add a comm condition which will send/recv data based
  // on whether some condition is met at run time - in this case whether it is the first execution or not.
  CommCondition condition;

  // for SmallMessages - if we don't copy the complete patch, we need to know the range so we can store all segments properly
  IntVector patchLow;
  IntVector patchHigh;

  // eliminate copy, assignment and move
  DetailedDep( const DetailedDep & )            = delete;
  DetailedDep& operator=( const DetailedDep & ) = delete;
  DetailedDep( DetailedDep && )                 = delete;
  DetailedDep& operator=( DetailedDep && )      = delete;


}; // DetailedDep

std::ostream& operator<<( std::ostream& out, const Uintah::DetailedDep& task );

} // namespace Uintah

#endif // CCA_COMPONENTS_SCHEDULERS_DETAILED_DEP_H


