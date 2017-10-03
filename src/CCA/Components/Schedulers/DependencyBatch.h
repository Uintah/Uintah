/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#ifndef CCA_COMPONENTS_SCHEDULERS_DEPENDENCY_BATCH_H
#define CCA_COMPONENTS_SCHEDULERS_DEPENDENCY_BATCH_H

#include <CCA/Components/Schedulers/DetailedTasks.h>

#include <list>
#include <map>
#include <vector>

namespace Uintah {

class DetailedDep;
class ProcessorGroup;
class Variable;
class VarLabel;


class DependencyBatch {

public:

  DependencyBatch( int            to
                 , DetailedTask * fromTask
                 , DetailedTask * toTask
                 )
    : m_from_task(fromTask)
    , m_to_rank(to)
  {
    m_to_tasks.push_back(toTask);
  }

  ~DependencyBatch();

  // Initialize receiving information for makeMPIRequest() and received() so that it can receive again.
  void reset();

  // The first thread calling this will return true, all others will return false.
  bool makeMPIRequest();

  // Tells this batch that it has actually been received and
  // awakens anybody blocked in makeMPIRequest().
  void received( const ProcessorGroup * pg );

  // Add invalid variables to dep batch. These variables will be marked as valid when MPI completes.
  void addVar( Variable * var );

  DependencyBatch          * m_comp_next{nullptr};
  DetailedTask             * m_from_task;
  DetailedDep              * m_head{nullptr};
  std::list<DetailedTask*>   m_to_tasks;
  int                        m_message_tag{-1};
  int                        m_to_rank{-1};


private:

  // eliminate copy, assignment and move
  DependencyBatch( const DependencyBatch & )            = delete;
  DependencyBatch& operator=( const DependencyBatch & ) = delete;
  DependencyBatch( DependencyBatch && )                 = delete;
  DependencyBatch& operator=( DependencyBatch && )      = delete;

  bool m_received{false};
  std::atomic<bool> m_made_mpi_request{false};

  std::vector<Variable*> m_to_vars;

};

} // namespace Uintah

#endif //CCA_COMPONENTS_SCHEDULERS_DEPENDENCY_BATCH_H
