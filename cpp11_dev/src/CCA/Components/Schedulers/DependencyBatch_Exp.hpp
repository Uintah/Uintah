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

#ifndef CCA_COMPONENTS_SCHEDULERS_DEPENDENCY_BATCH_EXP_H
#define CCA_COMPONENTS_SCHEDULERS_DEPENDENCY_BATCH_EXP_H

#include <CCA/Components/Schedulers/DetailedDependency_Exp.hpp>
#include <Core/Grid/Variables/Variable.h>

#include <list>
#include <mutex>
#include <vector>


namespace Uintah {

class DetailedTask;
class ProcessorGroup;


class DependencyBatch
{

public:

  DependencyBatch( int            to
                 , DetailedTask * fromTask
                 , DetailedTask * toTask
                 )
  : m_comp_next{nullptr}
  , m_from_task{fromTask}
  , m_head{nullptr}
  , m_message_tag{-1}
  , m_to_proc{to}
  , m_received{false}
  , m_made_mpi_request{false}
  {
    m_to_tasks.push_back(toTask);
  }

  ~DependencyBatch();

  // The first thread calling this will return true, all others
  // will return false.
  bool makeMPIRequest();

  // Tells this batch that it has actually been received and
  // awakens anybody blocked in makeMPIRequest().
  void received( const ProcessorGroup * pg );

  bool wasReceived() { return m_received; }

  // Initialize receiving information for makeMPIRequest() and received()
  // so that it can receive again.
  void reset();

  // Add invalid variables to the dependency batch.
  // These variables will be marked as valid when MPI completes.
  void addVar( Variable* var ) { m_to_vars.push_back(var); }

  DependencyBatch          * m_comp_next{};
  DetailedTask             * m_from_task{};
  std::list<DetailedTask*>   m_to_tasks{};
  DetailedDependency       * m_head{};
  int                        m_message_tag{};
  int                        m_to_proc{};


private:

  // eliminate copy, assignment and move
  DependencyBatch( const DependencyBatch & )            = delete;
  DependencyBatch& operator=( const DependencyBatch & ) = delete;
  DependencyBatch( DependencyBatch && )                 = delete;
  DependencyBatch& operator=( DependencyBatch && )      = delete;

  volatile bool  m_received{};
  volatile bool  m_made_mpi_request{};
  std::mutex     m_lock{};
  std::set<int>  m_receive_listeners{};
  std::vector<Variable*> m_to_vars{};

}; // DependencyBatch


struct InternalDependency {

  InternalDependency(       DetailedTask  * prerequisiteTask
                    ,       DetailedTask  * dependentTask
                    , const VarLabel      * var
                    ,       unsigned long   satisfiedGeneration
                    )
  : m_prerequisite_task{prerequisiteTask}
  , m_dependent_task{dependentTask}
  , m_satisfied_generation{satisfiedGeneration}
  {
    addVarLabel(var);
  }

  void addVarLabel( const VarLabel* var ) { m_var_labels.insert(var); }

  DetailedTask * m_prerequisite_task{};
  DetailedTask * m_dependent_task{};
  unsigned long  m_satisfied_generation{};

  std::set<const VarLabel*, VarLabel::Compare>  m_var_labels{};

}; // InternalDependency


} // namespace Uintah

#endif  // CCA_COMPONENTS_SCHEDULERS_DEPENDENCY_BATCH_EXP_H
