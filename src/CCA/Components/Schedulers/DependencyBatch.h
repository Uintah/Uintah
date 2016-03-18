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

#ifndef CCA_COMPONENTS_SCHEDULERS_DEPENDENCY_BATCH_H
#define CCA_COMPONENTS_SCHEDULERS_DEPENDENCY_BATCH_H

#include <CCA/Components/Schedulers/DetailedDep.h>

#include <Core/Grid/Variables/Variable.h>

#include <list>
#include <map>
#include <vector>

#include <mutex>

namespace Uintah {

class DetailedTask;
class ProcessorGroup;


class DependencyBatch
{
public:

  DependencyBatch( int to
                 , DetailedTask* fromTask
                 , DetailedTask* toTask
                 )
    : comp_next(0)
    , fromTask(fromTask)
    , head(0), messageTag(-1)
    , to(to), received_(false)
    , madeMPIRequest_(false)
  {
    toTasks.push_back(toTask);
  }

  ~DependencyBatch();

  // The first thread calling this will return true, all others
  // will return false.
  bool makeMPIRequest();

  // Tells this batch that it has actually been received and
  // awakens anybody blocked in makeMPIRequest().
  void received( const ProcessorGroup * pg );

  bool wasReceived() { return received_; }

  // Initialize receiving information for makeMPIRequest() and received()
  // so that it can receive again.
  void reset();

  //Add invalid variables to the dependency batch.  These variables will be marked
  //as valid when MPI completes.
  void addVar( Variable* var ) { toVars.push_back(var); }

  void addReceiveListener( int mpiSignal );

  DependencyBatch          * comp_next;
  DetailedTask             * fromTask;
  std::list<DetailedTask*>   toTasks;
  DetailedDep              * head;
  int                        messageTag;
  int                        to;

  //scratch pad to store wait times for debugging
  static std::map<std::string,double> waittimes;

private:

  volatile bool  received_;
  volatile bool  madeMPIRequest_;
  std::mutex     lock_;
  std::set<int>  receiveListeners_;

  // eliminate copy, assignment and move
  DependencyBatch( const DependencyBatch & )            = delete;
  DependencyBatch& operator=( const DependencyBatch & ) = delete;
  DependencyBatch( DependencyBatch && )                 = delete;
  DependencyBatch& operator=( DependencyBatch && )      = delete;

  std::vector<Variable*> toVars;

}; // DependencyBatch


struct InternalDependency {

  InternalDependency( DetailedTask* prerequisiteTask
                    , DetailedTask* dependentTask
                    , const VarLabel* var
                    , long satisfiedGeneration
                    )
    : prerequisiteTask(prerequisiteTask)
    , dependentTask(dependentTask)
    , satisfiedGeneration(satisfiedGeneration)
  {
    addVarLabel(var);
  }

  void addVarLabel( const VarLabel* var ) { vars.insert(var); }

  DetailedTask * prerequisiteTask;
  DetailedTask * dependentTask;
  unsigned long  satisfiedGeneration;

  std::set<const VarLabel*, VarLabel::Compare>  vars;

}; // InternalDependency


} // namespace Uintah

#endif  //CCA_COMPONENTS_SCHEDULERS_DEPENDENCY_BATCH_H
