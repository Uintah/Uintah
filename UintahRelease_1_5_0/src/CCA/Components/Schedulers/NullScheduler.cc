/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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



#include <CCA/Components/Schedulers/NullScheduler.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/ReductionVariable.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>

using namespace Uintah;
using namespace std;
using namespace SCIRun;

static DebugStream dbg("NullScheduler", false);

NullScheduler::NullScheduler(const ProcessorGroup* myworld,
			     Output* oport)
  : SchedulerCommon(myworld, oport)
{
  d_generation = 0;
  delt = VarLabel::create("delT",
			  ReductionVariable<double, Reductions::Min<double> >::getTypeDescription());
  firstTime=true;
}

NullScheduler::~NullScheduler()
{
}

SchedulerP
NullScheduler::createSubScheduler()
{
  NullScheduler* newsched = scinew NullScheduler(d_myworld, m_outPort);
  UintahParallelPort* lbp = getPort("load balancer");
  newsched->attachPort("load balancer", lbp);
  return newsched;
}

void
NullScheduler::verifyChecksum()
{
  // Not used in NullScheduler
}

void 
NullScheduler::advanceDataWarehouse(const GridP& grid)
{
  for(int i=0;i<(int)dws.size();i++)
    if( !dws[i] )
      dws[i] = scinew OnDemandDataWarehouse(d_myworld, this, 0, grid);
}

void
NullScheduler::actuallyCompile()
{
  if( dts_ )
    delete dts_;
  if(graph.getNumTasks() == 0){
    dts_=0;
    return;
  }

  UintahParallelPort* lbp = getPort("load balancer");
  LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
  dts_ = graph.createDetailedTasks(lb, useInternalDeps(), const_cast<Grid*>(getLastDW()->getGrid()), 
                                                          const_cast<Grid*>(get_dw(0)->getGrid()));

  if(dts_->numTasks() == 0){
    cerr << "WARNING: Scheduler executed, but no tasks\n";
  }
  
  lb->assignResources(*dts_);

  graph.createDetailedDependencies(dts_, lb);
  releasePort("load balancer");

  dts_->assignMessageTags(d_myworld->myrank());
}

void
NullScheduler::execute()
{
  if(dts_ == 0){
    cerr << "NullScheduler skipping execute, no tasks\n";
    return;
  }
  if(firstTime){
    firstTime=false;
    dws[dws.size()-1]->put(delt_vartype(1.0), delt);
  }
}

void
NullScheduler::scheduleParticleRelocation(const LevelP&,
					  const VarLabel*,
					  const vector<vector<const VarLabel*> >&,
					  const VarLabel*,
					  const vector<vector<const VarLabel*> >&,
					  const VarLabel* /*particleIDLabel*/,
					  const MaterialSet*)
{
}
