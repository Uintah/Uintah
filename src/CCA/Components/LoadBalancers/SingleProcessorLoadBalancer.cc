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



#include <CCA/Components/LoadBalancers/SingleProcessorLoadBalancer.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <Core/Grid/Level.h>

using namespace Uintah;
using namespace std;

SingleProcessorLoadBalancer::SingleProcessorLoadBalancer( const ProcessorGroup* myworld )
   : LoadBalancerCommon( myworld )
{
}

SingleProcessorLoadBalancer::~SingleProcessorLoadBalancer()
{
}

void SingleProcessorLoadBalancer::assignResources(DetailedTasks& graph)
{
  int ntasks = graph.numTasks();
  for(int i=0;i<ntasks;i++)
    graph.getTask(i)->assignResource(0);
}

int SingleProcessorLoadBalancer::getPatchwiseProcessorAssignment(const Patch*)
{
   return 0;
}

const PatchSet*
SingleProcessorLoadBalancer::createPerProcessorPatchSet(const LevelP& level)
{
  return level->allPatches();
}


void
SingleProcessorLoadBalancer::createNeighborhood(const GridP&)
{
  // Nothing to do
}

bool
SingleProcessorLoadBalancer::inNeighborhood(const PatchSubset*,
					    const MaterialSubset*)
{
  return true;
}

bool
SingleProcessorLoadBalancer::inNeighborhood(const Patch*)
{
  return true;
}
