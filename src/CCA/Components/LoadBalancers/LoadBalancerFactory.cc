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

#include <CCA/Components/LoadBalancers/LoadBalancerFactory.h>
#include <CCA/Components/LoadBalancers/DynamicLoadBalancer.h>
#include <CCA/Components/LoadBalancers/RoundRobinLoadBalancer.h>
#include <CCA/Components/LoadBalancers/SimpleLoadBalancer.h>
#include <CCA/Components/LoadBalancers/SingleProcessorLoadBalancer.h>
#include <CCA/Components/LoadBalancers/ParticleLoadBalancer.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <iostream>

using namespace std;
using namespace Uintah;

LoadBalancerCommon* LoadBalancerFactory::create(ProblemSpecP& ps, 
                                                const ProcessorGroup* world)
{
  LoadBalancerCommon* bal = 0;
  string loadbalancer = "";
  IntVector layout(1,1,1);
  
  ProblemSpecP lb_ps = ps->findBlock("LoadBalancer");
  if (lb_ps)
    lb_ps->getAttribute("type",loadbalancer);

  // Default settings
  if (Uintah::Parallel::usingMPI()) {
    if (loadbalancer == "")
      loadbalancer = "SimpleLoadBalancer";
  }
  else // No MPI
    if (loadbalancer == "")
      loadbalancer = "SingleProcessorLoadBalancer";


  if (world->myrank() == 0)
    cout << "Load Balancer: \t\t" << loadbalancer << endl;

  if(loadbalancer == "SingleProcessorLoadBalancer"){
    bal = scinew SingleProcessorLoadBalancer(world);
  } else if(loadbalancer == "RoundRobinLoadBalancer" || 
            loadbalancer == "RoundRobin" || 
            loadbalancer == "roundrobin"){
    bal = scinew RoundRobinLoadBalancer(world);
  } else if(loadbalancer == "SimpleLoadBalancer") {
    bal = scinew SimpleLoadBalancer(world);
  } else if(loadbalancer == "DLB") {
    bal = scinew DynamicLoadBalancer(world);
  } else if(loadbalancer == "PLB") {
    bal = scinew ParticleLoadBalancer(world);
  } else {
    bal = 0;   
    throw ProblemSetupException("Unknown load balancer", __FILE__, __LINE__);
  }
  
  return bal;

}
