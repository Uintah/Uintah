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

#include <CCA/Components/LoadBalancers/LoadBalancerFactory.h>
#include <CCA/Components/LoadBalancers/DynamicLoadBalancer.h>
#include <CCA/Components/LoadBalancers/ParticleLoadBalancer.h>
#include <CCA/Components/LoadBalancers/RoundRobinLoadBalancer.h>
#include <CCA/Components/LoadBalancers/SimpleLoadBalancer.h>

#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <string>
#include <iostream>

using namespace Uintah;


LoadBalancerCommon*
LoadBalancerFactory::create( const ProblemSpecP   & ps
                           , const ProcessorGroup * world
                           )
{
  LoadBalancerCommon* bal  = nullptr;
  std::string loadbalancer = "";
  IntVector layout(1, 1, 1);

  ProblemSpecP lb_ps = ps->findBlock("LoadBalancer");
  if (lb_ps) {
    lb_ps->getAttribute("type", loadbalancer);
  }

  /////////////////////////////////////////////////////////////////////
  // Default setting - nothing specified in the input file
  if (loadbalancer == "") {
    loadbalancer = "Simple";
  }

  /////////////////////////////////////////////////////////////////////
  // Check for specific load balancer request from the input file
  // TODO: Replace all remaining input file usage of SimpleLoadBalancer-->Simple then fix this and corresponding XML spec - APH 09/17/16
  //        This is a few ARCHES input files, including 8-corner case that use this for outputNthproc, etc
  if (loadbalancer == "Simple" || loadbalancer == "simple" || loadbalancer == "SimpleLoadBalancer") {
    bal = scinew SimpleLoadBalancer(world);
  }

  else if (loadbalancer == "RoundRobin") {
    bal = scinew RoundRobinLoadBalancer(world);
  }

  else if (loadbalancer == "DLB") {
    bal = scinew DynamicLoadBalancer(world);
  }

  else if (loadbalancer == "PLB") {
    bal = scinew ParticleLoadBalancer(world);
  }
  else {
    bal = nullptr;

    std::ostringstream msg;
    msg << "\nERROR<Loadbalancer>: Unknown load balancer : " << loadbalancer << ".\n";
    throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
  }

  // Output which LOAD BALANCER will be used
  proc0cout << "Load Balancer: \t\t" << loadbalancer << std::endl;

  return bal;
}
