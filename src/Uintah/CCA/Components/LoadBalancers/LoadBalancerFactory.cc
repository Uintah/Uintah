/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <Packages/Uintah/CCA/Components/LoadBalancers/LoadBalancerFactory.h>
#include <Packages/Uintah/CCA/Components/LoadBalancers/DynamicLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/LoadBalancers/RoundRobinLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/LoadBalancers/SimpleLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/LoadBalancers/SingleProcessorLoadBalancer.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <iostream>

using std::cerr;
using std::cout;
using std::endl;

using namespace Uintah;

LoadBalancerCommon* LoadBalancerFactory::create(ProblemSpecP& ps, 
                                                const ProcessorGroup* world)
{
  LoadBalancerCommon* bal = 0;
  string loadbalancer = "";
  IntVector layout(1,1,1);
  
  ProblemSpecP lb_ps = ps->findBlock("LoadBalancer");
  if (lb_ps)
    lb_ps->get("type",loadbalancer);

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
  } else if( (loadbalancer == "DLB") ||
             (loadbalancer == "PLB") /* backward-compatibility*/) {
    bal = scinew DynamicLoadBalancer(world);
  } else {
    bal = 0;   
    throw ProblemSetupException("Unknown load balancer", __FILE__, __LINE__);
  }
  
  return bal;

}
