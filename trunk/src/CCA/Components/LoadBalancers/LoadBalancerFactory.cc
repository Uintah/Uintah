#include <CCA/Components/LoadBalancers/LoadBalancerFactory.h>
#include <CCA/Components/LoadBalancers/NirvanaLoadBalancer.h>
#include <CCA/Components/LoadBalancers/DynamicLoadBalancer.h>
#include <CCA/Components/LoadBalancers/RoundRobinLoadBalancer.h>
#include <CCA/Components/LoadBalancers/SimpleLoadBalancer.h>
#include <CCA/Components/LoadBalancers/SingleProcessorLoadBalancer.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
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
  } else if( (loadbalancer == "NirvanaLoadBalancer") ||
             (loadbalancer == "NLB") ) {
    bal = scinew NirvanaLoadBalancer(world, layout);
  } else if( (loadbalancer == "DLB") ||
             (loadbalancer == "PLB") /* backward-compatibility*/) {
    bal = scinew DynamicLoadBalancer(world);
  } else {
    bal = 0;   
    throw ProblemSetupException("Unknown load balancer", __FILE__, __LINE__);
  }
  
  return bal;

}
