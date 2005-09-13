#include <Packages/Uintah/CCA/Components/LoadBalancers/LoadBalancerFactory.h>
#include <Packages/Uintah/CCA/Components/LoadBalancers/NirvanaLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/LoadBalancers/ParticleLoadBalancer.h>
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
    cout << "Using Load Balancer " << loadbalancer << endl;

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
  } else if( (loadbalancer == "ParticleLoadBalancer") ||
             (loadbalancer == "PLB") ) {
    bal = scinew ParticleLoadBalancer(world);
  } else {
    bal = 0;   
    throw ProblemSetupException("Unknown load balancer", __FILE__, __LINE__);
  }
  
  return bal;

}
