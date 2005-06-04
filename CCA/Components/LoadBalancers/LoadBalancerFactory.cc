#include <Packages/Uintah/CCA/Components/LoadBalancers/LoadBalancerFactory.h>
#include <Packages/Uintah/CCA/Components/LoadBalancers/NirvanaLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/LoadBalancers/ParticleLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/LoadBalancers/RoundRobinLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/LoadBalancers/SimpleLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/LoadBalancers/SingleProcessorLoadBalancer.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <iostream>

using std::cerr;
using std::endl;

using namespace Uintah;

LoadBalancerCommon* LoadBalancerFactory::create(ProblemSpecP& ps, 
                                                const ProcessorGroup* world)
{
  LoadBalancerCommon* bal = 0;
  string loadbalancer = "";
  IntVector layout(1,1,1);
  
  ps->get("LoadBalancer",loadbalancer);

  // Default settings
  if (Uintah::Parallel::usingMPI()) {
    if (loadbalancer == "")
      loadbalancer = "SimpleLoadBalancer";
  }
  else // No MPI
    if (loadbalancer == "")
      loadbalancer = "SingleProcessorLoadBalancer";


  if(loadbalancer == "SingleProcessorLoadBalancer"){
    SingleProcessorLoadBalancer* splb 
      = scinew SingleProcessorLoadBalancer(world);
    bal = splb;
  } else if(loadbalancer == "RoundRobinLoadBalancer" || 
            loadbalancer == "RoundRobin" || 
            loadbalancer == "roundrobin"){
    RoundRobinLoadBalancer* rrlb 
      = scinew RoundRobinLoadBalancer(world);
    bal = rrlb;
  } else if(loadbalancer == "SimpleLoadBalancer") {
    SimpleLoadBalancer* slb
      = scinew SimpleLoadBalancer(world);
    bal = slb;
  } else if( (loadbalancer == "NirvanaLoadBalancer") ||
             (loadbalancer == "NLB") ) {
    NirvanaLoadBalancer* nlb
      = scinew NirvanaLoadBalancer(world, layout);
    bal = nlb;
  } else if( (loadbalancer == "ParticleLoadBalancer") ||
             (loadbalancer == "PLB") ) {
    ParticleLoadBalancer* plb 
      = scinew ParticleLoadBalancer(world);
    bal = plb;
  } else {
    bal = 0;   
    cerr << "Unknown load balancer: " + loadbalancer << endl;
  }
  
  return bal;

}
