/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#ifndef CCA_COMPONENTS_LOADBALANCERS_ROUNDROBINLOADBALANCER_H
#define CCA_COMPONENTS_LOADBALANCERS_ROUNDROBINLOADBALANCER_H

#include <CCA/Components/LoadBalancers/LoadBalancerCommon.h>
#include <Core/Parallel/UintahParallelComponent.h>

namespace Uintah {

/**************************************

CLASS
 RoundRobinLoadBalancer


GENERAL INFORMATION

 RoundRobinLoadBalancer.h

 Steven G. Parker
 Department of Computer Science
 University of Utah

 Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


KEYWORDS
 RoundRobinLoadBalancer

DESCRIPTION


 ****************************************/

class RoundRobinLoadBalancer : public LoadBalancerCommon {

public:

  RoundRobinLoadBalancer( const ProcessorGroup * myworld );

  ~RoundRobinLoadBalancer();

  virtual int getPatchwiseProcessorAssignment( const Patch * patch );

private:

  // eliminate copy, assignment and move
  RoundRobinLoadBalancer( const RoundRobinLoadBalancer & )            = delete;
  RoundRobinLoadBalancer& operator=( const RoundRobinLoadBalancer & ) = delete;
  RoundRobinLoadBalancer( RoundRobinLoadBalancer && )                 = delete;
  RoundRobinLoadBalancer& operator=( RoundRobinLoadBalancer && )      = delete;

};

}  // End namespace Uintah

#endif // CCA_COMPONENTS_LOADBALANCERS_ROUNDROBINLOADBALANCER_H

