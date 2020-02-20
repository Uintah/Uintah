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

#ifndef CCA_COMPONENTS_LOADBALANCERS_SIMPLELOADBALANCER_H
#define CCA_COMPONENTS_LOADBALANCERS_SIMPLELOADBALANCER_H

#include <CCA/Components/LoadBalancers/LoadBalancerCommon.h>

namespace Uintah {

/**************************************

 CLASS
   SimpleLoadBalancer


 GENERAL INFORMATION

 SimpleLoadBalancer.h

 Steven G. Parker
 Department of Computer Science
 University of Utah

 Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


 KEYWORDS
   SimpleLoadBalancer

 DESCRIPTION

 ****************************************/

class Patch;
class ProcessorGroup;

class SimpleLoadBalancer : public LoadBalancerCommon {

public:

  SimpleLoadBalancer(const ProcessorGroup* myworld);

  virtual ~SimpleLoadBalancer() {};

  virtual int getPatchwiseProcessorAssignment(const Patch * patch);

  //! The old processor is the same as the current for this load balancer.
  virtual int getOldProcessorAssignment(const Patch * patch)
  {
    return getPatchwiseProcessorAssignment(patch);
  }

  virtual bool needRecompile( const GridP& ) { return false; };

private:

  // eliminate copy, assignment and move
  SimpleLoadBalancer( const SimpleLoadBalancer & )            = delete;
  SimpleLoadBalancer& operator=( const SimpleLoadBalancer & ) = delete;
  SimpleLoadBalancer( SimpleLoadBalancer && )                 = delete;
  SimpleLoadBalancer& operator=( SimpleLoadBalancer && )      = delete;

};

}  // End namespace Uintah

#endif // CCA_COMPONENTS_LOADBALANCERS_SIMPLELOADBALANCER_H

