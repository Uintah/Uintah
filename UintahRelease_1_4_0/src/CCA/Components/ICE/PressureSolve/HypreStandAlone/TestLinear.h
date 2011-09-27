/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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


#ifndef __TESTLINEAR_H__
#define __TESTLINEAR_H__

#include "Param.h"
#include "util.h"

class TestLinear : public Param {
  /*_____________________________________________________________________
    class TestLinear:
    Test case with
    * Exact solution U = linear function with Dirichlet B.C. on the d-D
    unit square. U is smooth.
    * Diffusion a=1 (Laplace operator).
    _____________________________________________________________________*/
public:
  
  TestLinear(const Counter d,
             const Counter base);
  virtual ~TestLinear(void) {}
  
  /* Input functions */
  
  virtual double diffusion(const Location& x) const;     // Diffusion coeff
  virtual double rhs(const Location& x) const;           // RHS of PDE
  virtual double rhsBC(const Location& x) const;         // RHS of B.C.
  virtual double exactSolution(const Location& x) const; // Exact solution

};

#endif // __TESTLINEAR_H__
