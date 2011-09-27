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



#include <Core/Exceptions/ConvergenceFailure.h>
#include <Core/Geometry/IntVector.h>
#include <sstream>
#include <iostream>

using namespace Uintah;
using namespace SCIRun;



ConvergenceFailure::ConvergenceFailure(const string& message,
				       int numiterations,
				       double final_residual,
				       double target_residual,
                                       const char* file,
                                       int line)
{
  std::ostringstream s;
  s << "A ConvergenceFailure exception was thrown.\n"
    << file << ":" << line << "\n"
    << message << " failed to converge in " << numiterations << " iterations"
    << ", final residual=" << final_residual 
    << ", target_residual=" << target_residual;
  d_msg = s.str();
  
#ifdef EXCEPTIONS_CRASH
  cout << d_msg << "\n";
#endif
}


ConvergenceFailure::ConvergenceFailure(const ConvergenceFailure& copy)
  : d_msg(copy.d_msg)
{
}

ConvergenceFailure::~ConvergenceFailure()
{
}

const char* ConvergenceFailure::message() const
{
  return d_msg.c_str();
}

const char* ConvergenceFailure::type() const
{
  return "Packages/Uintah::Exceptions::ConvergenceFailure";
}






