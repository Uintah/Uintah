
#include <Packages/Uintah/Core/Exceptions/ConvergenceFailure.h>
#include <Core/Geometry/IntVector.h>
#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;


ConvergenceFailure::ConvergenceFailure(const string& message,
				       int numiterations,
				       double final_residual,
				       double target_residual)
{
  ostringstream s;
  s << message << " failed to converge in " << numiterations << " iterations"
    << ", final residual=" << final_residual 
    << ", target_residual=" << target_residual;
  d_msg = s.str();
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






