#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/NormalForceBC.h>

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

NormalForceBC::NormalForceBC(ProblemSpecP& ps)
{
  // Read and save the load curve information
  d_loadCurve = new LoadCurve<double>(ps);
}

NormalForceBC::~NormalForceBC()
{
  delete d_loadCurve;
}

// Get the type of this object for BC application
std::string NormalForceBC::getType() const
{
  return "NormalForce";
}
