
#include <Packages/Uintah/Core/Exceptions/OutFluxVolume.h>
#include <Core/Geometry/IntVector.h>
#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;


OutFluxVolume::OutFluxVolume(IntVector c,double fluxout,double vol, int indx)
{
  ostringstream s;
  s << " cell["<<c.x()<<"]["<<c.y()<<"]["<<c.z()
    << "], total_outflux (" << fluxout << ") > vol (" << vol << ")"
    << " matl indx "<< indx;

  d_msg =  "inFluxOutFluxVolume" + s.str() + "\nThis usually means that the timestep is too large\n";
  
}


OutFluxVolume::OutFluxVolume(const OutFluxVolume& copy)
  : d_msg(copy.d_msg)
{
}

OutFluxVolume::~OutFluxVolume()
{
}

const char* OutFluxVolume::message() const
{
  return d_msg.c_str();
}

const char* OutFluxVolume::type() const
{
  return "Packages/Uintah::Exceptions::OutFluxVolume";
}






