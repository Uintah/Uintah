
#include <Packages/Uintah/Core/Exceptions/OutFluxVolume.h>
#include <Core/Geometry/IntVector.h>
#include <sstream>

using namespace Uintah;
using namespace SCIRun;
using namespace std;


OutFluxVolume::OutFluxVolume(IntVector c,double fluxout,double vol)
{
  ostringstream s;
  s << " cell["<<c.x()<<"]["<<c.y()<<"]["<<c.z()
    << "], total_outflux (" << fluxout << ") > vol (" << vol << ")";

  d_msg =  "inFluxOutFluxVolume" + s.str();
  
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






