#include <Packages/Uintah/Core/GeometryPiece/NullGeometryPiece.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;
using std::ifstream;

NullGeometryPiece::NullGeometryPiece(ProblemSpecP& /*ps*/)
{
  setName("null");
  d_box = Box(Point(0.,0.,0.),Point(0.,0.,0.));
}

NullGeometryPiece::NullGeometryPiece(const string& /*file_name*/)
{
}

NullGeometryPiece::~NullGeometryPiece()
{
}

NullGeometryPiece*
NullGeometryPiece::clone()
{
  return scinew NullGeometryPiece(*this);
}

bool
NullGeometryPiece::inside(const Point& /*p*/) const
{
  return true;
}

Box
NullGeometryPiece::getBoundingBox() const
{
  return d_box;
}

