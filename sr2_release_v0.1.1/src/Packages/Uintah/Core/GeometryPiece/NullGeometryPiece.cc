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

const string NullGeometryPiece::TYPE_NAME = "null";

NullGeometryPiece::NullGeometryPiece(ProblemSpecP& /*ps*/)
{
  name_ = "Unnamed Null";
  d_box = Box(Point(0.,0.,0.),Point(0.,0.,0.));
}

NullGeometryPiece::NullGeometryPiece(const string& /*file_name*/)
{
}

NullGeometryPiece::~NullGeometryPiece()
{
}

void
NullGeometryPiece::outputHelper( ProblemSpecP & ps ) const
{
  ps->appendChild("null");
}

GeometryPieceP
NullGeometryPiece::clone() const
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

