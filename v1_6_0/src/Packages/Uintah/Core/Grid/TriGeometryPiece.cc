#include "TriGeometryPiece.h"
#include "GeometryPieceFactory.h"
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <iostream>

using namespace Uintah;
using namespace std;

TriGeometryPiece::TriGeometryPiece(ProblemSpecP &ps)
{

  std::string file;

  ps->require("file",file);
  
}

TriGeometryPiece::~TriGeometryPiece()
{
}

bool TriGeometryPiece::inside(const Point &) const
{
    cerr << "TriGeometry piece not finished!\n";
    return false;
}

Box TriGeometryPiece::getBoundingBox() const
{
    cerr << "TriGeometry piece not finished!\n";
    return Box();
}

