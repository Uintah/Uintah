#include <Packages/Uintah/Core/Grid/FileGeometryPiece.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <fstream>

using namespace Uintah;
using namespace SCIRun;
using std::ifstream;

FileGeometryPiece::FileGeometryPiece(ProblemSpecP& ps)
{
  string file_name,var_name;
  ps->require("name",file_name);

  if (ps->get("var",var_name))
    readPoints(file_name,true);
  else
    readPoints(file_name);
}

FileGeometryPiece::FileGeometryPiece(const string& file_name)
{
  readPoints(file_name);
}

FileGeometryPiece::~FileGeometryPiece()
{
}

bool FileGeometryPiece::inside(const Point& p) const
{
  //Check p with the lower coordinates

  if (p == Max(p,d_box.lower()) && p == Min(p,d_box.upper()) )
    return true;
  else
    return false;
	       
}

Box FileGeometryPiece::getBoundingBox() const
{
  return d_box;
}

void FileGeometryPiece::readPoints(const string& f_name, bool var)
{
  ifstream source(f_name.c_str());

  while(!source.eof()) {
    double x,y,z,var;
    if (var == false) 
      source >> x >> y >> z;
    else {
      source >> x >> y >> z >> var;
      d_volume.push_back(var);
    }
    d_points.push_back(Point(x,y,z));
  }
  source.close();

  // Find the min and max points so that the bounding box can be determined.
  Point min(1e30,1e30,1e30),max(-1e30,-1e30,-1e30);
  vector<Point>::const_iterator itr;
  for (itr = d_points.begin(); itr != d_points.end(); ++itr) {
    min = Min(*itr,min);
    max = Max(*itr,max);
  }

  d_box = Box(min,max);
}

vector<Point>* FileGeometryPiece::getPoints()
{
  return &d_points;
}


vector<double>* FileGeometryPiece::getVolume()
{
  return &d_volume;
}
