#include <Packages/Uintah/Core/Grid/GeomPiece/FileGeometryPiece.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;
using std::ifstream;

FileGeometryPiece::FileGeometryPiece(ProblemSpecP& ps)
{
  setName("file");
  ps->require("name",d_file_name);
  ps->get("var",d_var_name);

  // We must first read in the min and max from file.0 so
  // that we can determine the BoundingBox for the geometry
  char fnum[5];
  sprintf(fnum,".%d",0);
  string file_name = d_file_name+fnum;
  ifstream source(file_name.c_str());
  if (!source ){
    throw ProblemSetupException("ERROR: opening MPM geometry file: \n The file must be in the same directory as sus");
  }
  double minx,miny,minz,maxx,maxy,maxz;
  source >> minx >> miny >> minz >> maxx >> maxy >> maxz;
  source.close();
  Point min(minx,miny,minz),max(maxx,maxy,maxz);
  Vector fudge(1.e-5,1.e-5,1.e-5);
  min = min - fudge;
  max = max + fudge;
  d_box = Box(min,max);
  cout << min << " " << max << endl;
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
  if (!source ){
    throw ProblemSetupException("ERROR: opening MPM geometry file: \n The file must be in the same directory as sus");
  }

  double x,y,z,vol;
  if (var == false) {
    while (source >> x >> y >> z) {
      d_points.push_back(Point(x,y,z));
    }
  } else {
    while(source >> x >> y >> z >> vol) {
      d_points.push_back(Point(x,y,z));
      d_volume.push_back(vol);
    }
  }
  source.close();

  // Find the min and max points so that the bounding box can be determined.
  Point min(1e30,1e30,1e30),max(-1e30,-1e30,-1e30);
  vector<Point>::const_iterator itr;
  for (itr = d_points.begin(); itr != d_points.end(); ++itr) {
    min = Min(*itr,min);
    max = Max(*itr,max);
  }
  Vector fudge(1.e-5,1.e-5,1.e-5);
  min = min - fudge;
  max = max + fudge;
  d_box = Box(min,max);
}

void FileGeometryPiece::readPoints(int pid)
{
  bool var=false;
  char fnum[5];
  sprintf(fnum,".%d",pid);
  string file_name = d_file_name+fnum;
  ifstream source(file_name.c_str());
  if (!source ){
    throw ProblemSetupException("ERROR: opening MPM geometry file:  The file must be in the same directory as sus");
  }

  double x,y,z,vol;
  double minx,miny,minz,maxx,maxy,maxz;
  source >> minx >> miny >> minz >> maxx >> maxy >> maxz;
  Point min(minx,miny,minz),max(maxx,maxy,maxz);
  Vector fudge(1.e-5,1.e-5,1.e-5);
  min = min - fudge;
  max = max + fudge;
  d_box = Box(min,max);

  if (var == false) {
    while (source >> x >> y >> z) {
      d_points.push_back(Point(x,y,z));
    }
  }
  else {
    while(source >> x >> y >> z >> vol) {
      d_points.push_back(Point(x,y,z));
      d_volume.push_back(vol);
    }
  }
  source.close();
}
