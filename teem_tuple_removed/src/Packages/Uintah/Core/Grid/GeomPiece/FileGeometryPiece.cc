#include <Packages/Uintah/Core/Grid/GeomPiece/FileGeometryPiece.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
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
  ps->get("var1",d_var_name1);
  ps->get("var2",d_var_name2);
  ps->get("var3",d_var_name3);
  d_var1_bool = false;
  d_var2_bool = false;
  d_var3_bool = false;
  if(d_var_name1 == "p.volume"){
    d_var1_bool=true;
  }
  if(d_var_name2 == "p.externalforce"){
    d_var2_bool=true;
  }
  if(d_var_name3 == "p.fiberdir"){
    d_var3_bool=true;
  }

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
}

FileGeometryPiece::~FileGeometryPiece()
{
}

FileGeometryPiece* FileGeometryPiece::clone()
{
  return scinew FileGeometryPiece(*this);
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

void FileGeometryPiece::readPoints(int pid)
{
  char fnum[5];
  sprintf(fnum,".%d",pid);
  string file_name = d_file_name+fnum;
  ifstream source(file_name.c_str());
  if (!source ){
    throw ProblemSetupException("ERROR: opening MPM geometry file:  The file must be in the same directory as sus");
  }

  double x,y,z,vol,fx,fy,fz,fibx,fiby,fibz;
  double minx,miny,minz,maxx,maxy,maxz;
  source >> minx >> miny >> minz >> maxx >> maxy >> maxz;
  Point min(minx,miny,minz),max(maxx,maxy,maxz);
  Vector fudge(1.e-5,1.e-5,1.e-5);
  min = min - fudge;
  max = max + fudge;
  d_box = Box(min,max);

  if(d_var1_bool==true && d_var2_bool == false && d_var3_bool == false){
    while(source >> x >> y >> z >> vol) {
      d_points.push_back(Point(x,y,z));
      d_volume.push_back(vol);
    }
  }
  if(d_var1_bool==false && d_var2_bool == true && d_var3_bool == false){
    while(source >> x >> y >> z >> fx >> fy >> fz) {
      d_points.push_back(Point(x,y,z));
      d_forces.push_back(Vector(-fx,-fy,-fz));
    }
  }
  if(d_var1_bool==false && d_var2_bool == true && d_var3_bool == true){
    while(source >> x >> y >> z >> fx >> fy >> fz >> fibx >> fiby >> fibz) {
      d_points.push_back(Point(x,y,z));
      d_forces.push_back(Vector(-fx,-fy,-fz));
      d_fiberdirs.push_back(Vector(fibx,fiby,fibz));
    }
  }
  if(d_var1_bool==false && d_var2_bool == false && d_var3_bool == true){
    while(source >> x >> y >> z >> fibx >> fiby >> fibz) {
      d_points.push_back(Point(x,y,z));
      d_fiberdirs.push_back(Vector(fibx,fiby,fibz));
    }
  }
  if(d_var1_bool==true && d_var2_bool == true && d_var3_bool == true){
    while(source >> x >> y >> z >> vol >> fx >> fy >> fz >> fibx >> fiby >> fibz) {
      d_volume.push_back(vol);
      d_points.push_back(Point(x,y,z));
      d_forces.push_back(Vector(-fx,-fy,-fz));
      d_fiberdirs.push_back(Vector(fibx,fiby,fibz));
    }
  }
  if(d_var1_bool==false && d_var2_bool == false && d_var3_bool == false){
    while (source >> x >> y >> z) {
      d_points.push_back(Point(x,y,z));
    }
  }
  source.close();
}

int FileGeometryPiece::createPoints()
{
  cout << "You should be reading points .. not creating them" << endl;  
  return 0;
}
