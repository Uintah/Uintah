#include <Packages/Uintah/Core/GeometryPiece/FileGeometryPiece.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>
#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

static string numbered_str(const string & s, int is)
{
  ostringstream b;
  b << s << is;
  return b.str();
}

FileGeometryPiece::FileGeometryPiece(ProblemSpecP& ps)
{
  setName("file");
  ps->require("name",d_file_name);
  
  for(ProblemSpecP varblock = ps->findBlock("var");
      varblock;varblock=varblock->findNextBlock("var")) {
    string next_var_name("EMPTY!");
    varblock->get(next_var_name);
    if      (next_var_name=="p.volume")        d_vars.push_back(IVvolume);
    else if (next_var_name=="p.externalforce") d_vars.push_back(IVextforces);
    else if (next_var_name=="p.fiberdir")      d_vars.push_back(IVfiberdirn);
    else 
      throw ProblemSetupException("Unexpected field variable of '"+next_var_name+"'");
  }
  
  cerr << "reading: positions";
  for(list<InputVar>::const_iterator vit(d_vars.begin());vit!=d_vars.end();vit++) {
    if       (*vit==IVvolume) {
      cerr << " volume";
    } else if(*vit==IVextforces) {
      cerr << " externalforce";
    } else if(*vit==IVfiberdirn) {
      cerr << " fiberdirn";
    }
  }
  cerr << endl;
  
  d_file_format = FFSplit; // default expects input to have been been processed with pfs

  string fformat_txt;
  if(ps->get("format",fformat_txt)) {
    if     (fformat_txt=="split") d_file_format = FFSplit;
    else if(fformat_txt=="text")  d_file_format = FFText;
    else if(fformat_txt=="lsb")   d_file_format = FFLSBBin;
    else if(fformat_txt=="msb")   d_file_format = FFMSBBin;
    else if(fformat_txt=="gzip")  d_file_format = FFGzip;
    else
      throw ProblemSetupException("Unexpected file geometry format");
  }
  
  if(d_file_format==FFSplit) {
    // read points now to find bounding box
    
    // We must first read in the min and max from file.0 so
    // that we can determine the BoundingBox for the geometry
    string file_name = numbered_str(d_file_name+".", 0);
    ifstream source(file_name.c_str());
    if (!source ){
      throw ProblemSetupException("ERROR: opening MPM geometry file '"+file_name+"'\nFailed to find point file");
    }
    
    Point min, max;
    source >> min(0) >> min(1) >> min(2) >> max(0) >> max(1) >> max(2);
    source.close();
    
    Vector fudge(1.e-5,1.e-5,1.e-5);
    min = min - fudge;
    max = max + fudge;
    d_box = Box(min,max);
    cerr << "bbox of " << min << " " << max << endl;
    
  } else {
    // if not using split format, have to read points now to find bounding box
    readPoints(-1);
  }
}

FileGeometryPiece::FileGeometryPiece(const string& /*file_name*/)
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

bool
FileGeometryPiece::read_line(istream & is, Point & xmin, Point & xmax)
{
  double x1,x2,x3;
  if(d_file_format==FFSplit || d_file_format==FFText) {
    double v1,v2,v3;
    
    // line always starts with coordinates
    is >> x1 >> x2 >> x3;
    if(is.eof()) return false; // out of points
    d_points.push_back(Point(x1,x2,x3));
    
    for(list<InputVar>::const_iterator vit(d_vars.begin());vit!=d_vars.end();vit++) {
      if       (*vit==IVvolume) {
        if(is >> v1) d_volume.push_back(v1);
      } else if(*vit==IVextforces) {
        if(is >> v1 >> v2 >> v3) d_forces.push_back(Vector(v1,v2,v3));
      } else if(*vit==IVfiberdirn) {
        if(is >> v1 >> v2 >> v3) d_fiberdirs.push_back(Vector(v1,v2,v3));
      }
      if(!is)
        throw ProblemSetupException("Failed while reading point text point file");
    }
    
  } else if(d_file_format==FFLSBBin || d_file_format==FFMSBBin) {
    // read unformatted binary numbers
    
    double v[3];
    
    // never changes, should save this !
    const bool iamlittle = isLittleEndian();
    const bool needflip = (iamlittle && (d_file_format==FFMSBBin)) || (!iamlittle && (d_file_format==FFLSBBin));
    
    is.read((char*)&x1, sizeof(double)); if(!is) return false;
    is.read((char*)&x2, sizeof(double));
    is.read((char*)&x3, sizeof(double));
    
    d_points.push_back(Point(x1,x2,x3));
    
    for(list<InputVar>::const_iterator vit(d_vars.begin());vit!=d_vars.end();vit++) {
      if (*vit==IVvolume) {
        if(is.read((char*)&v[0], sizeof(double))) {
          if(needflip) swapbytes(v[0]);
          d_volume.push_back(v[0]);
      } else if(*vit==IVextforces) {
        if(is.read((char*)&v[0], sizeof(double)*3)) {
          if(needflip) {
            swapbytes(v[0]);
            swapbytes(v[1]);
            swapbytes(v[2]);
          }
          d_forces.push_back(Vector(v[0],v[1],v[2]));
        }
      } else if(*vit==IVfiberdirn) {
        if(is.read((char*)&v[0], sizeof(double)*3)) {
          if(needflip) {
            swapbytes(v[0]);
            swapbytes(v[1]);
            swapbytes(v[2]);
          }
          d_fiberdirs.push_back(Vector(v[0],v[1],v[2]));
        }
      }
      if(!is)
        throw ProblemSetupException("Failed while reading point text point file");
      }
    }
    
  } else if(d_file_format==FFGzip) {
    throw ProblemSetupException("Sorry - gzip not implemented !");
  }
  
  xmin = Min(xmin, Point(x1,x2,x3));
  xmax = Max(xmax, Point(x1,x2,x3));
  return true;
}

void FileGeometryPiece::readPoints(int pid)
{
  if(d_file_format!=FFSplit && pid!=-1) return; // already read points

  ifstream source;
  
  Point minpt( 1e30, 1e30, 1e30);
  Point maxpt(-1e30,-1e30,-1e30);
  
  string file_name;
  if(d_file_format==FFSplit) {
    // pre-processed split point files
    char fnum[5];
    sprintf(fnum,".%d",pid);
    file_name = d_file_name+fnum;
    
    source.open(file_name.c_str());
    
    // re-reading these
    source >> minpt(0) >> minpt(1) >> minpt(2) >> maxpt(0) >> maxpt(1) >> maxpt(2);
    
  } else {
    file_name = d_file_name;
    source.open(file_name.c_str());
  }    
  
  if (!source ){
    throw ProblemSetupException("ERROR: opening MPM point file '"+file_name+"'\n:  The file must be in the same directory as sus");
  }
  
  int readpts = 0;
  while(source) {
    read_line(source, minpt, maxpt);
    if(source)
      readpts++;
  }
  
  if(d_file_format!=FFSplit) {
    Vector fudge(1.e-5,1.e-5,1.e-5);
    d_box = Box(minpt-fudge,maxpt+fudge);
  }
  
  cerr << "   Read " << readpts << " from file " << endl;
  cerr << "   Box = " << d_box << endl;
}

int FileGeometryPiece::createPoints()
{
  cerr << "You should be reading points .. not creating them" << endl;  
  return 0;
}
