/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/GeometryPiece/FileGeometryPiece.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Matrix3.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/Endian.h>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace Uintah;
using namespace std;

const string FileGeometryPiece::TYPE_NAME = "file";

static string
numbered_str(const string & s, int is)
{
  std::ostringstream b;
  b << s << is;
  return b.str();
}
//______________________________________________________________________
//  bulletproofing
void FileGeometryPiece::checkFileType(std::ifstream & source, string& fileType, string& file_name){

  int c;
  while((c = source.get()) != EOF && c <= 127) ;

  if(c == EOF ) {  
    // the file is ascii
    if( fileType != "text" ){
      std::ostringstream warn;
      warn << "ERROR: opening geometry file (" << file_name+")\n" 
           << "In the ups file you've specified that the file format is bin or " << fileType << "\n"
           << "However this is a text file.\n";
      throw ProblemSetupException(warn.str(),__FILE__, __LINE__);
    }
  } else{
    // the file is binary
    if( fileType != "bin" && fileType != "lsb" && fileType != "msb" ){
      std::ostringstream warn;
      warn << "ERROR: opening geometry file (" << file_name+")\n" 
           << "In the ups file you've specified that the file format is bin or " << fileType << "\n"
           << "However this is a binary file.  Please correct this inconsistency.\n";
      throw ProblemSetupException(warn.str(),__FILE__, __LINE__);
    }
  }
  
  // Reset read pointer to beginning of file
  source.clear();
  source.seekg (0, std::ios::beg);

  // Check that file can now be read.
  ASSERT(!source.eof());
}
//______________________________________________________________________
//
FileGeometryPiece::FileGeometryPiece( ProblemSpecP & ps )
{
  name_ = "Unnamed " + TYPE_NAME + " from PS";
  ps->require("name",d_file_name);

  for(ProblemSpecP varblock = ps->findBlock("var");
      varblock;varblock=varblock->findNextBlock("var")) {
    string next_var_name("");
    varblock->get(next_var_name);
    d_vars.push_back(next_var_name);
  }
  
  proc0cout << "File Geometry Piece: reading positions ";
  for(list<string>::const_iterator vit(d_vars.begin());vit!=d_vars.end();vit++){
    if       (*vit=="p.volume") {
      proc0cout << " and volume";
    } else if(*vit=="p.temperature") {
      proc0cout << " and temperature";
    } else if(*vit=="p.color"){
      proc0cout << " and color";
    } else if(*vit=="p.externalforce") {
      proc0cout << " and externalforce";
    } else if(*vit=="p.fiberdir") {
      proc0cout << " and fiberdirn";
    } else if(*vit=="p.rvec1") {
      proc0cout << " and rvec1";
    } else if(*vit=="p.rvec2") {
      proc0cout << " and rvec2";
    } else if(*vit=="p.rvec3") {
      proc0cout << " and rvec3";
    } else if(*vit=="p.velocity") {
      proc0cout << " and velocity";
    }
  }
  proc0cout << endl;
  
  ps->getWithDefault("format",d_file_format,"text");
  if (d_file_format=="bin"){   
    d_file_format = isLittleEndian()?"lsb":"msb";
  }

  ps->getWithDefault("usePFS",d_usePFS,true);

  Point min(1e30,1e30,1e30), max(-1e30,-1e30,-1e30);
  if(d_usePFS){
    // We must first read in the min and max from file.0 so
    // that we can determine the BoundingBox for the geometry
    string file_name = numbered_str(d_file_name+".", 0);
    std::ifstream source(file_name.c_str());
    if (!source ){
      throw ProblemSetupException("ERROR: opening geometry file '"+file_name+"'\nFailed to find points file",
                                  __FILE__, __LINE__);
    }

    // bulletproofing
    checkFileType(source, d_file_format, file_name);
    // find the bounding box.

    read_bbox(source, min, max);
    source.close();
  } else {
    // If we do not use PFS then we should read the entire points file now.

    std::ifstream source(d_file_name.c_str());
    if (!source ){
      throw ProblemSetupException("ERROR: opening geometry file '"+d_file_name+"'\nFailed to find points file",
                                  __FILE__, __LINE__);
    }

    // bulletproofing
    checkFileType(source, d_file_format, d_file_name);

    // While the file is read the max and min are updated.
    while(source) {
      read_line(source, min, max);
    }
    source.close();
  }



  Vector fudge(1.e-5,1.e-5,1.e-5);
  min = min - fudge;
  max = max + fudge;
  d_box = Box(min,max);  
}
//______________________________________________________________________
//
FileGeometryPiece::FileGeometryPiece(const string& /*file_name*/)
{
  name_ = "Unnamed " + TYPE_NAME + " from file_name";
}

FileGeometryPiece::~FileGeometryPiece()
{
}
//______________________________________________________________________
//
void
FileGeometryPiece::outputHelper( ProblemSpecP & ps ) const
{
  ps->appendElement("name",d_file_name);
  ps->appendElement("format",d_file_format);
  
  for (list<string>::const_iterator it = d_vars.begin(); it != d_vars.end(); it++) {
    ps->appendElement("var",*it);
  }
}
//______________________________________________________________________
//
GeometryPieceP
FileGeometryPiece::clone() const
{
  return scinew FileGeometryPiece(*this);
}
//______________________________________________________________________
//
bool
FileGeometryPiece::inside(const Point& p, const bool defVal=false) const
{
  //Check p with the lower coordinates
  if (p == Max(p,d_box.lower()) && p == Min(p,d_box.upper()) )
    return true;
  else
    return false;
}
//______________________________________________________________________
//
Box
FileGeometryPiece::getBoundingBox() const
{
  return d_box;
}

//______________________________________________________________________
//
void
FileGeometryPiece::read_bbox(std::istream & source, Point & min, 
                             Point & max) const
{
  if(d_file_format=="text") {
    source >> min(0) >> min(1) >> min(2) >> max(0) >> max(1) >> max(2);
   
  } else {
    // FIXME: never changes, should save this !
    const bool iamlittle = isLittleEndian();
    const bool needflip = (iamlittle && (d_file_format=="msb")) || (!iamlittle && (d_file_format=="lsb"));
    double t;
    source.read((char *)&t, sizeof(double)); if(needflip) swapbytes(t); min(0) = t;
    source.read((char *)&t, sizeof(double)); if(needflip) swapbytes(t); min(1) = t;
    source.read((char *)&t, sizeof(double)); if(needflip) swapbytes(t); min(2) = t;
    source.read((char *)&t, sizeof(double)); if(needflip) swapbytes(t); max(0) = t;
    source.read((char *)&t, sizeof(double)); if(needflip) swapbytes(t); max(1) = t;
    source.read((char *)&t, sizeof(double)); if(needflip) swapbytes(t); max(2) = t;
  }
}
//______________________________________________________________________
//
bool
FileGeometryPiece::read_line(std::istream & is, Point & xmin, Point & xmax)
{
  double inf = std::numeric_limits<double>::infinity();
  double x1=inf;
  double x2=inf;
  double x3=inf;

  // CPTI and CPDI can pass the size matrix columns containing rvec1, rvec2, rvec3
  // Other interpolators will default to grid spacing and default orientation
  Matrix3 size(d_DX.x(),0.,0.,0.,d_DX.y(),0.,0.,0.,d_DX.z());
  // grid spacing for normalizing size
  Matrix3 gsize((1./d_DX.x()),0.,0.,0.,(1./d_DX.y()),0.,0.,0.,(1./d_DX.z()));
  bool file_has_size=false;
  bool file_has_volume=false;
 
  //__________________________________
  //  TEXT FILE
  if(d_file_format=="text") {
    double v1,v2,v3,vol;
    
    // line always starts with coordinates
    is >> x1 >> x2 >> x3;
    if(is.eof()){
     return false; // out of points
    }
    // Particle coordinates
    d_points.push_back(Point(x1,x2,x3));
    

    for(list<string>::const_iterator vit(d_vars.begin());vit!=d_vars.end();vit++) {
      if (*vit=="p.volume") {
        if(is >> vol) {
          //cout << "v1 = " << v1 << endl;
          d_volume.push_back(vol);
          file_has_volume=true;
        }
      } else if(*vit=="p.temperature") {
        if(is >> v1){
          d_temperature.push_back(v1);
        }
      } else if(*vit=="p.color") {
        if(is >> v1){
          d_color.push_back(v1);
        }
      } else if(*vit=="p.externalforce") {
        if(is >> v1 >> v2 >> v3){
          d_forces.push_back(Vector(v1,v2,v3));
        }
      } else if(*vit=="p.fiberdir") {
        if(is >> v1 >> v2 >> v3){
          d_fiberdirs.push_back(Vector(v1,v2,v3));
        }  
      } else if(*vit=="p.rvec1") {
        if(is >> v1 >> v2 >> v3){
          d_rvec1.push_back(Vector(v1,v2,v3));
          size(0,0)=v1;
          size(1,0)=v2;
          size(2,0)=v3;
          file_has_size=true;
        }  
      } else if(*vit=="p.rvec2") {
        if(is >> v1 >> v2 >> v3){
          d_rvec2.push_back(Vector(v1,v2,v3));
          size(0,1)=v1;
          size(1,1)=v2;
          size(2,1)=v3;
          file_has_size=true;
        }  
      } else if(*vit=="p.rvec3") {
        if(is >> v1 >> v2 >> v3){
          d_rvec3.push_back(Vector(v1,v2,v3));
          size(0,2)=v1;
          size(1,2)=v2;
          size(2,2)=v3;
          file_has_size=true;
        }  
      } else if(*vit=="p.velocity") {
        if(is >> v1 >> v2 >> v3){
          d_velocity.push_back(Vector(v1,v2,v3));
        }
      }

      // If the volume is provided, but not the size, the particle
      // size needs to be adjusted to be consistent with the volume
      if(file_has_volume && !file_has_size){
        double cbrtVol = cbrt(vol);
        double s1,s2,s3;
        s1=1.*cbrtVol; s2=0.; s3=0.;
        d_rvec1.push_back(Vector(s1,s2,s3));
        size(0,0)=s1;
        size(1,0)=s2;
        size(2,0)=s3;
        s1=0.; s2=1.*cbrtVol; s3=0.;
        d_rvec2.push_back(Vector(s1,s2,s3));
        size(0,1)=s1;
        size(1,1)=s2;
        size(2,1)=s3;
        s1=0.; s2=0.; s3=1.*cbrtVol;
        d_rvec3.push_back(Vector(s1,s2,s3));
        size(0,2)=s1;
        size(1,2)=s2;
        size(2,2)=s3;
        file_has_size=true;
      }

      if(!is) {
        std::ostringstream warn;
        warn << "Failed while reading point text point file \n"
             << "Position: "<< Point(x1,x2,x3) << "\n"; 
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
    }

    //__________________________________
    //  BINARY FILE
  } else if(d_file_format=="lsb" || d_file_format=="msb") {
    // read unformatted binary numbers
    
    double v[3];
    
    // never changes, should save this !
    const bool iamlittle = isLittleEndian();
    const bool needflip = (iamlittle && (d_file_format=="msb")) || (!iamlittle && (d_file_format=="lsb"));

    is.read((char*)&x1, sizeof(double)); 
    
    if(!is){
      return false;  // out of points
    }
    
    is.read((char*)&x2, sizeof(double));
    is.read((char*)&x3, sizeof(double));
    if(needflip) {
      swapbytes(x1);
      swapbytes(x2);
      swapbytes(x3);
    }
    d_points.push_back(Point(x1,x2,x3));
    //__________________________________
    //
    for(list<string>::const_iterator vit(d_vars.begin());vit!=d_vars.end();vit++) {
      if (*vit=="p.volume") {
        if(is.read((char*)&v[0], sizeof(double))) {
          if(needflip){ 
            swapbytes(v[0]);
          }
          d_volume.push_back(v[0]);
          file_has_volume=true;
        }
      } else if(*vit=="p.temperature") {
        if(is.read((char*)&v[0], sizeof(double))) {
          if(needflip){
            swapbytes(v[0]);
          }
          d_temperature.push_back(v[0]);
        }
      } else if(*vit=="p.color") {
        if(is.read((char*)&v[0], sizeof(double))) {
          if(needflip){
            swapbytes(v[0]);
          }
          d_color.push_back(v[0]);
        }
      } else if(*vit=="p.externalforce") {
        if(is.read((char*)&v[0], sizeof(double)*3)) {
          if(needflip) {
            swapbytes(v[0]);
            swapbytes(v[1]);
            swapbytes(v[2]);
          }
          d_forces.push_back(Vector(v[0],v[1],v[2]));
        }
      } else if(*vit=="p.fiberdir") {
        if(is.read((char*)&v[0], sizeof(double)*3)) {
          if(needflip) {
            swapbytes(v[0]);
            swapbytes(v[1]);
            swapbytes(v[2]);
          }
          d_fiberdirs.push_back(Vector(v[0],v[1],v[2]));
        }
      } else if(*vit=="p.rvec1") {
        if(is.read((char*)&v[0], sizeof(double)*3)) {
          if(needflip) {
            swapbytes(v[0]);
            swapbytes(v[1]);
            swapbytes(v[2]);
          }
          d_rvec1.push_back(Vector(v[0],v[1],v[2]));
          size(0,0)=v[0];
          size(1,0)=v[1];
          size(2,0)=v[2];
          file_has_size=true;
        }
      } else if(*vit=="p.rvec2") {
        if(is.read((char*)&v[0], sizeof(double)*3)) {
          if(needflip) {
            swapbytes(v[0]);
            swapbytes(v[1]);
            swapbytes(v[2]);
          }
          d_rvec2.push_back(Vector(v[0],v[1],v[2]));
          size(0,1)=v[0];
          size(1,1)=v[1];
          size(2,1)=v[2];
          file_has_size=true;
        }
      } else if(*vit=="p.rvec3") {
        if(is.read((char*)&v[0], sizeof(double)*3)) {
          if(needflip) {
            swapbytes(v[0]);
            swapbytes(v[1]);
            swapbytes(v[2]);
          }
          d_rvec3.push_back(Vector(v[0],v[1],v[2]));
          size(0,2)=v[0];
          size(1,2)=v[1];
          size(2,2)=v[2];
          file_has_size=true;
        }
      } else if(*vit=="p.velocity") {
        if(is.read((char*)&v[0], sizeof(double)*3)) {
          if(needflip) {
            swapbytes(v[0]);
            swapbytes(v[1]);
            swapbytes(v[2]);
          }
          d_velocity.push_back(Vector(v[0],v[1],v[2]));
        }  
      }          
      if(!is){
        std::ostringstream warn;
        warn << "Failed while reading point text point file \n"
             << "Position: "<< Point(x1,x2,x3) << "\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
      
    }
  }
  if(file_has_size){
    // CPTI and CPDI populate size matrix with Rvectors defining the particle domain in columns
    // proc0cout << endl << "<res> is ignored for CPDI and CPTI particle domain import." << endl;
    double vol = size.Determinant();
    // CPDI and CPTI check for negative volumes due to Rvector order
    if (vol < 0) {
      // switch r2 and r3 in size to get a positive volume 
      Matrix3 tmpsize(size(0,0),size(0,2),size(0,1),
                      size(1,0),size(1,2),size(1,1),
                      size(2,0),size(2,2),size(2,1));
      vol = tmpsize.Determinant();
      // normalize size matrix by grid cell dimensions 
      size = tmpsize;
    }
    // CPDI and CPTI volumes determined prior to grid cell size normalization
    if(d_useCPTI){
      vol=size.Determinant()/6.0;
    }
    // Imported volumes override calculated volume
    if(!file_has_volume){
      d_volume.push_back(vol);
    }
    // Size matrix normalized by the grid spacing for interpolators
    size = gsize*size;
    d_size.push_back(size);
  }

  xmin = Min(xmin, Point(x1,x2,x3));
  xmax = Max(xmax, Point(x1,x2,x3));
  return true;
}
//______________________________________________________________________
//
void
FileGeometryPiece::readPoints(int patchID)
{
  if(d_usePFS){
    std::ifstream source;
  
    Point minpt( 1e30, 1e30, 1e30);
    Point maxpt(-1e30,-1e30,-1e30);
  
    string file_name;
    char fnum[5];
  
    sprintf(fnum,".%d",patchID);
    file_name = d_file_name+fnum;

    source.open(file_name.c_str());

    // bulletproofing
    checkFileType(source, d_file_format, file_name);
  
    // ignore the first line of the file;
    // this has already been processed
    Point notUsed;
    read_bbox(source, notUsed, notUsed);

    while(source) {
      read_line(source, minpt, maxpt);
    }

  }
}
//______________________________________________________________________
//
unsigned int
FileGeometryPiece::createPoints()
{
  cerr << "You should be reading points .. not creating them" << endl;  
  return 0;
}
//______________________________________________________________________
//
void 
FileGeometryPiece::setCpti(bool useCPTI)
{
  d_useCPTI = useCPTI;
}
