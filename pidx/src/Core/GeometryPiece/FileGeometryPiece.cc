/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include <Core/GeometryPiece/FileGeometryPiece.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Util/Endian.h>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace Uintah;
using namespace SCIRun;
using std::cerr;
using std::cout;
using std::endl;

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
      warn << "ERROR: opening MPM geometry file (" << file_name+")\n" 
           << "In the ups file you've specified that the file format is bin or " << fileType << "\n"
           << "However this is a text file.\n";
      throw ProblemSetupException(warn.str(),__FILE__, __LINE__);
    }
  } else{
    // the file is binary
    if( fileType != "bin" && fileType != "lsb" && fileType != "msb" ){
      std::ostringstream warn;
      warn << "ERROR: opening MPM geometry file (" << file_name+")\n" 
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
  for(list<string>::const_iterator vit(d_vars.begin());vit!=d_vars.end();vit++) {
    if       (*vit=="p.volume") {
      proc0cout << " and volume";
    } else if(*vit=="p.temperature") {
      proc0cout << " and  temperature";
    } else if(*vit=="p.color"){
      proc0cout << " and  color";
    } else if(*vit=="p.externalforce") {
      proc0cout << " and  externalforce";
    } else if(*vit=="p.fiberdir") {
      proc0cout << " and  fiberdirn";
    } else if(*vit=="p.velocity") {
      proc0cout << " and  velocity";
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
      throw ProblemSetupException("ERROR: opening MPM geometry file '"+file_name+"'\nFailed to find points file",
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
      throw ProblemSetupException("ERROR: opening MPM geometry file '"+d_file_name+"'\nFailed to find points file",
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
FileGeometryPiece::inside(const Point& p) const
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
  double x1,x2,x3;
  //__________________________________
  //  TEXT FILE
  if(d_file_format=="text") {
    double v1,v2,v3;
    
    // line always starts with coordinates
    is >> x1 >> x2 >> x3;
    if(is.eof()){
     return false; // out of points
    }
    d_points.push_back(Point(x1,x2,x3));
    
    for(list<string>::const_iterator vit(d_vars.begin());vit!=d_vars.end();vit++) {
      if (*vit=="p.volume") {
        if(is >> v1) {
          d_volume.push_back(v1);
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
      } else if(*vit=="p.velocity") {
        if(is >> v1 >> v2 >> v3){
          d_velocity.push_back(Vector(v1,v2,v3));
        }
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
