/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#ifndef __GEOMETRY_OBJECT_H__
#define __GEOMETRY_OBJECT_H__

#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include   <list>
#include   <string>
#include   <map>
#include   <sstream>

#include <Core/Exceptions/InternalError.h>
namespace Uintah {

class GeometryPiece;

using namespace SCIRun;
using std::string;
using std::list;
using std::map;

/**************************************
	
CLASS
   GeometryObject
	
   Short description...
	
GENERAL INFORMATION
	
   GeometryObject.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   GeometryObject
	
DESCRIPTION
   Long description...
	
WARNING
	
****************************************/

class GeometryObject {
 
public:
  enum DataType
  {
    IntVector, Vector, Point, Double, Integer
  };
  struct DataItem
  {
    string name;
    DataType type;
    DataItem(string name, DataType datatype) : name(name), type(datatype) {};
  };
  //////////
  // Insert Documentation Here:
  GeometryObject(GeometryPieceP piece, ProblemSpecP&,list<DataItem>& data);

  //////////
  // Insert Documentation Here:
  ~GeometryObject() {}

  void outputProblemSpec(ProblemSpecP& ps);

  //////////
  // Insert Documentation Here:
  GeometryPieceP getPiece() const {
    return d_piece;
  }

  double getInitialData_double(const string& data_string) {
    if(d_double_data.find(data_string)==d_double_data.end())
    {
      std::stringstream msg;
      msg << "Geometry Object string '" << data_string << "' was not read during problemSetup";
      throw InternalError(msg.str(),__FILE__,__LINE__);
    }
    return d_double_data[data_string];
  }
  
  int getInitialData_int(const string& data_string) {
    if(d_int_data.find(data_string)==d_int_data.end())
    {
      std::stringstream msg;
      msg << "Geometry Object string '" << data_string << "' was not read during problemSetup";
      throw InternalError(msg.str(),__FILE__,__LINE__);
    }
    return d_int_data[data_string];
  }
  
  Uintah::Point getInitialData_Point(const string& data_string) {
    if(d_point_data.find(data_string)==d_point_data.end())
    {
      std::stringstream msg;
      msg << "Geometry Object string '" << data_string << "' was not read during problemSetup";
      throw InternalError(msg.str(),__FILE__,__LINE__);
    }
    return d_point_data[data_string];
  }
  
  Uintah::Vector getInitialData_Vector(const string& data_string) {
    if(d_vector_data.find(data_string)==d_vector_data.end())
    {
      std::stringstream msg;
      msg << "Geometry Object string '" << data_string << "' was not read during problemSetup";
      throw InternalError(msg.str(),__FILE__,__LINE__);
    }
    return d_vector_data[data_string];
  }
  
  Uintah::IntVector getInitialData_IntVector(const string& data_string) {
    if(d_intvector_data.find(data_string)==d_intvector_data.end())
    {
      std::stringstream msg;
      msg << "Geometry Object string '" << data_string << "' was not read during problemSetup";
      throw InternalError(msg.str(),__FILE__,__LINE__);
    }
    return d_intvector_data[data_string];
  }

private:
  GeometryPieceP     d_piece;
  map<string,int>    d_int_data;
  map<string,double> d_double_data;
  map<string,Uintah::Vector> d_vector_data;
  map<string,Uintah::IntVector> d_intvector_data;
  map<string,Uintah::Point>  d_point_data;

};

} // End namespace Uintah
      
#endif // __GEOMETRY_OBJECT_H__

