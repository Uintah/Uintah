/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

#include <Core/Geometry/Polygon.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using std::ostream;

namespace SCIRun {



Polygon::Polygon(const Polygon& p)
  {
  vector<Point>::const_iterator i,j;
  for( i = p.vertices.begin(), j = p.texcoords.begin();
       i != p.vertices.end(); i++, j++){
    vertices.push_back(*i);
    texcoords.push_back(*j);
    
  }
}

Polygon::Polygon(const vector<Point>& p, const vector<Point>& t)
  {
  vector<Point>::const_iterator i,j;
  for( i = p.begin(), j = t.begin(); i != p.end(); i++, j++ ){
    vertices.push_back(*i);
    texcoords.push_back(*j);
  }
}

Polygon::Polygon(const Point *p, const Point *t, int nPoints)
{
  int i;
  for( i = 0; i < nPoints; i++){
    vertices.push_back(p[i]);
    texcoords.push_back(t[i]);
  }
}

bool 
Polygon::operator==(const Polygon& p) const
{
  if( p.vertices.size() != vertices.size())
    return false;
  vector<Point>::const_iterator i,j;
  for( i = vertices.begin(), j = p.vertices.begin();
       i != vertices.end(); i++, j++){
    if( *i != *j )
      return false;
  }
  return true;
}
 
bool
Polygon::operator!=(const Polygon& p) const
{
  return !(*this == p );
}


Polygon&
Polygon::operator=(const Polygon& p)
{
  vector<Point>::const_iterator i;
  vertices.clear();
  for( i = p.vertices.begin(); i != p.vertices.end(); i++ ){
    vertices.push_back(*i);
  }
  return *this;
}

const Point& 
Polygon::operator[](int i) const
{
  return vertices[i];
}

const Point&
Polygon::getVertex(int i) const
{
  return vertices[i];
}
const Point&
Polygon::getTexCoord(int i) const
{
  return texcoords[i];
}



string
Polygon::get_string() const
{
  vector<Point>::const_iterator i;
  string r = "( ";
  for( i = vertices.begin(); i != vertices.end(); i++ )
  {
    r+= i->get_string();
    if (i < vertices.end() - 1) { r += ", "; }
  }
  r += ")";
  return r;
}

std::ostream& operator<<(std::ostream& os, const Polygon& p)
{
  os << p.get_string();
  return os;
}



} // End namespace SCIRun
