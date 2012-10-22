/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#include <Core/Geometry/Polygon.h>
#include <iostream>

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
