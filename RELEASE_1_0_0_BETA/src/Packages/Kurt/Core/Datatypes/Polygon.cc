#include <Packages/Kurt/Core/Datatypes/Polygon.h>
#include <Core/Containers/String.h>
#include <iostream>
using std::cerr;
using std::ostream;

namespace Kurt {
using namespace SCIRun;


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



clString
Polygon::string() const
{
  vector<Point>::const_iterator i;
  clString r("");
  r += "( ";
  for( i = vertices.begin(); i != vertices.end(); i++ ){
     r+=  i->string() + ", ";
  }
  r += ")";
  return r;
}

std::ostream& operator<<(std::ostream& os, const Polygon& p)
{
  os << p.string();
  return os;
}


} // End namespace Kurt

