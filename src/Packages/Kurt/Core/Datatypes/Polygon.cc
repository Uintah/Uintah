#include <Kurt/Datatypes/Polygon.h>
#include <SCICore/Containers/String.h>
#include <iostream>
using std::cerr;
using std::ostream;

namespace SCICore {
namespace Geometry {

  using SCICore::Containers::clString;


Polygon::Polygon(const Polygon& p)
  {
  vector<Point>::const_iterator i;
  for( i = p.vertices.begin(); i != p.vertices.end(); i++ ){
    vertices.push_back(*i);
  }
}

Polygon::Polygon(const vector<Point>& p)
  {
  vector<Point>::const_iterator i;
  for( i = p.begin(); i != p.end(); i++ ){
    vertices.push_back(*i);
  }
}

Polygon::Polygon(const Point *p, int nPoints)
{
  int i;
  for( i = 0; i < nPoints; i++)
    vertices.push_back(p[i]);
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



} //Geometry
} //SCICore
