#include <Packages/Wangxl/Core/Datatypes/Mesh/DCell.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/DVertex.h>
#include <Core/Geometry/Vector.h>

namespace Wangxl {

using namespace SCIRun;

/*void DCell::set_ratio() {
  double a0,a1,a2,b0,b1,b2,c0,c1,c2;
  double det,r,len;
  int i;
  Vector vec0, vec1, vec2, vec;
  Point a, b, c, d;
  a = vertex(0)->point();
  b = vertex(1)->point();
  c = vertex(2)->point();
  d = vertex(3)->point();
  vec0 = Cross(b-a,c-a)*pow((d-a).length(),2);
  vec1 = Cross(d-a,b-a)*pow((c-a).length(),2);
  vec2 = Cross(c-a,d-a)*pow((b-a).length(),2);
  a0 = b.x()-a.x(); a1 = b.y()-a.y(); a2 = b.z()-a.z();
  b0 = c.x()-a.x(); b1 = c.y()-a.y(); b2 = c.z()-a.z();
  c0 = d.x()-a.x(); c1 = d.y()-a.y(); c2 = d.z()-a.z();
  det = a0*b1*c2 + b0*c1*a2 + c0*a1*b2 - c0*b1*a2 - b0*a1*c2 - a0*c1*b2;
  vec = vec0+vec1;
  vec = vec+vec2;
  r = vec.length()/(2*det);
  d_center = a + vec/(2*det);
  len = std::min((a-c).length(),(b-d).length());
  for ( i = 0; i < 4; i++ ) {
    Point p0 = vertex(i)->point();
    Point p1 = vertex((i+1)&3)->point();
    len = std::min(len,(p0-p1).length());
  }
  d_ratio = r/len;
  }*/

void DCell::set_ratio() {
  double a0,a1,a2,b0,b1,b2,c0,c1,c2;
  double det,r,len;
  int i;
  Vector vec0, vec1, vec2, vec;
  Point a, b, c, d;
  Point p0, p1;
  a = vertex(0)->point();
  b = vertex(1)->point();
  c = vertex(2)->point();
  d = vertex(3)->point();
  vec0 = Cross(b-a,c-a)*pow((d-a).length(),2);
  vec1 = Cross(d-a,b-a)*pow((c-a).length(),2);
  vec2 = Cross(c-a,d-a)*pow((b-a).length(),2);
  a0 = b.x()-a.x(); a1 = b.y()-a.y(); a2 = b.z()-a.z();
  b0 = c.x()-a.x(); b1 = c.y()-a.y(); b2 = c.z()-a.z();
  c0 = d.x()-a.x(); c1 = d.y()-a.y(); c2 = d.z()-a.z();
  det = a0*b1*c2 + b0*c1*a2 + c0*a1*b2 - c0*b1*a2 - b0*a1*c2 - a0*c1*b2;
  vec = vec0+vec1;
  vec = vec+vec2;
  r = vec.length()/(2*det);
  //  d_center = Point(0,0,0);
  d_center = a + vec/(2*det);
  len = std::min((a-c).length(),(b-d).length());
  for ( i = 0; i < 4; i++ ) {
    p0 = vertex(i)->point();
    p1 = vertex((i+1)&3)->point();
    len = std::min(len,(p0-p1).length());
  }
  d_ratio = r/len;
}

}




