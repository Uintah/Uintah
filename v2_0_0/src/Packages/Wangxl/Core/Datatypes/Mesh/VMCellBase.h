#ifndef SCI_Wangxl_Datatypes_Mesh_VMCellBase_h
#define SCI_Wangxl_Datatypes_Mesh_VMCellBase_h

#include <iostream>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/VMVertexBase.h>
namespace Wangxl {

using namespace SCIRun;

class VMCellBase
{
public:

  // CONSTRUCTORS

  VMCellBase()
  {
    set_vertices();
    set_neighbors();
    d_ratio = 0;
  }

  VMCellBase(void* v0, void* v1, void* v2, void* v3)
  {
    set_vertices(v0, v1, v2, v3);
    set_neighbors();
    d_ratio = 0;
  }

  VMCellBase(void* v0, void* v1, void* v2, void* v3,
			    void* n0, void* n1, void* n2, void* n3)
  {
    set_vertices(v0, v1, v2, v3);
    set_neighbors(n0, n1, n2, n3);
    d_ratio = 0;
  }

  // ACCESS FUNCTIONS

  const void* vertex(int i) const
  {
    assert( i >= 0 && i <= 3 );
    return V[i];
  } 

  bool has_vertex(const void* v) const
  {
    return (V[0] == v) || (V[1] == v) || (V[2]== v) || (V[3]== v);
  }
    
  bool has_vertex(const void* v, int & i) const
    {
      if (v == V[0]) { i = 0; return true; }
      if (v == V[1]) { i = 1; return true; }
      if (v == V[2]) { i = 2; return true; }
      if (v == V[3]) { i = 3; return true; }
      return false;
    }
    
  int vertex_index(const void* v) const
  {
    if (v == V[0]) { return 0; }
    if (v == V[1]) { return 1; }
    if (v == V[2]) { return 2; }
    assert( v == V[3] );
    return 3;
  }

  void* neighbor(int i) const
  {
    assert( i >= 0 && i <= 3);
    return N[i];
  }
    
  bool has_neighbor(const void* n) const
  {
    return (N[0] == n) || (N[1] == n) || (N[2] == n) || (N[3] == n);
  }
    
  bool has_neighbor(const void* n, int & i) const
  {
    if(n == N[0]){ i = 0; return true; }
    if(n == N[1]){ i = 1; return true; }
    if(n == N[2]){ i = 2; return true; }
    if(n == N[3]){ i = 3; return true; }
    return false;
  }
    
  int cell_index(const void* n) const
  {
    if (n == N[0]) return 0;
    if (n == N[1]) return 1;
    if (n == N[2]) return 2;
    assert( n == N[3] );
    return 3;
  }
 
  // SETTING

  void set_vertex(int i, void* v)
  {
    assert( i >= 0 && i <= 3);
    V[i] = v;
  }
    
  void set_neighbor(int i, void* n)
  {
    assert( i >= 0 && i <= 3);
    N[i] = n;
  }

  void set_vertices()
  {
    V[0] = V[1] = V[2] = V[3] = 0;
  }
    
  void set_vertices(void* v0, void* v1, void* v2, void* v3)
  {
    V[0] = v0;
    V[1] = v1;
    V[2] = v2;
    V[3] = v3;
  }
    
  void set_neighbors()
  {
    N[0] = 0;
    N[1] = 0;
    N[2] = 0;
    N[3] = 0;
  }
    
  void set_neighbors(void* n0, void* n1, void* n2, void* n3)
  {
    N[0] = n0;
    N[1] = n1;
    N[2] = n2;
    N[3] = n3;
  }

  // CHECKING

  // the following trivial is_valid allows
  // the user of derived cell base classes 
  // to add their own purpose checking
  bool is_valid(bool, int ) const {return true;}

  void init() {d_ratio = 0.0;}


  //COMPUTE TETRA
  void set_tetra() {
    int i;
    if ( d_ratio != 0.0 ) return;
    double a0,a1,a2,b0,b1,b2,c0,c1,c2;
    double det,r,len;
    Vector vec0, vec1, vec2, vec;
    Point a, b, c, d;
    Point p0, p1;
    a = ((VMVertexBase*)vertex(0))->point();
    b = ((VMVertexBase*)vertex(1))->point();
    c = ((VMVertexBase*)vertex(2))->point();
    d = ((VMVertexBase*)vertex(3))->point();
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
      p0 = ((VMVertexBase*)vertex(i))->point();
      p1 = ((VMVertexBase*)vertex((i+1)&3))->point();
      len = std::min(len,(p0-p1).length());
    }
    d_ratio = r/len;    
  }

  void set_tetra0() {
    int i,j;
    //    if ( d_ratio != 0.0 ) return;
    double a0,a1,a2,b0,b1,b2,c0,c1,c2;
    double det,r,len, len0, len1;
    double vec0[3], vec1[3], vec2[3], vec[3];
    double a[3], b[3], c[3], d[3], ba[3], ca[3], da[3];
    double baca[3],daba[3],cada[3];
    double lenba2, lenca2, lenda2;
    double p0[3], p1[3];
    Point p;

    p = ((VMVertexBase*)vertex(0))->point();
    a[0] = p.x();a[1] = p.y();a[2] = p.z();
    p = ((VMVertexBase*)vertex(1))->point();
    b[0] = p.x();b[1] = p.y();b[2] = p.z();
    p = ((VMVertexBase*)vertex(2))->point();
    c[0] = p.x();c[1] = p.y();c[2] = p.z();
    p = ((VMVertexBase*)vertex(3))->point();
    d[0] = p.x();d[1] = p.y();d[2] = p.z();
    for ( i = 0; i < 3; i++ ) {
      ba[i] = b[i] - a[i];
      ca[i] = c[i] - a[i];
      da[i] = d[i] - a[i];
    }
    cross(ba,ca,baca); cross(da,ba,daba); cross(ca,da,cada);
    lenba2 = lenca2 = lenda2 = 0.0;
    for ( i = 0; i < 3; i++ ) {
      lenba2 += ba[i]*ba[i];
      lenca2 += ca[i]*ca[i];
      lenda2 += da[i]*da[i];
    }
    for ( i = 0; i < 3; i++ ) {
      vec0[i] = baca[i]*lenda2;
      vec1[i] = daba[i]*lenca2;
      vec2[i] = cada[i]*lenba2;
      vec[i] = vec0[i]+vec1[i]+vec2[i];
    }
    a0 = b[0]-a[0]; a1 = b[1]-a[1]; a2 = b[2]-a[2];
    b0 = c[0]-a[0]; b1 = c[1]-a[1]; b2 = c[2]-a[2];
    c0 = d[0]-a[0]; c1 = d[1]-a[1]; c2 = d[2]-a[2];
    det = a0*b1*c2 + b0*c1*a2 + c0*a1*b2 - c0*b1*a2 - b0*a1*c2 - a0*c1*b2;
    r = 0;
    for ( i = 0; i < 3; i++ ) {
      r += vec[i]*vec[i];
    }
    r = sqrt(r)/(2*det);
    for ( i = 0; i < 3; i++ ) {
      d_center0[i] = a[i] + vec[i]/(2*det);
    }
    len0 = len1 = 0;
    for ( i = 0; i < 3; i++ ) {
      len0 += ca[i]*ca[i];
      len1 += (b[i]-d[i])*(b[i]-d[i]);
    }
    len = std::min(len0,len1);
    for ( i = 0; i < 4; i++ ) {
      p = ((VMVertexBase*)vertex(i))->point();
      p0[0] = p.x(); p0[1] = p.y(); p0[2] = p.z();
      p = ((VMVertexBase*)vertex((i+1)&3))->point();
      p1[0] = p.x(); p1[1] = p.y(); p1[2] = p.z();
      len0 = 0;      
      for ( j = 0; j < 3; j++ ) len0 += (p0[j]-p1[j])*(p0[j]-p1[j]);
      if ( len0 < len ) len = len0;
    }
    d_ratio0 = r/sqrt(len);    
 }

  void cross(double v0[3], double v1[3], double v[3])
  {
    v[0] = v0[1]*v1[2]-v0[2]*v1[1];
    v[1] = v0[2]*v1[0]-v0[0]*v1[2];
    v[2] = v0[0]*v1[1]-v0[1]*v1[0];
  }


  double ratio() const { return d_ratio; }
  Point center() const { return d_center; }
  double ratio0() const { return d_ratio0; }
  void center0(double ct[3]) const {ct[0] = d_center0[0]; ct[1] = d_center0[1]; ct[2] = d_center0[2]; }

  Point get_point() const {
    Point a, b, c, d;
    Point p0, p1, pp0, pp1;
    double len, len0, len1;
    int i;
    a = ((VMVertexBase*)vertex(0))->point();
    b = ((VMVertexBase*)vertex(1))->point();
    c = ((VMVertexBase*)vertex(2))->point();
    d = ((VMVertexBase*)vertex(3))->point();    
    len0 = (a-c).length2();
    len1 = (b-d).length2();
    if ( len0 >= len1 ) {
      len = len0;
      p0 = a;
      p1 = c;
    }
    else {
      len = len1;
      p0 = b;
      p1 = d;
    }
    for ( i = 0; i < 4; i++ ) {
      pp0 = ((VMVertexBase*)vertex(i))->point();
      pp1 = ((VMVertexBase*)vertex((i+1)&3))->point();
      len0 = (pp0-pp1).length2();
      if ( len0 > len ) {
	len = len0;
	p0 = pp0;
	p1 = pp1;
      }
    }
    return Point((p0.asVector()+p1.asVector())/2);
  }

  Point centroid() {
    Point a, b, c, d;
    a = ((VMVertexBase*)vertex(0))->point();
    b = ((VMVertexBase*)vertex(1))->point();
    c = ((VMVertexBase*)vertex(2))->point();
    d = ((VMVertexBase*)vertex(3))->point();     
    return Point((a.asVector()+b.asVector()+c.asVector()+d.asVector())/4);
  }

private:
  void* N[4];
  void* V[4];

  Point d_center;
  double d_ratio;
  double d_center0[3];
  double d_ratio0;
};

/*std::istream& operator >> (std::istream& is, VMCellBase & )
  // non combinatorial information. Default = nothing
{
  return is;
}

std::ostream& operator<< (std::ostream& os, const VMCellBase & )
  // non combinatorial information. Default = nothing
{
  return os;
}
*/
}

#endif
