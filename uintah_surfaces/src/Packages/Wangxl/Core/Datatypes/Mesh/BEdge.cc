#include <Packages/Wangxl/Core/Datatypes/Mesh/BEdge.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/BFace.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/DVertex.h>

namespace Wangxl {

using namespace SCIRun;

BEdge::BEdge() { d_source = d_target = 0; d_root = 0; d_root_bface = 0; }
BEdge::BEdge(DVertex* v0, DVertex* v1){
    d_source = v0;
    d_target = v1;
    d_vertices.insert(v0);
    d_vertices.insert(v1);
    d_root = this;
    d_root_bface = 0;
}
BEdge::BEdge(DVertex* v0, DVertex* v1, BEdge* parent) {
    d_source = v0;
    d_target = v1;
    d_root = parent->root();
    d_root_bface = 0;
}
DVertex* BEdge::source() const { return d_source; }
DVertex* BEdge::target() const { return d_target; }
BEdge* BEdge::root() { return d_root; }
bool BEdge::is_root() { return d_root == this; }
void BEdge::set_pot(bool pot) { d_pot = pot; }
bool BEdge::is_pot() { return d_pot; }
bool BEdge::is_split() { if ( d_vertices.size() > 2 ) return true; else return false; }
// a temporary BEdge might not be a boundary edge
bool BEdge::is_bedge() { return !d_root_bface; }
void BEdge::get_vertices( vector<DVertex*>& bvertices ) {
  set<DVertex*,VertComp>::const_iterator it;
  for ( it = d_vertices.begin(); it != d_vertices.end(); it++ ) bvertices.push_back(*it);
}
//void BEdge::set_no_root() { d_root = 0; } // indicate the edge is on the boundary faces but not on boundary edges
void BEdge::set_root_bface(BFace* bface) { d_root_bface = bface; }

void BEdge::insert(DVertex* v) { 
  // this is a temporary BEdge created for recovering created missing face
  // the new vertex will go to the original BFace 
  if ( d_root_bface != 0 ) { 
    d_root_bface->add_vertex(v);
    return;
  }

  // for regular boundary BEdges, split them by inserting a new vertex
  d_root->split(v);
}

bool VertComp::operator()(const DVertex* v0, const DVertex* v1) const
{
  double a, b, c;
  Point p0 = v0->point();
  Point p1 = v1->point();
  a = fabs(p0.x() - p1.x());
  b = fabs(p0.y() - p1.y());
  c = fabs(p0.z() - p1.z());
  if ( a > b ) {
    if ( a > c )  return p0.x() < p1.x();
    else return p0.z() < p1.z();
  }
  else {
    if ( b < c ) return p0.z() < p1.z();
    else return p0.y() < p1.y();
  }
}

bool BEdge::get_split_point(Point& mp)
{
  DVertex *sv, *tv;
  Point sp, tp, p;
  Vector vec;
  double d, tmp;
  int k;
  sv = source();
  tv = target();
  sp = sv->point();
  tp = tv->point();
  mp = (Point)(sp.asVector() + tp.asVector())/2.0;

  if ( ( sv->is_input() && tv->is_input() ) || ( !sv->is_input() && !tv->is_input() ) || is_pot() ) return false;
  bool bsv = sv->is_input();
  bool btv = tv->is_input();
  cout << " pot: " << bsv << " " << btv << endl;
  if ( sv->is_input() ) p = sp;
  else p = tp;
  vec = mp - p;
  d = vec.length();
  tmp = log(d/d_size)/log(2.0);
  k = (int) (tmp+0.5);
  d = pow(2.0,k)*d_size;
  vec.normalize();
  mp = p + vec*d;
  return true; // the pow-of-two subdivision is done
}

}




