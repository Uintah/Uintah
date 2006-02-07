#ifndef SCI_Wangxl_Datatypes_Mesh_BFace_h
#define SCI_Wangxl_Datatypes_Mesh_BFace_h

#include <vector>
#include <Core/Geometry/Point.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/Triple.h>

namespace Wangxl {

using namespace SCIRun;

using std::vector;

class DVertex;
class BEdge;

class BFace {
public:
  BFace();
  BFace(DVertex* v0, DVertex* v1, DVertex* v2);
  BFace(DVertex* v0, DVertex* v1, DVertex* v2, BFace* root);
  ~BFace(){}
  void add_edge(BEdge* be);
  void add_vertex(DVertex* v);
  bool is_split();
  bool is_root();
  DVertex* vertex0() const { return d_v[0]; }
  DVertex* vertex1() const { return d_v[1]; }
  DVertex* vertex2() const { return d_v[2]; }
  //  int pnumber() { return d_vertices.size(); }
  void get_new_faces(vector< triple<DVertex*, DVertex*, DVertex*> >& nfaces, vector<int>& neighbors);
  /*triple<DVertex*,DVertex*,DVertex*> get_vertices() {
    DVertex *v0, *v1, *v2, *vs, *vt;
    v0 = d_bedges[0]->source();
    v1 = d_bedges[0]->target();
    vs = d_bedges[1]->source();
    vt = d_bedges[1]->target();
    if ( vs != v0 && vs != v1 ) v2 = vs;
    else if ( vt != v0 && vt != v1 ) v2 = vt;
    else {
      cout << " ERROR!!!!!!!!!!! in get face's vertices" << endl;
      assert(false);
    }
    return make_triple(v0, v1, v2);
    }*/

  void get_split_edges(BEdge* bedges[3]);

private:
  void set_2D();
  void get_2D(const Point& p, double& x, double& y);
  Point get_split_point();

  double trans[4][3];
private:
  vector<DVertex*> d_vertices; // points on the face except those splitting edges
  DVertex* d_v[3];
  BEdge* d_bedges[3];
};

}

#endif

