#ifndef SCI_Wangxl_Datatypes_Mesh_BEdge_h
#define SCI_Wangxl_Datatypes_Mesh_BEdge_h

#include <vector>
#include <set>
 
namespace Wangxl {

using std::vector;
using std::set;

class DVertex;
class BFace;

class VertComp
{
public:
  bool operator()(const DVertex* v0, const DVertex* v1) const;
};

class BEdge {
public:
  BEdge();
  BEdge(DVertex* v0, DVertex* v1);
  BEdge(DVertex* v0, DVertex* v1, BEdge* parent);
  ~BEdge(){}

  DVertex* source() const;
  DVertex* target() const;
  BEdge* root();
  bool is_root();
  void set_pot(bool pot);
  bool is_pot();
  bool is_split();
  bool is_bedge();
  void get_vertices( vector<DVertex*>& bvertices );
  void set_root_bface(BFace* bface);
  void insert(DVertex* v);

  bool get_split_point(Point& mp);
private:
  void split(DVertex* v){ d_vertices.insert(v); }

private:
  set<DVertex*,VertComp> d_vertices; // only for the root BEdge
  DVertex *d_source, *d_target; // for every BEdge
  BEdge *d_root; // root BEdge for this BEdge
  bool d_pot; // splitting flag
  BFace* d_root_bface; // only used for generated boundary edges
};

}

#endif
