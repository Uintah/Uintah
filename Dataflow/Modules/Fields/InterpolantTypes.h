#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/TriSurf.h>
#include <Core/Datatypes/ContourField.h>
#include <Core/Datatypes/PointCloud.h>

namespace SCIRun {

using std::vector;
using std::pair;


template <> const string find_type_name(vector<pair<TetVolMesh::Node::index_type, double> > *);
template <> const string find_type_name(vector<pair<TetVolMesh::Edge::index_type, double> > *);
template <> const string find_type_name(vector<pair<TetVolMesh::Face::index_type, double> > *);
template <> const string find_type_name(vector<pair<TetVolMesh::Cell::index_type, double> > *);
void Pio(Piostream &, TetVolMesh::Node::index_type &);
void Pio(Piostream &, TetVolMesh::Edge::index_type &);
void Pio(Piostream &, TetVolMesh::Face::index_type &);
void Pio(Piostream &, TetVolMesh::Cell::index_type &);

#if 0
template <> const string find_type_name(vector<pair<LatVolMesh::Node::index_type, double> > *);
template <> const string find_type_name(vector<pair<LatVolMesh::Edge::index_type, double> > *);
template <> const string find_type_name(vector<pair<LatVolMesh::Face::index_type, double> > *);
template <> const string find_type_name(vector<pair<LatVolMesh::Cell::index_type, double> > *);
void Pio(Piostream &, LatVolMesh::Node::index_type &);
void Pio(Piostream &, LatVolMesh::Edge::index_type &);
void Pio(Piostream &, LatVolMesh::Face::index_type &);
void Pio(Piostream &, LatVolMesh::Cell::index_type &);
#endif

#if 0
template <> Vector TetVol<vector<pair<TetVolMesh::Node::index_type, double> > >::cell_gradient(TetVolMesh::Cell::index_type);
template <> Vector TetVol<vector<pair<TetVolMesh::Edge::index_type, double> > >::cell_gradient(TetVolMesh::Cell::index_type);
template <> Vector TetVol<vector<pair<TetVolMesh::Face::index_type, double> > >::cell_gradient(TetVolMesh::Cell::index_type);
template <> Vector TetVol<vector<pair<TetVolMesh::Cell::index_type, double> > >::cell_gradient(TetVolMesh::Cell::index_type);
template <> Vector TetVol<vector<pair<LatVolMesh::Node::index_type, double> > >::cell_gradient(TetVolMesh::Cell::index_type);
template <> Vector TetVol<vector<pair<LatVolMesh::Edge::index_type, double> > >::cell_gradient(TetVolMesh::Cell::index_type);
template <> Vector TetVol<vector<pair<LatVolMesh::Face::index_type, double> > >::cell_gradient(TetVolMesh::Cell::index_type);
template <> Vector TetVol<vector<pair<LatVolMesh::Cell::index_type, double> > >::cell_gradient(TetVolMesh::Cell::index_type);
#endif

#if 0
template <> bool LatticeVol<vector<pair<TetVolMesh::Node::index_type, double> > >::get_gradient(Vector &, Point &);
template <> bool LatticeVol<vector<pair<TetVolMesh::Edge::index_type, double> > >::get_gradient(Vector &, Point &);
template <> bool LatticeVol<vector<pair<TetVolMesh::Face::index_type, double> > >::get_gradient(Vector &, Point &);
template <> bool LatticeVol<vector<pair<TetVolMesh::Cell::index_type, double> > >::get_gradient(Vector &, Point &);
template <> bool LatticeVol<vector<pair<LatVolMesh::Node::index_type, double> > >::get_gradient(Vector &, Point &);
template <> bool LatticeVol<vector<pair<LatVolMesh::Edge::index_type, double> > >::get_gradient(Vector &, Point &);
template <> bool LatticeVol<vector<pair<LatVolMesh::Face::index_type, double> > >::get_gradient(Vector &, Point &);
template <> bool LatticeVol<vector<pair<LatVolMesh::Cell::index_type, double> > >::get_gradient(Vector &, Point &);
#endif

}
