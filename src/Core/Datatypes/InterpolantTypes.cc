#include <Core/Datatypes/InterpolantTypes.h>

namespace SCIRun {

using std::vector;
using std::pair;


template <> const string find_type_name(vector<pair<TetVolMesh::Node::index_type, double> > *)
{ return "vector<pair<TetVolMesh::Node::index_type"; }

template <> const string find_type_name(vector<pair<TetVolMesh::Edge::index_type, double> > *)
{ return "vector<pair<TetVolMesh::Edge::index_type"; }

template <> const string find_type_name(vector<pair<TetVolMesh::Face::index_type, double> > *)
{ return "vector<pair<TetVolMesh::Face::index_type"; }

template <> const string find_type_name(vector<pair<TetVolMesh::Cell::index_type, double> > *)
{ return "vector<pair<TetVolMesh::Cell::index_type"; }

void Pio(Piostream &s, TetVolMesh::Node::index_type &i)
{ Pio(s, (unsigned int &)i); }
void Pio(Piostream &s, TetVolMesh::Edge::index_type &i)
{ Pio(s, (unsigned int &)i); }
void Pio(Piostream &s, TetVolMesh::Face::index_type &i)
{ Pio(s, (unsigned int &)i); }
void Pio(Piostream &s, TetVolMesh::Cell::index_type &i)
{ Pio(s, (unsigned int &)i); }



#if 0
template <> const string find_type_name(vector<pair<LatVolMesh::Node::index_type, double> > *)
{ return "vector<pair<LatVolMesh::Node::index_type"; }

template <> const string find_type_name(vector<pair<LatVolMesh::Edge::index_type, double> > *)
{ return "vector<pair<LatVolMesh::Edge::index_type"; }

template <> const string find_type_name(vector<pair<LatVolMesh::Face::index_type, double> > *)
{ return "vector<pair<LatVolMesh::Face::index_type"; }

template <> const string find_type_name(vector<pair<LatVolMesh::Cell::index_type, double> > *)
{ return "vector<pair<LatVolMesh::Cell::index_type"; }


void Pio(Piostream &s, LatVolMesh::Node::index_type &i)
{
  s.begin_cheap_delim();
  Pio(s, i.i_);
  Pio(s, i.j_);
  Pio(s, i.k_);
  s.end_cheap_delim();
}
void Pio(Piostream &s, LatVolMesh::Edge::index_type &i)
{
  s.begin_cheap_delim();
  Pio(s, i.i_);
  s.end_cheap_delim();
}
void Pio(Piostream &s, LatVolMesh::Face::index_type &i)
{
  s.begin_cheap_delim();
  Pio(s, i.i_);
  s.end_cheap_delim();
}
void Pio(Piostream &s, LatVolMesh::Cell::index_type &i)
{
  s.begin_cheap_delim();
  Pio(s, i.i_);
  Pio(s, i.j_);
  Pio(s, i.k_);
  s.end_cheap_delim();
}

#endif


#if 0
template <> Vector TetVol<vector<pair<TetVolMesh::Node::index_type, double> > >::cell_gradient(TetVolMesh::Cell::index_type) { return Vector(0.0, 0.0, 0.0); }
template <> Vector TetVol<vector<pair<TetVolMesh::Edge::index_type, double> > >::cell_gradient(TetVolMesh::Cell::index_type) { return Vector(0.0, 0.0, 0.0); }
template <> Vector TetVol<vector<pair<TetVolMesh::Face::index_type, double> > >::cell_gradient(TetVolMesh::Cell::index_type) { return Vector(0.0, 0.0, 0.0); }
template <> Vector TetVol<vector<pair<TetVolMesh::Cell::index_type, double> > >::cell_gradient(TetVolMesh::Cell::index_type) { return Vector(0.0, 0.0, 0.0); }
template <> Vector TetVol<vector<pair<LatVolMesh::Node::index_type, double> > >::cell_gradient(TetVolMesh::Cell::index_type) { return Vector(0.0, 0.0, 0.0); }
template <> Vector TetVol<vector<pair<LatVolMesh::Edge::index_type, double> > >::cell_gradient(TetVolMesh::Cell::index_type) { return Vector(0.0, 0.0, 0.0); }
template <> Vector TetVol<vector<pair<LatVolMesh::Face::index_type, double> > >::cell_gradient(TetVolMesh::Cell::index_type) { return Vector(0.0, 0.0, 0.0); }
template <> Vector TetVol<vector<pair<LatVolMesh::Cell::index_type, double> > >::cell_gradient(TetVolMesh::Cell::index_type) { return Vector(0.0, 0.0, 0.0); }


template <> bool LatticeVol<vector<pair<TetVolMesh::Node::index_type, double> > >::get_gradient(Vector &, Point &) { return false; }
template <> bool LatticeVol<vector<pair<TetVolMesh::Edge::index_type, double> > >::get_gradient(Vector &, Point &) { return false; }
template <> bool LatticeVol<vector<pair<TetVolMesh::Face::index_type, double> > >::get_gradient(Vector &, Point &) { return false; }
template <> bool LatticeVol<vector<pair<TetVolMesh::Cell::index_type, double> > >::get_gradient(Vector &, Point &) { return false; }
template <> bool LatticeVol<vector<pair<LatVolMesh::Node::index_type, double> > >::get_gradient(Vector &, Point &) { return false; }
template <> bool LatticeVol<vector<pair<LatVolMesh::Edge::index_type, double> > >::get_gradient(Vector &, Point &) { return false; }
template <> bool LatticeVol<vector<pair<LatVolMesh::Face::index_type, double> > >::get_gradient(Vector &, Point &) { return false; }
template <> bool LatticeVol<vector<pair<LatVolMesh::Cell::index_type, double> > >::get_gradient(Vector &, Point &) { return false; }
#endif

#if 0
template class GenericField<TetVolMesh, vector<vector<pair<TetVolMesh::Node::index_type, double> > > >;
template class GenericField<TetVolMesh, vector<vector<pair<TetVolMesh::Edge::index_type, double> > > >;
template class GenericField<TetVolMesh, vector<vector<pair<TetVolMesh::Face::index_type, double> > > >;
template class GenericField<TetVolMesh, vector<vector<pair<TetVolMesh::Cell::index_type, double> > > >;
template class GenericField<TetVolMesh, vector<vector<pair<LatVolMesh::Node::index_type, double> > > >;
template class GenericField<TetVolMesh, vector<vector<pair<LatVolMesh::Edge::index_type, double> > > >;
template class GenericField<TetVolMesh, vector<vector<pair<LatVolMesh::Face::index_type, double> > > >;
template class GenericField<TetVolMesh, vector<vector<pair<LatVolMesh::Cell::index_type, double> > > >;

template class GenericField<LatVolMesh, FData3d<vector<pair<TetVolMesh::Node::index_type, double> > > >;
template class GenericField<LatVolMesh, FData3d<vector<pair<TetVolMesh::Edge::index_type, double> > > >;
template class GenericField<LatVolMesh, FData3d<vector<pair<TetVolMesh::Face::index_type, double> > > >;
template class GenericField<LatVolMesh, FData3d<vector<pair<TetVolMesh::Cell::index_type, double> > > >;
template class GenericField<LatVolMesh, FData3d<vector<pair<LatVolMesh::Node::index_type, double> > > >;
template class GenericField<LatVolMesh, FData3d<vector<pair<LatVolMesh::Edge::index_type, double> > > >;
template class GenericField<LatVolMesh, FData3d<vector<pair<LatVolMesh::Face::index_type, double> > > >;
template class GenericField<LatVolMesh, FData3d<vector<pair<LatVolMesh::Cell::index_type, double> > > >;
#endif

template class GenericField<TriSurfMesh, vector<vector<pair<TetVolMesh::Node::index_type, double> > > >;
template class GenericField<TriSurfMesh, vector<vector<pair<TetVolMesh::Edge::index_type, double> > > >;
template class GenericField<TriSurfMesh, vector<vector<pair<TetVolMesh::Face::index_type, double> > > >;
template class GenericField<TriSurfMesh, vector<vector<pair<TetVolMesh::Cell::index_type, double> > > >;
#if 0
template class GenericField<TriSurfMesh, vector<vector<pair<LatVolMesh::Node::index_type, double> > > >;
template class GenericField<TriSurfMesh, vector<vector<pair<LatVolMesh::Edge::index_type, double> > > >;
template class GenericField<TriSurfMesh, vector<vector<pair<LatVolMesh::Face::index_type, double> > > >;
template class GenericField<TriSurfMesh, vector<vector<pair<LatVolMesh::Cell::index_type, double> > > >;
#endif

template class GenericField<PointCloudMesh, vector<vector<pair<TetVolMesh::Node::index_type, double> > > >;
template class GenericField<PointCloudMesh, vector<vector<pair<TetVolMesh::Edge::index_type, double> > > >;
template class GenericField<PointCloudMesh, vector<vector<pair<TetVolMesh::Face::index_type, double> > > >;
template class GenericField<PointCloudMesh, vector<vector<pair<TetVolMesh::Cell::index_type, double> > > >;
#if 0
template class GenericField<PointCloudMesh, vector<vector<pair<LatVolMesh::Node::index_type, double> > > >;
template class GenericField<PointCloudMesh, vector<vector<pair<LatVolMesh::Edge::index_type, double> > > >;
template class GenericField<PointCloudMesh, vector<vector<pair<LatVolMesh::Face::index_type, double> > > >;
template class GenericField<PointCloudMesh, vector<vector<pair<LatVolMesh::Cell::index_type, double> > > >;
#endif


template class GenericField<ContourMesh, vector<vector<pair<TetVolMesh::Node::index_type, double> > > >;
template class GenericField<ContourMesh, vector<vector<pair<TetVolMesh::Edge::index_type, double> > > >;
template class GenericField<ContourMesh, vector<vector<pair<TetVolMesh::Face::index_type, double> > > >;
template class GenericField<ContourMesh, vector<vector<pair<TetVolMesh::Cell::index_type, double> > > >;
#if 0
template class GenericField<ContourMesh, vector<vector<pair<LatVolMesh::Node::index_type, double> > > >;
template class GenericField<ContourMesh, vector<vector<pair<LatVolMesh::Edge::index_type, double> > > >;
template class GenericField<ContourMesh, vector<vector<pair<LatVolMesh::Face::index_type, double> > > >;
template class GenericField<ContourMesh, vector<vector<pair<LatVolMesh::Cell::index_type, double> > > >;
#endif

#if 0
template class TetVol<vector<pair<TetVolMesh::Node::index_type, double> > >;
template class TetVol<vector<pair<TetVolMesh::Edge::index_type, double> > >;
template class TetVol<vector<pair<TetVolMesh::Face::index_type, double> > >;
template class TetVol<vector<pair<TetVolMesh::Cell::index_type, double> > >;
template class TetVol<vector<pair<LatVolMesh::Node::index_type, double> > >;
template class TetVol<vector<pair<LatVolMesh::Edge::index_type, double> > >;
template class TetVol<vector<pair<LatVolMesh::Face::index_type, double> > >;
template class TetVol<vector<pair<LatVolMesh::Cell::index_type, double> > >;

template class LatticeVol<vector<pair<TetVolMesh::Node::index_type, double> > >;
template class LatticeVol<vector<pair<TetVolMesh::Edge::index_type, double> > >;
template class LatticeVol<vector<pair<TetVolMesh::Face::index_type, double> > >;
template class LatticeVol<vector<pair<TetVolMesh::Cell::index_type, double> > >;
template class LatticeVol<vector<pair<LatVolMesh::Node::index_type, double> > >;
template class LatticeVol<vector<pair<LatVolMesh::Edge::index_type, double> > >;
template class LatticeVol<vector<pair<LatVolMesh::Face::index_type, double> > >;
template class LatticeVol<vector<pair<LatVolMesh::Cell::index_type, double> > >;
#endif

template class TriSurf<vector<pair<TetVolMesh::Node::index_type, double> > >;
template class TriSurf<vector<pair<TetVolMesh::Edge::index_type, double> > >;
template class TriSurf<vector<pair<TetVolMesh::Face::index_type, double> > >;
template class TriSurf<vector<pair<TetVolMesh::Cell::index_type, double> > >;
#if 0
template class TriSurf<vector<pair<LatVolMesh::Node::index_type, double> > >;
template class TriSurf<vector<pair<LatVolMesh::Edge::index_type, double> > >;
template class TriSurf<vector<pair<LatVolMesh::Face::index_type, double> > >;
template class TriSurf<vector<pair<LatVolMesh::Cell::index_type, double> > >;
#endif

template class PointCloud<vector<pair<TetVolMesh::Node::index_type, double> > >;
template class PointCloud<vector<pair<TetVolMesh::Edge::index_type, double> > >;
template class PointCloud<vector<pair<TetVolMesh::Face::index_type, double> > >;
template class PointCloud<vector<pair<TetVolMesh::Cell::index_type, double> > >;
#if 0
template class PointCloud<vector<pair<LatVolMesh::Node::index_type, double> > >;
template class PointCloud<vector<pair<LatVolMesh::Edge::index_type, double> > >;
template class PointCloud<vector<pair<LatVolMesh::Face::index_type, double> > >;
template class PointCloud<vector<pair<LatVolMesh::Cell::index_type, double> > >;
#endif

template class ContourField<vector<pair<TetVolMesh::Node::index_type, double> > >;
template class ContourField<vector<pair<TetVolMesh::Edge::index_type, double> > >;
template class ContourField<vector<pair<TetVolMesh::Face::index_type, double> > >;
template class ContourField<vector<pair<TetVolMesh::Cell::index_type, double> > >;
#if 0
template class ContourField<vector<pair<LatVolMesh::Node::index_type, double> > >;
template class ContourField<vector<pair<LatVolMesh::Edge::index_type, double> > >;
template class ContourField<vector<pair<LatVolMesh::Face::index_type, double> > >;
template class ContourField<vector<pair<LatVolMesh::Cell::index_type, double> > >;
#endif

}


