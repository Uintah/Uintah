#include <Dataflow/Modules/Fields/InterpolantTypes.h>

namespace SCIRun {

using std::vector;
using std::pair;


template <> const string find_type_name(vector<pair<TetVolMesh::node_index, double> > *)
{ return "vector<pair<TetVolMesh::node_index"; }

template <> const string find_type_name(vector<pair<TetVolMesh::edge_index, double> > *)
{ return "vector<pair<TetVolMesh::edge_index"; }

template <> const string find_type_name(vector<pair<TetVolMesh::face_index, double> > *)
{ return "vector<pair<TetVolMesh::face_index"; }

template <> const string find_type_name(vector<pair<TetVolMesh::cell_index, double> > *)
{ return "vector<pair<TetVolMesh::cell_index"; }

void Pio(Piostream &s, TetVolMesh::node_index &i)
{ Pio(s, (unsigned int &)i); }
void Pio(Piostream &s, TetVolMesh::edge_index &i)
{ Pio(s, (unsigned int &)i); }
void Pio(Piostream &s, TetVolMesh::face_index &i)
{ Pio(s, (unsigned int &)i); }
void Pio(Piostream &s, TetVolMesh::cell_index &i)
{ Pio(s, (unsigned int &)i); }



#if 0
template <> const string find_type_name(vector<pair<LatVolMesh::node_index, double> > *)
{ return "vector<pair<LatVolMesh::node_index"; }

template <> const string find_type_name(vector<pair<LatVolMesh::edge_index, double> > *)
{ return "vector<pair<LatVolMesh::edge_index"; }

template <> const string find_type_name(vector<pair<LatVolMesh::face_index, double> > *)
{ return "vector<pair<LatVolMesh::face_index"; }

template <> const string find_type_name(vector<pair<LatVolMesh::cell_index, double> > *)
{ return "vector<pair<LatVolMesh::cell_index"; }


void Pio(Piostream &s, LatVolMesh::node_index &i)
{
  s.begin_cheap_delim();
  Pio(s, i.i_);
  Pio(s, i.j_);
  Pio(s, i.k_);
  s.end_cheap_delim();
}
void Pio(Piostream &s, LatVolMesh::edge_index &i)
{
  s.begin_cheap_delim();
  Pio(s, i.i_);
  s.end_cheap_delim();
}
void Pio(Piostream &s, LatVolMesh::face_index &i)
{
  s.begin_cheap_delim();
  Pio(s, i.i_);
  s.end_cheap_delim();
}
void Pio(Piostream &s, LatVolMesh::cell_index &i)
{
  s.begin_cheap_delim();
  Pio(s, i.i_);
  Pio(s, i.j_);
  Pio(s, i.k_);
  s.end_cheap_delim();
}

#endif


#if 0
template <> Vector TetVol<vector<pair<TetVolMesh::node_index, double> > >::cell_gradient(TetVolMesh::cell_index) { return Vector(0.0, 0.0, 0.0); }
template <> Vector TetVol<vector<pair<TetVolMesh::edge_index, double> > >::cell_gradient(TetVolMesh::cell_index) { return Vector(0.0, 0.0, 0.0); }
template <> Vector TetVol<vector<pair<TetVolMesh::face_index, double> > >::cell_gradient(TetVolMesh::cell_index) { return Vector(0.0, 0.0, 0.0); }
template <> Vector TetVol<vector<pair<TetVolMesh::cell_index, double> > >::cell_gradient(TetVolMesh::cell_index) { return Vector(0.0, 0.0, 0.0); }
template <> Vector TetVol<vector<pair<LatVolMesh::node_index, double> > >::cell_gradient(TetVolMesh::cell_index) { return Vector(0.0, 0.0, 0.0); }
template <> Vector TetVol<vector<pair<LatVolMesh::edge_index, double> > >::cell_gradient(TetVolMesh::cell_index) { return Vector(0.0, 0.0, 0.0); }
template <> Vector TetVol<vector<pair<LatVolMesh::face_index, double> > >::cell_gradient(TetVolMesh::cell_index) { return Vector(0.0, 0.0, 0.0); }
template <> Vector TetVol<vector<pair<LatVolMesh::cell_index, double> > >::cell_gradient(TetVolMesh::cell_index) { return Vector(0.0, 0.0, 0.0); }


template <> bool LatticeVol<vector<pair<TetVolMesh::node_index, double> > >::get_gradient(Vector &, Point &) { return false; }
template <> bool LatticeVol<vector<pair<TetVolMesh::edge_index, double> > >::get_gradient(Vector &, Point &) { return false; }
template <> bool LatticeVol<vector<pair<TetVolMesh::face_index, double> > >::get_gradient(Vector &, Point &) { return false; }
template <> bool LatticeVol<vector<pair<TetVolMesh::cell_index, double> > >::get_gradient(Vector &, Point &) { return false; }
template <> bool LatticeVol<vector<pair<LatVolMesh::node_index, double> > >::get_gradient(Vector &, Point &) { return false; }
template <> bool LatticeVol<vector<pair<LatVolMesh::edge_index, double> > >::get_gradient(Vector &, Point &) { return false; }
template <> bool LatticeVol<vector<pair<LatVolMesh::face_index, double> > >::get_gradient(Vector &, Point &) { return false; }
template <> bool LatticeVol<vector<pair<LatVolMesh::cell_index, double> > >::get_gradient(Vector &, Point &) { return false; }
#endif

#if 0
template class GenericField<TetVolMesh, vector<vector<pair<TetVolMesh::node_index, double> > > >;
template class GenericField<TetVolMesh, vector<vector<pair<TetVolMesh::edge_index, double> > > >;
template class GenericField<TetVolMesh, vector<vector<pair<TetVolMesh::face_index, double> > > >;
template class GenericField<TetVolMesh, vector<vector<pair<TetVolMesh::cell_index, double> > > >;
template class GenericField<TetVolMesh, vector<vector<pair<LatVolMesh::node_index, double> > > >;
template class GenericField<TetVolMesh, vector<vector<pair<LatVolMesh::edge_index, double> > > >;
template class GenericField<TetVolMesh, vector<vector<pair<LatVolMesh::face_index, double> > > >;
template class GenericField<TetVolMesh, vector<vector<pair<LatVolMesh::cell_index, double> > > >;

template class GenericField<LatVolMesh, FData3d<vector<pair<TetVolMesh::node_index, double> > > >;
template class GenericField<LatVolMesh, FData3d<vector<pair<TetVolMesh::edge_index, double> > > >;
template class GenericField<LatVolMesh, FData3d<vector<pair<TetVolMesh::face_index, double> > > >;
template class GenericField<LatVolMesh, FData3d<vector<pair<TetVolMesh::cell_index, double> > > >;
template class GenericField<LatVolMesh, FData3d<vector<pair<LatVolMesh::node_index, double> > > >;
template class GenericField<LatVolMesh, FData3d<vector<pair<LatVolMesh::edge_index, double> > > >;
template class GenericField<LatVolMesh, FData3d<vector<pair<LatVolMesh::face_index, double> > > >;
template class GenericField<LatVolMesh, FData3d<vector<pair<LatVolMesh::cell_index, double> > > >;
#endif

template class GenericField<TriSurfMesh, vector<vector<pair<TetVolMesh::node_index, double> > > >;
template class GenericField<TriSurfMesh, vector<vector<pair<TetVolMesh::edge_index, double> > > >;
template class GenericField<TriSurfMesh, vector<vector<pair<TetVolMesh::face_index, double> > > >;
template class GenericField<TriSurfMesh, vector<vector<pair<TetVolMesh::cell_index, double> > > >;
#if 0
template class GenericField<TriSurfMesh, vector<vector<pair<LatVolMesh::node_index, double> > > >;
template class GenericField<TriSurfMesh, vector<vector<pair<LatVolMesh::edge_index, double> > > >;
template class GenericField<TriSurfMesh, vector<vector<pair<LatVolMesh::face_index, double> > > >;
template class GenericField<TriSurfMesh, vector<vector<pair<LatVolMesh::cell_index, double> > > >;
#endif

template class GenericField<PointCloudMesh, vector<vector<pair<TetVolMesh::node_index, double> > > >;
template class GenericField<PointCloudMesh, vector<vector<pair<TetVolMesh::edge_index, double> > > >;
template class GenericField<PointCloudMesh, vector<vector<pair<TetVolMesh::face_index, double> > > >;
template class GenericField<PointCloudMesh, vector<vector<pair<TetVolMesh::cell_index, double> > > >;
#if 0
template class GenericField<PointCloudMesh, vector<vector<pair<LatVolMesh::node_index, double> > > >;
template class GenericField<PointCloudMesh, vector<vector<pair<LatVolMesh::edge_index, double> > > >;
template class GenericField<PointCloudMesh, vector<vector<pair<LatVolMesh::face_index, double> > > >;
template class GenericField<PointCloudMesh, vector<vector<pair<LatVolMesh::cell_index, double> > > >;
#endif


template class GenericField<ContourMesh, vector<vector<pair<TetVolMesh::node_index, double> > > >;
template class GenericField<ContourMesh, vector<vector<pair<TetVolMesh::edge_index, double> > > >;
template class GenericField<ContourMesh, vector<vector<pair<TetVolMesh::face_index, double> > > >;
template class GenericField<ContourMesh, vector<vector<pair<TetVolMesh::cell_index, double> > > >;
#if 0
template class GenericField<ContourMesh, vector<vector<pair<LatVolMesh::node_index, double> > > >;
template class GenericField<ContourMesh, vector<vector<pair<LatVolMesh::edge_index, double> > > >;
template class GenericField<ContourMesh, vector<vector<pair<LatVolMesh::face_index, double> > > >;
template class GenericField<ContourMesh, vector<vector<pair<LatVolMesh::cell_index, double> > > >;
#endif

#if 0
template class TetVol<vector<pair<TetVolMesh::node_index, double> > >;
template class TetVol<vector<pair<TetVolMesh::edge_index, double> > >;
template class TetVol<vector<pair<TetVolMesh::face_index, double> > >;
template class TetVol<vector<pair<TetVolMesh::cell_index, double> > >;
template class TetVol<vector<pair<LatVolMesh::node_index, double> > >;
template class TetVol<vector<pair<LatVolMesh::edge_index, double> > >;
template class TetVol<vector<pair<LatVolMesh::face_index, double> > >;
template class TetVol<vector<pair<LatVolMesh::cell_index, double> > >;

template class LatticeVol<vector<pair<TetVolMesh::node_index, double> > >;
template class LatticeVol<vector<pair<TetVolMesh::edge_index, double> > >;
template class LatticeVol<vector<pair<TetVolMesh::face_index, double> > >;
template class LatticeVol<vector<pair<TetVolMesh::cell_index, double> > >;
template class LatticeVol<vector<pair<LatVolMesh::node_index, double> > >;
template class LatticeVol<vector<pair<LatVolMesh::edge_index, double> > >;
template class LatticeVol<vector<pair<LatVolMesh::face_index, double> > >;
template class LatticeVol<vector<pair<LatVolMesh::cell_index, double> > >;
#endif

template class TriSurf<vector<pair<TetVolMesh::node_index, double> > >;
template class TriSurf<vector<pair<TetVolMesh::edge_index, double> > >;
template class TriSurf<vector<pair<TetVolMesh::face_index, double> > >;
template class TriSurf<vector<pair<TetVolMesh::cell_index, double> > >;
#if 0
template class TriSurf<vector<pair<LatVolMesh::node_index, double> > >;
template class TriSurf<vector<pair<LatVolMesh::edge_index, double> > >;
template class TriSurf<vector<pair<LatVolMesh::face_index, double> > >;
template class TriSurf<vector<pair<LatVolMesh::cell_index, double> > >;
#endif

template class PointCloud<vector<pair<TetVolMesh::node_index, double> > >;
template class PointCloud<vector<pair<TetVolMesh::edge_index, double> > >;
template class PointCloud<vector<pair<TetVolMesh::face_index, double> > >;
template class PointCloud<vector<pair<TetVolMesh::cell_index, double> > >;
#if 0
template class PointCloud<vector<pair<LatVolMesh::node_index, double> > >;
template class PointCloud<vector<pair<LatVolMesh::edge_index, double> > >;
template class PointCloud<vector<pair<LatVolMesh::face_index, double> > >;
template class PointCloud<vector<pair<LatVolMesh::cell_index, double> > >;
#endif

template class ContourField<vector<pair<TetVolMesh::node_index, double> > >;
template class ContourField<vector<pair<TetVolMesh::edge_index, double> > >;
template class ContourField<vector<pair<TetVolMesh::face_index, double> > >;
template class ContourField<vector<pair<TetVolMesh::cell_index, double> > >;
#if 0
template class ContourField<vector<pair<LatVolMesh::node_index, double> > >;
template class ContourField<vector<pair<LatVolMesh::edge_index, double> > >;
template class ContourField<vector<pair<LatVolMesh::face_index, double> > >;
template class ContourField<vector<pair<LatVolMesh::cell_index, double> > >;
#endif

}


