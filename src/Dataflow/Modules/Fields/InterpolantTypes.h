#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/TriSurf.h>
#include <Core/Datatypes/ContourField.h>
#include <Core/Datatypes/PointCloud.h>

namespace SCIRun {

using std::vector;
using std::pair;


template <> const string find_type_name(vector<pair<TetVolMesh::node_index, double> > *);
template <> const string find_type_name(vector<pair<TetVolMesh::edge_index, double> > *);
template <> const string find_type_name(vector<pair<TetVolMesh::face_index, double> > *);
template <> const string find_type_name(vector<pair<TetVolMesh::cell_index, double> > *);
void Pio(Piostream &, TetVolMesh::node_index &);
void Pio(Piostream &, TetVolMesh::edge_index &);
void Pio(Piostream &, TetVolMesh::face_index &);
void Pio(Piostream &, TetVolMesh::cell_index &);

#if 0
template <> const string find_type_name(vector<pair<LatVolMesh::node_index, double> > *);
template <> const string find_type_name(vector<pair<LatVolMesh::edge_index, double> > *);
template <> const string find_type_name(vector<pair<LatVolMesh::face_index, double> > *);
template <> const string find_type_name(vector<pair<LatVolMesh::cell_index, double> > *);
void Pio(Piostream &, LatVolMesh::node_index &);
void Pio(Piostream &, LatVolMesh::edge_index &);
void Pio(Piostream &, LatVolMesh::face_index &);
void Pio(Piostream &, LatVolMesh::cell_index &);
#endif

#if 0
// TetVol

template <>
GenericField<TetVolMesh, vector<vector<pair<TetVolMesh::node_index, double> > > >::interp_type *
GenericField<TetVolMesh, vector<vector<pair<TetVolMesh::node_index, double> > > >::query_interpolate() const;

template <>
GenericField<TetVolMesh, vector<vector<pair<TetVolMesh::edge_index, double> > > >::interp_type *
GenericField<TetVolMesh, vector<vector<pair<TetVolMesh::edge_index, double> > > >::query_interpolate() const;

template <>
GenericField<TetVolMesh, vector<vector<pair<TetVolMesh::face_index, double> > > >::interp_type *
GenericField<TetVolMesh, vector<vector<pair<TetVolMesh::face_index, double> > > >::query_interpolate() const;

template <>
GenericField<TetVolMesh, vector<vector<pair<TetVolMesh::cell_index, double> > > >::interp_type *
GenericField<TetVolMesh, vector<vector<pair<TetVolMesh::cell_index, double> > > >::query_interpolate() const;

template <>
GenericField<TetVolMesh, vector<vector<pair<LatVolMesh::node_index, double> > > >::interp_type *
GenericField<TetVolMesh, vector<vector<pair<LatVolMesh::node_index, double> > > >::query_interpolate() const;

template <>
GenericField<TetVolMesh, vector<vector<pair<LatVolMesh::edge_index, double> > > >::interp_type *
GenericField<TetVolMesh, vector<vector<pair<LatVolMesh::edge_index, double> > > >::query_interpolate() const;

template <>
GenericField<TetVolMesh, vector<vector<pair<LatVolMesh::face_index, double> > > >::interp_type *
GenericField<TetVolMesh, vector<vector<pair<LatVolMesh::face_index, double> > > >::query_interpolate() const;

template <>
GenericField<TetVolMesh, vector<vector<pair<LatVolMesh::cell_index, double> > > >::interp_type *
GenericField<TetVolMesh, vector<vector<pair<LatVolMesh::cell_index, double> > > >::query_interpolate() const;

// LatVol

template <>
GenericField<LatVolMesh, FData3d<vector<pair<TetVolMesh::node_index, double> > > >::interp_type *
GenericField<LatVolMesh, FData3d<vector<pair<TetVolMesh::node_index, double> > > >::query_interpolate() const;

template <>
GenericField<LatVolMesh, FData3d<vector<pair<TetVolMesh::edge_index, double> > > >::interp_type *
GenericField<LatVolMesh, FData3d<vector<pair<TetVolMesh::edge_index, double> > > >::query_interpolate() const;

template <>
GenericField<LatVolMesh, FData3d<vector<pair<TetVolMesh::face_index, double> > > >::interp_type *
GenericField<LatVolMesh, FData3d<vector<pair<TetVolMesh::face_index, double> > > >::query_interpolate() const;

template <>
GenericField<LatVolMesh, FData3d<vector<pair<TetVolMesh::cell_index, double> > > >::interp_type *
GenericField<LatVolMesh, FData3d<vector<pair<TetVolMesh::cell_index, double> > > >::query_interpolate() const;

template <>
GenericField<LatVolMesh, FData3d<vector<pair<LatVolMesh::node_index, double> > > >::interp_type *
GenericField<LatVolMesh, FData3d<vector<pair<LatVolMesh::node_index, double> > > >::query_interpolate() const;

template <>
GenericField<LatVolMesh, FData3d<vector<pair<LatVolMesh::edge_index, double> > > >::interp_type *
GenericField<LatVolMesh, FData3d<vector<pair<LatVolMesh::edge_index, double> > > >::query_interpolate() const;

template <>
GenericField<LatVolMesh, FData3d<vector<pair<LatVolMesh::face_index, double> > > >::interp_type *
GenericField<LatVolMesh, FData3d<vector<pair<LatVolMesh::face_index, double> > > >::query_interpolate() const;

template <>
GenericField<LatVolMesh, FData3d<vector<pair<LatVolMesh::cell_index, double> > > >::interp_type *
GenericField<LatVolMesh, FData3d<vector<pair<LatVolMesh::cell_index, double> > > >::query_interpolate() const;
#endif

// TriSurf

template <>
GenericField<TriSurfMesh, vector<vector<pair<TetVolMesh::node_index, double> > > >::interp_type *
GenericField<TriSurfMesh, vector<vector<pair<TetVolMesh::node_index, double> > > >::query_interpolate() const;

template <>
GenericField<TriSurfMesh, vector<vector<pair<TetVolMesh::edge_index, double> > > >::interp_type *
GenericField<TriSurfMesh, vector<vector<pair<TetVolMesh::edge_index, double> > > >::query_interpolate() const;

template <>
GenericField<TriSurfMesh, vector<vector<pair<TetVolMesh::face_index, double> > > >::interp_type *
GenericField<TriSurfMesh, vector<vector<pair<TetVolMesh::face_index, double> > > >::query_interpolate() const;

template <>
GenericField<TriSurfMesh, vector<vector<pair<TetVolMesh::cell_index, double> > > >::interp_type *
GenericField<TriSurfMesh, vector<vector<pair<TetVolMesh::cell_index, double> > > >::query_interpolate() const;

#if 0
template <>
GenericField<TriSurfMesh, vector<vector<pair<LatVolMesh::node_index, double> > > >::interp_type *
GenericField<TriSurfMesh, vector<vector<pair<LatVolMesh::node_index, double> > > >::query_interpolate() const;

template <>
GenericField<TriSurfMesh, vector<vector<pair<LatVolMesh::edge_index, double> > > >::interp_type *
GenericField<TriSurfMesh, vector<vector<pair<LatVolMesh::edge_index, double> > > >::query_interpolate() const;

template <>
GenericField<TriSurfMesh, vector<vector<pair<LatVolMesh::face_index, double> > > >::interp_type *
GenericField<TriSurfMesh, vector<vector<pair<LatVolMesh::face_index, double> > > >::query_interpolate() const;

template <>
GenericField<TriSurfMesh, vector<vector<pair<LatVolMesh::cell_index, double> > > >::interp_type *
GenericField<TriSurfMesh, vector<vector<pair<LatVolMesh::cell_index, double> > > >::query_interpolate() const;
#endif

// ContourField

template <>
GenericField<ContourMesh, vector<vector<pair<TetVolMesh::node_index, double> > > >::interp_type *
GenericField<ContourMesh, vector<vector<pair<TetVolMesh::node_index, double> > > >::query_interpolate() const;

template <>
GenericField<ContourMesh, vector<vector<pair<TetVolMesh::edge_index, double> > > >::interp_type *
GenericField<ContourMesh, vector<vector<pair<TetVolMesh::edge_index, double> > > >::query_interpolate() const;

template <>
GenericField<ContourMesh, vector<vector<pair<TetVolMesh::face_index, double> > > >::interp_type *
GenericField<ContourMesh, vector<vector<pair<TetVolMesh::face_index, double> > > >::query_interpolate() const;

template <>
GenericField<ContourMesh, vector<vector<pair<TetVolMesh::cell_index, double> > > >::interp_type *
GenericField<ContourMesh, vector<vector<pair<TetVolMesh::cell_index, double> > > >::query_interpolate() const;

#if 0
template <>
GenericField<ContourMesh, vector<vector<pair<LatVolMesh::node_index, double> > > >::interp_type *
GenericField<ContourMesh, vector<vector<pair<LatVolMesh::node_index, double> > > >::query_interpolate() const;

template <>
GenericField<ContourMesh, vector<vector<pair<LatVolMesh::edge_index, double> > > >::interp_type *
GenericField<ContourMesh, vector<vector<pair<LatVolMesh::edge_index, double> > > >::query_interpolate() const;

template <>
GenericField<ContourMesh, vector<vector<pair<LatVolMesh::face_index, double> > > >::interp_type *
GenericField<ContourMesh, vector<vector<pair<LatVolMesh::face_index, double> > > >::query_interpolate() const;

template <>
GenericField<ContourMesh, vector<vector<pair<LatVolMesh::cell_index, double> > > >::interp_type *
GenericField<ContourMesh, vector<vector<pair<LatVolMesh::cell_index, double> > > >::query_interpolate() const;
#endif

// PointCloud

template <>
GenericField<PointCloudMesh, vector<vector<pair<TetVolMesh::node_index, double> > > >::interp_type *
GenericField<PointCloudMesh, vector<vector<pair<TetVolMesh::node_index, double> > > >::query_interpolate() const;

template <>
GenericField<PointCloudMesh, vector<vector<pair<TetVolMesh::edge_index, double> > > >::interp_type *
GenericField<PointCloudMesh, vector<vector<pair<TetVolMesh::edge_index, double> > > >::query_interpolate() const;

template <>
GenericField<PointCloudMesh, vector<vector<pair<TetVolMesh::face_index, double> > > >::interp_type *
GenericField<PointCloudMesh, vector<vector<pair<TetVolMesh::face_index, double> > > >::query_interpolate() const;

template <>
GenericField<PointCloudMesh, vector<vector<pair<TetVolMesh::cell_index, double> > > >::interp_type *
GenericField<PointCloudMesh, vector<vector<pair<TetVolMesh::cell_index, double> > > >::query_interpolate() const;

#if 0
template <>
GenericField<PointCloudMesh, vector<vector<pair<LatVolMesh::node_index, double> > > >::interp_type *
GenericField<PointCloudMesh, vector<vector<pair<LatVolMesh::node_index, double> > > >::query_interpolate() const;

template <>
GenericField<PointCloudMesh, vector<vector<pair<LatVolMesh::edge_index, double> > > >::interp_type *
GenericField<PointCloudMesh, vector<vector<pair<LatVolMesh::edge_index, double> > > >::query_interpolate() const;

template <>
GenericField<PointCloudMesh, vector<vector<pair<LatVolMesh::face_index, double> > > >::interp_type *
GenericField<PointCloudMesh, vector<vector<pair<LatVolMesh::face_index, double> > > >::query_interpolate() const;

template <>
GenericField<PointCloudMesh, vector<vector<pair<LatVolMesh::cell_index, double> > > >::interp_type *
GenericField<PointCloudMesh, vector<vector<pair<LatVolMesh::cell_index, double> > > >::query_interpolate() const;
#endif

#if 0
template <> Vector TetVol<vector<pair<TetVolMesh::node_index, double> > >::cell_gradient(TetVolMesh::cell_index);
template <> Vector TetVol<vector<pair<TetVolMesh::edge_index, double> > >::cell_gradient(TetVolMesh::cell_index);
template <> Vector TetVol<vector<pair<TetVolMesh::face_index, double> > >::cell_gradient(TetVolMesh::cell_index);
template <> Vector TetVol<vector<pair<TetVolMesh::cell_index, double> > >::cell_gradient(TetVolMesh::cell_index);
template <> Vector TetVol<vector<pair<LatVolMesh::node_index, double> > >::cell_gradient(TetVolMesh::cell_index);
template <> Vector TetVol<vector<pair<LatVolMesh::edge_index, double> > >::cell_gradient(TetVolMesh::cell_index);
template <> Vector TetVol<vector<pair<LatVolMesh::face_index, double> > >::cell_gradient(TetVolMesh::cell_index);
template <> Vector TetVol<vector<pair<LatVolMesh::cell_index, double> > >::cell_gradient(TetVolMesh::cell_index);
#endif

#if 0
template <> bool LatticeVol<vector<pair<TetVolMesh::node_index, double> > >::get_gradient(Vector &, Point &);
template <> bool LatticeVol<vector<pair<TetVolMesh::edge_index, double> > >::get_gradient(Vector &, Point &);
template <> bool LatticeVol<vector<pair<TetVolMesh::face_index, double> > >::get_gradient(Vector &, Point &);
template <> bool LatticeVol<vector<pair<TetVolMesh::cell_index, double> > >::get_gradient(Vector &, Point &);
template <> bool LatticeVol<vector<pair<LatVolMesh::node_index, double> > >::get_gradient(Vector &, Point &);
template <> bool LatticeVol<vector<pair<LatVolMesh::edge_index, double> > >::get_gradient(Vector &, Point &);
template <> bool LatticeVol<vector<pair<LatVolMesh::face_index, double> > >::get_gradient(Vector &, Point &);
template <> bool LatticeVol<vector<pair<LatVolMesh::cell_index, double> > >::get_gradient(Vector &, Point &);
#endif

}
