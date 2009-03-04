/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


//    File   : InsertHexSheet.h
//    Author : Jason Shepherd
//    Date   : January 2006

#if !defined(InsertHexSheet_h)
#define InsertHexSheet_h

#define GTB_BEGIN_NAMESPACE namespace gtb {
#define GTB_END_NAMESPACE }

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Basis/TriLinearLgn.h>
#include <Dataflow/Modules/Fields/IHSMeshUtilities.h>
#include <Dataflow/Modules/Fields/IHSMeshStructures.h>
#include <Dataflow/Modules/Fields/IHSKDTree.h>
#include <vector>
#include <set>

namespace SCIRun {

using std::copy;
using std::vector;
using std::set;    

using namespace std;
using namespace gtb;


// edges used to determine face normals
const int hex_normal_edges[6][2] = { 
  {0, 5}, {2, 6}, {10, 6}, {9, 7}, {1, 11}, {2, 9}};

class GuiInterface;

class InsertHexSheetAlgo : public DynamicAlgoBase
{
public:

  virtual void execute( 
      ProgressReporter *reporter, 
      FieldHandle hexfieldh, FieldHandle trifieldh, 
      FieldHandle& side1field, FieldHandle& side2field,
      bool add_to_side1, bool add_layer ) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    string ext);
};


template <class FIELD>
class InsertHexSheetAlgoHex : public InsertHexSheetAlgo
{
public:
    //! virtual interface. 
  virtual void execute(
      ProgressReporter *reporter, 
      FieldHandle hexfieldh, FieldHandle trifieldh, 
      FieldHandle& side1field, FieldHandle& side2field,
      bool add_to_side1, bool add_layer );
  
  void load_hex_mesh( FieldHandle hexfieldh );
  void load_tri_mesh( TriSurfMesh<TriLinearLgn<Point> > *sci_tri_mesh );
  
  void compute_intersections(
      ProgressReporter* mod,
      HexVolMesh<HexTrilinearLgn<Point> >* original_mesh,
      TriSurfMesh<TriLinearLgn<Point> > *tri_mesh,
      HexVolMesh<HexTrilinearLgn<Point> >*& side1_mesh,
      HexVolMesh<HexTrilinearLgn<Point> >*& side2_mesh, 
      bool add_to_side1, bool add_layer );
  
  bool interferes(const vector<Vector3> &p, const Vector3 &axis, int split);
  
  bool intersects( const HexMesh &hexmesh, int hex_index,
                   const TriangleMesh &trimesh, int face_index);
  void compute_intersections_KDTree( ProgressReporter* mod,
                                     vector<int> &crosses,
                                     const TriangleMesh& trimesh,
                                     const HexMesh& hexmesh);


  // Return the connected boundary faces.  They will be adjacent to
  // each other in the connected_faces vector, so (0,1) will be
  // paired, (2, 3) will be paired, etc.
  void separate_non_man_faces( vector<unsigned int>& connected_faces,
                               const typename FIELD::mesh_type &mesh,
                               typename FIELD::mesh_type::Edge::index_type non_man_edge_id,
                               typename FIELD::mesh_type::Face::array_type &non_man_boundary_faces );

  static bool pair_less(const pair<double, unsigned int> &a,
                        const pair<double, unsigned int> &b)
  {
    return a.first < b.first;
  }


  void node_get_faces(typename FIELD::mesh_type *mesh,
                      set<typename FIELD::mesh_type::Face::index_type> &result,
                      typename FIELD::mesh_type::Node::index_type node);
  
  void edge_get_faces( typename FIELD::mesh_type *mesh,
                       set<typename FIELD::mesh_type::Face::index_type> &result,
                       typename FIELD::mesh_type::Edge::index_type edge );

  class TriangleMeshFaceTree 
  {
public:
    
    typedef ::Box3 Box3;
    typedef ::Point3 Point3;
    
    TriangleMeshFaceTree(const TriangleMesh &m) : mesh(m) { }
    Box3 bounding_box(int i) const {
      return Box3::bounding_box(mesh.verts[mesh.faces[i].verts[0]].point,
                                mesh.verts[mesh.faces[i].verts[1]].point,
                                mesh.verts[mesh.faces[i].verts[2]].point);
    }
    
    const TriangleMesh &mesh;
  };

private:
  TriangleMesh trimesh;
  HexMesh hexmesh;
};


template <class FIELD>
void
InsertHexSheetAlgoHex<FIELD>::execute(
    ProgressReporter *mod, FieldHandle hexfieldh, FieldHandle trifieldh,
    FieldHandle& side1field, FieldHandle& side2field,
    bool add_to_side1, bool add_layer )
{
  typename FIELD::mesh_type *original_mesh =
      dynamic_cast<typename FIELD::mesh_type *>(hexfieldh->mesh().get_rep());
  TriSurfMesh<TriLinearLgn<Point> > *tri_mesh = 
      dynamic_cast<TriSurfMesh<TriLinearLgn<Point> >*>(trifieldh->mesh().get_rep());
  
  load_tri_mesh( tri_mesh );
  cerr << " Finished" << endl;
  mod->update_progress( 0.05 );

  load_hex_mesh( hexfieldh );
  cerr << " Finished" << endl;
  mod->update_progress( 0.15 );
  
  typename FIELD::mesh_type *mesh_rep = 
      dynamic_cast<typename FIELD::mesh_type *>(hexfieldh->mesh().get_rep());
  typename FIELD::mesh_type *side1_mesh = scinew typename FIELD::mesh_type();
  side1_mesh->copy_properties( mesh_rep );
  typename FIELD::mesh_type *side2_mesh = scinew typename FIELD::mesh_type();
  side2_mesh->copy_properties( mesh_rep );

  compute_intersections( mod, original_mesh, tri_mesh, side1_mesh, side2_mesh, add_to_side1, add_layer );

  side1field = scinew FIELD( side1_mesh );
  side2field = scinew FIELD( side2_mesh );
  side1field->copy_properties( hexfieldh.get_rep() );
  side2field->copy_properties( hexfieldh.get_rep() );
}
    

template <class FIELD>
void
InsertHexSheetAlgoHex<FIELD>::load_tri_mesh( TriSurfMesh<TriLinearLgn<Point> > *sci_tri_mesh )
{ 
  typename TriSurfMesh<TriLinearLgn<Point> >::Node::size_type num_nodes;
  typename TriSurfMesh<TriLinearLgn<Point> >::Elem::size_type num_tris;
  sci_tri_mesh->size( num_nodes );
  sci_tri_mesh->size( num_tris );
  
  cerr << endl << "Loading " << num_tris << " triangles...";
  typename TriSurfMesh<TriLinearLgn<Point> >::Node::iterator nbi, nei;
  sci_tri_mesh->begin( nbi );
  sci_tri_mesh->end( nei );
  unsigned int count = 0;
  while( nbi != nei )
  {
    if( count != *nbi )
        cout << "ERROR: Assumption of node id order is incorrect." << endl;
    Point p;
    sci_tri_mesh->get_center( p, *nbi );
    trimesh.add_point( p.x(), p.y(), p.z() );
    ++nbi;
    count++;
  }
  
  typename TriSurfMesh<TriLinearLgn<Point> >::Face::iterator bi, ei;
  sci_tri_mesh->begin(bi); 
  sci_tri_mesh->end(ei);
  while (bi != ei)
  {
    typename TriSurfMesh<TriLinearLgn<Point> >::Node::array_type onodes;
    sci_tri_mesh->get_nodes(onodes, *bi);
    int vi[3];
    vi[0] = (unsigned int)onodes[0];
    vi[1] = (unsigned int)onodes[1];
    vi[2] = (unsigned int)onodes[2];
    trimesh.add_tri( vi );
    ++bi;
  }
  
    // We've read all the data - build the actual structures now.
  std::vector<int> facemap, vertmap;
  trimesh.IdentityMap(facemap, num_tris);
  trimesh.IdentityMap(vertmap, num_nodes);
  trimesh.build_structures(facemap, vertmap);
}


template <class FIELD>
void
InsertHexSheetAlgoHex<FIELD>::load_hex_mesh( FieldHandle hexfieldh )
{
  typename FIELD::mesh_type *hex_mesh =
    dynamic_cast<typename FIELD::mesh_type *>(hexfieldh->mesh().get_rep());

  typename FIELD::mesh_type::Node::size_type num_nodes;
  typename FIELD::mesh_type::Elem::size_type num_hexes;
  hex_mesh->size( num_nodes );
  hex_mesh->size( num_hexes );
  
  cerr << "Loading " << num_hexes << " hexes...";
  hexmesh.hexes.resize( num_hexes );
  hexmesh.points.resize( num_nodes );
  
  typename FIELD::mesh_type::Node::iterator nbi, nei;
  hex_mesh->begin( nbi );
  hex_mesh->end( nei );
  unsigned int count = 0;
  while( nbi != nei )
  {
    if( count != *nbi )
        cout << "ERROR: Assumption of node id order is incorrect." << endl;
    Point p;
    hex_mesh->get_center( p, *nbi );
    hexmesh.points[count] = gtb::Point3( p.x(), p.y(), p.z() );
    ++nbi;
    count++;
  }
  
  hex_mesh->synchronize( Mesh::FACES_E );   
  typename FIELD::mesh_type::Elem::iterator bi, ei;
  hex_mesh->begin(bi); 
  hex_mesh->end(ei);
  count = 0;
  while (bi != ei)
  {
    if( count != *bi )
        cout << "ERROR: Assumption of hex id order is incorrect." << endl;
    
    typename FIELD::mesh_type::Node::array_type onodes;
     
    hex_mesh->get_nodes(onodes, *bi);
//     for (unsigned j=0; j<8; ++j)
//         hexmesh.hexes[count].verts[j] = onodes[j];
    hexmesh.hexes[count].verts[0] = onodes[0];
    hexmesh.hexes[count].verts[1] = onodes[1];
    hexmesh.hexes[count].verts[2] = onodes[3];
    hexmesh.hexes[count].verts[3] = onodes[2];
    hexmesh.hexes[count].verts[4] = onodes[4];
    hexmesh.hexes[count].verts[5] = onodes[5];
    hexmesh.hexes[count].verts[6] = onodes[7];
    hexmesh.hexes[count].verts[7] = onodes[6];
    ++bi;
    count++;
  }
}


//! \Brief projects points on the axis, tests overlap.
template <class FIELD>
bool
InsertHexSheetAlgoHex<FIELD>::interferes(const vector<Vector3> &p,
                                         const Vector3 &axis, int split )
{
  vector<Vector3> v(p.size());
  vector<float> d(p.size());
  for (unsigned i=0; i<p.size(); ++i)
  {
    // Project each point on axis by projected(d) = v.v^T.e
    v[i] = axis * axis.dot(p[i]);
    // Get the signed distance to points from null space of v
    d[i] = (float)v[i].dot(axis);
  }
  
  const float
      mnh = *min_element(d.begin(), d.begin()+split),
      mxh = *max_element(d.begin(), d.begin()+split),
      mnt = *min_element(d.begin()+split, d.end()),
      mxt = *max_element(d.begin()+split, d.end());
  const bool
      mntmxh = mnt <= mxh,
      mxhmxt = mxh <= mxt,
      mnhmnt = mnh <= mnt,
      mnhmxt = mnh <= mxt,
      mxtmxh = mxt <= mxh,
      mntmnh = mnt <= mnh;
  return (mntmxh && mxhmxt) || (mnhmnt && mntmxh) ||
      (mnhmxt && mxtmxh) || (mntmnh && mnhmxt);
}


template <class FIELD>
bool
InsertHexSheetAlgoHex<FIELD>::intersects(const HexMesh &hexmesh,
                                         int hex_index,
                                         const TriangleMesh &trimesh,
                                         int face_index)
{
  const TriangleMeshFace &face = trimesh.faces[face_index];
  const Hex &hex = hexmesh.hexes[hex_index];

  Vector3 triangle_edges[3] = {
      trimesh.verts[face.verts[1]].point-trimesh.verts[face.verts[0]].point,
      trimesh.verts[face.verts[2]].point-trimesh.verts[face.verts[1]].point,
      trimesh.verts[face.verts[0]].point-trimesh.verts[face.verts[2]].point
  };
  
  Vector3 hex_edges[12] = {
      hexmesh.points[hex.verts[0]] - hexmesh.points[hex.verts[1]],
      hexmesh.points[hex.verts[2]] - hexmesh.points[hex.verts[3]],
      hexmesh.points[hex.verts[4]] - hexmesh.points[hex.verts[5]],
      hexmesh.points[hex.verts[6]] - hexmesh.points[hex.verts[7]],
      hexmesh.points[hex.verts[0]] - hexmesh.points[hex.verts[2]],
      hexmesh.points[hex.verts[1]] - hexmesh.points[hex.verts[3]],
      hexmesh.points[hex.verts[4]] - hexmesh.points[hex.verts[6]],
      hexmesh.points[hex.verts[5]] - hexmesh.points[hex.verts[7]],
      hexmesh.points[hex.verts[0]] - hexmesh.points[hex.verts[4]],
      hexmesh.points[hex.verts[1]] - hexmesh.points[hex.verts[5]],
      hexmesh.points[hex.verts[2]] - hexmesh.points[hex.verts[6]],
      hexmesh.points[hex.verts[3]] - hexmesh.points[hex.verts[7]]
  };

  vector<Vector3> ps(11);
  for (int i=0; i<8; ++i)
      ps[i] = hexmesh.points[hex.verts[i]] - Point3(0,0,0);
  for (int i=8; i<11; ++i)
      ps[i] = trimesh.verts[face.verts[i-8]].point - Point3(0,0,0);
  for (int i=0; i<3; ++i)
      triangle_edges[i].normalize();
  for (int i=0; i<12; ++i)
  {
    hex_edges[i].normalize();
    for (int j=0; j<3; ++j)
        if (!interferes(ps, hex_edges[i].cross(triangle_edges[j]), 8))
            return false;
  }

  for (int i=0; i<6; ++i)
    if (!interferes(ps, hex_edges[hex_normal_edges[i][0]].
                    cross(hex_edges[hex_normal_edges[i][1]]), 8))
      return false;
  return interferes(ps, triangle_edges[1].cross(triangle_edges[0]), 8);
}


template <class FIELD>
void
InsertHexSheetAlgoHex<FIELD>::compute_intersections_KDTree(
    ProgressReporter *mod, vector<int> &crosses, 
    const TriangleMesh& trimesh, const HexMesh& hexmesh )
{
  vector<int> kdfi;
  for (unsigned i=0; i<trimesh.faces.size(); i++) 
  {
    kdfi.push_back(i);
  }
  TriangleMeshFaceTree kdtreebbox(trimesh);
  gtb::BoxKDTree<int, TriangleMeshFaceTree> kdtree(kdfi, kdtreebbox);
  
  int total_hexes = (int)hexmesh.hexes.size();
  
  for (int h=0; h<(int)hexmesh.hexes.size(); h++) 
  {  
    Box3 hbbox = Box3::bounding_box(hexmesh.points[hexmesh.hexes[h].verts[0]],
                                    hexmesh.points[hexmesh.hexes[h].verts[1]],
                                    hexmesh.points[hexmesh.hexes[h].verts[2]]);
    hbbox.update(hexmesh.points[hexmesh.hexes[h].verts[3]]);
    hbbox.update(hexmesh.points[hexmesh.hexes[h].verts[4]]);
    hbbox.update(hexmesh.points[hexmesh.hexes[h].verts[5]]);
    hbbox.update(hexmesh.points[hexmesh.hexes[h].verts[6]]);
    hbbox.update(hexmesh.points[hexmesh.hexes[h].verts[7]]);
    
    vector<int> possible;
    kdtree.GetIntersectedBoxes(kdtreebbox, hbbox, possible);
    
		for (int i=0; i<(int)possible.size(); i++) 
    {  
			if (intersects(hexmesh, h, trimesh, possible[i])) 
      {
        crosses[h] = 0;
        break;
      }
    }

    if( h%100 == 0 )
    {
      double temp = 0.15 + 0.5*h/total_hexes;
      mod->update_progress( temp );
    }
  }
}


template <class FIELD>
void
InsertHexSheetAlgoHex<FIELD>::compute_intersections(
    ProgressReporter* mod,
    HexVolMesh<HexTrilinearLgn<Point> >* original_mesh,
    TriSurfMesh<TriLinearLgn<Point> > *tri_mesh,
    HexVolMesh<HexTrilinearLgn<Point> >*& side1_mesh,
    HexVolMesh<HexTrilinearLgn<Point> >*& side2_mesh,
    bool add_to_side1, bool add_layer )
{
#ifdef HAVE_HASH_MAP
  typedef hash_map<unsigned int,
    typename FIELD::mesh_type::Node::index_type,
    hash<unsigned int>,
    equal_to<unsigned int> > hash_type;
#else
  typedef map<unsigned int,
    typename FIELD::mesh_type::Node::index_type,
    less<unsigned int> > hash_type;
#endif
#ifdef HAVE_HASH_MAP
  typedef hash_map<unsigned int,
    typename FIELD::mesh_type::Edge::index_type,
    hash<unsigned int>,
    equal_to<unsigned int> > hash_type2;
#else
  typedef map<unsigned int,
    typename FIELD::mesh_type::Edge::index_type,
    less<unsigned int> > hash_type2;
#endif
#ifdef HAVE_HASH_MAP
  typedef hash_map<unsigned int,
    typename FIELD::mesh_type::Face::index_type,
    hash<unsigned int>,
    equal_to<unsigned int> > hash_type3;
#else
  typedef map<unsigned int,
    typename FIELD::mesh_type::Face::index_type,
    less<unsigned int> > hash_type3;
#endif

  vector<int> crosses(hexmesh.hexes.size(), -1);
  vector<int> hexes(hexmesh.hexes.size());
  for (unsigned i=0; i<hexmesh.hexes.size(); ++i)
      hexes[i] = i;
  vector<int> faces(trimesh.faces.size());
  
  for (unsigned i=0; i<trimesh.faces.size(); ++i)
    faces[i] = i;

  compute_intersections_KDTree( mod, crosses, trimesh, hexmesh);
  
  // Flood the two sides.
  mod->update_progress( 0.65 );
  
  for (int side=0; side<2; side++) 
  {
    int start = -1;
    for (unsigned i=0; i<crosses.size(); i++) 
    {
      if (crosses[i] < 0) 
      {
        start=(int)i;
        break;
      }
    }
    
    if (start==-1) 
    {
      cerr<<"couldn't find hex to start flood from!"<<endl;
      break;
    }
    
    vector<int> toprocess;
    toprocess.push_back(start);
    crosses[start] = side+1;
    
    while (toprocess.size()) 
    {
      int h = toprocess.back();
      toprocess.resize(toprocess.size()-1);

      typename FIELD::mesh_type::Cell::array_type neighbors;
      original_mesh->get_neighbors( neighbors, h );
      typename FIELD::mesh_type::Cell::array_type::iterator iter = neighbors.begin();
      unsigned int i;
      
      if( neighbors.size() > 6 )
        cerr << "ERROR: More than six neighbors reported..." << h << endl;

      for( i = 0; i < neighbors.size(); i++ )
      {
        int hnbr = neighbors[i];
        if (crosses[hnbr] < 0) 
        {
          crosses[hnbr] = side+1;
          toprocess.push_back(hnbr);
        }
      }
    }
  }

  mod->update_progress( 0.70 );

    //check for non_manifold elements in the input mesh... 
  unsigned int k; 
  original_mesh->synchronize( Mesh::NODE_NEIGHBORS_E | Mesh::EDGES_E | Mesh::FACE_NEIGHBORS_E | Mesh::FACES_E );

  bool nmedges_clear = false;
  do
  {
    hash_type2 nmedgemap1, nmedgemap2, nmedgemap3;  
//  set<typename FIELD::mesh_type::Face::index_type> nmboundary_faces;
    
    for( k = 0; k < crosses.size(); ++k )
    {
      if( add_to_side1 )
      {
        if( crosses[k] == 2 )
        {
            // Get all the faces in the cell.
          typename FIELD::mesh_type::Face::array_type faces;
          original_mesh->get_faces( faces, k );
          
            // Check each face for neighbors. 
          typename FIELD::mesh_type::Face::array_type::iterator fiter = faces.begin();
          
          while( fiter != faces.end() )
          {
            typename FIELD::mesh_type::Cell::index_type nci;
            typename FIELD::mesh_type::Face::index_type fi = *fiter;
            ++fiter;
            
            if( original_mesh->get_neighbor( nci, k, fi ) )
            {
              if( crosses[nci] != 2 )
              {
                  //This face is on the boundary of the cut mesh...
//              nmboundary_faces.insert( fi );
                
                typename FIELD::mesh_type::Edge::array_type face_edges;
                original_mesh->get_edges( face_edges, fi );
                typename hash_type2::iterator esearch;
                for( unsigned int i = 0; i < face_edges.size(); i++ )
                {
                  esearch = nmedgemap1.find( face_edges[i] );
                  if( esearch == nmedgemap1.end() )
                  {
                    nmedgemap1[face_edges[i]] = face_edges[i];
                  }
                  else
                  {
                    esearch = nmedgemap2.find( face_edges[i] );
                    if( esearch == nmedgemap2.end() )
                    {
                      nmedgemap2[face_edges[i]] = face_edges[i];
                    }
                    else
                    {
                      esearch = nmedgemap3.find( face_edges[i] );
                      if( esearch == nmedgemap3.end() )
                      {
                        nmedgemap3[face_edges[i]] = face_edges[i];
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      else
      {
        if( crosses[k] == 1 )
        {
            // Get all the faces in the cell.
          typename FIELD::mesh_type::Face::array_type faces;
          original_mesh->get_faces( faces, k );
          
            // Check each face for neighbors. 
          typename FIELD::mesh_type::Face::array_type::iterator fiter = faces.begin();
          
          while( fiter != faces.end() )
          {
            typename FIELD::mesh_type::Cell::index_type nci;
            typename FIELD::mesh_type::Face::index_type fi = *fiter;
            ++fiter;
            
            if( original_mesh->get_neighbor( nci, k, fi ) )
            {
              if( crosses[nci] != 1 )
              {
                  //This face is on the boundary of the cut mesh...
                typename FIELD::mesh_type::Edge::array_type face_edges;
                original_mesh->get_edges( face_edges, fi );
                typename hash_type2::iterator esearch;
                for( unsigned int i = 0; i < face_edges.size(); i++ )
                {
                  esearch = nmedgemap1.find( face_edges[i] );
                  if( esearch == nmedgemap1.end() )
                  {
                    nmedgemap1[face_edges[i]] = face_edges[i];
                  }
                  else
                  {
                    esearch = nmedgemap2.find( face_edges[i] );
                    if( esearch == nmedgemap2.end() )
                    {
                      nmedgemap2[face_edges[i]] = face_edges[i];
                    }
                    else
                    {
                      esearch = nmedgemap3.find( face_edges[i] );
                      if( esearch == nmedgemap3.end() )
                      {
                        nmedgemap3[face_edges[i]] = face_edges[i];
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if( nmedgemap3.size() != 0 )
    { 
      cout << "WARNING: New mesh will contain " << nmedgemap3.size() << " non-manifold edges.\n";
      typename hash_type2::iterator nmsearch = nmedgemap3.begin();
      while( nmsearch != nmedgemap3.end() )
      {
        typename FIELD::mesh_type::Elem::array_type attached_hexes;
        typename FIELD::mesh_type::Edge::index_type this_edge = (*nmsearch).first;
        original_mesh->get_elems( attached_hexes, this_edge );
        
        typename FIELD::mesh_type::Elem::index_type hex_to_change = 0;
        for( unsigned int i = 0; i < attached_hexes.size(); i++ )
        {
          typename FIELD::mesh_type::Elem::index_type hex_id = attached_hexes[i];
//NOTE TO JS1: This is kind of random... We can/should probably be a lot smarter about how we add these....
          if( add_to_side1 )
          {
            if( crosses[hex_id] == 2 )
            {
              hex_to_change = hex_id;
            }
          }
          else
          {
            if( crosses[hex_id] == 1 )
            {
              hex_to_change = hex_id;
            }
          } 
        }
        
        crosses[hex_to_change] = 0;
//end NOTE TO JS1
        
        ++nmsearch;
      }
    }
    else
    {
      nmedges_clear = true;
    }
    
  }while( nmedges_clear == false );
      
        // Need to add elements from the three sets of elements.
      hash_type side1_nodemap, side2_nodemap, side1_reverse_map, side2_reverse_map;
  for( k = 0; k < crosses.size(); ++k )
  {
    typename FIELD::mesh_type::Node::array_type onodes;
    typename FIELD::mesh_type::Elem::index_type elem_id = k;
    original_mesh->get_nodes( onodes, elem_id );
    typename FIELD::mesh_type::Node::array_type nnodes(onodes.size());
    
    if( crosses[k] == 0 )
    {
//         //add to intersect_mesh
//       for (unsigned int i = 0; i < onodes.size(); i++)
//       {
//         if( intersect_nodemap.find((unsigned int)onodes[i]) == intersect_nodemap.end())
//         {
//           Point np;
//           original_mesh->get_center( np, onodes[i] );
//           const typename FIELD::mesh_type::Node::index_type nodeindex =
//               intersect_mesh->add_point( np );
//           intersect_nodemap[(unsigned int)onodes[i]] = nodeindex;
//           nnodes[i] = nodeindex;
//         }
//         else
//         {
//           nnodes[i] = intersect_nodemap[(unsigned int)onodes[i]];
//         }
//       }
//       intersect_mesh->add_elem( nnodes );
      if( add_to_side1 )
      {
        for (unsigned int i = 0; i < onodes.size(); i++)
        {
          if( side1_nodemap.find((unsigned int)onodes[i]) == side1_nodemap.end())
          {
            Point np;
            original_mesh->get_center( np, onodes[i] );
            const typename FIELD::mesh_type::Node::index_type nodeindex =
                side1_mesh->add_point( np );
            side1_nodemap[(unsigned int)onodes[i]] = nodeindex;
            side1_reverse_map[nodeindex] = (unsigned int)onodes[i];
            nnodes[i] = nodeindex;
          }
          else
          {
            nnodes[i] = side1_nodemap[(unsigned int)onodes[i]];
          }
        }
        side1_mesh->add_elem( nnodes );
      }
      else
      {
          // Add to side2_mesh.
        for (unsigned int i = 0; i < onodes.size(); i++)
        {
          if( side2_nodemap.find((unsigned int)onodes[i]) == side2_nodemap.end())
          {
            Point np;
            original_mesh->get_center( np, onodes[i] );
            const typename FIELD::mesh_type::Node::index_type nodeindex =
                side2_mesh->add_point( np );
            side2_nodemap[(unsigned int)onodes[i]] = nodeindex;
            side2_reverse_map[nodeindex] = (unsigned int)onodes[i];
            nnodes[i] = nodeindex;
          }
          else
          {
            nnodes[i] = side2_nodemap[(unsigned int)onodes[i]];
          }
        }
        side2_mesh->add_elem( nnodes );
      }
    }
    else if( crosses[k] == 1 )
    {
        // Add to side1_mesh.
      for (unsigned int i = 0; i < onodes.size(); i++)
      {
        if( side1_nodemap.find((unsigned int)onodes[i]) == side1_nodemap.end())
        {
          Point np;
          original_mesh->get_center( np, onodes[i] );
          const typename FIELD::mesh_type::Node::index_type nodeindex =
              side1_mesh->add_point( np );
          side1_nodemap[(unsigned int)onodes[i]] = nodeindex;
          side1_reverse_map[nodeindex] = (unsigned int)onodes[i];
          nnodes[i] = nodeindex;
        }
        else
        {
          nnodes[i] = side1_nodemap[(unsigned int)onodes[i]];
        }
      }
      side1_mesh->add_elem( nnodes );
    }
    else
    {
        // Add to side2_mesh.
      for (unsigned int i = 0; i < onodes.size(); i++)
      {
        if( side2_nodemap.find((unsigned int)onodes[i]) == side2_nodemap.end())
        {
          Point np;
          original_mesh->get_center( np, onodes[i] );
          const typename FIELD::mesh_type::Node::index_type nodeindex =
              side2_mesh->add_point( np );
          side2_nodemap[(unsigned int)onodes[i]] = nodeindex;
          side2_reverse_map[nodeindex] = (unsigned int)onodes[i];
          nnodes[i] = nodeindex;
        }
        else
        {
          nnodes[i] = side2_nodemap[(unsigned int)onodes[i]];
        }
      }
      side2_mesh->add_elem( nnodes );
    }
  }
  
  mod->update_progress( 0.75 );
  if( add_layer )
  {  
    cout << "Adding the new layers of hexes...";
    typename HexVolMesh<HexTrilinearLgn<Point> >::Node::size_type s1_node_size;
    typename HexVolMesh<HexTrilinearLgn<Point> >::Node::size_type s2_node_size;
    typename hash_type::iterator node_iter; 
    vector<typename FIELD::mesh_type::Node::index_type> oi_node_list;  
    hash_type shared_vertex_map;
    unsigned int count = 0;
    
    if( s1_node_size < s2_node_size )
    {
      typename hash_type::iterator hitr = side1_nodemap.begin();
      while( hitr != side1_nodemap.end() )
      {
        node_iter = side2_nodemap.find( (*hitr).first );
        if( node_iter != side2_nodemap.end() )
        {
            // Want this one.
          oi_node_list.push_back( (*hitr).first );
          count++;
        }
        ++hitr;
      }
    }
    else
    {
      typename hash_type::iterator hitr = side2_nodemap.begin();
      while( hitr != side2_nodemap.end() )
      {
        node_iter = side1_nodemap.find( (*hitr).first );
        if( node_iter != side1_nodemap.end() )
        {
            // Want this one.
          oi_node_list.push_back( (*hitr).first );
          shared_vertex_map[(*hitr).first] = (*hitr).first;
          count++;
        }
        ++hitr;
      }
    }
    
    tri_mesh->synchronize( Mesh::LOCATE_E );
    hash_type new_map1, new_map2;
    unsigned int i; 
    
    for( i = 0; i < oi_node_list.size(); i++ )
    {
      typename FIELD::mesh_type::Node::index_type this_node = oi_node_list[i];
      Point n_p;
      original_mesh->get_center( n_p, this_node );
      
      Point new_result;
      typename FIELD::mesh_type::Face::index_type face_id;
      tri_mesh->find_closest_elem( new_result, face_id, n_p );
      Vector dist_vect = 1.5*( new_result - n_p );
      
        // Finding the closest face can be slow.  Update the progress meter
        // to let the user know that we are performing calculations and the 
        // process has not hung.
      if( i%50 == 0 )
      {
        double temp = 0.75 + 0.25*( (double)i/(double)oi_node_list.size() );
        mod->update_progress( temp );
      }
      
        // Add the new node to the clipped mesh.
      Point new_point( new_result );
      typename FIELD::mesh_type::Node::index_type this_index1 = side1_mesh->add_point( new_point ); 
      typename FIELD::mesh_type::Node::index_type this_index2 = side2_mesh->add_point( new_point );
      
        // Create a map for the new node to a node on the boundary of
        // the clipped mesh.
      typename hash_type::iterator node_iter; 
      node_iter = new_map1.find( side1_nodemap[this_node] );
      if( node_iter == new_map1.end() )
      {
        new_map1[side1_nodemap[this_node]] = this_index1;
      }
      else
      {
        cout << "ERROR\n";
      }
      
      node_iter = new_map2.find( side2_nodemap[this_node] );
      if( node_iter == new_map2.end() )
      {
        new_map2[side2_nodemap[this_node]] = this_index2;
      }
      else
      {
        cout << "ERROR2\n";
      }
      
      if( add_to_side1 )
      {  
        Point p;
        side1_mesh->get_point( p, side1_nodemap[this_node] );
        double x = p.x(), y = p.y(), z = p.z();
        p.x( dist_vect.x() + x );
        p.y( dist_vect.y() + y );
        p.z( dist_vect.z() + z );
        side1_mesh->set_point( p, side1_nodemap[this_node] );
      }
      else
      {
        Point p;
        side2_mesh->get_point( p, side2_nodemap[this_node] );
        double x = p.x(), y = p.y(), z = p.z();
        p.x( dist_vect.x() + x );
        p.y( dist_vect.y() + y );
        p.z( dist_vect.z() + z );
        side2_mesh->set_point( p, side2_nodemap[this_node] );
      }
    }
    cout << "\nFound " << count << " nodes along the shared boundary...\n";
    
    side1_mesh->synchronize( Mesh::NODE_NEIGHBORS_E | Mesh::EDGES_E | Mesh::FACE_NEIGHBORS_E | Mesh::FACES_E );
    side2_mesh->synchronize( Mesh::NODE_NEIGHBORS_E | Mesh::EDGES_E | Mesh::FACE_NEIGHBORS_E | Mesh::FACES_E );
    
    set<typename FIELD::mesh_type::Face::index_type> boundary_faces;
    
      //Walk all the cells in the smallest clipped mesh to find the boundary faces...
    typename FIELD::mesh_type::Cell::iterator citer; side1_mesh->begin(citer);
    typename FIELD::mesh_type::Cell::iterator citere; side1_mesh->end(citere);
    hash_type3 face_list;
    hash_type2 edgemap1, edgemap2, edgemap3;
    
    while( citer != citere )
    {
      typename FIELD::mesh_type::Cell::index_type ci = *citer;
      ++citer;
      
        // Get all the faces in the cell.
      typename FIELD::mesh_type::Face::array_type faces;
      side1_mesh->get_faces( faces, ci );
      
        // Check each face for neighbors.
      typename FIELD::mesh_type::Face::array_type::iterator fiter = faces.begin();
      
      while( fiter != faces.end() )
      {
        typename FIELD::mesh_type::Cell::index_type nci;
        typename FIELD::mesh_type::Face::index_type fi = *fiter;
        ++fiter;
        
        if( !side1_mesh->get_neighbor( nci, ci, fi ) )
        {
            // Faces with no neighbors are on the boundary...
            //    make sure that this face isn't on the original boundary
          
          typename FIELD::mesh_type::Node::array_type face_nodes;
          side1_mesh->get_nodes( face_nodes, fi );
          typename hash_type::iterator search1, search2, search3, search4, search_end; 
          search_end = shared_vertex_map.end();
          search1 = shared_vertex_map.find( side1_reverse_map[face_nodes[0]] );
          search2 = shared_vertex_map.find( side1_reverse_map[face_nodes[1]] );
          search3 = shared_vertex_map.find( side1_reverse_map[face_nodes[2]] );
          search4 = shared_vertex_map.find( side1_reverse_map[face_nodes[3]] );
          if( search1 != search_end && search2 != search_end &&
              search3 != search_end && search4 != search_end )
          {
            bool ok_to_add_face = true;
            typename FIELD::mesh_type::Face::index_type old_face;
            if( !original_mesh->get_face( old_face, side1_reverse_map[face_nodes[0]], 
                                          side1_reverse_map[face_nodes[1]],
                                          side1_reverse_map[face_nodes[2]], 
                                          side1_reverse_map[face_nodes[3]] ) )
            {
              cout << "ERROR3" << endl;
              ok_to_add_face = false;
            }
            
//NOTE TO JS: Testing (1 - added)...                
            boundary_faces.insert( fi );
            face_list[fi] = fi;
            
              //check for non-manifold edges... 
            typename FIELD::mesh_type::Edge::array_type face_edges;
            side1_mesh->get_edges( face_edges, fi );
            typename hash_type2::iterator esearch;
            for( i = 0; i < face_edges.size(); i++ )
            {
              esearch = edgemap1.find( face_edges[i] );
              if( esearch == edgemap1.end() )
              {
                edgemap1[face_edges[i]] = face_edges[i];
              }
              else
              {
                esearch = edgemap2.find( face_edges[i] );
                if( esearch == edgemap2.end() )
                {
                  edgemap2[face_edges[i]] = face_edges[i];
                }
                else
                {
                  ok_to_add_face = false;
                  esearch = edgemap3.find( face_edges[i] );
                  if( esearch == edgemap3.end() )
                  {
                    edgemap3[face_edges[i]] = face_edges[i];
                  }
                }
              }
            }
            
            if( ok_to_add_face )
            { 
              typename hash_type3::iterator test_iter = face_list.find( fi );
              if( test_iter == face_list.end() )
              {
                face_list[fi] = fi;
//NOTE TO JS: Testing (2 - commented out...)
//                  boundary_faces.insert( fi );
              }
            }
          }
        }
      }
    }
    
      //special casing for projecting faces connected to non-manifold edges...
    if( edgemap3.size() != 0 )
    { 
      cout << "WARNING: Clipped mesh contains " << edgemap3.size() << " non-manifold edges.\n    Ids are:";
      cout << "Boundary has " << boundary_faces.size() << " faces.\n";
      
      typename hash_type2::iterator nm_search = edgemap3.begin();
      typename hash_type2::iterator nm_search_end = edgemap3.end();
      while( nm_search != nm_search_end )
      {
        typename FIELD::mesh_type::Edge::index_type this_edge = (*nm_search).first;
//          cout << " " << this_edge;
        
 //        typename FIELD::mesh_type::Face::array_type edges_faces;
// //        get_faces( side1_mesh, edges_faces, this_edge );
//         edge_get_faces( side1_mesh, edges_faces, this_edge );
//         vector<typename FIELD::mesh_type::Face::index_type> face_list2;
//         for( i = 0; i < edges_faces.size(); i++ )
//         {
//           typename FIELD::mesh_type::Elem::array_type faces_hexes;
//           side1_mesh->get_elems( faces_hexes, edges_faces[i] );
//           if( faces_hexes.size() == 1 )
//           {
//             face_list2.push_back( edges_faces[i] );
//           }
//         }
        
//NOTE TO JS: Need to remove this section of code once the non-manifold maps are set up...
//           for( i = 0; i < face_list2.size(); i++ )
//           {
//             typename hash_type3::iterator face_iter;
//             face_iter = face_list.find( face_list2[i] );
//             if( face_iter != face_list.end() )
//             {
//               face_list.erase( face_iter );
//             }
//           }
//end NOTE TO JS...
        
        bool multiproject_node1 = false;
        bool multiproject_node2 = false;
        typename FIELD::mesh_type::Node::array_type problem_nodes;
        side1_mesh->get_nodes( problem_nodes, this_edge );
        typename FIELD::mesh_type::Node::index_type node1 = problem_nodes[0], 
            node2 = problem_nodes[1];
        typename FIELD::mesh_type::Elem::array_type edges_hexes, node1_hexes, node2_hexes;
        side1_mesh->get_elems( edges_hexes, this_edge );
        side1_mesh->get_elems( node1_hexes, node1 );
        side1_mesh->get_elems( node2_hexes, node2 );
        
//         vector<unsigned int> connected_faces;
//         separate_non_man_faces( connected_faces, *side1_mesh, 
//                                 this_edge, face_list2 );
        
//        cout << endl;
//            cout << "Paired faces for edge " << this_edge << ":\n";
//            for( i = 0; i < connected_faces.size(); i++ )
//            {
//              cout << connected_faces[i++] << " and ";
//              cout << connected_faces[i] << endl;
//            }
        
        if( node1_hexes.size() == edges_hexes.size() )
        {
          cout << " Don't need to multiproject node1 (" << node1 << ")\n";
        }
        else
        {
          cout << " Need to multiproject node1 (" << node1 << ")\n";
          multiproject_node1 = true;
        }
        if( node2_hexes.size() == edges_hexes.size() )
        {
          cout << " Don't need to multiproject node2 (" << node2 << ")\n";
        }
        else
        {
          cout << " Need to multiproject node2 (" << node2 << ")\n";
          multiproject_node2 = true;
        }
        
//         if( multiproject_node1 )
//         {
//           set<typename FIELD::mesh_type::Face::index_type> node1s_faces;
//           node_get_faces( side1_mesh, node1s_faces, node1 );
//           set<typename FIELD::mesh_type::Face::index_type> local_boundary;
          
//           typename set<typename FIELD::mesh_type::Face::index_type>::iterator n1f_iter = node1s_faces.begin();
//           typename set<typename FIELD::mesh_type::Face::index_type>::iterator n1f_end = node1s_faces.end();
//             //separate the non-boundary faces from the array...   
          
// //            cout << "Local boundary: ";
//           while( n1f_iter != n1f_end )
//           {
//             if( boundary_faces.find( *n1f_iter ) != boundary_faces.end() )
//             {
//               local_boundary.insert( *n1f_iter );
// //                cout << *n1f_iter << ", ";
//             }
// //               else
// //                   cout << "*" << *n1f_iter << ", ";
            
//             ++n1f_iter;
//           }
// //            cout << endl;
          
//           set<typename FIELD::mesh_type::Face::index_type> local_projection_patch;
//           for( i = 0; i < connected_faces.size(); i++ )
//           {
//             local_projection_patch.clear();
//             typename FIELD::mesh_type::Face::index_type face1, face2;
//             face1 = connected_faces[i++];
//             face2 = connected_faces[i];
// //              cout << face1 << ", " << face2 << endl;
            
//             if( local_boundary.find( face1 ) != local_boundary.end() )
//             {
//               local_projection_patch.insert( face1 );
//             }
//             else
//             {
//               cout << "ERROR: Unexpected case1 in local patch projection.\n";
//             }
            
//             if( local_boundary.find( face2 ) != local_boundary.end() )
//             {
//               local_projection_patch.insert( face2 );
//             }
//             else
//             {
//               cout << "ERROR: Unexpected case2 in local patch projection.\n";
//             }
            
//               //find the connected faces to face1 and face2 in the local boundary...
//             typename FIELD::mesh_type::Edge::array_type edges1, edges2;
//             side1_mesh->get_edges( edges1, face1 );
//             side1_mesh->get_edges( edges2, face2 );
            
//             set<typename FIELD::mesh_type::Face::index_type> face_compare1, face_compare2;
// //              set<typename FIELD::mesh_type::Face::index_type> face_compare3;
//             unsigned int i, j;
//             for( i = 0; i < 4; i++ )
//             {
//               if( edges1[i] == this_edge )
//                   continue;
              
//               set<typename FIELD::mesh_type::Face::index_type> test_faces;                
//               edge_get_faces( side1_mesh, test_faces, edges1[i] );
              
//               typename set<typename FIELD::mesh_type::Face::index_type>::iterator tf_begin = test_faces.begin();
//               typename set<typename FIELD::mesh_type::Face::index_type>::iterator tf_end = test_faces.end();
//               while( tf_begin != tf_end )
//               {
// //                  if( face_compare1.find( *tf_begin ) == face_compare1.end() )
// //                  {
//                 face_compare1.insert( *tf_begin );
// //                  }
//                 ++tf_begin;
//               }
//             }
            
//             for( i = 0; i < 4; i++ )
//             {
//               if( edges2[i] == this_edge )
//                   continue;
              
//               set<typename FIELD::mesh_type::Face::index_type> test_faces;
//               edge_get_faces( side1_mesh, test_faces, edges2[i] );
//               typename set<typename FIELD::mesh_type::Face::index_type>::iterator tf_begin = test_faces.begin();
//               typename set<typename FIELD::mesh_type::Face::index_type>::iterator tf_end = test_faces.end();
//               while( tf_begin != tf_end )
//               {
//                 if( face_compare1.find( *tf_begin ) != face_compare1.end() )
//                 {
//                   face_compare2.insert( *tf_begin );
//                 }
//                 ++tf_begin;
//               }
//             }
            
//             typename set<typename FIELD::mesh_type::Face::index_type>::iterator fc2_begin = face_compare2.begin();
//             typename set<typename FIELD::mesh_type::Face::index_type>::iterator fc2_end = face_compare2.end();
//             while( fc2_begin != fc2_end )
//             {
//               if( local_boundary.find( *fc2_begin ) != local_boundary.end() )
//               {
//                 bool add_face = true;
//                 for( j = 0; j < face_list2.size(); j++ )
//                 {
//                   if( *fc2_begin == face_list2[j] )
//                   {
//                     add_face = false;
//                   }
//                 }
                
//                 if( add_face )
//                 {
//                   local_projection_patch.insert( *fc2_begin );
// //                    face_compare3.insert( *fc2_begin );
//                 }
//               }
//               ++fc2_begin;
//             }
            
// //               if( face_compare3.size() )
// //               {
// //                 cout << "ADDED " << face_compare3.size() << " faces to the local projection patch...\n";
// //               }
            
//               //create the new node and add a map to the node from each of the faces in the new pairing and old node...
//             typename set<typename FIELD::mesh_type::Face::index_type>::iterator lpp_begin = local_projection_patch.begin();
//             typename set<typename FIELD::mesh_type::Face::index_type>::iterator lpp_end = local_projection_patch.end();              
            
//             if( local_projection_patch.size() != 3 )
//             {
//               cout << "ERROR " << this_edge << ": Didn't find 3 faces for a local projection patch. " << local_projection_patch.size() << endl;
//               while( lpp_begin != lpp_end )
//               {
//                 cout << *lpp_begin << ", ";
//                 ++lpp_begin;
//               }
//               lpp_begin = local_projection_patch.begin();
//               cout << endl;
//             }
            
//             while( lpp_begin != lpp_end )
//             {
//               cout << *lpp_begin << " ";
              
//               if( non_man_projection_map.find( *lpp_begin ) != non_man_projection_map.end() )
//               {
//                 cout << "NOTE TO JS: adding a node to an existing non-manifold face..." << *lpp_begin << " " << node1 << "\n";
                
//                   //add the node to the existing non_man_projection_map for this face...
//                 if( i == 0 )
//                 {
//                     //use the previously created node...
//                   if( node1 == 0 )
//                       cout << "ERROR: UNEXPECTED ID...\n";
                  
// //NOTE TO JS: NEED TO MAP The other meshs node....
//                   NonManNodalMap* this_map = non_man_projection_map[*lpp_begin];
//                   if( this_map->node1 == 0 )
//                   {
//                     this_map->node1 = node1;
//                     this_map->fromNode1 = new_map1[node1];
//                     this_map->otherMeshNode1 = new_map2[side2_nodemap[side1_reverse_map[node1]]];
//                   }
//                   else if( this_map->node2 == 0 )
//                   {
//                     this_map->node2 = node1;
//                     this_map->fromNode2 = new_map1[node1];
//                     this_map->otherMeshNode2 = new_map2[side2_nodemap[side1_reverse_map[node1]]];
//                   }
//                   else if( this_map->node3 == 0 )
//                   {
//                     this_map->node3 = node1;
//                     this_map->fromNode3 = new_map1[node1];
//                     this_map->otherMeshNode3= new_map2[side2_nodemap[side1_reverse_map[node1]]];
//                   }
//                   else if( this_map->node4 == 0 )
//                   {
//                     this_map->node4 = node1;
//                     this_map->fromNode4 = new_map1[node1];
//                     this_map->otherMeshNode4 = new_map2[side2_nodemap[side1_reverse_map[node1]]];
//                   }
//                   else
//                       cout << "ERROR: UNEXPECTED NODE ID...\n";
                  
//                 }
//                 else
//                 {
//                   if( node1 == 0 )
//                       cout << "ERROR: UNEXPECTED ID...\n";     
                  
//                   Point n_p;
//                   original_mesh->get_center( n_p, node1 );
                  
//                   Point new_result;
//                   typename FIELD::mesh_type::Face::index_type face_id;
//                   tri_mesh->find_closest_elem( new_result, face_id, n_p );
                  
//                     // Add the new node to the clipped mesh.
//                   Point new_point( new_result );
//                   typename FIELD::mesh_type::Node::index_type new_node = side1_mesh->add_point( new_point );
//                   typename FIELD::mesh_type::Node::index_type other_meshs_node = side2_mesh->add_point( new_point );
                  
//                   NonManNodalMap* this_map = non_man_projection_map[*lpp_begin];
//                   if( this_map->node1 == 0 )
//                   {
//                     this_map->node1 = node1;
//                     this_map->fromNode1 = new_node;
//                     this_map->otherMeshNode1 = other_meshs_node;
//                   }
//                   else if( this_map->node2 == 0 )
//                   {
//                     this_map->node2 = node1;
//                     this_map->fromNode2 = new_node;
//                     this_map->otherMeshNode2 = other_meshs_node;
//                   }
//                   else if( this_map->node3 == 0 )
//                   {
//                     this_map->node3 = node1;
//                     this_map->fromNode3 = new_node;
//                     this_map->otherMeshNode3 = other_meshs_node;
//                   }
//                   else if( this_map->node4 == 0 )
//                   {
//                     this_map->node4 = node1;
//                     this_map->fromNode4 = new_node;
//                     this_map->otherMeshNode4 = other_meshs_node;
//                   }
//                   else
//                       cout << "ERROR: UNEXPECTED NODE ID...\n";
                  
//                 }
//               }
//               else
//               {
//                   //add the face to the non_man_projection_map with the correct nodes being mapped...
//                 if( i == 0 )
//                 {
//                     //use the previously created node...
//                   if( node1 == 0 )
//                       cout << "ERROR: UNEXPECTED ID...\n";
                  
// //NOTE TO JS: Need to map the other meshs node...
//                   NonManNodalMap* this_map = new NonManNodalMap( *lpp_begin );
//                   non_man_projection_map[*lpp_begin] = this_map;
//                   this_map->node1 = node1;
//                   this_map->fromNode1 = new_map1[node1];
//                   this_map->otherMeshNode1 = new_map2[side2_nodemap[side1_reverse_map[node1]]];
//                 }
//                 else
//                 {
//                     //need to create a new node...
//                   if( node1 == 0 )
//                       cout << "ERROR: UNEXPECTED ID...\n";
//                   Point n_p;
//                   original_mesh->get_center( n_p, node1 );
                  
//                   Point new_result;
//                   typename FIELD::mesh_type::Face::index_type face_id;
//                   tri_mesh->find_closest_elem( new_result, face_id, n_p );
                  
//                     // Add the new node to the clipped mesh.
//                   Point new_point( new_result );
//                   typename FIELD::mesh_type::Node::index_type new_node = side1_mesh->add_point( new_point ); 
//                   typename FIELD::mesh_type::Node::index_type other_meshs_node = side2_mesh->add_point( new_point );
                  
//                   NonManNodalMap* this_map = new NonManNodalMap( *lpp_begin );
//                   non_man_projection_map[*lpp_begin] = this_map;
//                   this_map->node1 = node1;
//                   this_map->fromNode1 = new_node;
//                   this_map->otherMeshNode1 = other_meshs_node;
//                 }
//               }
              
//               ++lpp_begin;
//             }
//             cout << endl;
//           }
          
// //NOTE TO JS: remove the code which special-case removes the faces from the face_list...
// //NOTE TO JS: delete the created NonManNodalMaps
//         }
//         if( multiproject_node2 )
//         {
//           set<typename FIELD::mesh_type::Face::index_type> node2s_faces;
//           node_get_faces( side1_mesh, node2s_faces, node2 );
//           set<typename FIELD::mesh_type::Face::index_type> local_boundary;
          
//           typename set<typename FIELD::mesh_type::Face::index_type>::iterator n2f_iter = node2s_faces.begin();
//           typename set<typename FIELD::mesh_type::Face::index_type>::iterator n2f_end = node2s_faces.end();
//             //separate the non-boundary faces from the array...   
          
// //            cout << "Local boundary: ";
//           while( n2f_iter != n2f_end )
//           {
//             if( boundary_faces.find( *n2f_iter ) != boundary_faces.end() )
//             {
//               local_boundary.insert( *n2f_iter );
// //                cout << *n2f_iter << ", ";
//             }
// //               else
// //                   cout << "*" << *n2f_iter << ", ";
            
//             ++n2f_iter;
//           }
// //            cout << endl;
          
//           set<typename FIELD::mesh_type::Face::index_type> local_projection_patch;
//           for( i = 0; i < connected_faces.size(); i++ )
//           {
//             local_projection_patch.clear();
//             typename FIELD::mesh_type::Face::index_type face1, face2;
//             face1 = connected_faces[i++];
//             face2 = connected_faces[i];
// //              cout << face1 << ", " << face2 << endl;
            
//             if( local_boundary.find( face1 ) != local_boundary.end() )
//             {
//               local_projection_patch.insert( face1 );
//             }
//             else
//             {
//               cout << "ERROR: Unexpected case1 in local patch projection.\n";
//             }
            
//             if( local_boundary.find( face2 ) != local_boundary.end() )
//             {
//               local_projection_patch.insert( face2 );
//             }
//             else
//             {
//               cout << "ERROR: Unexpected case2 in local patch projection.\n";
//             }
            
//               //find the connected faces to face1 and face2 in the local boundary...
//             typename FIELD::mesh_type::Edge::array_type edges1, edges2;
//             side1_mesh->get_edges( edges1, face1 );
//             side1_mesh->get_edges( edges2, face2 );
            
//             set<typename FIELD::mesh_type::Face::index_type> face_compare1, face_compare2;
// //              set<typename FIELD::mesh_type::Face::index_type> face_compare3;
//             unsigned int i, j;
//             for( i = 0; i < 4; i++ )
//             {
//               if( edges1[i] == this_edge )
//                   continue;
              
//               set<typename FIELD::mesh_type::Face::index_type> test_faces;                
//               edge_get_faces( side1_mesh, test_faces, edges1[i] );
              
//               typename set<typename FIELD::mesh_type::Face::index_type>::iterator tf_begin = test_faces.begin();
//               typename set<typename FIELD::mesh_type::Face::index_type>::iterator tf_end = test_faces.end();
//               while( tf_begin != tf_end )
//               {
// //                  if( face_compare1.find( *tf_begin ) == face_compare1.end() )
// //                  {
//                 face_compare1.insert( *tf_begin );
// //                  }
//                 ++tf_begin;
//               }
//             }
            
//             for( i = 0; i < 4; i++ )
//             {
//               if( edges2[i] == this_edge )
//                   continue;
              
//               set<typename FIELD::mesh_type::Face::index_type> test_faces;
//               edge_get_faces( side1_mesh, test_faces, edges2[i] );
//               typename set<typename FIELD::mesh_type::Face::index_type>::iterator tf_begin = test_faces.begin();
//               typename set<typename FIELD::mesh_type::Face::index_type>::iterator tf_end = test_faces.end();
//               while( tf_begin != tf_end )
//               {
//                 if( face_compare1.find( *tf_begin ) != face_compare1.end() )
//                 {
//                   face_compare2.insert( *tf_begin );
//                 }
//                 ++tf_begin;
//               }
//             }
            
//             typename set<typename FIELD::mesh_type::Face::index_type>::iterator fc2_begin = face_compare2.begin();
//             typename set<typename FIELD::mesh_type::Face::index_type>::iterator fc2_end = face_compare2.end();
//             while( fc2_begin != fc2_end )
//             {
//               if( local_boundary.find( *fc2_begin ) != local_boundary.end() )
//               {
//                 bool add_face = true;
//                 for( j = 0; j < face_list2.size(); j++ )
//                 {
//                   if( *fc2_begin == face_list2[j] )
//                   {
//                     add_face = false;
//                   }
//                 }
                
//                 if( add_face )
//                 {
//                   local_projection_patch.insert( *fc2_begin );
// //                    face_compare3.insert( *fc2_begin );
//                 }
//               }
//               ++fc2_begin;
//             }
            
// //               if( face_compare3.size() )
// //               {
// //                 cout << "ADDED " << face_compare3.size() << " faces to the local projection patch...\n";
// //               }
            
//               //create the new node and add a map to the node from each of the faces in the new pairing and old node...
//             typename set<typename FIELD::mesh_type::Face::index_type>::iterator lpp_begin = local_projection_patch.begin();
//             typename set<typename FIELD::mesh_type::Face::index_type>::iterator lpp_end = local_projection_patch.end();              
            
//             if( local_projection_patch.size() != 3 )
//             {
//               cout << "ERROR " << this_edge << ": Didn't find 3 faces for a local projection patch. " << local_projection_patch.size() << endl;
//               while( lpp_begin != lpp_end )
//               {
//                 cout << *lpp_begin << ", ";
//                 ++lpp_begin;
//               }
//               lpp_begin = local_projection_patch.begin();
//               cout << endl;
//             }
            
//             while( lpp_begin != lpp_end )
//             {
//               cout << *lpp_begin << " ";
//               if( non_man_projection_map.find( *lpp_begin ) != non_man_projection_map.end() )
//               {
//                 cout << "NOTE TO JS: adding a node to an existing non-manifold face..." << *lpp_begin << " " << node2 << "\n";
                
//                   //add the node to the existing non_man_projection_map for this face...
//                 if( i == 0 )
//                 {
//                     //use the previously created node...
//                   if( node2 == 0 )
//                       cout << "ERROR: UNEXPECTED ID...\n";
                  
// //NOTE TO JS: NEED TO MAP The other meshs node....
//                   NonManNodalMap* this_map = non_man_projection_map[*lpp_begin];
//                   if( this_map->node1 == 0 )
//                   {
//                     this_map->node1 = node2;
//                     this_map->fromNode1 = new_map1[node2];
//                     this_map->otherMeshNode1 = new_map2[side2_nodemap[side1_reverse_map[node2]]];
//                   }
//                   else if( this_map->node2 == 0 )
//                   {
//                     this_map->node2 = node2;
//                     this_map->fromNode2 = new_map1[node2];
//                     this_map->otherMeshNode2 = new_map2[side2_nodemap[side1_reverse_map[node2]]];
//                   }
//                   else if( this_map->node3 == 0 )
//                   {
//                     this_map->node3 = node2;
//                     this_map->fromNode3 = new_map1[node2];
//                     this_map->otherMeshNode3= new_map2[side2_nodemap[side1_reverse_map[node2]]];
//                   }
//                   else if( this_map->node4 == 0 )
//                   {
//                     this_map->node4 = node2;
//                     this_map->fromNode4 = new_map1[node2];
//                     this_map->otherMeshNode4 = new_map2[side2_nodemap[side1_reverse_map[node2]]];
//                   }
//                   else
//                       cout << "ERROR: UNEXPECTED NODE ID...\n";
                  
//                 }
//                 else
//                 {
//                   if( node2 == 0 )
//                       cout << "ERROR: UNEXPECTED ID...\n";     
                  
//                   Point n_p;
//                   original_mesh->get_center( n_p, node2 );
                  
//                   Point new_result;
//                   typename FIELD::mesh_type::Face::index_type face_id;
//                   tri_mesh->find_closest_elem( new_result, face_id, n_p );
                  
//                     // Add the new node to the clipped mesh.
//                   Point new_point( new_result );
//                   typename FIELD::mesh_type::Node::index_type new_node = side1_mesh->add_point( new_point );
//                   typename FIELD::mesh_type::Node::index_type other_meshs_node = side2_mesh->add_point( new_point );
                  
//                   NonManNodalMap* this_map = non_man_projection_map[*lpp_begin];
//                   if( this_map->node1 == 0 )
//                   {
//                     this_map->node1 = node2;
//                     this_map->fromNode1 = new_node;
//                     this_map->otherMeshNode1 = other_meshs_node;
//                   }
//                   else if( this_map->node2 == 0 )
//                   {
//                     this_map->node2 = node2;
//                     this_map->fromNode2 = new_node;
//                     this_map->otherMeshNode2 = other_meshs_node;
//                   }
//                   else if( this_map->node3 == 0 )
//                   {
//                     this_map->node3 = node2;
//                     this_map->fromNode3 = new_node;
//                     this_map->otherMeshNode3 = other_meshs_node;
//                   }
//                   else if( this_map->node4 == 0 )
//                   {
//                     this_map->node4 = node2;
//                     this_map->fromNode4 = new_node;
//                     this_map->otherMeshNode4 = other_meshs_node;
//                   }
//                   else
//                       cout << "ERROR: UNEXPECTED NODE ID...\n";
                  
//                 }
//               }
//               else
//               {
//                   //add the face to the non_man_projection_map with the correct nodes being mapped...
//                 if( i == 0 )
//                 {
//                     //use the previously created node...
//                   if( node2 == 0 )
//                       cout << "ERROR: UNEXPECTED ID...\n";
                  
// //NOTE TO JS: Need to map the other meshs node...
//                   NonManNodalMap* this_map = new NonManNodalMap( *lpp_begin );
//                   non_man_projection_map[*lpp_begin] = this_map;
//                   this_map->node1 = node2;
//                   this_map->fromNode1 = new_map1[node2];
//                   this_map->otherMeshNode1 = new_map2[side2_nodemap[side1_reverse_map[node2]]];
//                 }
//                 else
//                 {
//                     //need to create a new node...
//                   if( node2 == 0 )
//                       cout << "ERROR: UNEXPECTED ID...\n";
//                   Point n_p;
//                   original_mesh->get_center( n_p, node2 );
                  
//                   Point new_result;
//                   typename FIELD::mesh_type::Face::index_type face_id;
//                   tri_mesh->find_closest_elem( new_result, face_id, n_p );
                  
//                     // Add the new node to the clipped mesh.
//                   Point new_point( new_result );
//                   typename FIELD::mesh_type::Node::index_type new_node = side1_mesh->add_point( new_point ); 
//                   typename FIELD::mesh_type::Node::index_type other_meshs_node = side2_mesh->add_point( new_point );
                  
//                   NonManNodalMap* this_map = new NonManNodalMap( *lpp_begin );
//                   non_man_projection_map[*lpp_begin] = this_map;
//                   this_map->node1 = node2;
//                   this_map->fromNode1 = new_node;
//                   this_map->otherMeshNode1 = other_meshs_node;
//                 }
//               }
              
//               ++lpp_begin;
//             }
//             cout << endl;
//           }
          
// //NOTE TO JS: remove the code which special-case removes the faces from the face_list...
// //NOTE TO JS: delete the created NonManNodalMaps
//         }
        
        ++nm_search;
      }
      cout << endl;
    }
    
//    int non_man_counter = 0;  
    typename hash_type3::iterator fiter = face_list.begin();
    typename hash_type3::iterator fiter_end = face_list.end();
    while( fiter != fiter_end )
    {
      typename FIELD::mesh_type::Node::array_type face_nodes;
      typename FIELD::mesh_type::Face::index_type face_id = (*fiter).first;
      side1_mesh->get_nodes( face_nodes, face_id );
      
      typename FIELD::mesh_type::Node::array_type nnodes1(8);
      typename FIELD::mesh_type::Node::array_type nnodes2(8);
      
      nnodes1[0] = face_nodes[3];
      nnodes1[1] = face_nodes[2];
      nnodes1[2] = face_nodes[1];
      nnodes1[3] = face_nodes[0];
      
      nnodes2[0] = side2_nodemap[side1_reverse_map[face_nodes[0]]];
      nnodes2[1] = side2_nodemap[side1_reverse_map[face_nodes[1]]];
      nnodes2[2] = side2_nodemap[side1_reverse_map[face_nodes[2]]];
      nnodes2[3] = side2_nodemap[side1_reverse_map[face_nodes[3]]];
      
//       if( non_man_projection_map.find( (*fiter).first ) != non_man_projection_map.end() )
//       {
//         non_man_counter++;
        
//         NonManNodalMap* this_map = non_man_projection_map[(*fiter).first];
        
//         if( face_nodes[3] == this_map->node1 )
//         {
//           nnodes1[4] = this_map->fromNode1;
//           nnodes2[7] = this_map->otherMeshNode1;
//         }
//         else if( face_nodes[3] == this_map->node2 )
//         {
//           nnodes1[4] = this_map->fromNode2;
//           nnodes2[7] = this_map->otherMeshNode2;
//         }
//         else if( face_nodes[3] == this_map->node3 )
//         {
//           nnodes1[4] = this_map->fromNode3;
//           nnodes2[7] = this_map->otherMeshNode3;
//         }
//         else if( face_nodes[3] == this_map->node4 )
//         {
//           nnodes1[4] = this_map->fromNode4;
//           nnodes2[7] = this_map->otherMeshNode4;
//         }
//         else
//         {
//           nnodes1[4] = new_map1[face_nodes[3]];
//           nnodes2[7] = new_map2[side2_nodemap[side1_reverse_map[face_nodes[3]]]];
//         }
        
//         if( face_nodes[2] == this_map->node1 )
//         {
//           nnodes1[5] = this_map->fromNode1;
//           nnodes2[6] = this_map->otherMeshNode1;
//         }
//         else if( face_nodes[2] == this_map->node2 )
//         {
//           nnodes1[5] = this_map->fromNode2;
//           nnodes2[6] = this_map->otherMeshNode2;
//         }
//         else if( face_nodes[2] == this_map->node3 )
//         {
//           nnodes1[5] = this_map->fromNode3;
//           nnodes2[6] = this_map->otherMeshNode3;
//         }
//         else if( face_nodes[2] == this_map->node4 )
//         {
//           nnodes1[5] = this_map->fromNode4;
//           nnodes2[6] = this_map->otherMeshNode4;
//         }
//         else
//         {
//           nnodes1[5] = new_map1[face_nodes[2]];
//           nnodes2[6] = new_map2[side2_nodemap[side1_reverse_map[face_nodes[2]]]];
//         }
        
//         if( face_nodes[1] == this_map->node1 )
//         {
//           nnodes1[6] = this_map->fromNode1;
//           nnodes2[5] = this_map->otherMeshNode1;
//         }
//         else if( face_nodes[1] == this_map->node2 )
//         {
//           nnodes1[6] = this_map->fromNode2;
//           nnodes2[5] = this_map->otherMeshNode2;
//         }
//         else if( face_nodes[1] == this_map->node3 )
//         {
//           nnodes1[6] = this_map->fromNode3;
//           nnodes2[5] = this_map->otherMeshNode3;
//         }
//         else if( face_nodes[1] == this_map->node4 )
//         {
//           nnodes1[6] = this_map->fromNode4;
//           nnodes2[5] = this_map->otherMeshNode4;
//         }
//         else
//         {
//           nnodes1[6] = new_map1[face_nodes[1]];
//           nnodes2[5] = new_map2[side2_nodemap[side1_reverse_map[face_nodes[1]]]];
//         }
        
//         if( face_nodes[0] == this_map->node1 )
//         {
//           nnodes1[7] = this_map->fromNode1;
//           nnodes2[4] = this_map->otherMeshNode1;
//         }
//         else if( face_nodes[0] == this_map->node2 )
//         {
//           nnodes1[7] = this_map->fromNode2;
//           nnodes2[4] = this_map->otherMeshNode2;
//         }
//         else if( face_nodes[0] == this_map->node3 )
//         {
//           nnodes1[7] = this_map->fromNode3;
//           nnodes2[4] = this_map->otherMeshNode3;
//         }
//         else if( face_nodes[0] == this_map->node4 )
//         {
//           nnodes1[7] = this_map->fromNode4;
//           nnodes2[4] = this_map->otherMeshNode4;
//         }
//         else
//         {
//           nnodes1[7] = new_map1[face_nodes[0]];
//           nnodes2[4] = new_map2[side2_nodemap[side1_reverse_map[face_nodes[0]]]];
//         }
//       }
//       else
//       {
        nnodes1[4] = new_map1[face_nodes[3]];
        nnodes1[5] = new_map1[face_nodes[2]];
        nnodes1[6] = new_map1[face_nodes[1]];
        nnodes1[7] = new_map1[face_nodes[0]];
        
        nnodes2[4] = new_map2[side2_nodemap[side1_reverse_map[face_nodes[0]]]];
        nnodes2[5] = new_map2[side2_nodemap[side1_reverse_map[face_nodes[1]]]];
        nnodes2[6] = new_map2[side2_nodemap[side1_reverse_map[face_nodes[2]]]];
        nnodes2[7] = new_map2[side2_nodemap[side1_reverse_map[face_nodes[3]]]];
//      }
      
      side1_mesh->add_elem( nnodes1 );
      side2_mesh->add_elem( nnodes2 );
      
      ++fiter;
    }
//    cout << "Created " << non_man_counter << " non-manifold hexes " << non_man_projection_map.size() << endl;
  }
  cout << "Finished\n";
  
    // Force all the synch data to be rebuilt on next synch call.
  side1_mesh->unsynchronize();
  side2_mesh->unsynchronize();
  
  typename HexVolMesh<HexTrilinearLgn<Point> >::Elem::size_type side1_size;
  typename HexVolMesh<HexTrilinearLgn<Point> >::Elem::size_type side2_size;
  side1_mesh->size( side1_size );
  side2_mesh->size( side2_size );
  
  cout << "Side1 has " << side1_size << " hexes." << endl;
  cout << "Side2 has " << side2_size << " hexes." << endl << endl;
  mod->update_progress( 0.99 );
}


template <class FIELD>
void
InsertHexSheetAlgoHex<FIELD>::separate_non_man_faces(
    vector<unsigned int>& connected_faces,
    const typename FIELD::mesh_type &mesh,
    typename FIELD::mesh_type::Edge::index_type non_man_edge_id,
    typename FIELD::mesh_type::Face::array_type &non_man_boundary_faces )
{
  ASSERTMSG(non_man_boundary_faces.size(), "No faces to pair.");  
  ASSERTMSG((non_man_boundary_faces.size() % 2 == 0 && non_man_boundary_faces.size() != 2),
            "Can't pair odd number of faces, or a set of only two faces.");
  
  typename FIELD::mesh_type::Node::array_type edge_nodes;
  mesh.get_nodes(edge_nodes, non_man_edge_id);
  Point ep0, ep1;
  mesh.get_center(ep0, edge_nodes[0]);
  mesh.get_center(ep1, edge_nodes[1]);
  Vector evec = ep1 - ep0;
  
  vector<Vector> outvectors;
  for (unsigned int i = 0 ; i < non_man_boundary_faces.size(); i++)
  {
    typename FIELD::mesh_type::Edge::array_type edges;
    mesh.get_edges(edges, non_man_boundary_faces[i]);
    for (unsigned int j = 0; j < edges.size(); j++)
    {
      if (edges[j] != non_man_edge_id)
      {
        typename FIELD::mesh_type::Node::array_type nodes;
        mesh.get_nodes(nodes, edges[j]);
        if (nodes[0] == edge_nodes[0] || nodes[0] == edge_nodes[1])
        {
            // nodes[0] -> nodes[1];
          Point p0, p1;
          mesh.get_center(p0, nodes[0]);
          mesh.get_center(p1, nodes[1]);
          outvectors.push_back(p1 - p0);
          break;
        }
        else if (nodes[1] == edge_nodes[0] || nodes[1] == edge_nodes[1])
        {
            // nodes[1] -> nodes[0];
          Point p0, p1;
          mesh.get_center(p0, nodes[0]);
          mesh.get_center(p1, nodes[1]);
          outvectors.push_back(p0 - p1);
          break;
        }
      }
    }
  }
  ASSERTMSG(outvectors.size() == non_man_boundary_faces.size(),
            "A boundary face wasn't really on the non-manifold edge.");
  
    // We'll be tricky and toss the center of the nearest element on,
    // since it will be closer than the other side.
  typename FIELD::mesh_type::Elem::array_type elems;
  mesh.get_elems(elems, non_man_edge_id);
  for (unsigned int i = 0; i < elems.size(); i++)
  {
    typename FIELD::mesh_type::Face::array_type faces;
    mesh.get_faces(faces, elems[i]);
    bool found = false;
    for (unsigned int j = 0; j < faces.size(); j++)
    {
      if (faces[j] == non_man_boundary_faces[0])
      {
        found = true;
      }
    }
    if (found)
    {
      Point p;
      mesh.get_center(p, elems[i]);
      outvectors.push_back(p - ep0);
      break;
    }
  }

  // Project all of these vectors to the edge plane.
  const double elen = evec.length();
  for (unsigned int i = 0; i < outvectors.size(); i++)
  {
    const Vector eproj = evec * (Dot(evec, outvectors[i]) / elen);
    outvectors[i] = outvectors[i] - eproj;
  }

  vector<pair<double, unsigned int> > angles;
  angles.push_back(pair<double, unsigned int>(0.0, non_man_boundary_faces[0]));
  const double length0 = outvectors[0].length();
  // Compute the angle between outvectors[0] and outvectors[i];
  for (unsigned int i = 1; i < outvectors.size(); i++)
  {
    const double len = length0 * outvectors[i].length();
    const double q = acos(Dot(outvectors[0], outvectors[i]) / len);
    const Vector c = Cross(outvectors[0], outvectors[i]);
    const double angle = (Dot(c, evec) < 0.0) ? q : (2 * M_PI - q);
    if (i < non_man_boundary_faces.size())
    {
      angles.push_back(pair<double, unsigned int>(angle,
                                                  non_man_boundary_faces[i]));
    }
    else
    {
      angles.push_back(pair<double, unsigned int>(angle, (unsigned int)-1));
    }
  }
  
  sort(angles.begin(), angles.end(), pair_less);
  
  // If 0-1 are connected then our pairs are (1,2) (3,4) ... (n-1, 0).
  // Else our pairs are (0,1) (2,3), 4,5) etc.
  // Determine which case it is, return those in the connected faces.
  int offset = 0;
  if (angles[1].second == (unsigned int)-1)
  {
    offset = 1;
    angles.erase(angles.begin()+1);
  }
  else
  {
    ASSERTMSG(angles[angles.size()-1].second == (unsigned int)-1,
              "Elem center wasn't adjacent to the face.");
    angles.erase(angles.end()-1);
  }

  // Fill in the results.
  connected_faces.resize(non_man_boundary_faces.size());
  for (unsigned int i = 0; i < non_man_boundary_faces.size(); i++)
  {
    connected_faces[i] = angles[(i+offset)%angles.size()].second;
  }
}



template <class FIELD>
void
InsertHexSheetAlgoHex<FIELD>::node_get_faces(typename FIELD::mesh_type *mesh,
                      set<typename FIELD::mesh_type::Face::index_type> &result,
                      typename FIELD::mesh_type::Node::index_type node)
{
  result.clear();

  typename FIELD::mesh_type::Elem::array_type elems;
  mesh->get_elems(elems, node);

  for (unsigned int i = 0; i < elems.size(); i++)
  {
    typename FIELD::mesh_type::Face::array_type faces;
    mesh->get_faces( faces, elems[i] );
    for (unsigned int j = 0; j < faces.size(); j++)
    {
      typename FIELD::mesh_type::Node::array_type nodes;
      mesh->get_nodes(nodes, faces[j]);

      for (unsigned int k = 0; k < nodes.size(); k++)
      {
        if (nodes[k] == node)
        {
          result.insert(faces[j]);
          break;
        }
      }
    }
  }
}


template <class FIELD>
void
InsertHexSheetAlgoHex<FIELD>::edge_get_faces( typename FIELD::mesh_type *mesh,
                                              set<typename FIELD::mesh_type::Face::index_type> &result,
                                              typename FIELD::mesh_type::Edge::index_type edge )
{
  result.clear();

  typename FIELD::mesh_type::Elem::array_type elems;
  mesh->get_elems( elems, edge );

  for( unsigned int i = 0; i < elems.size(); i++ )
  {
    typename FIELD::mesh_type::Face::array_type faces;
    mesh->get_faces( faces, elems[i] );
    for( unsigned int j = 0; j < faces.size(); j++ )
    {
      typename FIELD::mesh_type::Edge::array_type edges;
      mesh->get_edges( edges, faces[j]);

      for( unsigned int k = 0; k < edges.size(); k++ )
      {
        if( edges[k] == edge)
        {
          result.insert( faces[j] );
          break;
        } 
      } 
    }
  }
}

} // end namespace SCIRun
#endif // InsertHexSheet_h
