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

namespace SCIRun {

using std::copy;
using std::vector;

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
void InsertHexSheetAlgoHex<FIELD>::execute(
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
void InsertHexSheetAlgoHex<FIELD>::load_tri_mesh( TriSurfMesh<TriLinearLgn<Point> > *sci_tri_mesh )
{ 
//  TriSurfMesh<TriLinearLgn<Point> > *sci_tri_mesh = dynamic_cast<TriSurfMesh<TriLinearLgn<Point> >*>(trifieldh->mesh().get_rep());
  
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
  
    // we've read all the data - build the actual structures now
 	std::vector<int> facemap, vertmap;
 	trimesh.IdentityMap(facemap, num_tris);
 	trimesh.IdentityMap(vertmap, num_nodes);
  trimesh.build_structures(facemap, vertmap);
}

template <class FIELD>
void InsertHexSheetAlgoHex<FIELD>::load_hex_mesh( FieldHandle hexfieldh )
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
    for (unsigned j=0; j<8; ++j)
        hexmesh.hexes[count].verts[j] = onodes[j];
    ++bi;
    count++;
  }
}

//! \brief projects points on the axis, tests overlap
template <class FIELD>
bool InsertHexSheetAlgoHex<FIELD>::interferes(
    const vector<Vector3> &p, const Vector3 &axis, int split )
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
  
  float
      mnh = *min_element(d.begin(), d.begin()+split),
      mxh = *max_element(d.begin(), d.begin()+split),
      mnt = *min_element(d.begin()+split, d.end()),
      mxt = *max_element(d.begin()+split, d.end());
  bool 
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
bool InsertHexSheetAlgoHex<FIELD>::intersects(
    const HexMesh &hexmesh, int hex_index, 
    const TriangleMesh &trimesh, int face_index)
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
void InsertHexSheetAlgoHex<FIELD>::compute_intersections_KDTree(
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
void InsertHexSheetAlgoHex<FIELD>::compute_intersections(
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

  vector<int> crosses(hexmesh.hexes.size(), -1);
  vector<int> hexes(hexmesh.hexes.size());
  for (unsigned i=0; i<hexmesh.hexes.size(); ++i)
      hexes[i] = i;
  vector<int> faces(trimesh.faces.size());
  
  for (unsigned i=0; i<trimesh.faces.size(); ++i)
      faces[i] = i;

	compute_intersections_KDTree( mod, crosses, trimesh, hexmesh);
  
    //flood the two sides
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

//need to add elements from the three sets of elements...
  hash_type side1_nodemap, side2_nodemap, side1_reverse_map, side2_reverse_map;
  for( unsigned int k = 0; k < crosses.size(); ++k )
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
          //add to side2_mesh
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
        //add to side1_mesh  
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
        //add to side2_mesh
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
            //want this one...
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
            //want this one...
          oi_node_list.push_back( (*hitr).first );
          shared_vertex_map[(*hitr).first] = (*hitr).first;
          count++;
        }
        ++hitr;
      }
    }

    tri_mesh->synchronize( Mesh::LOCATE_E );
    map<typename FIELD::mesh_type::Node::index_type, typename FIELD::mesh_type::Node::index_type> new_map1;
    map<typename FIELD::mesh_type::Node::index_type, typename FIELD::mesh_type::Node::index_type> new_map2;
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
    
        //since finding the closest face can be slow, update the progress meter 
        // to let the user know that we are performing calculations and the 
        // process has not hung...
      if( i%50 == 0 )
      {
        double temp = 0.75 + 0.25*( (double)i/(double)oi_node_list.size() );
        mod->update_progress( temp );
      }
    
        //add the new node to the clipped mesh
      Point new_point( new_result );
      typename FIELD::mesh_type::Node::index_type this_index1 = side1_mesh->add_point( new_point ); 
      typename FIELD::mesh_type::Node::index_type this_index2 = side2_mesh->add_point( new_point );
    
        //create a map for the new node to a node on the boundary of the clipped mesh...
      new_map1[side1_nodemap[this_node]] = this_index1;
      new_map2[side2_nodemap[this_node]] = this_index2;

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

    side1_mesh->synchronize( Mesh::NODE_NEIGHBORS_E | Mesh::FACE_NEIGHBORS_E | Mesh::FACES_E ); 
    side2_mesh->synchronize( Mesh::NODE_NEIGHBORS_E | Mesh::FACE_NEIGHBORS_E | Mesh::FACES_E );
  
    typename HexVolMesh<HexTrilinearLgn<Point> >::Elem::size_type clipped1_size;
    typename HexVolMesh<HexTrilinearLgn<Point> >::Elem::size_type clipped2_size;
    side1_mesh->size( clipped1_size );
    side2_mesh->size( clipped2_size );

    if( clipped1_size < clipped2_size )
    {
        //Walk all the cells in the smallest clipped mesh to find the boundary faces...
      typename FIELD::mesh_type::Cell::iterator citer; side1_mesh->begin(citer);
      typename FIELD::mesh_type::Cell::iterator citere; side1_mesh->end(citere);
      
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
            typename FIELD::mesh_type::Face::index_type old_face;
            
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
              typename FIELD::mesh_type::Node::array_type nnodes1(8);
              typename FIELD::mesh_type::Node::array_type nnodes2(8);
//NOTE TO JS:NEED to find a way to ensure the ordering is always correct...

              nnodes1[0] = face_nodes[3];
              nnodes1[1] = face_nodes[2];
              nnodes1[2] = face_nodes[1];
              nnodes1[3] = face_nodes[0];
              nnodes1[4] = new_map1[face_nodes[3]];
              nnodes1[5] = new_map1[face_nodes[2]];
              nnodes1[6] = new_map1[face_nodes[1]];
              nnodes1[7] = new_map1[face_nodes[0]];              
              side1_mesh->add_elem( nnodes1 );
              
              nnodes2[0] = side2_nodemap[side1_reverse_map[face_nodes[0]]];
              nnodes2[1] = side2_nodemap[side1_reverse_map[face_nodes[1]]];
              nnodes2[2] = side2_nodemap[side1_reverse_map[face_nodes[2]]];
              nnodes2[3] = side2_nodemap[side1_reverse_map[face_nodes[3]]];
              nnodes2[4] = new_map2[side2_nodemap[side1_reverse_map[face_nodes[0]]]];
              nnodes2[5] = new_map2[side2_nodemap[side1_reverse_map[face_nodes[1]]]];
              nnodes2[6] = new_map2[side2_nodemap[side1_reverse_map[face_nodes[2]]]];
              nnodes2[7] = new_map2[side2_nodemap[side1_reverse_map[face_nodes[3]]]];
              side2_mesh->add_elem( nnodes2 );
            }
          }
        }
      }
    }
    else
    {
        //Walk all the cells in the smallest clipped mesh to find the boundary faces...
      typename FIELD::mesh_type::Cell::iterator citer; side2_mesh->begin(citer);
      typename FIELD::mesh_type::Cell::iterator citere; side2_mesh->end(citere);
      
      while( citer != citere )
      {
        typename FIELD::mesh_type::Cell::index_type ci = *citer;
        ++citer;
        
          // Get all the faces in the cell.
        typename FIELD::mesh_type::Face::array_type faces;
        side2_mesh->get_faces( faces, ci );
        
          // Check each face for neighbors.
        typename FIELD::mesh_type::Face::array_type::iterator fiter = faces.begin();
        
        while( fiter != faces.end() )
        {
          typename FIELD::mesh_type::Cell::index_type nci;
          typename FIELD::mesh_type::Face::index_type fi = *fiter;
          ++fiter;
          
          if( !side2_mesh->get_neighbor( nci, ci, fi ) )
          {
              // Faces with no neighbors are on the boundary...
              //    make sure that this face isn't on the original boundary
            typename FIELD::mesh_type::Face::index_type old_face;
            
            typename FIELD::mesh_type::Node::array_type face_nodes;
            side2_mesh->get_nodes( face_nodes, fi );
            typename hash_type::iterator search1, search2, search3, search4, search_end; 
            search_end = shared_vertex_map.end();
            search1 = shared_vertex_map.find( side2_reverse_map[face_nodes[0]] );
            search2 = shared_vertex_map.find( side2_reverse_map[face_nodes[1]] );
            search3 = shared_vertex_map.find( side2_reverse_map[face_nodes[2]] );
            search4 = shared_vertex_map.find( side2_reverse_map[face_nodes[3]] );
            if( search1 != search_end && search2 != search_end &&
                search3 != search_end && search4 != search_end )
            {
              typename FIELD::mesh_type::Node::array_type nnodes1(8);
              typename FIELD::mesh_type::Node::array_type nnodes2(8);
//NOTE TO JS:NEED to find a way to ensure the ordering is always correct...           
//               if( add_to_side1 )
//               {
//                 nnodes2[0] = face_nodes[0];
//                 nnodes2[1] = face_nodes[1];
//                 nnodes2[2] = face_nodes[2];
//                 nnodes2[3] = face_nodes[3];
//                 nnodes2[4] = new_map2[face_nodes[0]];
//                 nnodes2[5] = new_map2[face_nodes[1]];
//                 nnodes2[6] = new_map2[face_nodes[2]];
//                 nnodes2[7] = new_map2[face_nodes[3]];
//                 side2_mesh->add_elem( nnodes2 );
//               }
//               else
//               {
                nnodes2[0] = face_nodes[3];
                nnodes2[1] = face_nodes[2];
                nnodes2[2] = face_nodes[1];
                nnodes2[3] = face_nodes[0];
                nnodes2[4] = new_map2[face_nodes[3]];
                nnodes2[5] = new_map2[face_nodes[2]];
                nnodes2[6] = new_map2[face_nodes[1]];
                nnodes2[7] = new_map2[face_nodes[0]];              
                side2_mesh->add_elem( nnodes2 );
//               }
              
              nnodes1[0] = side1_nodemap[side2_reverse_map[face_nodes[0]]];
              nnodes1[1] = side1_nodemap[side2_reverse_map[face_nodes[1]]];
              nnodes1[2] = side1_nodemap[side2_reverse_map[face_nodes[2]]];
              nnodes1[3] = side1_nodemap[side2_reverse_map[face_nodes[3]]];
              nnodes1[4] = new_map1[side1_nodemap[side2_reverse_map[face_nodes[0]]]];
              nnodes1[5] = new_map1[side1_nodemap[side2_reverse_map[face_nodes[1]]]];
              nnodes1[6] = new_map1[side1_nodemap[side2_reverse_map[face_nodes[2]]]];
              nnodes1[7] = new_map1[side1_nodemap[side2_reverse_map[face_nodes[3]]]];
              side1_mesh->add_elem( nnodes1 );
            }
          }
        }
      }
    }
  }
  cout << "Finished\n";

    //force all the synch data to be rebuilt on next synch call.
  side1_mesh->unsynchronize();
  side2_mesh->unsynchronize();
  
//  typename HexVolMesh<HexTrilinearLgn<Point> >::Elem::size_type original_size;
  typename HexVolMesh<HexTrilinearLgn<Point> >::Elem::size_type side1_size;
  typename HexVolMesh<HexTrilinearLgn<Point> >::Elem::size_type side2_size;
  side1_mesh->size( side1_size );
  side2_mesh->size( side2_size );
//  original_mesh->size( original_size );
  
//  cout << "Hexmesh has " << original_size << " hexes." << endl;
  cout << "Side1 has " << side1_size << " hexes." << endl;
  cout << "Side2 has " << side2_size << " hexes." << endl << endl;
  mod->update_progress( 0.99 );
}

} // end namespace SCIRun
#endif // InsertHexSheet_h
