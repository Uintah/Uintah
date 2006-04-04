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
      FieldHandle& intersectfield ) = 0;

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
      FieldHandle& intersectfield );
  
  void write_tri_off_file( FieldHandle trifieldh );
  void write_hexes_to_file( FieldHandle hexfieldh );

  void load_hex_mesh( FieldHandle hexfieldh );
  void load_tri_mesh( FieldHandle trifieldh );
  
  void compute_intersections( 
    HexVolMesh<HexTrilinearLgn<Point> >* original_mesh,
    HexVolMesh<HexTrilinearLgn<Point> >*& intersect_mesh,
    HexVolMesh<HexTrilinearLgn<Point> >*& side1_mesh,
    HexVolMesh<HexTrilinearLgn<Point> >*& side2_mesh );

  bool interferes(const vector<Vector3> &p, const Vector3 &axis, int split);
  
  bool intersects( const HexMesh &hexmesh, int hex_index,
                   const TriangleMesh &trimesh, int face_index);
  void compute_intersections_KDTree( vector<int> &crosses,
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
  
//   HexMesh side1;
//   HexMesh side2;
//   HexMesh intersect; 
  
//   const TriangleMesh &c_trimesh = gtb::trimesh;
//   const HexMesh &c_hexmesh = gtb::hexmesh;
};

template <class FIELD>
void InsertHexSheetAlgoHex<FIELD>::execute(
    ProgressReporter *mod, FieldHandle hexfieldh, FieldHandle trifieldh,
    FieldHandle& side1field, FieldHandle& side2field,
    FieldHandle& intersectfield )
{
  typename FIELD::mesh_type *original_mesh =
      dynamic_cast<typename FIELD::mesh_type *>(hexfieldh->mesh().get_rep());

//  FIELD *hexfield = dynamic_cast<FIELD*>( hexfieldh.get_rep() );
//for debugging...
//  write_tri_off_file( trifieldh );
//  write_hexes_to_file( hexfieldh );

  load_hex_mesh( hexfieldh );
  load_tri_mesh( trifieldh );

  typename FIELD::mesh_type *intersect_mesh = scinew typename FIELD::mesh_type();  
  typename FIELD::mesh_type *mesh_rep = 
      dynamic_cast<typename FIELD::mesh_type *>(hexfieldh->mesh().get_rep());
  intersect_mesh->copy_properties( mesh_rep );  
  typename FIELD::mesh_type *side1_mesh = scinew typename FIELD::mesh_type();
  side1_mesh->copy_properties( mesh_rep );
  typename FIELD::mesh_type *side2_mesh = scinew typename FIELD::mesh_type();
  side2_mesh->copy_properties( mesh_rep );

  compute_intersections( original_mesh, intersect_mesh, side1_mesh, side2_mesh );

  side1field = scinew FIELD( side1_mesh );
  side2field = scinew FIELD( side2_mesh );
  intersectfield = scinew FIELD( intersect_mesh );
  side1field->copy_properties( hexfieldh.get_rep() );
  side2field->copy_properties( hexfieldh.get_rep() );
  intersectfield->copy_properties( hexfieldh.get_rep() );

}
    
template <class FIELD>
void InsertHexSheetAlgoHex<FIELD>::load_tri_mesh( FieldHandle trifieldh )
{ 
  TriSurfMesh<TriLinearLgn<Point> > *sci_tri_mesh = dynamic_cast<TriSurfMesh<TriLinearLgn<Point> >*>(trifieldh->mesh().get_rep());
  
  typename TriSurfMesh<TriLinearLgn<Point> >::Node::size_type num_nodes;
  typename TriSurfMesh<TriLinearLgn<Point> >::Elem::size_type num_tris;
  sci_tri_mesh->size( num_nodes );
  sci_tri_mesh->size( num_tris );
  
//   read_a_line(f, line, 2048);
//   int vnum, fnum;
//   sscanf(line, "%d %d", &vnum, &fnum);
  
//   for (int i = 0; i < vnum; ++i)
//   {
//     if (!read_a_line(f, line, 2048))
//     {
//       cerr<<"Mesh::ReadOFF() - EOF" << endl;
//       Clear();	fclose(f);	return false;
//     }
//     float x,y,z;
//     sscanf(line, "%f %f %f", &x, &y, &z);
// 		verts.push_back(TriangleMeshVertex(Point3(x, y, z)));
//   }

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
//    fprintf( fp, "%f %f %f\n", p.x(), p.y(), p.z() );
    trimesh.add_point( p.x(), p.y(), p.z() );
    ++nbi;
    count++;
  }
  
//   for (int i = 0; i < fnum; ++i)
//   {
//     if (!read_a_line(f, line, 2048))
//     {
//       cerr<<"Mesh::ReadOFF() - EOF" << endl;
//       Clear();	fclose(f);	return false;
//     }
//     int n, vi[3];
//     sscanf(line, "%d %d %d %d", &n, vi, vi+1, vi+2);
//     if (n != 3)
//     {
//       cerr<<"Mesh::ReadOFF() - only triangle meshaes are suppored" << endl;
//         // continue anyway
//     }
// 		faces.push_back(TriangleMeshFace(vi));
//   }
  typename TriSurfMesh<TriLinearLgn<Point> >::Face::iterator bi, ei;
  sci_tri_mesh->begin(bi); 
  sci_tri_mesh->end(ei);
  while (bi != ei)
  {
    typename TriSurfMesh<TriLinearLgn<Point> >::Node::array_type onodes;
    sci_tri_mesh->get_nodes(onodes, *bi);
    int vi[3];
//     fprintf( fp, "3 %d %d %d\n", (unsigned int)onodes[0], 
//              (unsigned int)onodes[1],
//              (unsigned int)onodes[2] );
    vi[0] = (unsigned int)onodes[0];
    vi[1] = (unsigned int)onodes[1];
    vi[2] = (unsigned int)onodes[2];
//    trimesh.faces.push_back(TriangleMeshFace(vi));
    trimesh.add_tri( vi );
    ++bi;
  }
  
    // we've read all the data - build the actual structures now
 	std::vector<int> facemap, vertmap;
//  	trimesh.IdentityMap(facemap, fnum);
//  	trimesh.IdentityMap(vertmap, vnum);
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
  
//   ifstream in(filename);
//   int nverts, nhexes;
//   in >> nverts >> nhexes;
  
//   hexes.resize(nhexes);
//   points.resize(nverts); 
  hexmesh.hexes.resize( num_hexes );
  hexmesh.points.resize( num_nodes );
  
//   for (int i=0; i<nverts; ++i) 
//   {
//     float x, y, z;
//     in >> x >> y >> z;
//     points[i] = Point3(x,y,z);
//   }
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
//    fprintf( fp, "%f %f %f\n", p.x(), p.y(), p.z() );
    hexmesh.points[count] = gtb::Point3( p.x(), p.y(), p.z() );
    ++nbi;
    count++;
  }
  
//   for (int i=0; i<nhexes; ++i) 
//   {
//     for (unsigned j=0; j<8; ++j)
//         in >> hexes[i].verts[j];
//     for (unsigned j=0; j<6; ++j)
//         in >> hexes[i].acrossface[j];
//   }
//}
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
    typename FIELD::mesh_type::Cell::array_type neighbors;
     
    hex_mesh->get_nodes(onodes, *bi);
    hex_mesh->get_neighbors( neighbors, *bi );

    typename FIELD::mesh_type::Cell::array_type::iterator iter = neighbors.begin();
    unsigned int i;
    int n[6];
    for( i = 0; i < 6; i++ )
    {
      if( i < neighbors.size() )
          n[i] = neighbors[i];
      else
          n[i] = -1;
    }

    if( neighbors.size() > 6 )
        cout << "ERROR: More than six neighbors reported..." << count << endl;

    for (unsigned j=0; j<8; ++j)
        hexmesh.hexes[count].verts[j] = onodes[j];
    for (unsigned j=0; j<6; ++j)
        hexmesh.hexes[count].acrossface[j] = n[j];
//     fprintf( fp, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", 
//              (unsigned int)onodes[0], 
//              (unsigned int)onodes[1],
//              (unsigned int)onodes[4],
//              (unsigned int)onodes[5], 
//              (unsigned int)onodes[3],
//              (unsigned int)onodes[2],
//              (unsigned int)onodes[7], 
//              (unsigned int)onodes[6],
//              n[0], n[1], n[2], n[3], n[4], n[5] );
    ++bi;
    count++;
  }
}

template <class FIELD>
void InsertHexSheetAlgoHex<FIELD>::write_tri_off_file( FieldHandle trifieldh )
{
  FILE *fp = fopen( "trifield.off", "w" );
//  FIELD *trifield = dynamic_cast<FIELD*>( trifieldh.get_rep() );
  TriSurfMesh<TriLinearLgn<Point> > *tri_mesh = dynamic_cast<TriSurfMesh<TriLinearLgn<Point> >*>(trifieldh->mesh().get_rep());

  typename TriSurfMesh<TriLinearLgn<Point> >::Node::size_type num_nodes;
  typename TriSurfMesh<TriLinearLgn<Point> >::Elem::size_type num_tris;
  tri_mesh->size( num_nodes );
  tri_mesh->size( num_tris );
  
  fprintf( fp, "OFF\n" );
  fprintf( fp, "%d %d 0\n", (unsigned int)num_nodes, (unsigned int)num_tris );
  
  typename TriSurfMesh<TriLinearLgn<Point> >::Node::iterator nbi, nei;
  tri_mesh->begin( nbi );
  tri_mesh->end( nei );
  unsigned int count = 0;
  while( nbi != nei )
  {
    if( count != *nbi )
        cout << "ERROR: Assumption of node id order is incorrect." << endl;
    Point p;
    tri_mesh->get_center( p, *nbi );
    fprintf( fp, "%f %f %f\n", p.x(), p.y(), p.z() );
    ++nbi;
    count++;
  }
  
  typename TriSurfMesh<TriLinearLgn<Point> >::Face::iterator bi, ei;
  tri_mesh->begin(bi); 
  tri_mesh->end(ei);
  while (bi != ei)
  {
    typename TriSurfMesh<TriLinearLgn<Point> >::Node::array_type onodes;
    tri_mesh->get_nodes(onodes, *bi);
    fprintf( fp, "3 %d %d %d\n", (unsigned int)onodes[0], 
             (unsigned int)onodes[1],
             (unsigned int)onodes[2] );
    ++bi;
  }
  fclose( fp );
}

template <class FIELD>
void InsertHexSheetAlgoHex<FIELD>::write_hexes_to_file( FieldHandle hexfieldh )
{
  FILE *fp = fopen( "hexfield.txt", "w" );
//  FIELD *hexfield = dynamic_cast<FIELD*>( hexfieldh.get_rep() );
  typename FIELD::mesh_type *hex_mesh =
    dynamic_cast<typename FIELD::mesh_type *>(hexfieldh->mesh().get_rep());

  typename FIELD::mesh_type::Node::size_type num_nodes;
  typename FIELD::mesh_type::Elem::size_type num_hexes;
  hex_mesh->size( num_nodes );
  hex_mesh->size( num_hexes );
  
  fprintf( fp, "%d %d\n", (unsigned int)num_nodes, (unsigned int)num_hexes );
  
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
    fprintf( fp, "%f %f %f\n", p.x(), p.y(), p.z() );
    ++nbi;
    count++;
  }
  
  typename FIELD::mesh_type::Elem::iterator bi, ei;
  hex_mesh->begin(bi); 
  hex_mesh->end(ei);
  while (bi != ei)
  {
    typename FIELD::mesh_type::Node::array_type onodes;
    typename FIELD::mesh_type::Cell::array_type neighbors;
    
    hex_mesh->synchronize( Mesh::FACES_E );    
    hex_mesh->get_nodes(onodes, *bi);
    hex_mesh->get_neighbors( neighbors, *bi );

    typename FIELD::mesh_type::Cell::array_type::iterator iter = neighbors.begin();
    unsigned int i;
    int n[6];
    for( i = 0; i < 6; i++ )
    {
      if( i < neighbors.size() )
          n[i] = neighbors[i];
      else
          n[i] = -1;
    }

    if( neighbors.size() > 6 )
        cout << "ERROR: More than six neighbors reported..." << count << endl;

    fprintf( fp, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", 
             (unsigned int)onodes[0], 
             (unsigned int)onodes[1],
             (unsigned int)onodes[4],
             (unsigned int)onodes[5], 
             (unsigned int)onodes[3],
             (unsigned int)onodes[2],
             (unsigned int)onodes[7], 
             (unsigned int)onodes[6],
             n[0], n[1], n[2], n[3], n[4], n[5] );
    ++bi;
  }
  fclose( fp );
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
    vector<int> &crosses, const TriangleMesh& trimesh, const HexMesh& hexmesh )
{
	vector<int> kdfi;
	for (unsigned i=0; i<trimesh.faces.size(); i++) 
  {
		kdfi.push_back(i);
	}
	TriangleMeshFaceTree kdtreebbox(trimesh);
	gtb::BoxKDTree<int, TriangleMeshFaceTree> kdtree(kdfi, kdtreebbox);
  
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
	}
}

template <class FIELD>
void InsertHexSheetAlgoHex<FIELD>::compute_intersections( 
    HexVolMesh<HexTrilinearLgn<Point> >* original_mesh,
    HexVolMesh<HexTrilinearLgn<Point> >*& intersect_mesh,
    HexVolMesh<HexTrilinearLgn<Point> >*& side1_mesh,
    HexVolMesh<HexTrilinearLgn<Point> >*& side2_mesh )
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
//   side1.points.clear();
//   side1.hexes.clear();
//   side2.points.clear();
//   side2.hexes.clear();
//   intersect.points.clear();
//   intersect.hexes.clear();
  
  vector<int> hexes(hexmesh.hexes.size());
  for (unsigned i=0; i<hexmesh.hexes.size(); ++i)
      hexes[i] = i;
  vector<int> faces(trimesh.faces.size());
  
  for (unsigned i=0; i<trimesh.faces.size(); ++i)
      faces[i] = i;

  const TriangleMesh &c_trimesh = trimesh;
  const HexMesh &c_hexmesh = hexmesh;
  
  Box3 b = Box3::make_union(c_trimesh.bounding_box(),
                            c_hexmesh.bounding_box());
  
	compute_intersections_KDTree(crosses, trimesh, hexmesh);
  
    //flood the two sides
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
      
			for (int nbr=0; nbr<6; nbr++) 
      {
				int hnbr = hexmesh.hexes[h].acrossface[nbr];
				if (hnbr<0) continue;	// boundary
				if (crosses[hnbr] < 0) 
        {
					crosses[hnbr] = side+1;
					toprocess.push_back(hnbr);
				}
			}
		}
	}

//NOTE TO JS: for debugging...
    // inefficient, but whatever, it's just for drawing
//   side1.points = hexmesh.points;
//   side2.points = hexmesh.points;
//   intersect.points = hexmesh.points;
//   for( unsigned i=0; i<crosses.size(); ++i) 
//   {
//     if (crosses[i]==0)
//         intersect.hexes.push_back(hexmesh.hexes[i]);
//     else if (crosses[i] == 1)
//         side1.hexes.push_back(hexmesh.hexes[i]);
//     else 
//         side2.hexes.push_back(hexmesh.hexes[i]);
//   }
  
//   cout << endl;
//   cout << "Hexmesh has " << hexmesh.hexes.size() << " hexes." << endl;
//   cout << "Side1 has " << side1.hexes.size() << " hexes." << endl;
//   cout << "Side2 has " << side2.hexes.size() << " hexes." << endl;
//   cout << "Intersect has " << intersect.hexes.size() << " hexes." << endl << endl;
//end NOTE TO JS

//need to add elements from the three sets of elements...
  hash_type intersect_nodemap, side1_nodemap, side2_nodemap;
  for( unsigned int k = 0; k < crosses.size(); ++k )
  {
    typename FIELD::mesh_type::Node::array_type onodes;
    typename FIELD::mesh_type::Elem::index_type elem_id = k;
    original_mesh->get_nodes( onodes, elem_id );
    typename FIELD::mesh_type::Node::array_type nnodes(onodes.size());
  
    if( crosses[k] == 0 )
    {
        //add to intersect_mesh
      for (unsigned int i = 0; i < onodes.size(); i++)
      {
        if( intersect_nodemap.find((unsigned int)onodes[i]) == intersect_nodemap.end())
        {
          Point np;
          original_mesh->get_center( np, onodes[i] );
          const typename FIELD::mesh_type::Node::index_type nodeindex =
              intersect_mesh->add_point( np );
          intersect_nodemap[(unsigned int)onodes[i]] = nodeindex;
          nnodes[i] = nodeindex;
        }
        else
        {
          nnodes[i] = intersect_nodemap[(unsigned int)onodes[i]];
        }
      }
      intersect_mesh->add_elem( nnodes );
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
  typename HexVolMesh<HexTrilinearLgn<Point> >::Elem::size_type original_size;
  typename HexVolMesh<HexTrilinearLgn<Point> >::Elem::size_type side1_size;
  typename HexVolMesh<HexTrilinearLgn<Point> >::Elem::size_type side2_size;
  typename HexVolMesh<HexTrilinearLgn<Point> >::Elem::size_type intersect_size;
  side1_mesh->size( side1_size );
  side2_mesh->size( side2_size );
  intersect_mesh->size( intersect_size );
  original_mesh->size( original_size );
  
  cout << endl;
  cout << "Hexmesh has " << original_size << " hexes." << endl;
  cout << "Side1 has " << side1_size << " hexes." << endl;
  cout << "Side2 has " << side2_size << " hexes." << endl;
  cout << "Intersect has " << intersect_size << " hexes." << endl << endl;
}

} // end namespace SCIRun
#endif // InsertHexSheet_h
