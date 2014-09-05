//#include <string.h>
//#include <fstream>
#include <vector>
#include <Dataflow/Modules/Fields/IHSMeshUtilities.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Basis/TriLinearLgn.h>

using std::vector;

#ifndef GTB_MODEL_INCLUDED
#define GTB_MODEL_INCLUDED

GTB_BEGIN_NAMESPACE

template<class T>
class tModel {
public:
  virtual ~tModel();
  typedef tBox3<T> Box3;
  
  virtual const Box3 &bounding_box() const;
  virtual const tPoint3<T> &centroid() const;
  
protected:
  tModel();
  
  virtual void compute_bounding_box() const = 0;
  virtual void compute_centroid() const = 0;
  
  void invalidate_all();
  
  mutable Box3 _bounding_box;
  mutable tPoint3<T> _centroid;
  mutable bool _is_bounding_box_valid;
  mutable bool _is_centroid_valid;
};

typedef tModel<double> Model;

//begin inline functions...
template<class T>
inline tModel<T>::tModel()
        : _is_bounding_box_valid(false), _is_centroid_valid(false)
{
}

template<class T>
inline tModel<T>::~tModel()
{
}

template<class T>
inline const typename tModel<T>::Box3 &tModel<T>::bounding_box() const
{
	if (!_is_bounding_box_valid) {
		compute_bounding_box();
		_is_bounding_box_valid = true;
	} 
	return _bounding_box;
}

template<class T>
inline const tPoint3<T> &tModel<T>::centroid() const
{
	if (!_is_centroid_valid) {
		compute_centroid();
		_is_centroid_valid = true;
	}
	return _centroid;
}

template<class T>
inline void tModel<T>::invalidate_all()
{
  _is_bounding_box_valid = _is_centroid_valid = false;
}

GTB_END_NAMESPACE
#endif // GTB_MODEL_INCLUDED


#ifndef __HEXMESH_H
#define __HEXMESH_H

// an hexahedral cell stored in the following way:
//
//     /+-------------|
//    /	|6     	     /|7
//   /	|      	   -/ |
//  / 	|      	  /   |
// /-------------/    |
//2|   	|      	 |3   |
// |  	|     	 |    |
// |  	|     	 |    |
// |  	|     	 |    |
// |  	|        |    |
// |   	|--------|-----
// |  /- 4     	 |  -/ 5
// |/-         	 |-/
// /-------------/
//0             1
//
//NOTE: The following definition is incorrect...  The neighbors array is simply a random accounting of which hexes are adjacent through faces to other hexes.  Ordering is not important in this algorithm...
// Neighbors: 0: through face 0132
//            1: through face 5467
//            2: through face 0264
//            3: through face 1573
//            4: through face 2376
//            5: through face 4510

class Hex
{
public:
  int verts[8];
};

class HexMesh:public gtb::Model
{
public:
  HexMesh() {};
  
  std::vector<gtb::Point3> points;
  std::vector<Hex> hexes;
//   std::vector<int> glIndices;
  
  virtual void compute_bounding_box() const;
  virtual void compute_centroid() const;
};

using namespace std;
using namespace gtb;

void HexMesh::compute_bounding_box() const
{
  _bounding_box = Box3::bounding_box(points);
}

void HexMesh::compute_centroid() const
{
  _centroid = Point3::centroid(points);
}
#endif //__HEXMESH_H

#ifndef __TRIANGLE_MESH_H
#define __TRIANGLE_MESH_H

GTB_BEGIN_NAMESPACE

template <class T>
class tTriangleMeshVertex {
public:
  typedef T value_type;
  typedef tPoint3<T> Point3;
  typedef tVector3<T> Vector3;
  
	tTriangleMeshVertex(Point3 p) : point(p), normal(0,0,0), someface(-1) { }
	Point3 point;
	Vector3 normal;
  
	int someface;
};

class TriangleMeshFace {
public:  
	TriangleMeshFace(const int v[3]) {
		verts[0] = v[0];
		verts[1] = v[1];
		verts[2] = v[2];
		nbrs[0]  = nbrs[1] = nbrs[2] = -1;
	}
  
	int VertIndex(int v) const {
		if (verts[0] == v)	return 0;
		if (verts[1] == v)	return 1;
		if (verts[2] == v)	return 2;
		return -1;
	}
  
	int EdgeIndexCCW(int v0, int v1) const {
    
		int vi = VertIndex(v0);
		if (vi<0) return -1;
    
		if (verts[(vi+1)%3] == v1)
        return vi;
    
		return -1;
	}
  
	int verts[3];	// ccw order
	int nbrs[3];	// nbrs[0] shares verts 0,1, 1 shares 1,2, 2 shares 2,0
    // -1 - boundary
};

template <class T>
class tTriangleMesh : public tModel<T>  {
  
public:
  typedef T value_type;
  typedef tPoint3<T> Point3;
  typedef tVector3<T> Vector3;
  typedef tBox3<T> Box3;
	typedef tTriangleMeshVertex<T> TriangleMeshVertex;
  
  typedef std::vector<TriangleMeshVertex> vertex_list;
  typedef std::vector<TriangleMeshFace> face_list;
  
	tTriangleMesh();
	~tTriangleMesh();
  
//  bool ReadOFF(const char *fname);
  void add_point( double x, double y, double z );
  void add_tri( int* node_id_array );
  
    // clear the whole mesh
	void Clear();
  
    // these are dangerous if the mesh has been changed!
	int FaceIndex(const TriangleMeshFace &f) const { return (int)(&f - &faces[0]); }
	int VertexIndex(const TriangleMeshVertex &v) const { return (int)(&v - &verts[0]); }
  
	Vector3 FaceNormal(const TriangleMeshFace &f) const {
		Vector3 e1 = verts[f.verts[1]].point - verts[f.verts[0]].point;
		Vector3 e2 = verts[f.verts[2]].point - verts[f.verts[0]].point;
		Vector3 norm = e1.cross(e2);
		norm.normalize();
		return norm;
	}
  
	void SetNormals();
  
	void compute_bounding_box() const;
	void compute_centroid() const;
  
	class VertexFaceIterator {
public:
    
		VertexFaceIterator(tTriangleMesh<T> &_mesh, int vi);
		bool done();
		VertexFaceIterator& operator++();
		TriangleMeshFace& operator*();
    
private:
    
		bool first;
		int cur_face;
		int first_face;
		int vertex;
		tTriangleMesh<T> *mesh;
	};
  
	void build_structures(const std::vector<int> &facemap, const std::vector<int> &vertmap);
	void IdentityMap(std::vector<int> &map, int size);
  
  vertex_list	verts;
  face_list	faces;
  
  bool read_a_line(FILE* f, char* line, int maxlinelen);
};

typedef tTriangleMesh<double> TriangleMesh;

//begin inline definitions....
template <class T>
inline tTriangleMesh<T>::tTriangleMesh() 
{
	tModel<T>::invalidate_all();
}

template <class T>
inline tTriangleMesh<T>::~tTriangleMesh() 
{
}

// Model stuff
template <class T>
inline void tTriangleMesh<T>::compute_bounding_box() const {
  
  tModel<T>::_bounding_box = Box3(verts[0].point, verts[0].point);
	for (typename vertex_list::const_iterator v=verts.begin(); v!=verts.end(); ++v) {
    tModel<T>::_bounding_box.update(v->point);
	}
	tModel<T>::_is_bounding_box_valid = true;
}

template <class T>
inline void tTriangleMesh<T>::compute_centroid() const {
  
	tModel<T>::_centroid = Point3(0,0,0);
  
	for (typename vertex_list::const_iterator v=verts.begin(); v!=verts.end(); ++v) {
		tModel<T>::_centroid.add(v->point);
	}
	tModel<T>::_centroid.scalar_scale(1.0 / verts.size());
	tModel<T>::_is_centroid_valid = true;
}

template <class T>
inline void tTriangleMesh<T>::add_point( double x, double y, double z ) 
{
  verts.push_back(TriangleMeshVertex(Point3(x, y, z)));
}

template <class T>
inline void tTriangleMesh<T>::add_tri( int* node_id_array ) 
{
  faces.push_back(TriangleMeshFace(node_id_array));
}
// template <class T>
// inline bool tTriangleMesh<T>::load_mesh( FieldHandle trifieldh ) 
// {
// 	FILE* f(fopen(fname, "rb"));
// 	if (f==0) {
// 		cerr<<"couldn't open file "<<fname<<endl;
// 		return false;
// 	}
  
// 	char line[2048];
//     //Read the header
//   read_a_line(f, line, 2048);
// //    if (strnicmp(line, "off", 3) != 0)
//   if (strncmp(line, "OFF", 3) != 0)
//   {
// 		cerr<<"Mesh::ReadOFF() - Not a .off file" << endl;
// 		Clear();	fclose(f);	return false;
//   }
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
  
// 	fclose(f);
  
//     // we've read all the data - build the actual structures now
// 	std::vector<int> facemap, vertmap;
// 	IdentityMap(facemap, fnum);
// 	IdentityMap(vertmap, vnum);
//   build_structures(facemap, vertmap);
  
// 	return true;
// }

// template <class T>
// inline bool tTriangleMesh<T>::ReadOFF(const char *fname) 
// {
// 	FILE* f(fopen(fname, "rb"));
// 	if (f==0) {
// 		cerr<<"couldn't open file "<<fname<<endl;
// 		return false;
// 	}
  
// 	char line[2048];
//     //Read the header
//   read_a_line(f, line, 2048);
// //    if (strnicmp(line, "off", 3) != 0)
//   if (strncmp(line, "OFF", 3) != 0)
//   {
// 		cerr<<"Mesh::ReadOFF() - Not a .off file" << endl;
// 		Clear();	fclose(f);	return false;
//   }
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
  
// 	fclose(f);
  
//     // we've read all the data - build the actual structures now
// 	std::vector<int> facemap, vertmap;
// 	IdentityMap(facemap, fnum);
// 	IdentityMap(vertmap, vnum);
//   build_structures(facemap, vertmap);
  
// 	return true;
// }

template <class T>
inline void tTriangleMesh<T>::build_structures(const std::vector<int> &facemap, const std::vector<int> &vertmap)
{
    // convert all the indices to the array indices
	for (typename face_list::iterator f=faces.begin(); f!=faces.end(); ++f) {
		for (int i=0; i<3; i++) {
			f->verts[i] = vertmap[f->verts[i]];
		}
	}
  
    // set the somefaces
//	cerr<<"setting somefaces"<<endl;
	for (typename vertex_list::iterator v=verts.begin(); v!=verts.end(); ++v) {
		v->someface = -1;
	}
	for (typename face_list::iterator f=faces.begin(); f!=faces.end(); ++f) {
		for (int i=0; i<3; i++) {
			verts[f->verts[i]].someface = FaceIndex(*f);
		}
	}
  
    // build the adjacency info
//	cerr<<"finding neighbors"<<endl;
	vector< vector<int> > vertfaces(verts.size());
	for (unsigned i=0; i<vertfaces.size(); i++) {
		vertfaces[i].reserve(7);
	}
  
	for (typename face_list::iterator f=faces.begin(); f!=faces.end(); ++f) {
		for (int i=0; i<3; i++) {
			vertfaces[f->verts[i]].push_back(FaceIndex(*f));
		}
	}
  
	for (typename face_list::iterator f=faces.begin(); f!=faces.end(); ++f) {
		for (int i=0; i<3; i++) {
      
			int v0 = f->verts[i];
			int v1 = f->verts[(i+1)%3];
      
        // look for a face with the edge v1,v0
			bool found=false;
			for (unsigned vfi=0; vfi<vertfaces[v0].size(); vfi++) {
				int vf = vertfaces[v0][vfi];
				if (faces[vf].EdgeIndexCCW(v1, v0) != -1) {
					f->nbrs[i] = vf;
					if (found) {
						cerr<<"more than one matching triangle found: faces["<<vf<<"]"<<endl;
					}
					found=true;
//					break;
				}
			}
		}
	}
  
//	cerr<<"setting normals"<<endl;
	SetNormals();
  
//	cerr<<"done"<<endl;
  
	tModel<T>::invalidate_all();
}

// set the normals
template <class T>
inline void tTriangleMesh<T>::SetNormals() {
	for (typename vertex_list::iterator v=verts.begin(); v!=verts.end(); ++v) {
		v->normal = Vector3(0,0,0);
		for (VertexFaceIterator f(*this, VertexIndex(*v)); !f.done(); ++f) {
      
			Vector3 e1 = verts[(*f).verts[1]].point - verts[(*f).verts[0]].point;
			Vector3 e2 = verts[(*f).verts[2]].point - verts[(*f).verts[0]].point;
			Vector3 fn = e1.cross(e2);
			if (fn.length() == 0) {
				cerr<<"skipping normal of degenerate face "<<FaceIndex(*f)<<endl;
			} else {
				fn.normalize();
				v->normal += FaceNormal(*f);
			}
		}
		v->normal.normalize();
	}
}

// iterators
template<class T>
inline tTriangleMesh<T>::VertexFaceIterator::VertexFaceIterator(tTriangleMesh<T> &_mesh, int vi) {
  
	first=true;
	mesh = &_mesh;
	vertex = vi;
  
    // rotate clockwise as far as possible
	cur_face = mesh->verts[vi].someface;
	if (cur_face != -1) {
		do {
			TriangleMeshFace &f = mesh->faces[cur_face];
			int cindex = f.VertIndex(vi);
			if (f.nbrs[cindex] == -1) break;
			cur_face = f.nbrs[cindex];
		} while (cur_face != mesh->verts[vi].someface);
		first_face = cur_face;
	}
}

template<class T>
inline bool tTriangleMesh<T>::VertexFaceIterator::done() {
	if (cur_face == -1) return true;
	return false;
}

template<class T>
TriangleMeshFace& tTriangleMesh<T>::VertexFaceIterator::operator*() {
	return mesh->faces[cur_face];
}

template<class T>
inline typename tTriangleMesh<T>::VertexFaceIterator& tTriangleMesh<T>::VertexFaceIterator::operator++() {
  
	TriangleMeshFace &f = mesh->faces[cur_face];
	int nindex = (f.VertIndex(vertex) + 2) % 3;
	cur_face = f.nbrs[nindex];
  
	if (cur_face==first_face)
      cur_face = -1;
  
	first = false;
  
	return *this;
}

template <class T>
inline void tTriangleMesh<T>::Clear() 
{
	verts.clear();
	faces.clear();
}

template <class T>
inline void tTriangleMesh<T>::IdentityMap(std::vector<int> &map, int size) 
{
	map.resize(size);
	for (int i=0; i<size; i++) {
		map[i] = i;
	}
}

template <class T>
inline bool tTriangleMesh<T>::read_a_line(FILE* f, char* line, int maxlinelen) 
{
  do
  {
    fgets(line, maxlinelen, f);
  } while (!feof(f) && (line[0] == '#'));
  if (feof(f)) return false;
  else return true;
}

GTB_END_NAMESPACE
#endif // __TRIANGLE_MESH_H
