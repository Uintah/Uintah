#ifndef SCI_Wangxl_Datatypes_Mesh_FObject_h
#define SCI_Wangxl_Datatypes_Mesh_FObject_h

#include <Core/Datatypes/TriSurfField.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/BFace.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/BEdge.h>

namespace Wangxl {

using namespace SCIRun;

typedef TriSurfMesh::Node::iterator node_iterator;
typedef TriSurfMesh::Face::iterator face_iterator;
typedef TriSurfMesh::Node::array_type TriFace;

typedef pair<DVertex*,DVertex*> Edge;
typedef triple<DVertex*,DVertex*,DVertex*> Face;

struct EdgeEqual
{
  bool operator()( const Edge edge0, const Edge edge1 ) const
  {
    if ( edge0.first != edge1.first && edge0.first != edge1.second ) return false;
    if ( edge0.second != edge1.first && edge0.second != edge1.second ) return false;
    return true;
  }
};

struct EdgeHash {
  int operator() ( const Edge edge ) const
  {
    return ( int ) ( ( long )edge.first ^ ( long )edge.second );
  }
};

struct FaceEqual
{
  bool operator()( const Face face0, const Face face1 ) const
  {
    if ( face0.first != face1.first && face0.first != face1.second  && face0.first != face1.third ) return false;
    if ( face0.second != face1.first && face0.second != face1.second  && face0.second != face1.third ) return false;
    if ( face0.third != face1.first && face0.third != face1.second  && face0.third != face1.third ) return false;
    return true;
  }
};

struct FaceHash {
  int operator() ( const Face face ) const
  {
    return ( int ) ( ( long )face.first ^ ( long )face.second  ^ ( long )face.third );
  }
};

struct BEdgeEqual
{
  bool operator()( const BEdge* bedge0, const BEdge* bedge1 ) const
  {
    if ( bedge0->source() != bedge1->source() && bedge0->source() != bedge1->target() ) return false;
    if ( bedge0->target() != bedge1->source() && bedge0->target() != bedge1->target() ) return false;
    return true;
  }
};

struct BEdgeHash {
  int operator() ( const BEdge* bedge ) const
  {
    return ( int ) ( ( long )bedge->source() ^ ( long )bedge->target() );
  }
};

/*struct BFaceEqual
{
  bool operator()( const BFace* bface0, const BFace* bface1, const BFace* bface2 ) const
  {
    if ( bface0->vertex0() != bface1->vertex0() && bface0->vertex0() != bface1->vertex1() && bface0->vertex0() != bface1->vertex2() ) return false;
    if ( bface0->vertex1() != bface1->vertex0() && bface0->vertex1() != bface1->vertex1() && bface0->vertex1() != bface1->vertex2() ) return false;
    if ( bface0->vertex2() != bface1->vertex0() && bface0->vertex2() != bface1->vertex1() && bface0->vertex2() != bface1->vertex2() ) return false;
    return true;
  }
};

struct BFaceHash {
  int operator() ( const BFace* bface ) const
  {
    return ( int ) ( ( long )bface->vertex0() ^ ( long )bface->vertex1() ^ ( long )bface->vertex2() );
  }
};*/


struct MHash {
  int operator() ( const DVertex* v ) const
  {
    return ( int ) ( long )v;
  }
};

struct MEqual {
  int operator() ( const DVertex* v0, const DVertex* v1 ) const
  {
    return ( v0 == v1 );
  }
};

}

#endif
