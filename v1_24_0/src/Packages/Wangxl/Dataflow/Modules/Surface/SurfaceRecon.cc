/*
 *  SurfaceRecon.cc:
 *
 *  Written by:
 *   wangxl
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/Wangxl/share/share.h>

#include <Core/Geometry/Vector.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/MaskedLatVolField.h>
#include <Core/Datatypes/PointCloudField.h>

#include <Packages/Wangxl/Core/Datatypes/Graph/Graph.h>
#include <Packages/Wangxl/Core/Datatypes/Graph/Mst.h>
#include <Packages/Wangxl/Core/Datatypes/Graph/Dfs.h>
#include <Packages/Wangxl/Core/Datatypes/Graph/NodeMap.h>
#include <Packages/Wangxl/Core/Datatypes/Graph/EdgeMap.h>
#include <Packages/Wangxl/Core/Datatypes/Kdtree.h>
#include <Packages/Wangxl/Core/Datatypes/Point3.h>
#include <Core/Containers/RangeTree.h>
#include <Core/Containers/Array3.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <map>
#include <hash_map>
#include <queue>
#include <math.h>
#include <sys/stat.h>

namespace Wangxl {

using std::cerr;
using std::cout;
using std::ifstream;
using std::ofstream;
using std::endl;
using std::vector;
using std::list;
using std::map;
using std::hash_map;
using std::hash;
using std::equal_to;
using std::queue;
using std::min;

#ifndef LARGENUMBER
#define LARGENUMBER 1e300
#endif
#ifndef SMALLNUMBER
#define SMALLNUMBER -1e300
#endif

class SurfaceRecon;

class TPlane {
public:
  TPlane() { d_point = 0; d_origin = 0; d_normal = 0; }
  TPlane( Point3* pnt, Point3* org, Vector* nml ):d_point(pnt),d_origin(org),d_normal(nml){}
  ~TPlane() { delete d_point; delete d_origin; delete d_normal; }
  inline Point3* origin() { return d_origin; }
  inline Vector* normal() { return d_normal; }
  inline Point3* point() { return d_point; }
private:
  Point3* d_point;
  Point3* d_origin;
  Vector* d_normal;
};

class CubeNode {
public:
  CubeNode() { indexI = indexJ = indexK = 0; visited = false; fieldValue = 0.0; }
  CubeNode( LatVolMesh::NodeIndex nidx, double dis ) { indexI = nidx.i_; indexJ = nidx.j_; indexK = nidx.k_; fieldValue = dis; visited = false; }
  CubeNode( LatVolMesh::NodeIndex nidx, double dis, bool vst ) { indexI = nidx.i_; indexJ = nidx.j_; indexK = nidx.k_; fieldValue = dis; visited = vst; }
  ~CubeNode() {};
  inline bool isDone() { return visited; }
  inline void done() { visited = true; }
  inline int i() { return indexI; }
  inline int j() { return indexJ; }
  inline int k() { return indexK; }
  inline double value() { return fieldValue; }
private:
  int indexI;
  int indexJ;
  int indexK;
  double fieldValue;
  bool visited;
};

class SurfaceDfs : public Dfs {
public:
  SurfaceDfs( SurfaceRecon* ptr ) : Dfs() { reconPtr = ptr; }
  virtual ~SurfaceDfs(){}
  void run(  Graph& graph, GraphNode& startnode );
  virtual void pre_recusive_handler( Graph& graph, GraphEdge& edge, GraphNode& node );
private:
  SurfaceRecon* reconPtr;
};

struct Hash {
  int operator() ( const Point3* point ) const
  {
    return ( int ) ( long )point;
  }
};

struct Equal {
  bool operator() ( const Point3* point0, const Point3* point1 ) const
  {
    return ( point0 == point1 );
  }
};

using namespace SCIRun;

typedef PointCloudMesh::Node::iterator node_iterator;

class WangxlSHARE SurfaceRecon : public Module {
public:
  SurfaceRecon(GuiContext*);

  virtual ~SurfaceRecon();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
private:
  bool readPoints();
  void computeTangentPlane(); 
  void principalComponents( const list< Point3* >& nnPts, Vector* normal, Point3* centroid );
  void orientTangentPlane();
  void trackSurface();
  void cleanUP();
  bool getSignedDis( Point3* queryPnt, double& distance );
  void getFace( int index, unsigned int code[8] , unsigned int vert[4] );
  bool getNeighbor ( int face, CubeNode* node, LatVolMesh::NodeIndex& idx, int ni, int nj, int nk );
  unsigned int encode ( LatVolMesh::NodeIndex idx, int ni, int nj );
  //fuctions for computing pricipal components
  void jacobi( double a[][4],double d[], double v[][4] );
  void eigsrt( double d[], double v[][4] );

  list< Point3* > d_points, d_origins;
  KDTree< Point3, double > *d_pntTree;
  KDTree< Point3, double > *d_orgTree;
  RangeTree< Point3, double, true > *rd_pntTree;
  RangeTree< Point3, double, true > *rd_orgTree;

  Graph d_graph; // remeinian graph for tangent planes
  double d_dist; // nearest neighbor searching distant
  GraphNode d_startNode; //nodes from which to do normal propagation
  double minx, miny, minz, maxx, maxy, maxz; // bounding box of the poits
  hash_map< Point3*, GraphNode, Hash, Equal > d_pnmap; // point-node map
  hash_map< Point3*, TPlane*, Hash, Equal > d_otmap; // origin-tplane map
  NodeMap< TPlane* > d_ntmap; // node-tengent plane map
  EdgeMap< double > d_ewmap; // edge-weight map

  FieldIPort* d_iport;
  FieldOPort* d_oport;
  MaskedLatVolField<double> *lv;

  friend class SurfaceDfs;
};


DECLARE_MAKER(SurfaceRecon)
SurfaceRecon::SurfaceRecon(GuiContext* ctx)
  : Module("SurfaceRecon", ctx, Source, "Surface", "Wangxl")
{
  
}

SurfaceRecon::~SurfaceRecon(){
}

void
 SurfaceRecon::execute(){

  d_iport = (FieldIPort *)get_iport("point_cloud");
  d_oport = (FieldOPort *)get_oport("surface_mesh");

  if (!d_iport) {
    error("Unable to initialize iport 'point_cloud'.");
    return;
  }
  if (!d_oport) {
    error("Unable to initialize iport 'surface_mesh'.");
    return;
  }

  readPoints();
  computeTangentPlane();
  orientTangentPlane();
  trackSurface();
  cleanUP();
  FieldHandle fH(lv);
  d_oport->send(fH);
}

bool SurfaceRecon::readPoints()
{
  FieldHandle ihandle;
  Field* ifield;
  PointCloudField<double>* pfield;
  PointCloudMesh* pmesh;
  node_iterator ni, ni_end;
  Point point;
  double x, y, z;

  GraphNode node;
  Point3* pnt;
  minx = miny = minz = LARGENUMBER;
  maxx = maxy = maxz = SMALLNUMBER;

  if (!d_iport->get(ihandle) || !(ifield = ihandle.get_rep())) return false;
  pfield = (PointCloudField<double>*)ifield;
  pmesh =  pfield->get_typed_mesh().get_rep();
  pmesh->begin(ni);
  pmesh->end(ni_end);
  while ( ni != ni_end ) {
    pmesh->get_point(point,*ni);
    x = point.x();
    y = point.y();
    z = point.z();
    if ( x < minx ) minx = x;  if ( y < miny ) miny = y;  if ( z < minz ) minz = z;
    if ( x > maxx ) maxx = x;  if ( y > maxy ) maxy = y;  if ( z > maxz ) maxz = z;
    pnt = new Point3( x, y, z );
    d_points.push_back( pnt );
    node = d_graph.newNode(); // create a node in graph for each point
    d_pnmap[pnt] = node; // associate the node to this point
  }
  d_ntmap.init( d_graph, 0 );
  d_pntTree = new KDTree< Point3, double >( d_points, 3 );
  return true;
}

void SurfaceRecon::computeTangentPlane()
{
  Point3 *queryPnt, *centroid, *pnt;
  TPlane *tplane0, *tplane1;
  Vector *normal;
  GraphNode node0, node1;
  GraphEdge edge;
  list< Point3* > nnPts;
  list< Point3* >::const_iterator i,j,k;
  GraphNode::adj_nodes_iterator in;
  GraphNode::adj_edges_iterator ie;
  double maxz = SMALLNUMBER;
  bool add;
  bool found = false;

  for ( i = d_points.begin(); i != d_points.end(); i++ ) {
    queryPnt = *i;
    nnPts.clear();
    d_pntTree->querySphere( *queryPnt, d_dist, nnPts );
    node0 = d_pnmap[queryPnt];
    for ( j = nnPts.begin(); j != nnPts.end(); j++ ) {
      // add edge information to the rameinian graph
      pnt = *j;
      if ( pnt != queryPnt ){ // a point could not create a edge with itself
	node1 = d_pnmap[pnt];
	add = true;// a new edge would be created by default
	for ( in = node0.adj_nodes_begin(); in != node0.adj_nodes_end(); in++ )
	  if ( *in == node1 ) {
	    add = false;
	    break;
	  }
	if ( add ) d_graph.newEdge( node0, node1 );
      }
    }
    normal = new Vector;
    centroid = new Point3;

    principalComponents( nnPts, normal, centroid );
    if( centroid->z() > maxz ) { // get starting node for propagating normal vectors
      maxz = centroid->z();
      d_startNode = node0;
    }
    for ( j = d_origins.begin(); j != d_origins.end(); j++ ) {
      if ( ( **j - *centroid ).length() < 0.0001 ) {
	int stop = 1;
      }
    }
    cout << queryPnt->x() << " " << queryPnt->y() << " " << queryPnt->z() << endl;
    cout << centroid->x() << " " << centroid->y() << " " << centroid->z() << endl;
    d_origins.push_back( centroid );
    tplane0 = new TPlane( queryPnt, centroid, normal );
    for ( ie = node0.adj_edges_begin(); ie != node0.adj_edges_end(); ie++ ) {
      edge = *ie;
      node1 = node0.opposite(edge);
      tplane1 =  d_ntmap[node1];
      if ( tplane1 != 0 ) // the other tangent plane exists
	d_ewmap[edge] = 1 - fabs( Dot( *( tplane0->normal() ), *( tplane1->normal() ) ) );
    }
    d_ntmap[node0] = tplane0;
    d_otmap[centroid] = tplane0;
  }
  d_orgTree = new KDTree< Point3, double >( d_origins, 3 );

}

void SurfaceRecon::principalComponents( const list< Point3* >& nnPts, Vector* normal, Point3* centroid )
{
Point3* pnt;
//int nrot;
list< Point3* >::const_iterator i;
double mcovar[4][4] = { { 0,0,0,0 }, { 0,0,0,0 }, { 0,0,0,0 }, { 0,0,0,0 } };
double v[4][4] = { { 0,0,0,0 }, { 0,0,0,0 }, { 0,0,0,0 }, { 0,0,0,0 } };
double d[4] = { 0,0,0,0 };

  int num = nnPts.size();
  for ( i = nnPts.begin(); i != nnPts.end(); i++ ) {
    pnt = *i;
    centroid->x( centroid->x() + pnt->x() );
    centroid->y( centroid->y() + pnt->y() );
    centroid->z( centroid->z() + pnt->z() );
  }
  *centroid /= num;
    for ( i = nnPts.begin(); i != nnPts.end(); i++ ) {
    pnt = *i;
    mcovar[1][1] += ( pnt->x() - centroid->x() ) * ( pnt->x() - centroid->x() );
    mcovar[1][2] += ( pnt->y() - centroid->y() ) * ( pnt->x() - centroid->x() );
    mcovar[2][2] += ( pnt->y() - centroid->y() ) * ( pnt->y() - centroid->y() );
    mcovar[1][3] += ( pnt->z() - centroid->z() ) * ( pnt->x() - centroid->x() );
    mcovar[2][3] += ( pnt->z() - centroid->z() ) * ( pnt->y() - centroid->y() );
    mcovar[3][3] += ( pnt->z() - centroid->z() ) * ( pnt->z() - centroid->z() );
    }
  mcovar[1][1] /= num;
  mcovar[2][1] = mcovar[1][2] /= num;
  mcovar[2][2] /= num;
  mcovar[3][1] = mcovar[1][3] /= num;
  mcovar[3][2] = mcovar[2][3] /= num;
  mcovar[3][3] /= num;
  jacobi( mcovar, d, v );
  eigsrt ( d, v );
  Vector vec( v[1][3], v[2][3], v[3][3]);
  vec.normalize();
  *normal = vec;
} 

#define ROTATE(a,i,j,k,l) g=a[i][j];h=a[k][l];a[i][j]=g-s*(h+g*tau);\
	a[k][l]=h+s*(g-h*tau);

void SurfaceRecon::jacobi(double a[][4], double d[], double v[][4] )
{
int j,iq,ip,i;
int n = 3;
double tresh,theta,tau,t,sm,s,h,g,c;
double b[4], z[4];

  for (ip=1;ip<=n;ip++) {
    for (iq=1;iq<=n;iq++) v[ip][iq]=0.0;
    v[ip][ip]=1.0;
  }
  for (ip=1;ip<=n;ip++) {
    b[ip]=d[ip]=a[ip][ip];
    z[ip]=0.0;
  }
  //  *nrot=0;
  for (i=1;i<=50;i++) {
    sm=0.0;
    for (ip=1;ip<=n-1;ip++) {
      for (iq=ip+1;iq<=n;iq++)
	sm += fabs(a[ip][iq]);
    }
    if (sm == 0.0) return;
    if (i < 4)
      tresh=0.2*sm/(n*n);
    else
      tresh=0.0;
    for (ip=1;ip<=n-1;ip++) {
      for (iq=ip+1;iq<=n;iq++) {
	g=100.0*fabs(a[ip][iq]);
	if (i > 4 && fabs(d[ip])+g == fabs(d[ip])
	    && fabs(d[iq])+g == fabs(d[iq]))
	  a[ip][iq]=0.0;
	else if (fabs(a[ip][iq]) > tresh) {
	  h=d[iq]-d[ip];
	  if (fabs(h)+g == fabs(h))
	    t=(a[ip][iq])/h;
	  else {
	    theta=0.5*h/(a[ip][iq]);
	    t=1.0/(fabs(theta)+sqrt(1.0+theta*theta));
	    if (theta < 0.0) t = -t;
	  }
	  c=1.0/sqrt(1+t*t);
	  s=t*c;
	  tau=s/(1.0+c);
	  h=t*a[ip][iq];
	  z[ip] -= h;
	  z[iq] += h;
	  d[ip] -= h;
	  d[iq] += h;
	  a[ip][iq]=0.0;
	  for (j=1;j<=ip-1;j++) {
	    ROTATE(a,j,ip,j,iq)
	      }
	  for (j=ip+1;j<=iq-1;j++) {
	    ROTATE(a,ip,j,j,iq)
	      }
	  for (j=iq+1;j<=n;j++) {
	    ROTATE(a,ip,j,iq,j)
	      }
	  for (j=1;j<=n;j++) {
	    ROTATE(v,j,ip,j,iq)
	      }
	  //	  ++(*nrot);
	}
      }
    }
    for (ip=1;ip<=n;ip++) {
      b[ip] += z[ip];
      d[ip]=b[ip];
      z[ip]=0.0;
    }
  }
  cout << "Too many iterations in routine JACOBI" << endl;
}

#undef ROTATE


void SurfaceRecon::eigsrt( double d[], double v[][4] )
{
int k,j,i;
double p;
int n = 3;

  for (i=1;i<n;i++) {
    p=d[k=i];
    for (j=i+1;j<=n;j++)
      if (d[j] >= p) p=d[k=j];
    if (k != i) {
      d[k]=d[i];
      d[i]=p;
      for (j=1;j<=n;j++) {
	p=v[j][i];
	v[j][i]=v[j][k];
	v[j][k]=p;
      }
    }
  }
}

void SurfaceRecon::orientTangentPlane()
{
Vector* startNormal;
Mst msTree;
SurfaceDfs propagation( this );

  msTree.run( d_graph, d_ewmap );
  startNormal = d_ntmap[d_startNode]->normal();
  if ( startNormal->z() < 0.0 ) *startNormal = -*startNormal; // force the point with maximum z value has positive z normal
  propagation.run( d_graph, d_startNode );
}

void SurfaceRecon::trackSurface()
{
int ni, nj, nk; // dimension of the grid
int i, j, k, index;
Point3 minp(minx, miny, minz), maxp(maxx, maxy, maxz);
Point3 queryPnt, *org;
hash_map < unsigned int, CubeNode*,  hash< unsigned int >, equal_to< unsigned int > > NodeMap;
hash_map < unsigned int, CubeNode*,  hash< unsigned int >, equal_to< unsigned int > >::const_iterator iter;
unsigned int cube, face[4], code[8];
queue< unsigned int > CubeQueue;
CubeNode *node, *tmpnode;
double distance;
int number=0;
LatVolMesh::NodeIndex nidx;

  ni = ( int )( fabs( maxx - minx ) /* *1.25*/ / d_dist ) + 1;
  nj = ( int )( fabs( maxy - miny ) /* *1.25*/ / d_dist ) + 1;
  nk = ( int )( fabs( maxz - minz ) /* *1.25*/ / d_dist ) + 1;
  LatVolMeshHandle lvm = new LatVolMesh(ni, nj, nk, minp, maxp);
  lv = new MaskedLatVolField<double>(lvm, Field::NODE);

  org = *d_origins.begin();
  nidx.i_ = min( (int) ( ( org->x() - minx ) / ( fabs( maxx - minx ) / (ni-1) ) ), ni-2 );
  nidx.j_ = min( (int) ( ( org->y() - miny ) / ( fabs( maxy - miny ) / (nj-1) ) ), nj-2 );
  nidx.k_ = min( (int) ( ( org->z() - minz ) / ( fabs( maxz - minz ) / (nk-1) ) ), nk-2 );
  cube =  encode( nidx, ni, nj );
  lvm->get_center( queryPnt, nidx );
  if( getSignedDis( &queryPnt,distance ) ) { // compute first cube intersecting with the surface
    lv->fdata()[nidx] = distance;
    lv->mask()[nidx] = 1;
    cout << "vertex " << nidx.i_ << " " << nidx.j_ << " " << nidx.k_ << "  dis=" << distance << endl;
  }
  NodeMap[ cube ] = new CubeNode( nidx, distance, true );number++;
  CubeQueue.push ( cube );
  //      else continue; // this cube has been processed
  while ( !CubeQueue.empty() ) {
    cube = CubeQueue.front();
    CubeQueue.pop();
    node = NodeMap[cube];
    //computing the signed distances for every vertices of this cube if necessary
    index = 0;
    for ( i = 0; i <= 1; i++ )
      for ( j = 0; j <= 1; j++ )
	for ( k = 0; k <= 1; k++ ) {
	  nidx.i_ = node->i() + i;
	  nidx.j_ = node->j() + j;
	  nidx.k_ = node->k() + k;
	  code[index++] = cube = encode( nidx, ni, nj );
	  if ( NodeMap.find(cube) == NodeMap.end() ) {// a vertex that has never been checked
	    lvm->get_center( queryPnt, nidx );
	    if( getSignedDis( &queryPnt,distance ) ) {
	      lv->fdata()[nidx] = distance;
	      lv->mask()[nidx] = 1;
    cout << "vertex " << nidx.i_ << " " << nidx.j_ << " " << nidx.k_ << "  dis=" << distance << endl;
	    }
	    NodeMap[ cube ] = new CubeNode( nidx, distance );number++;
	  }
	}
    for ( i = 0; i <= 5; i++ ) {// for each face of the cube
      getFace( i, code, face );
      if ( !( ( ( NodeMap[face[0]]->value() > 0 ) && ( NodeMap[face[1]]->value() > 0 ) && ( NodeMap[face[2]]->value() > 0 ) &&( NodeMap[face[3]]->value() > 0 ) ) ||  ( ( NodeMap[face[0]]->value() < 0 ) && ( NodeMap[face[1]]->value() < 0 ) && ( NodeMap[face[2]]->value() < 0 ) &&( NodeMap[face[3]]->value() < 0 ) ) )) { // intersected 
	if (getNeighbor( i, node, nidx, ni, nj, nk ) ){
	  cube = encode( nidx, ni, nj );
	  if ( NodeMap.find( cube ) ==  NodeMap.end() ) { // neighbor cube has not been processed
	    lvm->get_center( queryPnt, nidx );
	    if( getSignedDis( &queryPnt,distance ) ) {
	      lv->fdata()[nidx] = distance;
	      lv->mask()[nidx] = 1;
cout << "vertex " << nidx.i_ << " " << nidx.j_ << " " << nidx.k_ << "  dis=" << distance << endl;
	    }
	    NodeMap[ cube ] = new CubeNode( nidx, distance, true );number++;
	    CubeQueue.push ( cube );
	  }
	  else {
	    tmpnode = NodeMap[ cube ];
	    if( !(tmpnode->isDone() ) ) {
	      tmpnode->done();
	      CubeQueue.push ( cube );
	    }
	  }
	}// end of neighbor testing
      }//end of intersecting
    } // end of 6 faces
  } // IntCubes is empty
  //  }// end for
  for ( iter = NodeMap.begin(); iter != NodeMap.end(); iter++ ) delete (*iter).second;
  cout << "Total= " << ni*nj*nk;
  cout << "Checked = " << number;
}

unsigned int SurfaceRecon::encode( LatVolMesh::NodeIndex idx, int ni, int nj )
{
  return ( idx.i_ + idx.j_ * ni + idx.k_ * ni * nj );
}

void SurfaceRecon::getFace( int index, unsigned int code[8], unsigned int face[4] ) 
{
  switch ( index ) {
  case 0:
    face[0] = code[0];
    face[1] = code[4];
    face[2] = code[6];
    face[3] = code[2];
    break;
  case 1:
    face[0] = code[1];
    face[1] = code[5];
    face[2] = code[7];
    face[3] = code[3];
    break;
  case 2:
    face[0] = code[0];
    face[1] = code[4];
    face[2] = code[5];
    face[3] = code[1];
    break;
  case 3:
    face[0] = code[4];
    face[1] = code[6];
    face[2] = code[7];
    face[3] = code[5];
    break;
  case 4:
    face[0] = code[2];
    face[1] = code[6];
    face[2] = code[7];
    face[3] = code[3];
    break;
  case 5:
    face[0] = code[0];
    face[1] = code[2];
    face[2] = code[3];
    face[3] = code[1];
    break;
  }
  return;
}

bool SurfaceRecon::getNeighbor ( int face, CubeNode* node, LatVolMesh::NodeIndex& nidx, int ni, int nj, int nk )
{
  
  switch ( face ) {
  case 0:
    if( node->k() == 0 ) return false;
    else {
      nidx.i_ = node->i();
      nidx.j_ = node->j();
      nidx.k_ = node->k()-1;
    }
    break;
  case 1:
    if( node->k() == nk-2 ) return false;
    else {
      nidx.i_ = node->i();
      nidx.j_ = node->j();
      nidx.k_ = node->k()+1;
    }
    break;
  case 2:
    if( node->j() == 0 ) return false;
    else {
      nidx.i_ = node->i();
      nidx.j_ = node->j()-1;
      nidx.k_ = node->k();
    }
    break;
  case 3:
    if( node->i() == ni-2 ) return false;
    else {
      nidx.i_ = node->i()+1;
      nidx.j_ = node->j();
      nidx.k_ = node->k();
    }
    break;
  case 4:
    if( node->j() == nj-2 ) return false;
    else {
      nidx.i_ = node->i();
      nidx.j_ = node->j()+1;
      nidx.k_ = node->k();
    }
    break;
  case 5:
    if( node->i() == 0 ) return false;
    else {
      nidx.i_ = node->i()-1;
      nidx.j_ = node->j();
      nidx.k_ = node->k();
    }
    break;
  }
  return true;
}

bool SurfaceRecon::getSignedDis( Point3* queryPnt, double& distance )
{
Point3 *org, *pnt, *korg;
Vector *nml;
Point3 proj;
list< Point3* > found;
  d_orgTree->queryNearest( *queryPnt, 1, found );
  korg = found.front();
  nml = d_otmap[korg]->normal();
  pnt =  d_otmap[korg]->point();
  distance = Dot( ( *queryPnt - *korg ), *nml );
  proj = *queryPnt - ( *nml ) * distance;
  if ( ( *pnt - proj ).length() < d_dist ) return true;
  return false;
}

void SurfaceRecon::cleanUP()
{
  d_points.clear();
  d_origins.clear();
  //  d_graph.clear();
  delete d_pntTree;
  delete d_orgTree;
}

void SurfaceDfs::run(  Graph& graph, GraphNode& startnode ) { 
  Dfs::start( startnode );
  Dfs::run( graph );
}
void SurfaceDfs::pre_recusive_handler( Graph& graph, GraphEdge& edge, GraphNode& node ){
  GraphNode curr;
  Vector* normal0;
  Vector* normal1;
  curr = node.opposite( edge );
  normal0 = reconPtr->d_ntmap[curr]->normal();
  normal1 = reconPtr->d_ntmap[node]->normal();
  double product = Dot( *normal0, *normal1 );
  if ( product < 0.0 ) *normal1 = -*normal1;
}


void
 SurfaceRecon::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Wangxl






