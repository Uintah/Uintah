#include <Packages/rtrt/Core/HierarchicalGrid.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <iostream>
#include <stdlib.h>
#include <Packages/rtrt/Core/Tri.h>

extern "C" {
#include <Packages/rtrt/Core/pcube.h>
}

using namespace rtrt;
using SCIRun::Thread;
using SCIRun::Time;

int HierarchicalGrid::L1Cells = 0;
int HierarchicalGrid::L2Cells = 0;
int HierarchicalGrid::L1CellsWithChildren = 0;
int HierarchicalGrid::L2CellsWithChildren = 0;
int HierarchicalGrid::LeafCells = 0;
int HierarchicalGrid::TotalLeafPrims = 0;

struct GridData
{
    int sx, sy, sz, ex, ey, ez;
};

HierarchicalGrid::HierarchicalGrid( Object* obj, int nsides,
				    int nSidesLevel2, int nSidesLevel3,
				    int minObjects1, int minObjects2 ):
    Grid( obj, nsides ),
    _nSidesLevel2( nSidesLevel2 ),
    _nSidesLevel3( nSidesLevel3 ),
    _minObjects1( minObjects1 ),
    _minObjects2( minObjects2 )
{
}

HierarchicalGrid::HierarchicalGrid( Object* obj, int nsides,
				    int nSidesLevel2, int nSidesLevel3,
				    int minObjects1, int minObjects2, 
				    const BBox &b ):
    Grid( obj, nsides ),
    _nSidesLevel2( nSidesLevel2 ),
    _nSidesLevel3( nSidesLevel3 ),
    _minObjects1( minObjects1 ),
    _minObjects2( minObjects2 )
{
    bbox=b;
}

HierarchicalGrid::~HierarchicalGrid( void )
{
    // There's probably a memory leak somewhere
}

//
// This is repeated from Grid.cc -- Might be good idea to merge the two
//
static inline void calc_se(const BBox& obj_bbox, const BBox& bbox,
			   const Vector& diag,
			   int nx, int ny, int nz,
			   int& sx, int& sy, int& sz,
			   int& ex, int& ey, int& ez)
{
    Vector s((obj_bbox.min()-bbox.min())/diag);
    Vector e((obj_bbox.max()-bbox.min())/diag);
    sx=(int)(s.x()*nx);
    sy=(int)(s.y()*ny);
    sz=(int)(s.z()*nz);
    ex=(int)(e.x()*nx);
    ey=(int)(e.y()*ny);
    ez=(int)(e.z()*nz);
    if(sx < 0 || ex >= nx){
	cerr << "NX out of bounds!\n";
	cerr << "sx=" << sx << ", ex=" << ex << '\n';
	cerr << "e=" << e << '\n';
	cerr << "obj_bbox=" << obj_bbox.min() << ", " << obj_bbox.max() << '\n';
	cerr << "bbox=" << bbox.min() << ", " << bbox.max() << '\n';
	cerr << "diag=" << diag << '\n';
	exit(1);
    }
    if(sy < 0 || ey >= ny){
	cerr << "NY out of bounds!\n";
	cerr << "sy=" << sy << ", ey=" << ey << '\n';
	exit(1);
    }
    if(sz < 0 || ez >= nz){
	cerr << "NZ out of bounds!\n";
	cerr << "sz=" << sz << ", ez=" << ez << '\n';
	cerr << "e=" << e << '\n';
	cerr << "obj_bbox=" << obj_bbox.min() << ", " << obj_bbox.max() << '\n';
	cerr << "bbox=" << bbox.min() << ", " << bbox.max() << '\n';
	cerr << "diag=" << diag << '\n';
	exit(1);
    }
}

static inline void sub_calc_se(const BBox& obj_bbox, const BBox& bbox,
			       const Vector& diag,
			       int nx, int ny, int nz,
			       int& sx, int& sy, int& sz,
			       int& ex, int& ey, int& ez)
{
  Vector s((obj_bbox.min()-bbox.min())/diag);
  Vector e((obj_bbox.max()-bbox.min())/diag);
  sx=(int)(s.x()*nx);
  sy=(int)(s.y()*ny);
  sz=(int)(s.z()*nz);
  ex=(int)(e.x()*nx);
  ey=(int)(e.y()*ny);
  ez=(int)(e.z()*nz);
  sx=Max(Min(sx, nx-1), 0);
  sy=Max(Min(sy, ny-1), 0);
  sz=Max(Min(sz, nz-1), 0);
  ex=Max(Min(ex, nx-1), 0);
  ey=Max(Min(ey, ny-1), 0);
  ez=Max(Min(ez, nz-1), 0);
}

void
HierarchicalGrid::subpreprocess( double maxradius, int& pp_offset, int& scratchsize, int level )
{
    double time=Time::currentSeconds();
    obj->preprocess(maxradius, pp_offset, scratchsize);
    time=Time::currentSeconds();

    Array1<Object*> prims;
    obj->collect_prims(prims);
    Array1<BBox> primsBBox( prims.size() );
    Array1<GridData> primsGridData( prims.size() );
    
    time=Time::currentSeconds();

    time=Time::currentSeconds();

    int ncells;
    if (level == 2) {
      ncells = _nSidesLevel2*_nSidesLevel2*_nSidesLevel2;
    } else if (level == 3) {
      ncells = _nSidesLevel3*_nSidesLevel3*_nSidesLevel3;
    }

    Vector diag(bbox.diagonal());
    bbox.extend(bbox.max()+diag*1.e-3);
    bbox.extend(bbox.min()-diag*1.e-3);
    diag=bbox.diagonal();
    double volume=diag.x()*diag.y()*diag.z();
    double c=cbrt(ncells/volume);
    nx=(int)(c*diag.x()+0.5);
    ny=(int)(c*diag.y()+0.5);
    nz=(int)(c*diag.z()+0.5);
    if(nx<2)
	nx=2;
    if(ny<2)
	ny=2;
    if(nz<2)
	nz=2;
    int ngrid=nx*ny*nz;

    if (level == 2) {
      L2Cells += ngrid;
    }

    if(counts)
	delete[] counts;
    if(grid)
	delete[] grid;
    counts = new int[2*ngrid];
    for( int i = 0; i < ngrid*2; i++ )
	counts[i] = 0;
    
    //
    // Allocate group of objects
    //
    Array1<Group*> objList( ngrid );
    objList.initialize( 0 );

    double itime=time;
    int nynz=ny*nz;

    real verts[3][3];
    real polynormal[3];

    Vector p0,p1,p2;

    for(int i=0;i<prims.size();i++){
	double tnow=Time::currentSeconds();
	if(tnow-itime > 5.0){
	    cerr << i << "/" << prims.size() << '\n';
	    itime=tnow;
	}
	BBox obj_bbox;
	prims[i]->compute_bounds( primsBBox[i], maxradius );
	sub_calc_se( primsBBox[i], bbox, diag, nx, ny, nz, 
		     primsGridData[i].sx, primsGridData[i].sy, primsGridData[i].sz,
		     primsGridData[i].ex, primsGridData[i].ey, primsGridData[i].ez );

	Tri *tri = dynamic_cast<Tri*>(prims[i]);

	if (tri) {
	    if (tri->isbad())
		continue;

//	    printf("Original vertices: %lf %lf %lf, %lf %lf %lf, %lf %lf %lf\n", tri->pt(0).x(),tri->pt(0).y(),tri->pt(0).z(),tri->pt(1).x(),tri->pt(1).y(),tri->pt(1).z(),tri->pt(2).x(),tri->pt(2).y(),tri->pt(2).z());

	    p0 = (tri->pt(0) - bbox.min())*(Vector(nx,ny,nz)/diag);
	    p1 = (tri->pt(1) - bbox.min())*(Vector(nx,ny,nz)/diag);
	    p2 = (tri->pt(2) - bbox.min())*(Vector(nx,ny,nz)/diag);

	    Vector n = Cross((p2-p0),(p1-p0));

	    n.normalize();
	
	    polynormal[0] = n.x();
	    polynormal[1] = n.y();
	    polynormal[2] = n.z();
	}

	for(int x=primsGridData[i].sx;x<=primsGridData[i].ex;x++){
	    for(int y=primsGridData[i].sy;y<=primsGridData[i].ey;y++){
		int idx=x*nynz+y*nz+primsGridData[i].sz;
		for(int z=primsGridData[i].sz;z<=primsGridData[i].ez;z++){
		  if (tri) {	
			
		    verts[0][0] = p0.x() - ((double)x+.5);
		    verts[1][0] = p1.x() - ((double)x+.5);
		    verts[2][0] = p2.x() - ((double)x+.5);
		    
		    verts[0][1] = p0.y() - ((double)y+.5);
		    verts[1][1] = p1.y() - ((double)y+.5);
		    verts[2][1] = p2.y() - ((double)y+.5);
		    
		    verts[0][2] = p0.z() - ((double)z+.5);
		    verts[1][2] = p1.z() - ((double)z+.5);
		    verts[2][2] = p2.z() - ((double)z+.5);
		    
//		    printf("Verts: %lf %lf %lf, %lf %lf %lf, %lf %lf %lf\n", verts[0][0], verts[0][1], verts[0][2], verts[1][0], verts[1][1], verts[1][2], verts[2][0], verts[2][1], verts[2][2]);
		    
		    if (fast_polygon_intersects_cube(3, verts, polynormal, 0, 0))
		    {
//		      printf("Added to Cell %d %d %d\n\n",x,y,z);
		      if( !objList[idx] )	 	
			objList[idx] = new Group();
		      objList[idx]->add( prims[i] );
		      counts[idx*2+1]++;
		    }
		    
		    idx++;
		    
		  } else {  // Not TRI
		    if( !objList[idx] )	 	
		      objList[idx] = new Group();
		    objList[idx]->add( prims[i] );
		    counts[idx*2+1]++;
		    idx++;
		  }
		}
	    }
	}
    }
    time=Time::currentSeconds();
    int total=0;
    Array1<int> whichCell( ngrid );
    whichCell.initialize( 0 );

    for(int i=0;i<ngrid;i++){
	int count=counts[i*2+1];
	counts[i*2]=total;
	if(level == 2 && count >= _minObjects2 ) {
	  whichCell[i] = count;
	  count = 1;
	  counts[i*2+1] = 1;
	}
	total+=count;
    }

    grid=new Object*[total];
    for(int i=0;i<total;i++)
	grid[i]=0;

    time=Time::currentSeconds();
    itime=time;
    Array1<int> current(ngrid);
    Array1<int> whichCellPos( ngrid );
    current.initialize(0);
    whichCellPos.initialize(0);

    int pos = 0;

    for (int idx=0; idx < objList.size(); idx++)
    {
      // Find grid position
      
      if (whichCell[idx] == 0)
      {
	if (objList[idx])
	  for (int j=0; j<objList[idx]->numObjects(); j++)
	  {
	    grid[pos] = objList[idx]->objs[j];
	    pos++;
	  }
      } else {
	grid[pos] = (Object*) 0;
	whichCellPos[idx] = pos;
	pos++;
      }
    }

    int i=0;
    double dx=diag.x()/nx;
    double dy=diag.y()/ny;
    double dz=diag.z()/nz;

    for (int ii=0; ii<nx; ii++) {
      for (int jj=0; jj<ny; jj++) {
	for (int kk=0; kk<nz; kk++, i++) {
	  if( whichCell[i] > 0 && !grid[whichCellPos[i]] ) {
	    BBox b(Point(bbox.min()+Vector(ii*dx, jj*dy, kk*dz)),
		   Point(bbox.min()+Vector((ii+1)*dx,(jj+1)*dy,(kk+1)*dz)));
	    HierarchicalGrid *g = new HierarchicalGrid( objList[i],
							nsides,
							_nSidesLevel2,
							_nSidesLevel3,
							_minObjects1,
							_minObjects2, b );
	    g->subpreprocess( maxradius, pp_offset, scratchsize, 3 );
	    grid[whichCellPos[i]] = g;
	    L2CellsWithChildren++;
	  } else {
	    LeafCells++;
	    TotalLeafPrims+=counts[i*2+1];
	  }
	}
      }
    }

    //
    // Deallocate group of objects
    //
    for( int i = 0; i < ngrid; i++ )
	delete objList[i];
    time=Time::currentSeconds();
    for(int i=0;i<ngrid;i++){
      if( ( current[i] != counts[i*2+1] ) ) {
	if( ( whichCell[i] > 0 ) && ( counts[i*2+1] != 1 ) ) {
	  cerr << "OOPS!\n";
	  cerr << "current: " << current[i] << '\n';
	  cerr << "counts: " << counts[i*2+1] << '\n';
	  cerr << "whichCell: " << whichCell[i] << '\n';
	  exit(1);
	}
      }
    }
    for(int i=0;i<total;i++){
      if(!grid[i]){
	cerr << "OOPS: grid[" << i << "]==0!\n";
	exit(1);
      }
    }
    time=Time::currentSeconds();
}

void 
HierarchicalGrid::preprocess( double maxradius, int& pp_offset, 
			      int& scratchsize )
{
    double time=Time::currentSeconds();
    obj->preprocess(maxradius, pp_offset, scratchsize);
    time=Time::currentSeconds();

    Array1<Object*> prims;
    obj->collect_prims(prims);
    Array1<BBox> primsBBox( prims.size() );
    Array1<GridData> primsGridData( prims.size() );
    time=Time::currentSeconds();

    bbox.reset();
    obj->compute_bounds( bbox, maxradius );

    time=Time::currentSeconds();

    int ncells = nsides*nsides*nsides;

    bbox.extend(bbox.min()-Vector(1.e-3, 1.e-3, 1.e-3));
    bbox.extend(bbox.max()+Vector(1.e-3, 1.e-3, 1.e-3));
    Vector diag(bbox.diagonal());
    bbox.extend(bbox.max()+diag*1.e-3);
    bbox.extend(bbox.min()-diag*1.e-3);
    diag=bbox.diagonal();
    double volume=diag.x()*diag.y()*diag.z();
    double c=cbrt(ncells/volume);
    nx=(int)(c*diag.x()+0.5);
    ny=(int)(c*diag.y()+0.5);
    nz=(int)(c*diag.z()+0.5);
    if(nx<2)
	nx=2;
    if(ny<2)
	ny=2;
    if(nz<2)
	nz=2;
    int ngrid=nx*ny*nz;

    L1Cells += ngrid;

    if(counts)
	delete[] counts;
    if(grid)
	delete[] grid;
    counts = new int[2*ngrid];
    for( int i = 0; i < ngrid*2; i++ )
	counts[i] = 0;
    
    //
    // Allocate group of objects
    //
    Array1<Group*> objList( ngrid );
    objList.initialize( 0 );

    double itime=time;
    int nynz=ny*nz;

    real verts[3][3];
    real polynormal[3];

    Vector p0,p1,p2;

    for(int i=0;i<prims.size();i++){
	double tnow=Time::currentSeconds();
	if(tnow-itime > 5.0){
	    cerr << i << "/" << prims.size() << '\n';
	    itime=tnow;
	}
	BBox obj_bbox;
	prims[i]->compute_bounds( primsBBox[i], maxradius );
	calc_se( primsBBox[i], bbox, diag, nx, ny, nz, 
		 primsGridData[i].sx, primsGridData[i].sy, primsGridData[i].sz,
		 primsGridData[i].ex, primsGridData[i].ey, primsGridData[i].ez );
	Tri *tri = dynamic_cast<Tri*>(prims[i]);

	if (tri) {
	    if (tri->isbad())
		continue;

//	    printf("Original vertices: %lf %lf %lf, %lf %lf %lf, %lf %lf %lf\n", tri->pt(0).x(),tri->pt(0).y(),tri->pt(0).z(),tri->pt(1).x(),tri->pt(1).y(),tri->pt(1).z(),tri->pt(2).x(),tri->pt(2).y(),tri->pt(2).z());

	    p0 = (tri->pt(0) - bbox.min())*(Vector(nx,ny,nz)/diag);
	    p1 = (tri->pt(1) - bbox.min())*(Vector(nx,ny,nz)/diag);
	    p2 = (tri->pt(2) - bbox.min())*(Vector(nx,ny,nz)/diag);

	    Vector n = Cross((p2-p0),(p1-p0));

	    n.normalize();
	
	    polynormal[0] = n.x();
	    polynormal[1] = n.y();
	    polynormal[2] = n.z();

	}

	for(int x=primsGridData[i].sx;x<=primsGridData[i].ex;x++){
	    for(int y=primsGridData[i].sy;y<=primsGridData[i].ey;y++){
		int idx=x*nynz+y*nz+primsGridData[i].sz;
		for(int z=primsGridData[i].sz;z<=primsGridData[i].ez;z++){
// 		    if( !objList[idx] ) 
//			objList[idx] = new Group();
//		    objList[idx]->add( prims[i] );
//		    counts[idx*2+1]++;
//		    idx++;
		  if (tri) {	
		    
		    verts[0][0] = p0.x() - ((double)x+.5);
		    verts[1][0] = p1.x() - ((double)x+.5);
		    verts[2][0] = p2.x() - ((double)x+.5);
		    
		    verts[0][1] = p0.y() - ((double)y+.5);
		    verts[1][1] = p1.y() - ((double)y+.5);
		    verts[2][1] = p2.y() - ((double)y+.5);
		    
		    verts[0][2] = p0.z() - ((double)z+.5);
		    verts[1][2] = p1.z() - ((double)z+.5);
		    verts[2][2] = p2.z() - ((double)z+.5);
		    
//		    printf("Verts: %lf %lf %lf, %lf %lf %lf, %lf %lf %lf\n", verts[0][0], verts[0][1], verts[0][2], verts[1][0], verts[1][1], verts[1][2], verts[2][0], verts[2][1], verts[2][2]);
		    
		    if (fast_polygon_intersects_cube(3, verts, polynormal, 0, 0))
		    {
//		      printf("Added to Cell %d %d %d\n\n",x,y,z);
		      if( !objList[idx] )	 	
			objList[idx] = new Group();
		      objList[idx]->add( prims[i] );
		      counts[idx*2+1]++;
		    }
		    
		    idx++;
		    
		  } else {  // Not TRI
		    if( !objList[idx] )	 	
		      objList[idx] = new Group();
		    objList[idx]->add( prims[i] );
		    counts[idx*2+1]++;
		    idx++;
		  }
		}
	    }
	}
    }
    time=Time::currentSeconds();
    int total=0;
    Array1<int> whichCell( ngrid );
    whichCell.initialize( 0 );

    for(int i=0;i<ngrid;i++){
	int count=counts[i*2+1];
	counts[i*2]=total;
	if( count >= _minObjects1 ) {
	  whichCell[i] = count;
	  count = 1;
	  counts[i*2+1] = 1;
	}
	total+=count;
    }

    grid=new Object*[total];
    for(int i=0;i<total;i++)
	grid[i]=0;

    time=Time::currentSeconds();
    itime=time;
    Array1<int> current(ngrid);
    Array1<int> whichCellPos( ngrid );
    current.initialize(0);
    whichCellPos.initialize(0);
//    for(int i=0;i<prims.size();i++){
//	double tnow=Time::currentSeconds();
//	if(tnow-itime > 5.0){
//	    cerr << i << "/" << prims.size() << '\n';
//	    itime=tnow;
//	}
	//
	// Array of objects that will be put in a grid at lower level
	//
	// Already computed and cached -- no need to do it again
	//
	// prims[i]->compute_bounds(obj_bbox, maxradius);

//	int sx = primsGridData[i].sx;
//	int sy = primsGridData[i].sy;
//	int sz = primsGridData[i].sz;
//	int ex = primsGridData[i].ex;
//	int ey = primsGridData[i].ey;
//	int ez = primsGridData[i].ez;
	//
	// Already computed and cached
	//
	// calc_se( primsBBox[i], bbox, diag, nx, ny, nz, sx, sy, sz, ex, ey, ez);
//	for(int x=sx;x<=ex;x++){
//	    for(int y=sy;y<=ey;y++){
//		int idx=x*nynz+y*nz+sz;
//		for(int z=sz;z<=ez;z++){
//		    int cur=current[idx];
//		    int pos=counts[idx*2]+cur;
//		    if( whichCell[idx] == 0 ){
//			grid[pos] = prims[i];
//			current[idx]++;
//		    } else {
			// int pos = counts[idx*2];
//			grid[pos] = ( Object* ) 0;
//			whichCellPos[idx] = pos;
//			current[idx] = 0;
//		    }
//		    idx++;
//		}
//	    }
//	}
//    }

    int pos = 0;

    for (int idx=0; idx < objList.size(); idx++)
    {
      // Find grid position
      
      if (whichCell[idx] == 0)
      {
	if (objList[idx])
	  for (int j=0; j<objList[idx]->numObjects(); j++)
	  {
	    grid[pos] = objList[idx]->objs[j];
	    pos++;
	  }
      } else {
	grid[pos] = (Object*) 0;
	whichCellPos[idx] = pos;
	pos++;
      }
    }
    
    int i=0;
    double dx=diag.x()/nx;
    double dy=diag.y()/ny;
    double dz=diag.z()/nz;

    for (int ii=0; ii<nx; ii++) {
      for (int jj=0; jj<ny; jj++) {
	for (int kk=0; kk<nz; kk++, i++) {
	  if ((i%100) == 50) {
	    cerr << "Completed "<<i<<"/"<<whichCell.size()<<" ("<<i*100./whichCell.size()<<"%)\n";
	    cerr <<"   "<<L1CellsWithChildren*100./L1Cells<<"% of L1 cells have children\n";
	    cerr <<"   "<<L2CellsWithChildren*100./L2Cells<<"% of L2 cells have children\n";
	    cerr <<"   "<<TotalLeafPrims*1./LeafCells<<" tris / leaf.\n";
	  }
	  if( whichCell[i] > 0 && !grid[whichCellPos[i]] ) {
	    BBox b(Point(bbox.min()+Vector(ii*dx, jj*dy, kk*dz)),
		   Point(bbox.min()+Vector((ii+1)*dx,(jj+1)*dy,(kk+1)*dz)));
	    HierarchicalGrid *g = new HierarchicalGrid( objList[i],
							nsides,
							_nSidesLevel2,
							_nSidesLevel3,
							_minObjects1,
							_minObjects2, b );
	    g->subpreprocess( maxradius, pp_offset, scratchsize, 2 );
	    grid[whichCellPos[i]] = g;
	    L1CellsWithChildren++;
	  } else {
	    LeafCells++;
	    TotalLeafPrims+=counts[i*2+1];
	  }
	}
      }
    }

    //
    // Deallocate group of objects
    //
    for( int i = 0; i < ngrid; i++ )
	delete objList[i];

    time=Time::currentSeconds();
    for(int i=0;i<ngrid;i++){
	if( ( current[i] != counts[i*2+1] ) ) {
	    if( ( whichCell[i] > 0 ) && ( counts[i*2+1] != 1 ) ) {
		cerr << "OOPS!\n";
		cerr << "current: " << current[i] << '\n';
		cerr << "counts: " << counts[i*2+1] << '\n';
		cerr << "whichCell: " << whichCell[i] << '\n';
		exit(1);
	    }
	}
    }
    for(int i=0;i<total;i++){
	if(!grid[i]){
	    cerr << "OOPS: grid[" << i << "]==0!\n";
	    exit(1);
	}
    }

    time=Time::currentSeconds();
}

