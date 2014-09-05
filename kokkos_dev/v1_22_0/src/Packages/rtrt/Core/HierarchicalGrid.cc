#include <Packages/rtrt/Core/HierarchicalGrid.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Names.h>
#include <Core/Thread/Parallel.h>
#include <Core/Geometry/Vector.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/WorkQueue.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <iostream>
#include <stdlib.h>
#include <Packages/rtrt/Core/Tri.h>
#include <Packages/rtrt/Core/TexturedTri.h>

extern "C" {
#include <Packages/rtrt/Core/pcube.h>
}

using namespace rtrt;
using namespace SCIRun;

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
				    int minObjects1, int minObjects2, int np ):
    Grid( obj, nsides ),
    _nSidesLevel2( nSidesLevel2 ),
    _nSidesLevel3( nSidesLevel3 ),
    _minObjects1( minObjects1 ),
    _minObjects2( minObjects2 ),
    _level( 1 ),
    np(np)
{
  work=0;
}

HierarchicalGrid::HierarchicalGrid( Object* obj, int nsides,
				    int nSidesLevel2, int nSidesLevel3,
				    int minObjects1, int minObjects2, 
				    const BBox &b, int level ):
    Grid( obj, nsides ),
    _nSidesLevel2( nSidesLevel2 ),
    _nSidesLevel3( nSidesLevel3 ),
    _minObjects1( minObjects1 ),
    _minObjects2( minObjects2 ),
    _level( level )
{
    bbox=b;
}

HierarchicalGrid::~HierarchicalGrid( void )
{
    // There's probably a memory leak somewhere
}

void HierarchicalGrid::calc_clipped_se(const BBox& obj_bbox, const BBox& bbox,
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
HierarchicalGrid::preprocess( double maxradius, int& pp_offset, 
			      int& scratchsize )
{
    if (was_preprocessed) return;
    was_preprocessed=true;

    if (Names::hasName(this)) std::cerr << "\n\n"
                              << "\n==========================================================\n"
					<< "* Building Hierarchical Grid for Object " << Names::getName(this)
                              << "\n==========================================================\n";

    obj->preprocess(maxradius, pp_offset, scratchsize);

    Array1<Object*> prims;
    obj->collect_prims(prims);
    Array1<BBox> primsBBox( prims.size() );
    Array1<GridData> primsGridData( prims.size() );

    if (_level == 1) {
      bbox.reset();
      obj->compute_bounds( bbox, maxradius );
    }

    int ncells = nsides*nsides*nsides;
    if (_level == 2) {
      ncells = _nSidesLevel2*_nSidesLevel2*_nSidesLevel2;
    } else if (_level == 3) {
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

    if (_level == 1) {
      L1Cells += ngrid;
    } else if (_level == 2) {
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

    double itime=Time::currentSeconds();
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

	if (_level == 1) {
	  calc_se( primsBBox[i], bbox, diag, nx, ny, nz, 
		   primsGridData[i].sx, primsGridData[i].sy, primsGridData[i].sz,
		   primsGridData[i].ex, primsGridData[i].ey, primsGridData[i].ez );
	} else {
	  calc_clipped_se( primsBBox[i], bbox, diag, nx, ny, nz, 
			   primsGridData[i].sx, primsGridData[i].sy, primsGridData[i].sz,
			   primsGridData[i].ex, primsGridData[i].ey, primsGridData[i].ez );
	}

	Tri *tri = dynamic_cast<Tri*>(prims[i]);

	if (tri) {
	    if (tri->isbad())
		continue;
	    p0 = (tri->pt(0) - bbox.min())*(Vector(nx,ny,nz)/diag);
	    p1 = (tri->pt(1) - bbox.min())*(Vector(nx,ny,nz)/diag);
	    p2 = (tri->pt(2) - bbox.min())*(Vector(nx,ny,nz)/diag);
	    Vector n = Cross((p2-p0),(p1-p0));
	    n.normalize();
	    polynormal[0] = n.x();
	    polynormal[1] = n.y();
	    polynormal[2] = n.z();
	}
        TexturedTri *ttri = dynamic_cast<TexturedTri*>(prims[i]);

        if (ttri) {
            if (ttri->isbad())
                continue;
            p0 = (ttri->pt(0) - bbox.min())*(Vector(nx,ny,nz)/diag);
            p1 = (ttri->pt(1) - bbox.min())*(Vector(nx,ny,nz)/diag);
            p2 = (ttri->pt(2) - bbox.min())*(Vector(nx,ny,nz)/diag);
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
		  if (tri || ttri) {	
		    verts[0][0] = p0.x() - ((double)x+.5);
		    verts[1][0] = p1.x() - ((double)x+.5);
		    verts[2][0] = p2.x() - ((double)x+.5);
		    verts[0][1] = p0.y() - ((double)y+.5);
		    verts[1][1] = p1.y() - ((double)y+.5);
		    verts[2][1] = p2.y() - ((double)y+.5);
		    verts[0][2] = p0.z() - ((double)z+.5);
		    verts[1][2] = p1.z() - ((double)z+.5);
		    verts[2][2] = p2.z() - ((double)z+.5);
		    if (fast_polygon_intersects_cube(3, verts, polynormal, 0, 0))
		    {
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
    int total=0;
    Array1<int> whichCell( ngrid );
    whichCell.initialize( 0 );

    for(int i=0;i<ngrid;i++){
	int count=counts[i*2+1];
	counts[i*2]=total;
	if(( _level == 1 && count >= _minObjects1 ) || 
	   ( _level == 2 && count >= _minObjects2 )) {
	  whichCell[i] = count;
	  count = 1;
	  counts[i*2+1] = 1;
	}
	total+=count;
    }

    grid=new Object*[total];

    total_ = total;
    if( total == 0 )
      {
	cout << "HGrid: error, total is 0 for " << this << "\n";
      }

    for(int i=0;i<total;i++)
	grid[i]=0;

    Array1<int> current(ngrid);
    Array1<int> whichCellPos( ngrid );
    current.initialize(0);
    whichCellPos.initialize(-1);

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


    if (_level == 1) {
      work = new WorkQueue("HGrid");
      work->refill(ngrid, 1, 5);
      
//      work->refill(ngrid, np, 5);
      pdata.dx = dx;
      pdata.dy = dy;
      pdata.dz = dz;
      pdata.maxradius = maxradius;
      pdata.pp_offset = pp_offset;
      pdata.scratchsize = scratchsize;
      pdata.whichCell = whichCell;
      pdata.whichCellPos = whichCellPos;
      pdata.objList = objList;
      Parallel<HierarchicalGrid> phelper(this, &HierarchicalGrid::gridit);
      Thread::parallel(phelper, 1, true);
//      Thread::parallel(phelper, np, true);
      delete work;
    } else {
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
							  _minObjects2, 
							  b, _level+1 );
	      g->preprocess( maxradius, pp_offset, scratchsize);
	      grid[whichCellPos[i]] = g;
	      if (_level == 2) {
		L2CellsWithChildren++;
	      }
	    } else {
	      LeafCells++;
	      TotalLeafPrims+=counts[i*2+1];
	    }
	  }
	}
      }
    }

    //
    // Deallocate group of objects
    //
    for( int i = 0; i < ngrid; i++ )
	delete objList[i];

    for(int i=0;i<ngrid;i++){
	if( ( current[i] != counts[i*2+1] ) ) {
	    if( ( whichCell[i] > 0 ) && ( counts[i*2+1] != 1 ) ) {
		cerr << "OOPS! in HGrid\n";
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
}

void HierarchicalGrid::gridit(int proc)
{
  int nynz=ny*nz;
  int sx, ex;
  while(work->nextAssignment(sx, ex)){
    for(int i=sx;i<ex;i++){
      int ii, jj, kk;
      ii=i/nynz;
      jj=(i-ii*nynz)/nz;
      kk=i-ii*nynz-jj*nz;
      if (proc==0 && (i%10 == 5)) {
	cerr << "HGrid parallel preprocess: ("<<i<<" of "<<sx<<"-"<<ex<<") of "<<nx*ny*nz <<"\n";
	if(L1Cells)
	  cerr <<"   "<<L1CellsWithChildren*100./L1Cells<<"% of L1 cells have children\n";
	else
	  cerr <<"   no  L1 children\n";
	if(L2Cells)
	  cerr <<"   "<<L2CellsWithChildren*100./L2Cells<<"% of L2 cells have children\n";
	else
	  cerr <<"   no L2 children\n";
	if(LeafCells)
	  cerr <<"   "<<TotalLeafPrims*1./LeafCells<<" objects / leaf.\n";
	else
	  cerr <<"   no leafs\n";
	
      }
      if( pdata.whichCell[i] > 0 && !grid[pdata.whichCellPos[i]] ) {
	BBox b(Point(bbox.min()+Vector(ii*pdata.dx, jj*pdata.dy, kk*pdata.dz)),
	       Point(bbox.min()+Vector((ii+1)*pdata.dx,(jj+1)*pdata.dy,
				       (kk+1)*pdata.dz)));
	HierarchicalGrid *g = new HierarchicalGrid( pdata.objList[i],
						    nsides,
						    _nSidesLevel2,
						    _nSidesLevel3,
						    _minObjects1,
						    _minObjects2, 
						    b, 2 );
	g->preprocess( pdata.maxradius, pdata.pp_offset, pdata.scratchsize);
	grid[pdata.whichCellPos[i]] = g;
	L1CellsWithChildren++;
      } else {
	LeafCells++;
	TotalLeafPrims+=counts[i*2+1];
      }
    }
  }
}
