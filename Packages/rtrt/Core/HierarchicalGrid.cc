
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

using namespace rtrt;
using SCIRun::Thread;
using SCIRun::Time;

int HierarchicalGrid::L1counter = 0;
int HierarchicalGrid::L2counter = 0;
int HierarchicalGrid::L3counter = 0;

HierarchicalGrid::HierarchicalGrid( Object* obj, int nsides,
				    int nSidesLevel2, int nSidesLevel3,
				    int minObjects1, int minObjects2,
				    int level ) :
    Grid( obj, nsides ),
    _nSidesLevel2( nSidesLevel2 ),
    _nSidesLevel3( nSidesLevel3 ),
    _minObjects1( minObjects1 ),
    _minObjects2( minObjects2 ),
    _level( level )
{
    //    cerr << "Hierarchical grid: L1 = " << nsides << " L2 = " << nSidesLevel2 << " L3 = " << nSidesLevel3 << endl;
    // cerr << "Min objects L1 = " << _minObjects1 << " Min objects L2 = " << _minObjects2 << endl;
    if( level == 1 )
	HierarchicalGrid::L1counter++;
    if( level == 2 )
	HierarchicalGrid::L2counter++;
    if( level == 3 )
	HierarchicalGrid::L3counter++;

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

void 
HierarchicalGrid::preprocess( double maxradius, int& pp_offset, 
			      int& scratchsize )
{
  if (_level == 1) {
    cerr << "L1\n";
  } else if (_level == 2) {
    cerr << "  L2\n";
  } else if (_level == 3) {
    cerr << "   L3\n";
  }
//    cerr << "Building hierarchical grid\n";
    double time=Time::currentSeconds();
    int numHierL2 = 0;
    int numHierL3 = 0;
    obj->preprocess(maxradius, pp_offset, scratchsize);
//    cerr << "1/8 Preprocess took " << Time::currentSeconds()-time << " seconds\n";
    time=Time::currentSeconds();

    Array1<Object*> prims;
    obj->collect_prims(prims);
//    cerr << "2/8 Collect prims took " << Time::currentSeconds()-time << " seconds\n";
    time=Time::currentSeconds();

    bbox.reset();
    obj->compute_bounds(bbox, maxradius);
//    cerr << "3/8 Compute bounds took " << Time::currentSeconds()-time << " seconds\n";
    time=Time::currentSeconds();

    int ncells = nsides*nsides*nsides;
    int ncellsL2 = _nSidesLevel2*_nSidesLevel2*_nSidesLevel2;
    int ncellsL3 = _nSidesLevel3*_nSidesLevel3*_nSidesLevel3;

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
//    cerr << "Computing " << nx << 'x' << ny << 'x' << nz << " grid for " << ngrid << " cells (wanted " << ncells << ")\n";


    if(counts)
	delete[] counts;
    if(grid)
	delete[] grid;
    counts = new int[2*ngrid];
//    cerr << "counts=" << counts << ":" << counts+2*ngrid << '\n';
    for( int i = 0; i < ngrid*2; i++ )
	counts[i] = 0;
    
    Array1<Group*> objList( ngrid );
    objList.initialize(0);

    double itime=time;
    int nynz=ny*nz;
    for(int i=0;i<prims.size();i++){
	double tnow=Time::currentSeconds();
	if(tnow-itime > 5.0){
	    cerr << i << "/" << prims.size() << '\n';
	    itime=tnow;
	}
	BBox obj_bbox;
	prims[i]->compute_bounds(obj_bbox, maxradius);
	int sx, sy, sz, ex, ey, ez;
	calc_se(obj_bbox, bbox, diag, nx, ny, nz, sx, sy, sz, ex, ey, ez);
	for(int x=sx;x<=ex;x++){
	    for(int y=sy;y<=ey;y++){
		int idx=x*nynz+y*nz+sz;
		for(int z=sz;z<=ez;z++){
 		    if (!objList[idx]) objList[idx]=new Group;
		    objList[idx]->add( prims[i] );
		    counts[idx*2+1]++;
		    idx++;
		}
	    }
	}
    }

//    cerr << "4/8 Counting cells took " << Time::currentSeconds()-time << " seconds\n";
    time=Time::currentSeconds();
    int total=0;
    Array1<int> whichCell( ngrid );
    whichCell.initialize( 0 );

    for(int i=0;i<ngrid;i++){
	int count=counts[i*2+1];
	counts[i*2]=total;
	if( _level == 1 ) {
	    if( count >= _minObjects1 ) {
//		cout << "Level1: Cell " << i << " has more than " << _minObjects1 << " objects in it: " << count << endl;
		whichCell[i] = count;
		count = 1;
		counts[i*2+1] = 1;
		numHierL2++;
		
	    }
	} else if( _level == 2 ) {
	    if( count >= _minObjects2 ) {
//		cout << "Level2: Cell " << i << " has more than " << _minObjects2 << " objects in it: " << count << endl;
		whichCell[i] = count;
		count = 1;
		counts[i*2+1] = 1;
		numHierL3++;
	    }
	}
	
	total+=count;
	
    }
//    cerr << "Allocating " << total << " grid cells (" << double(total)/prims.size() << " per object, " << double(total)/ngrid << " per cell)\n";
    grid=new Object*[total];
//    cerr << "grid=" << grid << ":" << grid+total << '\n';
    for(int i=0;i<total;i++)
	grid[i]=0;
//    cerr << "total=" << total << '\n';
//    cerr << "5/8 Calculating offsets took " << Time::currentSeconds()-time << " seconds\n";

    //=======================================================================
    /*
    Array1<Group*> objList( ngrid );
    for(int i=0;i<prims.size();i++){
	double tnow=Time::currentSeconds();
	if(tnow-itime > 5.0){
	    cerr << i << "/" << prims.size() << '\n';
	    itime=tnow;
	}
	BBox obj_bbox;
	//
	// Array of objects that will be put in a grid at lower level
	//
	
	prims[i]->compute_bounds(obj_bbox, maxradius);
	int sx, sy, sz, ex, ey, ez;
	calc_se(obj_bbox, bbox, diag, nx, ny, nz, sx, sy, sz, ex, ey, ez);
	for(int x=sx;x<=ex;x++){
	    for(int y=sy;y<=ey;y++){
		int idx=x*nynz+y*nz+sz;
		for(int z=sz;z<=ez;z++){
		    if( whichCell[idx] > 0 ){
			cout << "This cell " << idx << " contains more elements than it should " << endl;
			objList[idx]->add( prims[i] );
		    } 
		    idx++;
		}
	    }
	}
    }
    cerr << "6/8 Grouping grid took " << Time::currentSeconds()-time << " seconds\n";
    */

    //=======================================================================

    time=Time::currentSeconds();
    itime=time;
    Array1<int> current(ngrid);
    Array1<int> whichCellPos( ngrid );
    current.initialize(0);
    whichCellPos.initialize(0);
    for(int i=0;i<prims.size();i++){
	double tnow=Time::currentSeconds();
	if(tnow-itime > 5.0){
	    cerr << i << "/" << prims.size() << '\n';
	    itime=tnow;
	}
	BBox obj_bbox;
	//
	// Array of objects that will be put in a grid at lower level
	//
	prims[i]->compute_bounds(obj_bbox, maxradius);
	int sx, sy, sz, ex, ey, ez;
	calc_se(obj_bbox, bbox, diag, nx, ny, nz, sx, sy, sz, ex, ey, ez);
	for(int x=sx;x<=ex;x++){
	    for(int y=sy;y<=ey;y++){
		int idx=x*nynz+y*nz+sz;
		for(int z=sz;z<=ez;z++){
		    int cur=current[idx];
		    int pos=counts[idx*2]+cur;
		    if( whichCell[idx] == 0 ){
			grid[pos] = prims[i];
			current[idx]++;
		    } else {
			// int pos = counts[idx*2];
			grid[pos] = ( Object* ) 0;
			whichCellPos[idx] = pos;
			current[idx] = 0;
		    }
		    idx++;
		}
	    }
	}
    }

    for( int i = 0; i < whichCell.size(); i++ ) {
        if (_level == 1 && (i%100) == 1) cerr << "Completed "<<i<<"/"<<whichCell.size()<<" ("<<i*100./whichCell.size()<<"%)\n";
	if( whichCell[i] > 0 )
	    // grid[whichCellPos[i]] = ( Object* ) 0;
	  if(  !grid[whichCellPos[i]] ) {
		grid[whichCellPos[i]] = new HierarchicalGrid( objList[i],
							      nsides,
							      _nSidesLevel2,
							      _nSidesLevel3,
							      _minObjects1,
							      _minObjects2,  
							      _level + 1 );
		grid[whichCellPos[i]]->preprocess(maxradius, pp_offset, scratchsize);
	  } else 
		cerr << "Something is wrong!!!" << endl;
	
    }
	
//    cerr << "6/7 Filling grid took " << Time::currentSeconds()-time << " seconds\n";

//    cerr << "We allocated " << HierarchicalGrid::L1counter << " grids at level 1" << endl;
//    cerr << "We allocated " << HierarchicalGrid::L2counter << " grids at level 2" << endl;
//    cerr << "We allocated " << HierarchicalGrid::L3counter << " grids at level 2" << endl;
    
//    cerr << "it should be " << numHierL2 << " for Level 2" << endl;
//    cerr << "it should be " << numHierL3 << " for Level 3" << endl;

    
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
//    cerr << "7/7 Verifying grid took " << Time::currentSeconds()-time << " seconds\n";

//    time=Time::currentSeconds();
//    for( int i = 0 ; i < ngrid; i++ ){
//	Array1<Object*> objs;
//    }

//    cerr << "REBUILDING THE GRID -- building a hierarchy took " << Time::currentSeconds()-time << " seconds\n";
//    cerr << "Done building hierarchical grid\n";

}

