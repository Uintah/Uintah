#include <Packages/rtrt/Core/HTVolumeBrick.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Color.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Thread.h>
#include <Packages/rtrt/Core/VolumeDpy.h>
#include <stdio.h>
#include <fstream>
#include <unistd.h>
#include <iostream>

using namespace rtrt;
using namespace std;
using namespace SCIRun;

namespace rtrt {
  struct VMCell {
    float max;
    float min;
  };
} // end namespace rtrt

//int ans_count;
//char ans_vec[50000000];

void HTVolumeBrick::tetra_bounds(int *nodes, int *sx, int *ex,
				 int *sy, int *ey, int *sz, int *ez)
{
  Point min(MAXFLOAT,MAXFLOAT,MAXFLOAT);
  Point max(-MAXFLOAT,-MAXFLOAT,-MAXFLOAT);
  for(int j=0; j < 4; j++) {
    float *p=points+nodes[j]*4;
    min=Min(min, Point(p[0],p[1],p[2]));
    max=Max(max, Point(p[0],p[1],p[2]));
  }
  Vector mincell((min-this->min)*idatadiag*Vector(nx,ny,nz)
		 -Vector(1.e-3,1.e-3,1.e-3));
  Vector maxcell((max-this->min)*idatadiag*Vector(nx,ny,nz)
		 +Vector(1.e-3,1.e-3,1.e-3));
  *sx=(int)mincell.x();
  *ex=(int)maxcell.x();
  *sy=(int)mincell.y();
  *ey=(int)maxcell.y();
  *sz=(int)mincell.z();
  *ez=(int)maxcell.z();
}

bool HTVolumeBrick::vertex_in_tetra(const Point&  v, const Point& p0,
				    const Point& p1, const Point& p2,
				    const Point& p3)
{

  double x1=p0.x();
  double y1=p0.y();
  double z1=p0.z();
  double x2=p1.x();
  double y2=p1.y();
  double z2=p1.z();
  double x3=p2.x();
  double y3=p2.y();
  double z3=p2.z();
  double x4=p3.x();
  double y4=p3.y();
  double z4=p3.z();
  double a1=+x2*(y3*z4-y4*z3)+x3*(y4*z2-y2*z4)+x4*(y2*z3-y3*z2);
  double a2=-x3*(y4*z1-y1*z4)-x4*(y1*z3-y3*z1)-x1*(y3*z4-y4*z3);
  double a3=+x4*(y1*z2-y2*z1)+x1*(y2*z4-y4*z2)+x2*(y4*z1-y1*z4);
  double a4=-x1*(y2*z3-y3*z2)-x2*(y3*z1-y1*z3)-x3*(y1*z2-y2*z1);
  double iV6=1./(a1+a2+a3+a4);
  if(iV6 < 0) iV6=-iV6;

  double b1=-(y3*z4-y4*z3)-(y4*z2-y2*z4)-(y2*z3-y3*z2);
  double c1=+(x3*z4-x4*z3)+(x4*z2-x2*z4)+(x2*z3-x3*z2);
  double d1=-(x3*y4-x4*y3)-(x4*y2-x2*y4)-(x2*y3-x3*y2);
  Vector g1(b1*iV6, c1*iV6, d1*iV6);
  double b2=+(y4*z1-y1*z4)+(y1*z3-y3*z1)+(y3*z4-y4*z3);
  double c2=-(x4*z1-x1*z4)-(x1*z3-x3*z1)-(x3*z4-x4*z3);
  double d2=+(x4*y1-x1*y4)+(x1*y3-x3*y1)+(x3*y4-x4*y3);
  Vector g2(b2*iV6, c2*iV6, d2*iV6);
  double b3=-(y1*z2-y2*z1)-(y2*z4-y4*z2)-(y4*z1-y1*z4);
  double c3=+(x1*z2-x2*z1)+(x2*z4-x4*z2)+(x4*z1-x1*z4);
  double d3=-(x1*y2-x2*y1)-(x2*y4-x4*y2)-(x4*y1-x1*y4);
  Vector g3(b3*iV6, c3*iV6, d3*iV6);
  double b4=+(y2*z3-y3*z2)+(y3*z1-y1*z3)+(y1*z2-y2*z1);
  double c4=-(x2*z3-x3*z2)-(x3*z1-x1*z3)-(x1*z2-x2*z1);
  double d4=+(x2*y3-x3*y2)+(x3*y1-x1*y3)+(x1*y2-x2*y1);
  Vector g4(b4*iV6, c4*iV6, d4*iV6);
  a1*=iV6;
  a2*=iV6;
  a3*=iV6;
  a4*=iV6;

  if (Dot(v, g1) + a1 >= -1e-3 &&
      Dot(v, g2) + a2 >= -1e-3 &&
      Dot(v, g3) + a3 >= -1e-3 &&
      Dot(v, g4) + a4 >= -1e-3) return true;

  return false;
}

bool HTVolumeBrick::tetra_edge_in_box(const Point&  min, const Point&  max,
				      const Point& orig, const Vector& dir)
{

  double MIN, MAX;
  double xinv_dir=1./dir.x();
  if(dir.x() > 0){
    MIN=xinv_dir*(min.x()-orig.x());
    MAX=xinv_dir*(max.x()-orig.x());
  } else {
    MIN=xinv_dir*(max.x()-orig.x());
    MAX=xinv_dir*(min.x()-orig.x());
  }	
  double y0, y1;
  double yinv_dir=1./dir.y();
  if(dir.y() > 0){
    y0=yinv_dir*(min.y()-orig.y());
    y1=yinv_dir*(max.y()-orig.y());
  } else {
    y0=yinv_dir*(max.y()-orig.y());
    y1=yinv_dir*(min.y()-orig.y());
  }
  if(y0>MIN)
    MIN=y0;
  if(y1<MAX)
    MAX=y1;
  if(MAX<MIN) {
    return false;
  }
    
  double z0, z1;
  double zinv_dir=1./dir.z();
  if(dir.z() > 0){
    z0=zinv_dir*(min.z()-orig.z());
    z1=zinv_dir*(max.z()-orig.z());
  } else {
    z0=zinv_dir*(max.z()-orig.z());
    z1=zinv_dir*(min.z()-orig.z());
  }
  if(z0>MIN)
    MIN=z0;
  if(z1<MAX)
    MAX=z1;
  if(MAX<MIN) {
    return false;
  }

  if(MIN > 1.001 || MAX < -0.001) return false;

  return true;

}

bool HTVolumeBrick::intersect_voxel_tetra(int x, int y, int z, int* nodes)
{

  Point v0(this->min+Vector(x,y,z)*datadiag/Vector(nx,ny,nz));
  Point v1(this->min+Vector(x+1,y+1,z+1)*datadiag/Vector(nx,ny,nz));
  float *p=points+nodes[0]*4;
  Point p0(p[0],p[1],p[2]);
  p=points+nodes[1]*4;
  Point p1(p[0],p[1],p[2]);
  p=points+nodes[2]*4;
  Point p2(p[0],p[1],p[2]);
  p=points+nodes[3]*4;
  Point p3(p[0],p[1],p[2]);

  if(vertex_in_tetra(Point(v0.x(),v0.y(),v0.z()), p0, p1, p2, p3))
    return true;
  if(vertex_in_tetra(Point(v0.x(),v0.y(),v1.z()), p0, p1, p2, p3))
    return true;
  if(vertex_in_tetra(Point(v0.x(),v1.y(),v0.z()), p0, p1, p2, p3))
    return true;
  if(vertex_in_tetra(Point(v0.x(),v1.y(),v1.z()), p0, p1, p2, p3))
    return true;
  if(vertex_in_tetra(Point(v1.x(),v0.y(),v0.z()), p0, p1, p2, p3))
    return true;
  if(vertex_in_tetra(Point(v1.x(),v0.y(),v1.z()), p0, p1, p2, p3))
    return true;
  if(vertex_in_tetra(Point(v1.x(),v1.y(),v0.z()), p0, p1, p2, p3))
    return true;
  if(vertex_in_tetra(Point(v1.x(),v1.y(),v1.z()), p0, p1, p2, p3))
    return true;
  if(tetra_edge_in_box(v0, v1, p0, p1-p0)) return true;
  if(tetra_edge_in_box(v0, v1, p0, p2-p0)) return true;
  if(tetra_edge_in_box(v0, v1, p0, p3-p0)) return true;
  if(tetra_edge_in_box(v0, v1, p1, p2-p1)) return true;
  if(tetra_edge_in_box(v0, v1, p1, p3-p1)) return true;
  if(tetra_edge_in_box(v0, v1, p2, p3-p2)) return true;

  return false;

}

HTVolumeBrick::HTVolumeBrick(Material* matl, VolumeDpy* dpy,
			     char* filebase, int depth, int np,
			     double density)
  : VolumeBase(matl, dpy), depth(depth), filebase(filebase), work(0)
{
  if(depth<=0)
    depth=this->depth=1;
  ifstream in(filebase);
  if(!in){
    cerr << "Error opening input file: " << filebase << '\n';
    exit(1);
  }
  char buf2[200];
  in.getline(buf2,200);
  char *filestr="HTVolumeBrick file";
  if(strncmp(buf2, filestr, strlen(filestr)-1)){
    cerr << filebase << " is not a valid HTVolumeBrick file\n";
    exit(1);
  }
  in >> npts >> ntetra;
  in.getline(buf2,200);
  if(!in){
    cerr << "Error reading file header: " << filebase << '\n';
    exit(1);
  }
  points=new float[4*npts];
  in.read((char *)points, (int)sizeof(float)*4*npts);
  tetra=new int[4*ntetra];
  in.read((char *)tetra, (int)sizeof(int)*4*ntetra);
  if(!in){
    cerr << "Error reading file data: " << filebase << '\n';
    exit(1);
  }

  Point max=min=Point(points[0],points[1],points[2]);
  datamin=datamax=points[3];
  for(int i=1; i < npts; i++) {
    float *p=points+i*4;
    min=Min(min, Point(p[0],p[1],p[2]));
    max=Max(max, Point(p[0],p[1],p[2]));
    datamin=Min(datamin,p[3]);
    datamax=Max(datamax,p[3]);
  }
  datadiag=max-min;
  min+=-datadiag*1e-3;
  max+=datadiag*1e-3;
  datadiag=max-min;
  idatadiag=Vector(1/datadiag.x(),1/datadiag.y(),1/datadiag.z());
  float volume=datadiag.x()*datadiag.y()*datadiag.z();
  float k=cbrt(volume/(ntetra*density));
  nx=(int)(datadiag.x()/k+0.5);
  ny=(int)(datadiag.y()/k+0.5);
  nz=(int)(datadiag.z()/k+0.5);
  if(nx < 1) nx=1;
  if(ny < 1) ny=1;
  if(nz < 1) nz=1;
  sdiag=datadiag/Vector(nx,ny,nz);

#define L1 3
#define L2 6
  int totalx=(nx+L2*L1-1)/(L2*L1);
  int totaly=(ny+L2*L1-1)/(L2*L1);
  int totalz=(nz+L2*L1-1)/(L2*L1);

  xidx=new int[nx];

  for(int x=0;x<nx;x++){
    int m1x=x%L1;
    int xx=x/L1;
    int m2x=xx%L2;
    int m3x=xx/L2;
    xidx[x]=m3x*totaly*totalz*L2*L2*L2*L1*L1*L1+m2x*L2*L2*L1*L1*L1+m1x*L1*L1;
  }
  yidx=new int[ny];
  for(int y=0;y<ny;y++){
    int m1y=y%L1;
    int yy=y/L1;
    int m2y=yy%L2;
    int m3y=yy/L2;
    yidx[y]=m3y*totalz*L2*L2*L2*L1*L1*L1+m2y*L2*L1*L1*L1+m1y*L1;
  }
  zidx=new int[nz];
  for(int z=0;z<nz;z++){
    int m1z=z%L1;
    int zz=z/L1;
    int m2z=zz%L2;
    int m3z=zz/L2;
    zidx[z]=m3z*L2*L2*L2*L1*L1*L1+m2z*L1*L1*L1+m1z;
  }

  int totalsize=totalx*totaly*totalz*L2*L2*L2*L1*L1*L1;
#undef L1
#undef L2
		   
  xsize=new int[depth];
  ysize=new int[depth];
  zsize=new int[depth];
  int tx=nx;
  int ty=ny;
  int tz=nz;
  for(int i=depth-1;i>=0;i--){
    int nx=(int)(pow(tx, 1./(i+1))+.9);
    tx=(tx+nx-1)/nx;
    xsize[depth-i-1]=nx;
    int ny=(int)(pow(ty, 1./(i+1))+.9);
    ty=(ty+ny-1)/ny;
    ysize[depth-i-1]=ny;
    int nz=(int)(pow(tz, 1./(i+1))+.9);
    tz=(tz+nz-1)/nz;
    zsize[depth-i-1]=nz;
  }
  ixsize=new double[depth];
  iysize=new double[depth];
  izsize=new double[depth];
  cerr << "Calculating depths...\n";
  for(int i=0;i<depth;i++){
    cerr << "xsize=" << xsize[i] << ", ysize=" << ysize[i] << ", zsize=" << zsize[i] << '\n';
    ixsize[i]=1./xsize[i];
    iysize[i]=1./ysize[i];
    izsize[i]=1./zsize[i];
  }
  cerr << "X: ";
  tx=1;
  for(int i=depth-1;i>=0;i--){
    cerr << xsize[i] << ' ';
    tx*=xsize[i];
  }
  cerr << "(" << tx << ")\n";
  if(tx<nx){
    cerr << "TX TOO SMALL!\n";
    exit(1);
  }
  cerr << "Y: ";
  ty=1;
  for(int i=depth-1;i>=0;i--){
    cerr << ysize[i] << ' ';
    ty*=ysize[i];
  }
  cerr << "(" << ty << ")\n";
  if(ty<ny){
    cerr << "TY TOO SMALL!\n";
    exit(1);
  }
  cerr << "Z: ";
  tz=1;
  for(int i=depth-1;i>=0;i--){
    cerr << zsize[i] << ' ';
    tz*=zsize[i];
  }
  cerr << "(" << tz << ")\n";
  if(tz<nz){
    cerr << "TZ TOO SMALL!\n";
    exit(1);
  }

  nx=tx;
  ny=ty;
  nz=tz;

  if(depth==1){
    macrocells=0;
  } else {
    macrocells=new VMCell*[depth+1];
    macrocells[0]=0;
    macrocell_xidx=new int*[depth+1];
    macrocell_yidx=new int*[depth+1];
    macrocell_zidx=new int*[depth+1];
    macrocell_xidx[0]=0;
    macrocell_yidx[0]=0;
    macrocell_zidx[0]=0;
#define L1 3
#define L2 4
    int xs=1;
    int ys=1;
    int zs=1;
    int total_macrocells=0;
    int total_indices=0;
    for(int d=depth-1;d>=1;d--){
      xs*=xsize[d];
      ys*=ysize[d];
      zs*=zsize[d];
      //cerr << d << ": " << xs << ", " << ys << ", " << zs << "\n";
      total_indices+=xs+ys+zs;
      int totalx=(xs+L2*L1-1)/(L2*L1);
      int totaly=(ys+L2*L1-1)/(L2*L1);
      int totalz=(zs+L2*L1-1)/(L2*L1);
      //cerr << d << ": " << totalx << ", " << totaly << ", " << totalz << "\n";
      total_macrocells+=totalx*totaly*totalz;
    }
    xs=1;
    ys=1;
    zs=1;
    total_macrocells*=L1*L1*L1*L2*L2*L2;
    VMCell* p=new VMCell[total_macrocells];
    int* indices=new int[total_indices];
    cerr << "Allocating " << total_macrocells << " macrocells and " << total_indices << " indices\n";
    for(int d=depth-1;d>=1;d--){
      xs*=xsize[d];
      ys*=ysize[d];
      zs*=zsize[d];
      int totalx=(xs+L2*L1-1)/(L2*L1);
      int totaly=(ys+L2*L1-1)/(L2*L1);
      int totalz=(zs+L2*L1-1)/(L2*L1);
      macrocells[d]=p;
      p+=totalx*totaly*totalz;
      int* xidx=macrocell_xidx[d]=indices;
      indices+=xs;
      for(int x=0;x<xs;x++){
	int m1x=x%L1;
	int xx=x/L1;
	int m2x=xx%L2;
	int m3x=xx/L2;
	xidx[x]=m3x*totaly*totalz*L2*L2*L2*L1*L1*L1+m2x*L2*L2*L1*L1*L1+m1x*L1*L1;
      }
      int* yidx=macrocell_yidx[d]=indices;
      indices+=ys;
      for(int y=0;y<ys;y++){
	int m1y=y%L1;
	int yy=y/L1;
	int m2y=yy%L2;
	int m3y=yy/L2;
	yidx[y]=m3y*totalz*L2*L2*L2*L1*L1*L1+m2y*L2*L1*L1*L1+m1y*L1;
      }
      int* zidx=macrocell_zidx[d]=indices;
      indices+=zs;
      for(int z=0;z<zs;z++){
	int m1z=z%L1;
	int zz=z/L1;
	int m2z=zz%L2;
	int m3z=zz/L2;
	zidx[z]=m3z*L2*L2*L2*L1*L1*L1+m2z*L1*L1*L1+m1z;
      }
    }
#undef L1
#undef L2

    cerr << "Totalsize = " << totalsize << '\n';

    cerr << "nx = " << nx << ", ny = " << ny << ", nz = " << nz << "\n";

    char buf[200];
    sprintf(buf, "%s.lists", filebase);
    ifstream input(buf);
    if(!input) {

      // Setup bottom level of hierarchy
      int ncells=nx*ny*nz;
      int nynz=ny*nz;
      Array1<int> counts(ncells);
      counts.initialize(0);
      cerr << "Pass 1\n";
      int checksum_1=0;
      //ofstream out1("out1");
//ans_count=0;

      for(int i=0; i < ntetra; i++) {  // foreach tetra
	if(i%10000 == 0)
	  cerr << "tetra " << i << " of " << ntetra << '\n';
	int *nodes=tetra+i*4;
	// determine bounds of tetra
#if 0
	Point min(MAXFLOAT,MAXFLOAT,MAXFLOAT);
	Point max(-MAXFLOAT,-MAXFLOAT,-MAXFLOAT);
	for(int j=0; j < 4; j++) {
	  float *p=points+nodes[j]*4;
	  min=Min(min, Point(p[0],p[1],p[2]));
	  max=Max(max, Point(p[0],p[1],p[2]));
	}
	Vector mincell((min-this->min)*idatadiag*Vector(nx,ny,nz)
		       -Vector(1.e-3,1.e-3,1.e-3));
	Vector maxcell((max-this->min)*idatadiag*Vector(nx,ny,nz)
		       +Vector(1.e-3,1.e-3,1.e-3));
	//Vector mincell((min-this->min)/datadiag*Vector(nx,ny,nz));
	//Vector maxcell((max-this->min)/datadiag*Vector(nx,ny,nz));
	int sx=(int)mincell.x();
	int ex=(int)maxcell.x();
	int sy=(int)mincell.y();
	int ey=(int)maxcell.y();
	int sz=(int)mincell.z();
	int ez=(int)maxcell.z();
#else
	int sx, ex, sy, ey, sz, ez;
	tetra_bounds(nodes, &sx, &ex, &sy, &ey, &sz, &ez);
	//out1 << i << ' ' << sx << ' ' << sy << ' ' << sz << ' '
	//   << ex << ' ' << ey << ' ' << ex << '\n';
#endif
//	if (i == 28803) {
//	cerr << "sx=" << sx << ", sy=" << sy << ", sz=" << sz
//	     << ", ex=" << ex << ", ey=" << ey << ", ez=" << ez << '\n';
//	}
	for(int x=sx; x <= ex; x++) {     // foreach voxel in the tetra's
	  for(int y=sy; y <= ey; y++) {   // bounding box, determine if the
	    for(int z=sz; z <= ez; z++) { // tetra is REALLY in the voxel
	      if(intersect_voxel_tetra(x, y, z, nodes)) {
//ans_vec[ans_count++]=1;
		int idx=x*nynz+y*nz+z;
		counts[idx]++;
		checksum_1++;
	      } //else
//ans_vec[ans_count++]=0;
	    }
	  }
	}
      }
      cerr << "Pass 2\n";

      cells=new int[ncells];

      int sum=1;

      for(int i=0; i < ncells; i++) {
	if (counts[i] == 0) {
	  cells[i]=0;
	} else {
	  cells[i]=sum;
	  sum+=counts[i]+1;
	}
      }
    
      lists = new int[sum];
      int size=sum;

      sum=1;
      for(int i=0; i < ncells; i++) {
	if (counts[i] != 0) {
	  lists[sum]=counts[i];
	  sum+=counts[i]+1;
	}
      }

      cerr << "Allocating " << sum << " grid cells (" << double(sum)/ntetra << " per object, " << double(sum)/ncells << " per cell)\n";

      counts.initialize(1);
      cerr << "Pass 3\n";

      int checksum_2=0;
      //ofstream out2("out2");
//ans_count=0;

      for(int i=0; i < ntetra; i++) {
	if(i%10000 == 0)
	  cerr << "tetra " << i << " of " << ntetra << '\n';
	int *nodes=tetra+i*4;
	// determine bounds of tetra
#if 0
	Point min(MAXFLOAT,MAXFLOAT,MAXFLOAT);
	Point max(-MAXFLOAT,-MAXFLOAT,-MAXFLOAT);
	for(int j=0; j < 4; j++) {
	  float *p=points+nodes[j]*4;
	  min=Min(min, Point(p[0],p[1],p[2]));
	  max=Max(max, Point(p[0],p[1],p[2]));
	}
	Vector mincell((min-this->min)*idatadiag*Vector(nx,ny,nz)
		       -Vector(1.e-3,1.e-3,1.e-3));
	Vector maxcell((max-this->min)*idatadiag*Vector(nx,ny,nz)
		       +Vector(1.e-3,1.e-3,1.e-3));
	//Vector mincell((min-this->min)/datadiag*Vector(nx,ny,nz));
	//Vector maxcell((max-this->min)/datadiag*Vector(nx,ny,nz));
	int sx=(int)mincell.x();
	int ex=(int)maxcell.x();
	int sy=(int)mincell.y();
	int ey=(int)maxcell.y();
	int sz=(int)mincell.z();
	int ez=(int)maxcell.z();
#else
	int sx, ex, sy, ey, sz, ez;
	tetra_bounds(nodes, &sx, &ex, &sy, &ey, &sz, &ez);
	//out2 << i << ' ' << sx << ' ' << sy << ' ' << sz << ' '
	//   << ex << ' ' << ey << ' ' << ex << '\n';
#endif
//	if (i == 28803) {
//	cerr << "sx=" << sx << ", sy=" << sy << ", sz=" << sz
//	<< ", ex=" << ex << ", ey=" << ey << ", ez=" << ez << '\n';
//	}
	for(int x=sx; x <= ex; x++) {
	  for(int y=sy; y <= ey; y++) {
	    for(int z=sz; z <= ez; z++) {
	      if(intersect_voxel_tetra(x, y, z, nodes)) {
//if(ans_vec[ans_count++] != 1) cerr << "problem (1) at " << ans_count << "\n";
		int idx=x*nynz+y*nz+z;
		int loc=cells[idx]+counts[idx ];
//		if(loc >= cells[idx+1] && cells[idx+1] != 0)
//		  cerr << "BIGGIE: loc=" << loc << ", cells[" << idx+1
//		       << "]=" << cells[idx+1] << '\n';
		lists[loc]=i;
		counts[idx]++;
		checksum_2++;
	      } //else
//if(ans_vec[ans_count++] != 0) cerr << "problem (0) at " << ans_count << ", tetra=" << i  << " " << sx << " " << sy << " " << sz << " " << ex << " " << ey << " " <<ez << "\n";
	    }
	  }
	}
      }
      if(checksum_1 != checksum_2){
	cerr << "MISMATCH: " << checksum_1 << ' ' << checksum_2 << '\n';
      }

      cerr << "Writing .lists file\n";
      ofstream output(buf);
      output << ncells << '\n' << size << '\n';
      output.write((char *)cells, (int)sizeof(int)*ncells);
      output.write((char *)lists, (int)sizeof(int)*size);

    } else {

      int size, ncells;
      input >> ncells >> size;
      input.getline(buf2, 200);

      cells = new int[ncells];
      lists = new int[size];
      input.read((char *)cells, (int)sizeof(int)*ncells);
      input.read((char *)lists, (int)sizeof(int)*size);

      if(!input) {
	cerr << "ERROR reading lists file" << buf << '\n';
      }

    }

    cerr << "Building hierarchy\n";
#if 0
    VMCell top;
    calc_mcell(depth-1, 0, 0, 0, top);
    cerr << "Min: " << top.min << ", Max: " << top.max << '\n';
#else
    minmax.resize(nx,ny,nz);
    int nx=xsize[depth-1];
    int ny=ysize[depth-1];
    int nz=zsize[depth-1];
    int totaltop=nx*ny*nz;

    // <<<<< bigler >>>>>
    //work=WorkQueue("Building hierarchy", totaltop, np, false, 5);
    work.refill(totaltop, np, 5);
    Parallel<HTVolumeBrick> phelper(this, &HTVolumeBrick::parallel_calc_mcell);
    Thread::parallel(phelper, np, true);
#endif
    cerr << "done\n";
  }
}

HTVolumeBrick::~HTVolumeBrick()
{
}

void HTVolumeBrick::preprocess(double, int&, int&)
{
}

void HTVolumeBrick::calc_mcell(int depth, int startx, int starty, int startz,
			       VMCell& mcell)
{
  mcell.min=MAXFLOAT;
  mcell.max=-MAXFLOAT;
  int endx=startx+xsize[depth];
  int endy=starty+ysize[depth];
  int endz=startz+zsize[depth];
  if(depth==0){
    if(endx>nx)
      endx=nx;
    if(endy>ny)
      endy=ny;
    if(endz>nz)
      endz=nz;
    if(startx>=endx || starty>=endy || startz>=endz){
      /* This cell won't get used... */
      mcell.min=datamax;
      mcell.max=datamin;
      return;
    }
    int nynz=ny*nz;
    for(int ix=startx;ix<endx;ix++){
      for(int iy=starty;iy<endy;iy++){
	for(int iz=startz;iz<endz;iz++){
	  int idx=ix*nynz+iy*nz+iz;
	  int list=cells[idx];
	  int n=lists[list++];
	  float min=datamax;
	  float max=datamin;
	  
	  for (int i=0; i < n; i ++) {
	    int *nodes=tetra+4*lists[list+i];
	    for(int j=0; j < 4; j++) {
	      if(nodes[j] > npts)
		cerr << "nodes[" << j << "]=" << nodes[j] << "!!!!\n";
	      float *p=points+nodes[j]*4;
	      min=Min(min,p[3]);
	      max=Max(max,p[3]);
	    }
	  }
	  if(ix < nx && iy < ny && iz < nz){
	    minmax(ix,iy,iz).min=min;
	    minmax(ix,iy,iz).max=max;
	  }
	  mcell.min=Min(mcell.min, min);
	  mcell.max=Max(mcell.max, max);
	}
      }
    }
  } else {
    int nx=xsize[depth-1];
    int ny=ysize[depth-1];
    int nz=zsize[depth-1];
    VMCell* mcells=macrocells[depth];
    int* mxidx=macrocell_xidx[depth];
    int* myidx=macrocell_yidx[depth];
    int* mzidx=macrocell_zidx[depth];
    for(int x=startx;x<endx;x++){
      for(int y=starty;y<endy;y++){
	for(int z=startz;z<endz;z++){
	  VMCell tmp;
	  calc_mcell(depth-1, x*nx, y*ny, z*nz, tmp);
	  if(tmp.min < mcell.min)
	    mcell.min=tmp.min;
	  if(tmp.max > mcell.max)
	    mcell.max=tmp.max;
	  int idx=mxidx[x]+myidx[y]+mzidx[z];
	  mcells[idx]=tmp;
	}
      }
    }
  }
}

void HTVolumeBrick::parallel_calc_mcell(int)
{
  int ny=ysize[depth-1];
  int nz=zsize[depth-1];
  int nnx=xsize[depth-2];
  int nny=ysize[depth-2];
  int nnz=zsize[depth-2];
  VMCell* mcells=macrocells[depth-1];
  int* mxidx=macrocell_xidx[depth-1];
  int* myidx=macrocell_yidx[depth-1];
  int* mzidx=macrocell_zidx[depth-1];
  int s, e;
  while(work.nextAssignment(s, e)){
    for(int block=s;block<e;block++){
      int z=block%nz;
      int y=(block%(nz*ny))/nz;
      int x=(block/(ny*nz));
      VMCell tmp;
      calc_mcell(depth-2, x*nnx, y*nny, z*nnz, tmp);
      int idx=mxidx[x]+myidx[y]+mzidx[z];
      mcells[idx]=tmp;
    }
  }
}

void HTVolumeBrick::compute_bounds(BBox& bbox, double offset)
{
  bbox.extend(min-Vector(offset,offset,offset));
  bbox.extend(min+datadiag+Vector(offset,offset,offset));
}

namespace rtrt { 
extern int HitCell(const Ray& r, const Point& pmin, const Point& pmax, 
		   float rho[2][2][2], float iso, double tmin, double tmax, double& t);
extern Vector GradientCell(const Point& pmin, const Point& pmax,
			   const Point& p, float rho[2][2][2]);
} // end namespace rtrt

void HTVolumeBrick::isect(int depth, float isoval, double t,
			  double dtdx, double dtdy, double dtdz,
			  double next_x, double next_y, double next_z,
			  int ix, int iy, int iz,
			  int dix_dx, int diy_dy, int diz_dz,
			  int startx, int starty, int startz,
			  const Vector& cellcorner, const Vector& celldir,
			  const Ray& ray, HitInfo& hit,
			  DepthStats* st, PerProcessorContext* ppc)
{
  //cerr << "start depth: " << depth << "\n";
  int cx=xsize[depth];
  int cy=ysize[depth];
  int cz=zsize[depth];
  if(depth==0){
    Point o=ray.origin();
    Vector v=ray.direction();
    int nynz=ny*nz;
    for(;;){
      int gx=startx+ix;
      int gy=starty+iy;
      int gz=startz+iz;
      if(gx<nx && gy<ny && gz<nz){
	VMCell& botcell=minmax(gx,gy,gz);
	if(isoval > botcell.min && isoval < botcell.max) { 
	  int idx=gx*nynz+gy*nz+gz;
	  int list=cells[idx];
	  int n=lists[list++];
	  for (int i=0; i < n; i ++) {
	    int *nodes=tetra+4*lists[list+i];
	    // does the isovalue lie in this tetrahedron
	    float min=MAXFLOAT, max=-MAXFLOAT;
	    for(int j=0; j < 4; j++) {
	      float *p=points+nodes[j]*4;
	      min=Min(min, p[3]);
	      max=Max(max, p[3]);
	    }
	    if ( min < isoval && max > isoval) {
	      // does the ray intersect the tetrahedron
	      float *p1=points+nodes[0]*4;
	      float *p2=points+nodes[1]*4;
	      float *p3=points+nodes[2]*4;
	      float *p4=points+nodes[3]*4;
	      double x1=p1[0];
	      double y1=p1[1];
	      double z1=p1[2];
	      double x2=p2[0];
	      double y2=p2[1];
	      double z2=p2[2];
	      double x3=p3[0];
	      double y3=p3[1];
	      double z3=p3[2];
	      double x4=p4[0];
	      double y4=p4[1];
	      double z4=p4[2];
	      double a1=+x2*(y3*z4-y4*z3)+x3*(y4*z2-y2*z4)+x4*(y2*z3-y3*z2);
	      double a2=-x3*(y4*z1-y1*z4)-x4*(y1*z3-y3*z1)-x1*(y3*z4-y4*z3);
	      double a3=+x4*(y1*z2-y2*z1)+x1*(y2*z4-y4*z2)+x2*(y4*z1-y1*z4);
	      double a4=-x1*(y2*z3-y3*z2)-x2*(y3*z1-y1*z3)-x3*(y1*z2-y2*z1);
	      double iV6=1./(a1+a2+a3+a4);
	      if(iV6 < 0) iV6=-iV6;
	      
	      double b1=-(y3*z4-y4*z3)-(y4*z2-y2*z4)-(y2*z3-y3*z2);
	      double c1=+(x3*z4-x4*z3)+(x4*z2-x2*z4)+(x2*z3-x3*z2);
	      double d1=-(x3*y4-x4*y3)-(x4*y2-x2*y4)-(x2*y3-x3*y2);
	      Vector g1(b1*iV6, c1*iV6, d1*iV6);
	      double b2=+(y4*z1-y1*z4)+(y1*z3-y3*z1)+(y3*z4-y4*z3);
	      double c2=-(x4*z1-x1*z4)-(x1*z3-x3*z1)-(x3*z4-x4*z3);
	      double d2=+(x4*y1-x1*y4)+(x1*y3-x3*y1)+(x3*y4-x4*y3);
	      Vector g2(b2*iV6, c2*iV6, d2*iV6);
	      double b3=-(y1*z2-y2*z1)-(y2*z4-y4*z2)-(y4*z1-y1*z4);
	      double c3=+(x1*z2-x2*z1)+(x2*z4-x4*z2)+(x4*z1-x1*z4);
	      double d3=-(x1*y2-x2*y1)-(x2*y4-x4*y2)-(x4*y1-x1*y4);
	      Vector g3(b3*iV6, c3*iV6, d3*iV6);
	      double b4=+(y2*z3-y3*z2)+(y3*z1-y1*z3)+(y1*z2-y2*z1);
	      double c4=-(x2*z3-x3*z2)-(x3*z1-x1*z3)-(x1*z2-x2*z1);
	      double d4=+(x2*y3-x3*y2)+(x3*y1-x1*y3)+(x1*y2-x2*y1);
	      Vector g4(b4*iV6, c4*iV6, d4*iV6);
	      a1*=iV6;
	      a2*=iV6;
	      a3*=iV6;
	      a4*=iV6;
	      
	      double A=isoval - (Dot(o, g1) + a1) * p1[3]
		- (Dot(o, g2) + a2) * p2[3]
		- (Dot(o, g3) + a3) * p3[3]
		- (Dot(o, g4) + a4) * p4[3];
	      double B=Dot(v, g1) * p1[3] +
		Dot(v, g2) * p2[3] +
		Dot(v, g3) * p3[3] +
		Dot(v, g4) * p4[3];
	      
	      if( B < -1.e-6 || B > 1.e-6) {
		
		double t=A/B;
		
		Point hitpoint=o+v*t;
		
		if (Dot(hitpoint, g1) + a1 >= -1e-6 &&
		    Dot(hitpoint, g2) + a2 >= -1e-6 &&
		    Dot(hitpoint, g3) + a3 >= -1e-6 &&
		    Dot(hitpoint, g4) + a4 >= -1e-6) {
		  
		  if(hit.hit(this, (double) t)){
		    Vector* n=(Vector*)hit.scratchpad;
		    //*n=GradientCell(p0, p1, ray.origin()+ray.direction()*hit_t, rho);
		    *n=g1*p1[3]+g2*p2[3]+g3*p3[3]+g4*p4[3];
		    n->normalize();
		    // return;
		  }
		}
	      }
	    }
	  }
	}
      }
      if(next_x < next_y && next_x < next_z){
	// Step in x...
	t=next_x;
	next_x+=dtdx;
	ix+=dix_dx;
	if(ix<0 || ix>=cx || ix+startx>=nx)
	  break;
      } else if(next_y < next_z){
	t=next_y;
	next_y+=dtdy;
	iy+=diy_dy;
	if(iy<0 || iy>=cy || iy+starty>=ny)
	  break;
      } else {
	t=next_z;
	next_z+=dtdz;
	iz+=diz_dz;
	if(iz<0 || iz>=cz || iz+startz>=nz)
	  break;
      }
      if (hit.min_t < t) {
	break;
      }
    }
  } else {
    VMCell* mcells=macrocells[depth];
    int* mxidx=macrocell_xidx[depth];
    int* myidx=macrocell_yidx[depth];
    int* mzidx=macrocell_zidx[depth];
    for(;;){
      int gx=startx+ix;
      int gy=starty+iy;
      int gz=startz+iz;
      int idx=mxidx[gx]+myidx[gy]+mzidx[gz];
      VMCell& mcell=mcells[idx];
      //cerr << "doing macrocell: " << gx << ", " << gy << ", " << gz << ": " << mcell.min << ", " << mcell.max << '\n';
      if(mcell.max>isoval && mcell.min<isoval){
	// Do this cell...
	int new_cx=xsize[depth-1];
	int new_cy=ysize[depth-1];
	int new_cz=zsize[depth-1];
	int new_ix=(int)((cellcorner.x()+t*celldir.x()-ix)*new_cx);
	int new_iy=(int)((cellcorner.y()+t*celldir.y()-iy)*new_cy);
	int new_iz=(int)((cellcorner.z()+t*celldir.z()-iz)*new_cz);
	//cerr << "new: " << (cellcorner.x()+t*celldir.x()-ix)*new_cx
	//<< " " << (cellcorner.y()+t*celldir.y()-iy)*new_cy
	//<< " " << (cellcorner.z()+t*celldir.z()-iz)*new_cz
	//<< '\n';
	if(new_ix<0)
	  new_ix=0;
	else if(new_ix>=new_cx)
	  new_ix=new_cx-1;
	if(new_iy<0)
	  new_iy=0;
	else if(new_iy>=new_cy)
	  new_iy=new_cy-1;
	if(new_iz<0)
	  new_iz=0;
	else if(new_iz>=new_cz)
	  new_iz=new_cz-1;

	double new_dtdx=dtdx*ixsize[depth-1];
	double new_dtdy=dtdy*iysize[depth-1];
	double new_dtdz=dtdz*izsize[depth-1];
	const Vector dir(ray.direction());
	double new_next_x;
	if(dir.x() > 0){
	  new_next_x=next_x-dtdx+new_dtdx*(new_ix+1);
	} else {
	  new_next_x=next_x-new_ix*new_dtdx;
	}
	double new_next_y;
	if(dir.y() > 0){
	  new_next_y=next_y-dtdy+new_dtdy*(new_iy+1);
	} else {
	  new_next_y=next_y-new_iy*new_dtdy;
	}
	double new_next_z;
	if(dir.z() > 0){
	  new_next_z=next_z-dtdz+new_dtdz*(new_iz+1);
	} else {
	  new_next_z=next_z-new_iz*new_dtdz;
	}
	int new_startx=gx*new_cx;
	int new_starty=gy*new_cy;
	int new_startz=gz*new_cz;
	//cerr << "startz=" << startz << '\n';
	//cerr << "iz=" << iz << '\n';
	//cerr << "new_cz=" << new_cz << '\n';
	Vector cellsize(new_cx, new_cy, new_cz);
	isect(depth-1, isoval, t,
	      new_dtdx, new_dtdy, new_dtdz,
	      new_next_x, new_next_y, new_next_z,
	      new_ix, new_iy, new_iz,
	      dix_dx, diy_dy, diz_dz,
	      new_startx, new_starty, new_startz,
	      (cellcorner-Vector(ix, iy, iz))*cellsize, celldir*cellsize,
	      ray, hit, st, ppc);
      }
      if(next_x < next_y && next_x < next_z){
	// Step in x...
	t=next_x;
	next_x+=dtdx;
	ix+=dix_dx;
	if(ix<0 || ix>=cx)
	  break;
      } else if(next_y < next_z){
	t=next_y;
	next_y+=dtdy;
	iy+=diy_dy;
	if(iy<0 || iy>=cy)
	  break;
      } else {
	t=next_z;
	next_z+=dtdz;
	iz+=diz_dz;
	if(iz<0 || iz>=cz)
	  break;
      }
      if(hit.min_t < t)
	break;
    }
  }
  //cerr << "end depth: " << depth << "\n";
}

void HTVolumeBrick::intersect(Ray& ray, HitInfo& hit,
			      DepthStats* st, PerProcessorContext* ppc)
{
  const Vector dir(ray.direction());
  const Point orig(ray.origin());
  Point max(min+datadiag);
  double MIN, MAX;
  double xinv_dir=1./dir.x();
  int dix_dx;
  int ddx;
  if(dir.x() > 0){
    MIN=xinv_dir*(min.x()-orig.x());
    MAX=xinv_dir*(max.x()-orig.x());
    dix_dx=1;
    ddx=1;
  } else {
    MIN=xinv_dir*(max.x()-orig.x());
    MAX=xinv_dir*(min.x()-orig.x());
    dix_dx=-1;
    ddx=0;
  }	
  double y0, y1;
  int diy_dy;
  int ddy;
  double yinv_dir=1./dir.y();
  if(dir.y() > 0){
    y0=yinv_dir*(min.y()-orig.y());
    y1=yinv_dir*(max.y()-orig.y());
    diy_dy=1;
    ddy=1;
  } else {
    y0=yinv_dir*(max.y()-orig.y());
    y1=yinv_dir*(min.y()-orig.y());
    diy_dy=-1;
    ddy=0;
  }
  if(y0>MIN)
    MIN=y0;
  if(y1<MAX)
    MAX=y1;
  if(MAX<MIN) {
    return;
  }
    
  double z0, z1;
  int diz_dz;
  int ddz;
  double zinv_dir=1./dir.z();
  if(dir.z() > 0){
    z0=zinv_dir*(min.z()-orig.z());
    z1=zinv_dir*(max.z()-orig.z());
    diz_dz=1;
    ddz=1;
  } else {
    z0=zinv_dir*(max.z()-orig.z());
    z1=zinv_dir*(min.z()-orig.z());
    diz_dz=-1;
    ddz=0;
  }
  if(z0>MIN)
    MIN=z0;
  if(z1<MAX)
    MAX=z1;
  if(MAX<MIN) {
    return;
  }
  double t;
  if(MIN > 1.e-6){
    t=MIN;
  } else if(MAX > 1.e-6){
    t=0;
  } else {
    return;
  }
  if(t>1.e29) {
    return;
  }
  Point p(orig+dir*t);
  Vector s((p-min)*idatadiag);
  int cx=xsize[depth-1];
  int cy=ysize[depth-1];
  int cz=zsize[depth-1];
  int ix=(int)(s.x()*cx);
  int iy=(int)(s.y()*cy);
  int iz=(int)(s.z()*cz);
  if(ix>=cx)
    ix--;
  if(iy>=cy)
    iy--;
  if(iz>=cz)
    iz--;
  if(ix<0)
    ix++;
  if(iy<0)
    iy++;
  if(iz<0)
    iz++;

  double next_x, next_y, next_z;
  double dtdx, dtdy, dtdz;
  double icx=ixsize[depth-1];
  double x=min.x()+datadiag.x()*double(ix+ddx)*icx;
  next_x=(x-orig.x())*xinv_dir;
  dtdx=dix_dx*datadiag.x()*icx*xinv_dir;
  double icy=iysize[depth-1];
  double y=min.y()+datadiag.y()*double(iy+ddy)*icy;
  next_y=(y-orig.y())*yinv_dir;
  dtdy=diy_dy*datadiag.y()*icy*yinv_dir;
  double icz=izsize[depth-1];
  double z=min.z()+datadiag.z()*double(iz+ddz)*icz;
  next_z=(z-orig.z())*zinv_dir;
  dtdz=diz_dz*datadiag.z()*icz*zinv_dir;

  Vector cellsize(cx,cy,cz);
  Vector cellcorner((orig-min)*idatadiag*cellsize);
  Vector celldir(dir*idatadiag*cellsize);
  float isoval=dpy->isoval;

  //cerr << "start ray: " << orig << " " << dir << '\n';
  isect(depth-1, isoval, t, dtdx, dtdy, dtdz, next_x, next_y, next_z,
	ix, iy, iz, dix_dx, diy_dy, diz_dz,
	0, 0, 0,
	cellcorner, celldir,
	ray, hit, st, ppc);
  //cerr << "done\n\n";
}

Vector HTVolumeBrick::normal(const Point&, const HitInfo& hit)
{
  // We computed the normal at intersect time and tucked it
  // away in the scratchpad...
  Vector* n=(Vector*)hit.scratchpad;
  return *n;
}

void HTVolumeBrick::compute_hist(int nhist, int* hist,
				 float datamin, float datamax)
{
#if 0
  char buf[200];
  sprintf(buf, "%s.hist_%d", filebase, nhist);
  ifstream in(buf);
  if(in){
    for(int i=0;i<nhist;i++){
      in >> hist[i];
    }
  } else {
#endif
    float scale=(nhist-1)/(datamax-datamin);
    for(int i=0; i < ntetra; i++) {
      int *nodes=tetra+i*4;
      float min=MAXFLOAT;
      float max=-min;
      for(int j=0; j < 4; j++) {
	min=Min(min,points[nodes[j]*4+3]);
	max=Max(max,points[nodes[j]*4+3]);
      }
      int nmin=(int)((min-datamin)*scale);
      int nmax=(int)((max-datamin)*scale+.999999);
      if(nmax>=nhist)
	nmax=nhist-1;
      if(nmin<0)
	nmin=0;
      if(nmax>=nhist)
	nmax=nhist;
      for(int i=nmin;i<nmax;i++){
	hist[i]++;
      }
    }
#if 0
    ofstream out(buf);
    for(int i=0;i<nhist;i++){
      out << hist[i] << '\n';
    }
  }
#endif
  cerr << "Done building histogram\n";
}

void HTVolumeBrick::get_minmax(float& min, float& max)
{
  min=datamin;
  max=datamax;
}
