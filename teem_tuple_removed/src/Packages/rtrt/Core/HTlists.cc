#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/Array1.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <unistd.h>
#include <iostream>

using namespace std;

struct VMCell {
  float max;
  float min;
};

Point global_min;
Vector global_datadiag;
Vector global_idatadiag;
int global_nx, global_ny, global_nz;
float *global_points;

void tetra_bounds(int *nodes, int *sx, int *ex, int *sy, int *ey,
                  int *sz, int *ez)
{
  Point min(MAXFLOAT,MAXFLOAT,MAXFLOAT);
  Point max(-MAXFLOAT,-MAXFLOAT,-MAXFLOAT);
  for(int j=0; j < 4; j++) {
    float *p=global_points+nodes[j]*4;
    min=Min(min, Point(p[0],p[1],p[2]));
    max=Max(max, Point(p[0],p[1],p[2]));
  }
  Vector mincell((min-global_min)*global_idatadiag
                 *Vector(global_nx,global_ny,global_nz)
		 -Vector(1.e-3,1.e-3,1.e-3));
  Vector maxcell((max-global_min)*global_idatadiag
                 *Vector(global_nx,global_ny,global_nz)
		 +Vector(1.e-3,1.e-3,1.e-3));
  *sx=(int)mincell.x();
  *ex=(int)maxcell.x();
  *sy=(int)mincell.y();
  *ey=(int)maxcell.y();
  *sz=(int)mincell.z();
  *ez=(int)maxcell.z();
}

bool vertex_in_tetra(const Point&  v, const Point& p0, const Point& p1,
                     const Point& p2, const Point& p3)
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

bool tetra_edge_in_box(const Point&  min, const Point&  max,
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

bool intersect_voxel_tetra(int x, int y, int z, int* nodes)
{

  Point v0(global_min+Vector(x,y,z)*global_datadiag
           /Vector(global_nx,global_ny,global_nz));
  Point v1(global_min+Vector(x+1,y+1,z+1)*global_datadiag
           /Vector(global_nx,global_ny,global_nz));
  float *p=global_points+nodes[0]*4;
  Point p0(p[0],p[1],p[2]);
  p=global_points+nodes[1]*4;
  Point p1(p[0],p[1],p[2]);
  p=global_points+nodes[2]*4;
  Point p2(p[0],p[1],p[2]);
  p=global_points+nodes[3]*4;
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

int main (int argc, char *argv[])
{

  global_nx = atoi(argv[2]);
  global_ny = atoi(argv[3]);
  global_nz = atoi(argv[4]);

  ifstream in(argv[1]);
  if(!in){
    cerr << "Error opening input file: " << argv[1] << '\n';
    exit(1);
  }
  char buf2[200];
  in.getline(buf2,200);
  char *filestr="HTVolumeBrick file";
  if(strncmp(buf2, filestr, strlen(filestr)-1)){
    cerr << argv[1] << " is not a valid HTVolumeBrick file\n";
    exit(1);
  }
  int npts, ntetra;
  in >> npts >> ntetra;
  in.getline(buf2,200);
  if(!in){
    cerr << "Error reading file header: " << argv[1] << '\n';
    exit(1);
  }
  global_points=new float[4*npts];
  in.read((char *)global_points, (int)sizeof(float)*4*npts);
  int *tetra=new int[4*ntetra];
  in.read((char *)tetra, (int)sizeof(int)*4*ntetra);
  if(!in){
    cerr << "Error reading file data: " << argv[1] << '\n';
    exit(1);
  }

  Point max=global_min=Point(global_points[0],global_points[1],
                             global_points[2]);
  for(int i=1; i < npts; i++) {
    float *p=global_points+i*4;
    global_min=Min(global_min, Point(p[0],p[1],p[2]));
    max=Max(max, Point(p[0],p[1],p[2]));
  }
  global_datadiag=max-global_min;
  global_min+=-global_datadiag*1e-3;
  max+=global_datadiag*1e-3;
  global_datadiag=max-global_min;
  global_idatadiag=Vector(1/global_datadiag.x(),1/global_datadiag.y(),
                          1/global_datadiag.z());

  char buf[200];
  sprintf(buf, "%s.lists", argv[1]);

    // Setup bottom level of hierarchy
    int ncells=global_nx*global_ny*global_nz;
    int nynz=global_ny*global_nz;
    Array1<int> counts(ncells);
    counts.initialize(0);
    cerr << "Pass 1\n";
    int checksum_1=0;

    for(int i=0; i < ntetra; i++) {  // foreach tetra
      if(i%10000 == 0)
        cerr << "tetra " << i << " of " << ntetra << '\n';
      int *nodes=tetra+i*4;
      // determine bounds of tetra
      int sx, ex, sy, ey, sz, ez;
      tetra_bounds(nodes, &sx, &ex, &sy, &ey, &sz, &ez);
      for(int x=sx; x <= ex; x++) {     // foreach voxel in the tetra's
        for(int y=sy; y <= ey; y++) {   // bounding box, determine if the
          for(int z=sz; z <= ez; z++) { // tetra is REALLY in the voxel
            if(intersect_voxel_tetra(x, y, z, nodes)) {
	      int idx=x*nynz+y*global_nz+z;
	      counts[idx]++;
	      checksum_1++;
	    }
	  }
	}
      }
    }
    cerr << "Pass 2\n";

    int *cells=new int[ncells];

    int sum=1;

    for(int i=0; i < ncells; i++) {
      if (counts[i] == 0) {
        cells[i]=0;
      } else {
        cells[i]=sum;
	sum+=counts[i]+1;
      }
    }
    
    int *lists = new int[sum];
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

    for(int i=0; i < ntetra; i++) {
      if(i%10000 == 0)
        cerr << "tetra " << i << " of " << ntetra << '\n';
      int *nodes=tetra+i*4;
      // determine bounds of tetra
      int sx, ex, sy, ey, sz, ez;
      tetra_bounds(nodes, &sx, &ex, &sy, &ey, &sz, &ez);
      for(int x=sx; x <= ex; x++) {
	for(int y=sy; y <= ey; y++) {
	  for(int z=sz; z <= ez; z++) {
	    if(intersect_voxel_tetra(x, y, z, nodes)) {
	      int idx=x*nynz+y*global_nz+z;
	      int loc=cells[idx]+counts[idx ];
	      lists[loc]=i;
	      counts[idx]++;
	      checksum_2++;
	    }
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

}
