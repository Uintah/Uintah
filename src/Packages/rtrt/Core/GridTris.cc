
#include <Packages/rtrt/Core/GridTris.h>

#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/visinfo/visinfo.h>
extern "C" {
#include <Packages/rtrt/Core/pcube.h>
}

#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Runnable.h>

#include <iostream>

#include <GL/glx.h>
#include <GL/glu.h>
#include <X11/Xlib.h>

#include <stdlib.h>
#include <fstream>

using namespace rtrt;
using namespace SCIRun;
using namespace std;

GridTris::GridTris(Material* matl, int ncells, int depth,
		   const string& filename)
  : Object(this), fallbackMaterial(matl), ncells(ncells), depth(depth),
    preprocessed(false), filename(filename)
{
  inv_ncells=1./ncells;
}

GridTris::~GridTris()
{
}

void 
GridTris::io(SCIRun::Piostream &)
{
  ASSERTFAIL("Pio for GridTris not implemented");
}

void GridTris::calc_se(Tri& t, int totalcells,
		       int& sx, int& sy, int& sz,
		       int& ex, int& ey, int& ez)
{
  Vert& v0 = verts[t.idx[0]];
  Vert& v1 = verts[t.idx[1]];
  Vert& v2 = verts[t.idx[2]];
  Point p0(v0.x[0], v0.x[1], v0.x[2]);
  Point p1(v1.x[0], v1.x[1], v1.x[2]);
  Point p2(v2.x[0], v2.x[1], v2.x[2]);
  Point ss(Min(Min(p0, p1), p2));
  Point ee(Max(Max(p0, p1), p2));
  Vector s = (ss-min)*inv_diag;
  Vector e = (ee-min)*inv_diag;
  sx=(int)(s.x()*totalcells);
  sy=(int)(s.y()*totalcells);
  sz=(int)(s.z()*totalcells);
  ex=(int)(e.x()*totalcells);
  ey=(int)(e.y()*totalcells);
  ez=(int)(e.z()*totalcells);
  if(sx < 0 || ex >= totalcells){
    cerr << "NX out of bounds!\n";
    cerr << "sx=" << sx << ", ex=" << ex << '\n';
    cerr << "e=" << e << '\n';
    cerr << "bbox=" << min << ", " << max << '\n';
    cerr << "diag=" << diag << '\n';
    exit(1);
  }
  if(sy < 0 || ey >= totalcells){
    cerr << "NY out of bounds!\n";
    cerr << "sy=" << sy << ", ey=" << ey << '\n';
    exit(1);
  }
  if(sz < 0 || ez >= totalcells){
    cerr << "NZ out of bounds!\n";
    cerr << "sz=" << sz << ", ez=" << ez << '\n';
    exit(1);
  }
}

bool GridTris::intersects(const Tri& tri, int totalcells, int x, int y, int z)
{
  Vert& v0 = verts[tri.idx[0]];
  Vert& v1 = verts[tri.idx[1]];
  Vert& v2 = verts[tri.idx[2]];
  Point p0(v0.x[0], v0.x[1], v0.x[2]);
  Point p1(v1.x[0], v1.x[1], v1.x[2]);
  Point p2(v2.x[0], v2.x[1], v2.x[2]);
  Vector vv0 = (p0-min)*Vector(totalcells, totalcells, totalcells)*inv_diag;
  Vector vv1 = (p1-min)*Vector(totalcells, totalcells, totalcells)*inv_diag;
  Vector vv2 = (p2-min)*Vector(totalcells, totalcells, totalcells)*inv_diag;
  Vector n = Cross((vv2-vv0),(vv1-vv0));
  n.normalize();
  real polynormal[3];
  polynormal[0] = n.x();
  polynormal[1] = n.y();
  polynormal[2] = n.z();
  real verts[3][3];
  verts[0][0] = vv0.x() - ((double)x+.5);
  verts[1][0] = vv1.x() - ((double)x+.5);
  verts[2][0] = vv2.x() - ((double)x+.5);
  verts[0][1] = vv0.y() - ((double)y+.5);
  verts[1][1] = vv1.y() - ((double)y+.5);
  verts[2][1] = vv2.y() - ((double)y+.5);
  verts[0][2] = vv0.z() - ((double)z+.5);
  verts[1][2] = vv1.z() - ((double)z+.5);
  verts[2][2] = vv2.z() - ((double)z+.5);
  return fast_polygon_intersects_cube(3, verts, polynormal, 0, 0);
}

void GridTris::preprocess(double, int&, int&)
{
  if (preprocessed) return;
  preprocessed = true;

  int totalcells=1;
  for(int i=0;i<=depth;i++)
    totalcells*=ncells;
  int totalsize=totalcells*totalcells*totalcells;

  // Try to read it from a file...
  if(isCached()){
    ostringstream fname;
    fname << filename << ".gridtri_" << ncells << "_" << depth;
    ifstream in(fname.str().c_str());
    cerr << "Reading gridtris from: " << fname.str() << '\n';
    double start = SCIRun::Time::currentSeconds();
    streampos ss = in.tellg();
    int have_fallback;
    in.read((char*)&have_fallback, sizeof(int));
    if(!have_fallback)
      clearFallback();

    long numVerts;
    in.read((char*)&numVerts, sizeof(long));
    verts.resize(numVerts);
    cerr << "Reading verts, " << (double)numVerts*sizeof(Vert)/1024./1024. << "M\n";
    in.read((char*)&verts[0], numVerts*sizeof(Vert));

    long numTris;
    in.read((char*)&numTris, sizeof(long));
    tris.resize(numTris);
    cerr << "Reading tris, " << (double)numTris*sizeof(Tri)/1024./1024. << "M\n";
    in.read((char*)&tris[0], numTris*sizeof(Tri));
    
    in.read((char*)&min, sizeof(min));
    in.read((char*)&max, sizeof(max));
    in.read((char*)&diag, sizeof(diag));
    inv_diag = Vector(1./diag.x(), 1./diag.y(), 1./diag.z());
    counts.resize(totalcells, totalcells, totalcells);
    cerr << "Reading cells, " << (double)counts.get_datasize()/1024./1024. << "M\n";
    in.read((char*)counts.get_dataptr(), long(counts.get_datasize()+sizeof(int)));
    long totalcells;
    in.read((char*)&totalcells, sizeof(long));
    cells.resize(totalcells);
    cerr << "Reading lists, " << (double)totalcells*sizeof(long)/1024./1024. << "M\n";
    in.read((char*)&cells[0], long(sizeof(int)*totalcells));
    if(depth==0){
      macrocells=0;
    } else {
      macrocells=new BrickArray3<bool>[depth+1];
      int size=ncells;
      for(int d=depth;d>=1;d--){
	macrocells[d].resize(size, size, size);
	cerr << "Reading macrocells for depth " << d << '\n';
	in.read((char*)macrocells[d].get_dataptr(), long(macrocells[d].get_datasize()));
	size*=ncells;
      }
    }
    if(!in){
      cerr << "ERROR reading cached gridtri structure!\n";
      exit(1);
    }
    streampos se = in.tellg();
    double dt = SCIRun::Time::currentSeconds()-start;
    double rate = double(se-ss)/dt/1024./1024.;
    cerr << "Read file in " << dt << " seconds (" << rate << " MB/sec)\n";
  } else {  // Not cached
    cerr << "Building GridTris for " << tris.size() << " triangles\n";
    float time=SCIRun::Time::currentSeconds();
  
    vector<Vert>::iterator iter = verts.begin();
    if(iter == verts.end()){
      cerr << "No vertices!\n";
      return;
    }
    min = max = Point(iter->x[0], iter->x[1], iter->x[2]);
    for(;iter != verts.end(); iter++){
      Point p(iter->x[0], iter->x[1], iter->x[2]);
      min=Min(min, p);
      max=Max(max, p);
    }
    min -= Vector(1.e-5, 1.e-5, 1.e-5);
    max += Vector(1.e-5, 1.e-5, 1.e-5);
    diag = max-min;
    max += diag*1.e-5;
    min -= diag*1.e-5;
    diag = max-min;
    inv_diag = Vector(1./diag.x(), 1./diag.y(), 1./diag.z());
    cerr << "0/6 bounds took " << SCIRun::Time::currentSeconds()-time << " seconds\n";
    time=SCIRun::Time::currentSeconds();

    cerr << "Computing " << totalcells << 'x' << totalcells << 'x' << totalcells << " grid for " << totalsize << " cells\n";
  
    counts.resize(totalcells, totalcells, totalcells);
    counts.initialize(0);
    cerr << "1/6 allocation took " << SCIRun::Time::currentSeconds()-time << " seconds\n";
    time=SCIRun::Time::currentSeconds();
  
    double itime=time;
    int tt=0;
    for(int i=0;i<(int)tris.size();i++){
      double tnow=SCIRun::Time::currentSeconds();
      if(tnow-itime > 5.0){
	cerr << i << "/" << tris.size() << '\n';
	itime=tnow;
      }
      int sx, sy, sz, ex, ey, ez;
      Tri& tri = tris[i];
      calc_se(tri, totalcells, sx, sy, sz, ex, ey, ez);
      for(int x=sx;x<=ex;x++){
	for(int y=sy;y<=ey;y++){
	  for(int z=sz;z<=ez;z++){
	    if(intersects(tri, totalcells, x, y, z)){
	      counts(x,y,z)++;
	      tt++;
	    }
	  }
	}
      }
    }
    
    cerr << "2/6 Counting cells took " << SCIRun::Time::currentSeconds()-time << " seconds\n";
    time=SCIRun::Time::currentSeconds();

    int total=0;
    int* ptr = counts.get_dataptr();
    unsigned long datasize = counts.get_datasize()/sizeof(int);
    for(unsigned long i=0;i<datasize;i++){
      int count=ptr[i];
      ptr[i]=total;
      total+=count;
    }
    ptr[datasize]=total;
    cerr << "Allocating " << total << " grid indices (" << double(total)/double(tris.size()) << " per tri, " << double(total)/totalsize << " per cell)\n";
    cells.resize(total);
    for(int i=0;i<total;i++)
      cells[i]=-1234;
    cerr << "3/6 Calculating offsets took " << SCIRun::Time::currentSeconds()-time << " seconds\n";
    time=SCIRun::Time::currentSeconds();
    itime=time;
    BrickArray3<int> current(totalcells, totalcells, totalcells);
    current.initialize(0);
    for(unsigned int i=0;i<tris.size();i++){
      double tnow=SCIRun::Time::currentSeconds();
      if(tnow-itime > 5.0){
	cerr << i << "/" << tris.size() << '\n';
	itime=tnow;
      }
      int sx, sy, sz, ex, ey, ez;
      Tri& tri = tris[i];
      calc_se(tri, totalcells, sx, sy, sz, ex, ey, ez);
      for(int x=sx;x<=ex;x++){
	for(int y=sy;y<=ey;y++){
	  for(int z=sz;z<=ez;z++){
	    if(intersects(tri, totalcells, x, y, z)){
	      int cur=current(x,y,z)++;
	      int pos=counts(x,y,z)+cur;
	      cells[pos]=i;
	    }
	  }
	}
      }
    }
    cerr << "4/6 Filling grid took " << SCIRun::Time::currentSeconds()-time << " seconds\n";
    time=SCIRun::Time::currentSeconds();
    int* ptr2=current.get_dataptr();
    for(unsigned int i=0;i<datasize;i++){
      int diff = ptr[i+1]-ptr[i];
      if(ptr2[i] != diff){
	cerr << "OOPS!\n";
	cerr << "current: " << ptr2[i] << '\n';
	cerr << "counts: " << ptr[i] << '\n';
	exit(1);
      }
    }
    for(int i=0;i<total;i++){
      if(cells[i]==-1234){
	cerr << "OOPS: cells[" << i << "]==-1234!\n";
	exit(1);
      }
    }
    cerr << "5/6 Verifying grid took " << SCIRun::Time::currentSeconds()-time << " seconds\n";
    time=SCIRun::Time::currentSeconds();
    if(depth==0){
      macrocells=0;
    } else {
      macrocells=new BrickArray3<bool>[depth+1];
      int size=ncells;
      for(int d=depth;d>=1;d--){
	macrocells[d].resize(size, size, size);
	macrocells[d].initialize(false);
	size*=ncells;
      }
      bool haveit;
      int ntris;
      calc_mcell(depth, 0, 0, 0, haveit, ntris);
      if(ntris != total || !haveit){
	cerr << "Mcell went wrong!\n";
	cerr << "mcell: " << ntris << '\n';
	cerr << "total: " << total << '\n';
	exit(1);
      }
      for(int d=1;d<=depth;d++){
	BrickArray3<bool>& mcells = macrocells[depth];
	int d1=mcells.dim1();
	int d2=mcells.dim2();
	int d3=mcells.dim3();
	int haveit=0;
	for(int x=0;x<d1;x++){
	  for(int y=0;y<d2;y++){
	    for(int z=0;z<d3;z++){
	      if(mcells(x,y,z))
		haveit++;
	    }
	  }
	}
	int tot = d1*d2*d3;
	cerr << "Depth " << d << " is " << 100.*double(haveit)/double(tot) << "% occupied\n";
      }
      cerr << "6/6 Calculating macrocells took " << SCIRun::Time::currentSeconds()-time << " seconds\n";

      if(filename.length() != 0){
	ostringstream fname;
	fname << filename << ".gridtri_" << ncells << "_" << depth;
	ofstream out(fname.str().c_str());
	if(out){
	  cerr << "Writing gridtris to: " << fname.str() << '\n';
	  double start = SCIRun::Time::currentSeconds();
	  streampos ss = out.tellp();
	  int have_fallback = fallbackMaterial?1:0;
	  out.write((char*)&have_fallback, sizeof(int));

	  long numVerts = (long)verts.size();
	  out.write((char*)&numVerts, sizeof(long));
	  cerr << "Writing verts, " << (double)numVerts*sizeof(Vert)/1024./1024. << "M\n";
	  out.write((char*)&verts[0], numVerts*sizeof(Vert));

	  long numTris = (long)tris.size();
	  out.write((char*)&numTris, sizeof(long));
	  cerr << "Writing tris, " << (double)numTris*sizeof(Tri)/1024./1024. << "M\n";
	  out.write((char*)&tris[0], numTris*sizeof(Tri));

	  out.write((char*)&min, sizeof(min));
	  out.write((char*)&max, sizeof(max));
	  out.write((char*)&diag, sizeof(diag));
	  cerr << "Writing cells, " << (double)counts.get_datasize()/1024./1024. << "M\n";
	  out.write((char*)counts.get_dataptr(), long(counts.get_datasize()+sizeof(int)));
	  long totalcells = (long)cells.size();
	  out.write((char*)&totalcells, sizeof(long));
	  cells.resize(totalcells);
	  cerr << "Writing lists, " << (double)totalcells*sizeof(long)/1024./1024. << "M\n";
	  out.write((char*)&cells[0], long(sizeof(int)*totalcells));
	  int size=ncells;
	  for(int d=depth;d>=1;d--){
	    cerr << "Writing macrocells for depth " << d << '\n';
	    out.write((char*)macrocells[d].get_dataptr(), long(macrocells[d].get_datasize()));
	    size*=ncells;
	  }
	  if(!out){
	    cerr << "Error writing gridtri to file: " << fname.str() << '\n';
	    exit(1);
	  }
	  streampos se = out.tellp();
	  cerr << "Closing file\n";
	  out.close();
	  double dt = SCIRun::Time::currentSeconds()-start;
	  double rate = double(se-ss)/dt/1024./1024.;
	  cerr << "Wrote file in " << dt << " seconds (" << rate << " MB/sec)\n";
	} else {
	  cerr << "WARNING, not saving gridtri structure\n";
	}
      }
    }
  }
  cerr << "Done building GridTris\n";
}

void GridTris::calc_mcell(int depth, int startx, int starty, int startz,
			  bool& haveit, int& ntris)
{
  haveit=false;
  ntris=0;

  int sx=startx*ncells;
  int sy=starty*ncells;
  int sz=startz*ncells;
  if(depth==0){
    for(int x=0;x<ncells;x++){
      for(int y=0;y<ncells;y++){
	for(int z=0;z<ncells;z++){
	  int& s=counts(sx+x, sy+y, sz+z);
	  int e = *((&s)+1);
	  int diff = e-s;
	  if(diff){
	    ntris += diff;
	    haveit=true;
	  }
	}
      }
    }
  } else {
    BrickArray3<bool>& mcells = macrocells[depth];
    for(int x=0;x<ncells;x++){
      for(int y=0;y<ncells;y++){
	for(int z=0;z<ncells;z++){
	  bool child_haveit;
	  int child_ntris;
	  calc_mcell(depth-1, sx+x, sy+y, sz+z, child_haveit, child_ntris);
	  ntris += child_ntris;
	  mcells(sx+x, sy+y, sz+z) = child_haveit;
	}
      }
    }
  }
  if(ntris)
    haveit=true;
}

void GridTris::intersect(Ray& ray, HitInfo& hit,
			 DepthStats*, PerProcessorContext*)
{
  const Vector& dir(ray.direction());
  const Point& orig(ray.origin());
  //cerr << "orig: " << orig << '\n';
  //cerr << "dir: " << dir << '\n';
  double MIN, MAX;
  double inv_dirx=1./dir.x();
  int dix_dx;
  int ddx;
  if(dir.x() > 0){
    MIN=inv_dirx*(min.x()-orig.x());
    MAX=inv_dirx*(max.x()-orig.x());
    dix_dx=1;
    ddx=1;
  } else {
    MIN=inv_dirx*(max.x()-orig.x());
    MAX=inv_dirx*(min.x()-orig.x());
    dix_dx=-1;
    ddx=0;
  }	
  double y0, y1;
  int diy_dy;
  int ddy;
  double inv_diry=1./dir.y();
  if(dir.y() > 0){
    y0=inv_diry*(min.y()-orig.y());
    y1=inv_diry*(max.y()-orig.y());
    diy_dy=1;
    ddy=1;
  } else {
    y0=inv_diry*(max.y()-orig.y());
    y1=inv_diry*(min.y()-orig.y());
    diy_dy=-1;
    ddy=0;
  }
  if(y0>MIN)
    MIN=y0;
  if(y1<MAX)
    MAX=y1;
  if(MAX<MIN)
    return;
  
  double z0, z1;
  int diz_dz;
  int ddz;
  double inv_dirz=1./dir.z();
  if(dir.z() > 0){
    z0=inv_dirz*(min.z()-orig.z());
    z1=inv_dirz*(max.z()-orig.z());
    diz_dz=1;
    ddz=1;
  } else {
    z0=inv_dirz*(max.z()-orig.z());
    z1=inv_dirz*(min.z()-orig.z());
    diz_dz=-1;
    ddz=0;
  }
  if(z0>MIN)
    MIN=z0;
  if(z1<MAX)
    MAX=z1;
  if(MAX<MIN)
    return;
  double t;
  if(MIN > 1.e-6){
    t=MIN;
  } else if(MAX > 1.e-6){
    t=0;
  } else {
    return;
  }
  if(t>1.e29)
    return;
  Point p(orig+dir*t);
  Vector s((p-min)*inv_diag);
  //cerr << "min=" << min << ", max=" << max << ", orig=" << orig << ", dir=" << dir << ", t=" << t << '\n';
  int ix=(int)(s.x()*ncells);
  int iy=(int)(s.y()*ncells);
  int iz=(int)(s.z()*ncells);
  //  cerr << "s=" << s << ", ix=" << ix << ", iy=" << iy << ", iz=" << iz << '\n';
  if(ix>=ncells)
    ix--;
  if(iy>=ncells)
    iy--;
  if(iz>=ncells)
    iz--;
  if(ix<0)
    ix++;
  if(iy<0)
    iy++;
  if(iz<0)
    iz++;
  
  double next_x, next_y, next_z;
  double dtdx, dtdy, dtdz;
  double x=min.x()+diag.x()*double(ix+ddx)*inv_ncells;
  next_x=(x-orig.x())*inv_dirx;
  dtdx=dix_dx*diag.x()*inv_ncells*inv_dirx;
  double y=min.y()+diag.y()*double(iy+ddy)*inv_ncells;
  next_y=(y-orig.y())*inv_diry;
  dtdy=diy_dy*diag.y()*inv_ncells*inv_diry;
  double z=min.z()+diag.z()*double(iz+ddz)*inv_ncells;
  next_z=(z-orig.z())*inv_dirz;
  dtdz=diz_dz*diag.z()*inv_ncells*inv_dirz;
  
  Vector cellcorner((orig-min)*inv_diag*ncells);
  Vector celldir(dir*inv_diag*ncells);
  isect(depth, t, dtdx, dtdy, dtdz, next_x, next_y, next_z,
	0, 0, 0, ix, iy, iz, dix_dx, diy_dy, diz_dz,
	cellcorner, celldir,
	ray, hit);
}

void GridTris::isect(int depth, double t,
		     double dtdx, double dtdy, double dtdz,
		     double next_x, double next_y, double next_z,
		     int sx, int sy, int sz, int ix, int iy, int iz,
		     int dix_dx, int diy_dy, int diz_dz,
		     const Vector& cellcorner, const Vector& celldir,
		     const Ray& ray, HitInfo& hit)
{
  //cerr << "Starting depth " << depth << ", sx=" << sx << ", sy=" << sy << ", sz=" << sz << ", ix=" << ix << ", iy=" << iy << ", iz=" << iz << '\n';
  if(depth==0){
    for(;;){
      int& s=counts(sx+ix,sy+iy,sz+iz);
      int e = *((&s)+1);
      //cerr << "t=" << t << ": " << ix << ", " << iy << ", " << iz << ", " << next_x << ", " << next_y << ", " << next_z << ", s=" << s << ", e=" << e << '\n';
      for(int i=s;i<e;i++){
	Tri& tri = tris[cells[i]];
	Vert& v0 = verts[tri.idx[0]];
	Vert& v1 = verts[tri.idx[1]];
	Vert& v2 = verts[tri.idx[2]];

#if 0
	// For some strange reason, this seems to be slower...
	float e1[3], e2[3];
	float o[3];
	for(int i=0;i<3;i++){
	  e1[i]=v1.x[i]-v0.x[i];
	  e2[i]=v2.x[i]-v0.x[i];
	  o[i]=v0.x[i]-ray.origin()(i);
	}
	float e1e2[3];
	e1e2[0]=e1[1]*e2[2]-e1[2]*e2[1];
	e1e2[1]=e1[2]*e2[0]-e1[0]*e2[2];
	e1e2[2]=e1[0]*e2[1]-e1[1]*e2[0];
	const Vector& dir = ray.direction();

	float det = e1e2[0]*(float)dir.x()+e1e2[1]*(float)dir.y()+e1e2[2]*(float)dir.z();
	if(det>1.f-9 || det < -1.f-9){
	  float idet=1.f/det;

	  float dx[3];
	  dx[0]=dir(1)*o[2]-dir(2)*o[1];
	  dx[1]=dir(2)*o[0]-dir(0)*o[2];
	  dx[2]=dir(0)*o[1]-dir(1)*o[0];
	  float A = -(dx[0]*e2[0] + dx[1]*e2[1] + dx[2]*e2[2])*idet;
	  if(A>0.0f && A<1.0f){
	    float B= (dx[0]*e1[0] + dx[1]*e1[1] + dx[2]*e1[2])*idet;
	    if(B>0.0f && A+B<1.0f){
	      double t= (e1e2[0]*o[0] + e1e2[1]*o[1] + e1e2[2]*o[2])*idet;
	      if (hit.hit(this, t)) {
		HitRecord* h = (HitRecord*)hit.scratchpad;
		h->u=A;
		h->v=B;
		h->idx=cells[i];
	      }
	    }
	  }
	}
#else
	//st->tri_isect++;
	Point p1(v0.x[0], v0.x[1], v0.x[2]);
	Point p2(v1.x[0], v1.x[1], v1.x[2]);
	Point p3(v2.x[0], v2.x[1], v2.x[2]);
	Vector e1(p2-p1);
	Vector e2(p3-p1);
	Vector dir(ray.direction());
	Vector o(p1-ray.origin());

	Vector e1e2(Cross(e1, e2));
	double det=Dot(e1e2, dir);
	if(det>1.e-9 || det < -1.e-9){
	  double idet=1./det;
	  
	  Vector DX(Cross(dir, o));
	  double A=-Dot(DX, e2)*idet;
	  if(A>0.0 && A<1.0){
	    double B=Dot(DX, e1)*idet;
	    if(B>0.0 && A+B<1.0){
	      double t=Dot(e1e2, o)*idet;
		if (hit.hit(this, t)) {
		  HitRecord* h = (HitRecord*)hit.scratchpad;
		  h->u=A;
		  h->v=B;
		  h->idx=cells[i];
		}
	    }
	  }
	}
#endif	
      }
      if(next_x < next_y && next_x < next_z){
	// Step in x...
	t=next_x;
	next_x+=dtdx;
	ix+=dix_dx;
	if(ix<0 || ix>=ncells)
	  break;
      } else if(next_y < next_z){
	t=next_y;
	next_y+=dtdy;
	iy+=diy_dy;
	if(iy<0 || iy>=ncells)
	  break;
      } else {
	t=next_z;
	next_z+=dtdz;
	iz+=diz_dz;
	if(iz<0 || iz >=ncells)
	  break;
      }
      if(hit.min_t < t)
	break;
    }
  } else {
    BrickArray3<bool>& mcells=macrocells[depth];
    for(;;){
      //cerr << "t=" << t << ": " << ix << ", " << iy << ", " << iz << ", " << next_x << ", " << next_y << ", " << next_z << ", mcells=" << mcells(sx+ix, sy+iy, sz+iz) << '\n';
      if(mcells(sx+ix, sy+iy, sz+iz)){
	// Do this cell...
	int new_ix=(int)((cellcorner.x()+t*celldir.x()-ix)*ncells);
	int new_iy=(int)((cellcorner.y()+t*celldir.y()-iy)*ncells);
	int new_iz=(int)((cellcorner.z()+t*celldir.z()-iz)*ncells);
	//cerr << "new_ix=" << new_ix << ", new_iy=" << new_iy << ", new_iz=" << new_iz << '\n';
	if(new_ix<0)
	  new_ix=0;
	else if(new_ix>=ncells)
	  new_ix=ncells-1;
	if(new_iy<0)
	  new_iy=0;
	else if(new_iy>=ncells)
	  new_iy=ncells-1;
	if(new_iz<0)
	  new_iz=0;
	else if(new_iz>=ncells)
	  new_iz=ncells-1;

	double new_dtdx=dtdx*inv_ncells;
	double new_dtdy=dtdy*inv_ncells;
	double new_dtdz=dtdz*inv_ncells;
	const Vector& dir(ray.direction());
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
	isect(depth-1, t,
	      new_dtdx, new_dtdy, new_dtdz,
	      new_next_x, new_next_y, new_next_z,
	      (sx+ix)*ncells, (sy+iy)*ncells, (sz+iz)*ncells,
	      new_ix, new_iy, new_iz,
	      dix_dx, diy_dy, diz_dz,
	      (cellcorner-Vector(ix, iy, iz))*ncells, celldir*ncells,
	      ray, hit);
      }
      if(next_x < next_y && next_x < next_z){
	// Step in x...
	t=next_x;
	next_x+=dtdx;
	ix+=dix_dx;
	if(ix<0 || ix>=ncells)
	  break;
      } else if(next_y < next_z){
	t=next_y;
	next_y+=dtdy;
	iy+=diy_dy;
	if(iy<0 || iy>=ncells)
	  break;
      } else {
	t=next_z;
	next_z+=dtdz;
	iz+=diz_dz;
	if(iz<0 || iz >=ncells)
	  break;
      }
      if(hit.min_t < t)
	break;
    }
  }
  //cerr << "Done depth " << depth << '\n';  
}

void GridTris::compute_bounds(BBox& bbox, double)
{
  if(preprocessed){
    bbox.extend(min);
    bbox.extend(max);
  } else {
    for(vector<Vert>::iterator iter = verts.begin(); iter != verts.end();
	iter++)
      bbox.extend(Point(iter->x[0], iter->x[1], iter->x[2]));
  }
}

Vector GridTris::normal(const Point&, const HitInfo& hit)
{
  HitRecord* h = (HitRecord*)hit.scratchpad;
  Tri& t = tris[h->idx];
  return Vector(t.n[0], t.n[1], t.n[2]);
}

void GridTris::shade(Color& result, const Ray& ray,
			const HitInfo& hit, int depth,
			double atten, const Color& accumcolor,
			Context* cx)
{
  if(fallbackMaterial)
    fallbackMaterial->shade(result, ray, hit, depth, atten, accumcolor, cx);
  else {
    // extract barycoords
    HitRecord* h = (HitRecord*)hit.scratchpad;
    Tri& tri = tris[h->idx];
    Vert& v0 = verts[tri.idx[0]];
    Vert& v1 = verts[tri.idx[1]];
    Vert& v2 = verts[tri.idx[2]];
  
    // blend colors;
    Color diff_color = 
      (1.-h->u-h->v)*Color(v0.color[0], v0.color[1], v0.color[2])+
      h->u*Color(v1.color[0], v1.color[1], v1.color[2])+
      h->v*Color(v2.color[0], v2.color[1], v2.color[2]);
    diff_color = diff_color*Color(1./255, 1./255, 1./255);
    
    Color spec_color(.2,.2,.2);
    phongshade(result,diff_color,spec_color,80,0,ray,hit,depth,atten,
	       accumcolor,cx);
  }
}

void GridTris::addVertex(float x[3], unsigned char c[3])
{
  Vert v;
  for(int i=0;i<3;i++)
    v.x[i]=x[i];
  for(int i=0;i<3;i++)
    v.color[i]=c[i];
  verts.push_back(v);
}

void GridTris::addTri(int i0, int i1, int i2)
{
  ASSERT(i0 >= 0 && i1 >= 0 && i2 >= 0);
  ASSERT(i0 < (int)verts.size() && i1 < (int)verts.size() && i2 < (int)verts.size());

  Vert& v0 = verts[i0];
  Vert& v1 = verts[i1];
  Vert& v2 = verts[i2];
  Point pt0 (v0.x[0], v0.x[1], v0.x[2]);
  Point pt1 (v1.x[0], v1.x[1], v1.x[2]);
  Point pt2 (v2.x[0], v2.x[1], v2.x[2]);
	  
  Vector n = Cross(pt2-pt0,pt1-pt0);
  // remove "bad" triangles
  if (n.length() < 1.E-16)
    return;

  n.normalize();
  
  Tri tri;
  tri.idx[0]=i0;
  tri.idx[1]=i1;
  tri.idx[2]=i2;
  tri.n[0]=n.x();
  tri.n[1]=n.y();
  tri.n[2]=n.z();
  tris.push_back(tri);
}

void GridTris::transform(Transform& T)
{
  for(vector<Vert>::iterator iter = verts.begin(); iter != verts.end();
      iter++){
    Vert& vt = *iter;
    Vector v(vt.x[0], vt.x[1], vt.x[2]);
    T.project_inplace(v);
    vt.x[0] = v.x(); vt.x[1] = v.y(); vt.x[2] = v.z();
  }
  for(vector<Tri>::iterator iter = tris.begin(); iter != tris.end(); iter++){
    Tri& tri = *iter;
    Vector v(tri.n[0], tri.n[1], tri.n[2]);
    T.project_normal_inplace(v);
    v.normalize();
    tri.n[0] = v.x(); tri.n[1] = v.y(); tri.n[2] = v.z();
  }
}

bool GridTris::isCached()
{
  if(filename.length() != 0){
    ostringstream fname;
    fname << filename << ".gridtri_" << ncells << "_" << depth;
    ifstream in(fname.str().c_str());
    if(in)
      return true;
  }
  return false;
}
