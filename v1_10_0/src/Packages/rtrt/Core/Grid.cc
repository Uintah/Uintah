#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/TexturedTri.h>
#include <Packages/rtrt/Core/Tri.h>
#include <Packages/rtrt/Core/Names.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <iostream>
#include <stdlib.h>
#include "Color.h"
#include "LambertianMaterial.h"

using namespace rtrt;
using namespace SCIRun;

SCIRun::Persistent* grid_maker() {
  return new Grid;
}

// initialize the static member type_id
SCIRun::PersistentTypeID Grid::type_id("Grid", "Object", grid_maker);

Grid::Grid(Object* obj, int nsides)
    : Object(0), obj(obj), nsides(nsides)
{
  if (obj == 0) 
    ASSERTFAIL("Trying to preprocess a Grid with no objects");

  grid=0;
  counts=0;
  
  set_matl(new LambertianMaterial(Color(1,0,0)));

}

Grid::~Grid()
{
}

void Grid::calc_se(const BBox& obj_bbox, const BBox& bbox,
			  const Vector& diag,
			  int nx, int ny, int nz,
			  int& sx, int& sy, int& sz,
			  int& ex, int& ey, int& ez)
{
  //    cerr << "obbx min " << obj_bbox.min() << endl;
  //    cerr << "obbx max " << obj_bbox.max() << endl;
  //    cerr << "bbx min " << bbox.min() << endl;
  //    cerr << "bbx max " << bbox.max() << endl;
  //    cerr << "diag " << diag << endl;
  //    cerr << "nx, ny, nz " << nx << ", " << ny << ", " << nz << endl;
    
  Vector s((obj_bbox.min()-bbox.min())/diag);
  //    cerr << "s " << s << endl;
  Vector e((obj_bbox.max()-bbox.min())/diag);
  //    cerr << "s " << s << endl;
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
    cerr << "nz=" << nz << '\n';
    cerr << "obj_bbox=" << obj_bbox.min() << ", " << obj_bbox.max() << '\n';
    cerr << "bbox=" << bbox.min() << ", " << bbox.max() << '\n';
    cerr << "diag=" << diag << '\n';
    exit(1);
  }
}

void Grid::preprocess(double maxradius, int& pp_offset, int& scratchsize)
{
  if (was_preprocessed) return;
  was_preprocessed=true;
  
  if (Names::hasName(this)) std::cerr << "\n\n"
                             << "\n==========================================================\n"
			     << "* Building Regular Grid for Object " << Names::getName(this)
			     << "\n==========================================================\n";

  //    cerr << "Building grid\n";
    double time=Time::currentSeconds();
    obj->preprocess(maxradius, pp_offset, scratchsize);
    //    cerr << "1/7 Preprocess took " << Time::currentSeconds()-time << " seconds\n";
    time=Time::currentSeconds();

    Array1<Object*> prims;
    obj->collect_prims(prims);
    //    cerr << "2/7 Collect prims took " << Time::currentSeconds()-time << " seconds\n";
    time=Time::currentSeconds();

    bbox.reset();
    obj->compute_bounds(bbox, maxradius);
    //    cerr << "3/7 Compute bounds took " << Time::currentSeconds()-time << " seconds\n";
    time=Time::currentSeconds();

    int ncells=nsides*nsides*nsides;
    bbox.extend(bbox.min()-Vector(1.e-3, 1.e-3, 1.e-3));
    bbox.extend(bbox.max()+Vector(1.e-3, 1.e-3, 1.e-3));
    Vector diag(bbox.diagonal());
    bbox.extend(bbox.max()+diag*1.e-3);
    bbox.extend(bbox.min()-diag*1.e-3);
    diag=bbox.diagonal();
    double volume=diag.x()*diag.y()*diag.z();
    double c=cbrt(ncells/volume);

    float fx = c*diag.x();
    float fy = c*diag.y();
    float fz = c*diag.z();

    if(fx<2){
      fy *= sqrt( fx/2 );
      fz *= sqrt( fx/2 );
      fx=2;
    }
    if(fy<2){
      fx *= sqrt( fy/2 );
      if( fx < 2 ) { 
	fx = 2; 
	fz *= fy/2;
      } else {
	fz *= sqrt( fy/2 );
      }
      fy=2;
    }
    if(fz<2){
      fx *= sqrt( fz/2 );
      if( fx < 2 ) { 
	fx = 2; 
	fy *= fz/2;
      } else {
	fy *= sqrt( fz/2 );
      }
      if( fy < 2 ){
	fy = 2;
      }
      fz=2;
    }

    nx=(int)(fx+.5);
    ny=(int)(fy+.5);
    nz=(int)(fz+.5);
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
    counts=new int[2*ngrid];
    //    cerr << "counts=" << counts << ":" << counts+2*ngrid << '\n';
    for(int i=0;i<ngrid*2;i++)
	counts[i]=0;

    double itime=time;
    int nynz=ny*nz;

    Vector p0,p1,p2,n;
    real verts[3][3];
    real polynormal[3];
 
    for(int i=0;i<prims.size();i++){
	double tnow=Time::currentSeconds();
	if(tnow-itime > 5.0){
	  //	    cerr << i << "/" << prims.size() << '\n';
	    itime=tnow;
	}
	BBox obj_bbox;
	prims[i]->compute_bounds(obj_bbox, maxradius);

        Tri *tri = dynamic_cast<Tri*>(prims[i]);
        if (tri) {
            if (tri->isbad()){
               //cerr << "WARNING -- tri isbad() true!!" << endl;
                continue;
            }
            p0 = (tri->pt(0) - bbox.min())*(Vector(nx,ny,nz)/diag);
            p1 = (tri->pt(1) - bbox.min())*(Vector(nx,ny,nz)/diag);
            p2 = (tri->pt(2) - bbox.min())*(Vector(nx,ny,nz)/diag);
            n = Cross((p2-p0),(p1-p0));
            n.normalize();
            polynormal[0] = n.x();
            polynormal[1] = n.y();
            polynormal[2] = n.z();
        }
       
        
        TexturedTri *ttri = dynamic_cast<TexturedTri*>(prims[i]);
        if (ttri) {
            if (ttri->isbad()){
               //cerr << "WARNING -- ttri isbad() true!!" << endl;
                continue;
            }
            p0 = (ttri->pt(0) - bbox.min())*(Vector(nx,ny,nz)/diag);
            p1 = (ttri->pt(1) - bbox.min())*(Vector(nx,ny,nz)/diag);
            p2 = (ttri->pt(2) - bbox.min())*(Vector(nx,ny,nz)/diag);
            Vector n = Cross((p2-p0),(p1-p0));
            n.normalize();
            polynormal[0] = n.x();
            polynormal[1] = n.y();
            polynormal[2] = n.z();
        }
    
        int sx, sy, sz, ex, ey, ez;
        calc_se(obj_bbox, bbox, diag, nx, ny, nz, sx, sy, sz, ex, ey, ez);
        for(int x=sx;x<=ex;x++){
            for(int y=sy;y<=ey;y++){
                int idx=x*nynz+y*nz+sz;
		for(int z=sz;z<=ez;z++){
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
                          counts[idx*2+1]++;
                       }
                       idx++;
                    }
                    else {
                       counts[idx*2+1]++;
		       idx++;
                    }
		}
	    }
	}
    }

    //    cerr << "4/7 Counting cells took " << Time::currentSeconds()-time << " seconds\n";
    time=Time::currentSeconds();
    int total=0;
    for(int i=0;i<ngrid;i++){
	int count=counts[i*2+1];
	counts[i*2]=total;
	total+=count;
    }
    //    cerr << "Allocating " << total << " grid cells (" << double(total)/prims.size() << " per object, " << double(total)/ngrid << " per cell)\n";
    grid=new Object*[total];
    //    cerr << "grid=" << grid << ":" << grid+total << '\n';
    for(int i=0;i<total;i++)
	grid[i]=0;
    //    cerr << "total=" << total << '\n';
    //    cerr << "5/7 Calculating offsets took " << Time::currentSeconds()-time << " seconds\n";
    time=Time::currentSeconds();
    itime=time;
    Array1<int> current(ngrid);
    current.initialize(0);

    for(int i=0;i<prims.size();i++){
	double tnow=Time::currentSeconds();
	if(tnow-itime > 5.0){
	    cerr << i << "/" << prims.size() << '\n';
	    itime=tnow;
	}
	BBox obj_bbox;
	prims[i]->compute_bounds(obj_bbox, maxradius);

        Tri *tri = dynamic_cast<Tri*>(prims[i]);

        if (tri) {
            if (tri->isbad()){
               //cerr << "WARNING -- tri isbad() true!!" << endl;
                continue;
            }
            p0 = (tri->pt(0) - bbox.min())*(Vector(nx,ny,nz)/diag);
            p1 = (tri->pt(1) - bbox.min())*(Vector(nx,ny,nz)/diag);
            p2 = (tri->pt(2) - bbox.min())*(Vector(nx,ny,nz)/diag);
            n = Cross((p2-p0),(p1-p0));
            n.normalize();
            polynormal[0] = n.x();
            polynormal[1] = n.y();
            polynormal[2] = n.z();
        }
        TexturedTri *ttri = dynamic_cast<TexturedTri*>(prims[i]);

        if (ttri) {
            if (ttri->isbad()){
               //cerr << "WARNING -- ttri isbad() true!!" << endl;
                continue;
            }
            p0 = (ttri->pt(0) - bbox.min())*(Vector(nx,ny,nz)/diag);
            p1 = (ttri->pt(1) - bbox.min())*(Vector(nx,ny,nz)/diag);
            p2 = (ttri->pt(2) - bbox.min())*(Vector(nx,ny,nz)/diag);
            Vector n = Cross((p2-p0),(p1-p0));
            n.normalize();
            polynormal[0] = n.x();
            polynormal[1] = n.y();
            polynormal[2] = n.z();
        }


	int sx, sy, sz, ex, ey, ez;
	calc_se(obj_bbox, bbox, diag, nx, ny, nz, sx, sy, sz, ex, ey, ez);
	for(int x=sx;x<=ex;x++){
	    for(int y=sy;y<=ey;y++){
		int idx=x*nynz+y*nz+sz;
		for(int z=sz;z<=ez;z++){
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
                          int cur=current[idx];
                          int pos=counts[idx*2]+cur;
                          grid[pos]=prims[i];
                          current[idx]++;
                       }
                       idx++;
                    }
                    else {
		       int cur=current[idx];
		       int pos=counts[idx*2]+cur;
		       grid[pos]=prims[i];
		       current[idx]++;
		       idx++;
                    }
		}
	    }
	}
    }
    //    cerr << "6/7 Filling grid took " << Time::currentSeconds()-time << " seconds\n";
    time=Time::currentSeconds();
    for(int i=0;i<ngrid;i++){
	if(current[i] != counts[i*2+1]){
	    cerr << "OOPS!\n";
	    cerr << "current: " << current[i] << '\n';
	    cerr << "counts: " << counts[i*2+1] << '\n';
	    exit(1);
	}
    }
    for(int i=0;i<total;i++){
	if(!grid[i]){
	    cerr << "OOPS: grid[" << i << "]==0!\n";
	    exit(1);
	}
    }
    //    cerr << "7/7 Verifying grid took " << Time::currentSeconds()-time << " seconds\n";
    //    cerr << "Done building grid\n";
}

void Grid::intersect(Ray& ray, HitInfo& hit,
		    DepthStats* st, PerProcessorContext* ppc)
{
  if (ray.already_tested[0] == this ||
      ray.already_tested[1] == this ||
      ray.already_tested[2] == this ||
      ray.already_tested[3] == this)
    return;
  else {
    ray.already_tested[3] = ray.already_tested[2];
    ray.already_tested[2] = ray.already_tested[1];
    ray.already_tested[1] = ray.already_tested[0];
    ray.already_tested[0] = this;
  }
  const Vector dir(ray.direction()+Vector(1.e-8,1.e-8,1.e-8));
  const Point orig(ray.origin());
  Point min(bbox.min());
  Point max(bbox.max());
  Vector diag(bbox.diagonal());
  double MIN, MAX;
  double inv_dir=1./dir.x();
  int didx_dx, dix_dx;
  int ddx;
  int nynz=ny*nz;
  if(dir.x() > 0){
    MIN=inv_dir*(min.x()-orig.x());
    MAX=inv_dir*(max.x()-orig.x());
    didx_dx=nynz;
    dix_dx=1;
    ddx=1;
  } else {
    MIN=inv_dir*(max.x()-orig.x());
    MAX=inv_dir*(min.x()-orig.x());
    didx_dx=-nynz;
    dix_dx=-1;
    ddx=0;
  }	
  double y0, y1;
  int didx_dy, diy_dy;
  int ddy;
  if(dir.y() > 0){
    double inv_dir=1./dir.y();
    y0=inv_dir*(min.y()-orig.y());
    y1=inv_dir*(max.y()-orig.y());
    didx_dy=nz;
    diy_dy=1;
    ddy=1;
  } else {
    double inv_dir=1./dir.y();
    y0=inv_dir*(max.y()-orig.y());
    y1=inv_dir*(min.y()-orig.y());
    didx_dy=-nz;
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
  int didx_dz, diz_dz;
  int ddz;
  if(dir.z() > 0){
    double inv_dir=1./dir.z();
    z0=inv_dir*(min.z()-orig.z());
    z1=inv_dir*(max.z()-orig.z());
    didx_dz=1;
    diz_dz=1;
    ddz=1;
  } else {
    double inv_dir=1./dir.z();
    z0=inv_dir*(max.z()-orig.z());
    z1=inv_dir*(min.z()-orig.z());
    didx_dz=-1;
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
  Vector s((p-min)/diag);
  int ix=(int)(s.x()*nx);
  int iy=(int)(s.y()*ny);
  int iz=(int)(s.z()*nz);
  if(ix>=nx)
    ix--;
  if(iy>=ny)
    iy--;
  if(iz>=nz)
    iz--;
  if(ix<0)
    ix++;
  if(iy<0)
    iy++;
  if(iz<0)
    iz++;
  
  
  //      hit.hit(this,t);
  //      return;
  
  
  int idx=ix*nynz+iy*nz+iz;
  
  double next_x, next_y, next_z;
  double dtdx, dtdy, dtdz;
  double x=min.x()+diag.x()*double(ix+ddx)/nx;
  next_x=(x-orig.x())/dir.x();
  dtdx=dix_dx*diag.x()/nx/dir.x();
  double y=min.y()+diag.y()*double(iy+ddy)/ny;
  next_y=(y-orig.y())/dir.y();
  dtdy=diy_dy*diag.y()/ny/dir.y();
  double z=min.z()+diag.z()*double(iz+ddz)/nz;
  next_z=(z-orig.z())/dir.z();
  dtdz=diz_dz*diag.z()/nz/dir.z();
  for(;;){
    int n=counts[idx*2+1];
    int s=counts[idx*2];
    for(int i=0;i<n;i++)
      grid[s+i]->intersect(ray, hit, st, ppc);
    if(next_x < next_y && next_x < next_z){
      // Step in x...
      t=next_x;
      next_x+=dtdx;
      ix+=dix_dx;
      idx+=didx_dx;
      if(ix<0 || ix>=nx)
	break;
    } else if(next_y < next_z){
      t=next_y;
      next_y+=dtdy;
      iy+=diy_dy;
      idx+=didx_dy;
      if(iy<0 || iy>=ny)
	break;
    } else {
      t=next_z;
      next_z+=dtdz;
      iz+=diz_dz;
      idx+=didx_dz;
      if(iz<0 || iz >=nz)
	break;
    }
    if(hit.min_t < t)
      break;
  }
}

void Grid::light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			   DepthStats* st, PerProcessorContext* ppc)
{
  if (ray.already_tested[0] == this ||
      ray.already_tested[1] == this ||
      ray.already_tested[2] == this ||
      ray.already_tested[3] == this)
    return;
  else {
    ray.already_tested[3] = ray.already_tested[2];
    ray.already_tested[2] = ray.already_tested[1];
    ray.already_tested[1] = ray.already_tested[0];
    ray.already_tested[0] = this;
  }

  const Vector dir(ray.direction()+Vector(1.e-8,1.e-8,1.e-8));
  const Point orig(ray.origin());
  Point min(bbox.min());
  Point max(bbox.max());
  Vector diag(bbox.diagonal());
  double MIN, MAX;
  double inv_dir=1./dir.x();
  int didx_dx, dix_dx;
  int ddx;
  int nynz=ny*nz;
  if(dir.x() > 0){
    MIN=inv_dir*(min.x()-orig.x());
    MAX=inv_dir*(max.x()-orig.x());
    didx_dx=nynz;
    dix_dx=1;
    ddx=1;
  } else {
    MIN=inv_dir*(max.x()-orig.x());
    MAX=inv_dir*(min.x()-orig.x());
    didx_dx=-nynz;
    dix_dx=-1;
    ddx=0;
  }	
  double y0, y1;
  int didx_dy, diy_dy;
  int ddy;
  if(dir.y() > 0){
    double inv_dir=1./dir.y();
    y0=inv_dir*(min.y()-orig.y());
    y1=inv_dir*(max.y()-orig.y());
    didx_dy=nz;
    diy_dy=1;
    ddy=1;
  } else {
    double inv_dir=1./dir.y();
    y0=inv_dir*(max.y()-orig.y());
    y1=inv_dir*(min.y()-orig.y());
    didx_dy=-nz;
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
  int didx_dz, diz_dz;
  int ddz;
  if(dir.z() > 0){
    double inv_dir=1./dir.z();
    z0=inv_dir*(min.z()-orig.z());
    z1=inv_dir*(max.z()-orig.z());
    didx_dz=1;
    diz_dz=1;
    ddz=1;
  } else {
    double inv_dir=1./dir.z();
    z0=inv_dir*(max.z()-orig.z());
    z1=inv_dir*(min.z()-orig.z());
    didx_dz=-1;
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
  Vector s((p-min)/diag);
  int ix=(int)(s.x()*nx);
  int iy=(int)(s.y()*ny);
  int iz=(int)(s.z()*nz);
  if(ix>=nx)
    ix--;
  if(iy>=ny)
    iy--;
  if(iz>=nz)
    iz--;
  if(ix<0)
    ix++;
  if(iy<0)
    iy++;
  if(iz<0)
    iz++;
  
  
  //      hit.hit(this,t);
  //      return;
  
  
  int idx=ix*nynz+iy*nz+iz;
  
  double next_x, next_y, next_z;
  double dtdx, dtdy, dtdz;
  double x=min.x()+diag.x()*double(ix+ddx)/nx;
  next_x=(x-orig.x())/dir.x();
  dtdx=dix_dx*diag.x()/nx/dir.x();
  double y=min.y()+diag.y()*double(iy+ddy)/ny;
  next_y=(y-orig.y())/dir.y();
  dtdy=diy_dy*diag.y()/ny/dir.y();
  double z=min.z()+diag.z()*double(iz+ddz)/nz;
  next_z=(z-orig.z())/dir.z();
  dtdz=diz_dz*diag.z()/nz/dir.z();
  for(;;){
    int n=counts[idx*2+1];
    int s=counts[idx*2];
    for(int i=0;i<n;i++) {
      grid[s+i]->light_intersect(ray, hit, atten, st, ppc);
      if (hit.was_hit) return;
    }
    if(next_x < next_y && next_x < next_z){
      // Step in x...
      t=next_x;
      next_x+=dtdx;
      ix+=dix_dx;
      idx+=didx_dx;
      if(ix<0 || ix>=nx)
	break;
    } else if(next_y < next_z){
      t=next_y;
      next_y+=dtdy;
      iy+=diy_dy;
      idx+=didx_dy;
      if(iy<0 || iy>=ny)
	break;
    } else {
      t=next_z;
      next_z+=dtdz;
      iz+=diz_dz;
      idx+=didx_dz;
      if(iz<0 || iz >=nz)
	break;
    }
    if(hit.min_t < t)
      break;
  }
}

void Grid::softshadow_intersect(Light* light, Ray& ray,
				HitInfo& hit, double dist, Color& atten,
				DepthStats* st, PerProcessorContext* ppc)
{
  if (ray.already_tested[0] == this ||
      ray.already_tested[1] == this ||
      ray.already_tested[2] == this ||
      ray.already_tested[3] == this)
    return;
  else {
    ray.already_tested[3] = ray.already_tested[2];
    ray.already_tested[2] = ray.already_tested[1];
    ray.already_tested[1] = ray.already_tested[0];
    ray.already_tested[0] = this;
  }

  const Vector dir(ray.direction()+Vector(1.e-8,1.e-8,1.e-8));
  const Point orig(ray.origin());
  Point min(bbox.min());
  Point max(bbox.max());
  Vector diag(bbox.diagonal());
  double MIN, MAX;
  double inv_dir=1./dir.x();
  int didx_dx, dix_dx;
  int ddx;
  int nynz=ny*nz;
  if(dir.x() > 0){
    MIN=inv_dir*(min.x()-orig.x());
    MAX=inv_dir*(max.x()-orig.x());
    didx_dx=nynz;
    dix_dx=1;
    ddx=1;
  } else {
    MIN=inv_dir*(max.x()-orig.x());
    MAX=inv_dir*(min.x()-orig.x());
    didx_dx=-nynz;
    dix_dx=-1;
    ddx=0;
  }	
  double y0, y1;
  int didx_dy, diy_dy;
  int ddy;
  if(dir.y() > 0){
    double inv_dir=1./dir.y();
    y0=inv_dir*(min.y()-orig.y());
    y1=inv_dir*(max.y()-orig.y());
    didx_dy=nz;
    diy_dy=1;
    ddy=1;
  } else {
    double inv_dir=1./dir.y();
    y0=inv_dir*(max.y()-orig.y());
    y1=inv_dir*(min.y()-orig.y());
    didx_dy=-nz;
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
  int didx_dz, diz_dz;
  int ddz;
  if(dir.z() > 0){
    double inv_dir=1./dir.z();
    z0=inv_dir*(min.z()-orig.z());
    z1=inv_dir*(max.z()-orig.z());
    didx_dz=1;
    diz_dz=1;
    ddz=1;
  } else {
    double inv_dir=1./dir.z();
    z0=inv_dir*(max.z()-orig.z());
    z1=inv_dir*(min.z()-orig.z());
    didx_dz=-1;
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
  Vector s((p-min)/diag);
  int ix=(int)(s.x()*nx);
  int iy=(int)(s.y()*ny);
  int iz=(int)(s.z()*nz);
  if(ix>=nx)
    ix--;
  if(iy>=ny)
    iy--;
  if(iz>=nz)
    iz--;
  if(ix<0)
    ix++;
  if(iy<0)
    iy++;
  if(iz<0)
    iz++;
  
  
  //      hit.hit(this,t);
  //      return;
  
  
  int idx=ix*nynz+iy*nz+iz;
  
  double next_x, next_y, next_z;
  double dtdx, dtdy, dtdz;
  double x=min.x()+diag.x()*double(ix+ddx)/nx;
  next_x=(x-orig.x())/dir.x();
  dtdx=dix_dx*diag.x()/nx/dir.x();
  double y=min.y()+diag.y()*double(iy+ddy)/ny;
  next_y=(y-orig.y())/dir.y();
  dtdy=diy_dy*diag.y()/ny/dir.y();
  double z=min.z()+diag.z()*double(iz+ddz)/nz;
  next_z=(z-orig.z())/dir.z();
  dtdz=diz_dz*diag.z()/nz/dir.z();
  for(;;){
    int n=counts[idx*2+1];
    int s=counts[idx*2];
    for(int i=0;i<n;i++) {
      grid[s+i]->softshadow_intersect(light, ray, hit, dist, atten, st, ppc);
      if (hit.was_hit) return;
    }
    if(next_x < next_y && next_x < next_z){
      // Step in x...
      t=next_x;
      next_x+=dtdx;
      ix+=dix_dx;
      idx+=didx_dx;
      if(ix<0 || ix>=nx)
	break;
    } else if(next_y < next_z){
      t=next_y;
      next_y+=dtdy;
      iy+=diy_dy;
      idx+=didx_dy;
      if(iy<0 || iy>=ny)
	break;
    } else {
      t=next_z;
      next_z+=dtdz;
      iz+=diz_dz;
      idx+=didx_dz;
      if(iz<0 || iz >=nz)
	break;
    }
    if(hit.min_t < t)
      break;
  }
}

void Grid::animate(double t, bool&changed)
{
    obj->animate(t, changed);
}

void Grid::collect_prims(Array1<Object*>& prims)
{
    prims.add(this);
}

void Grid::compute_bounds(BBox& bbox, double offset)
{
    obj->compute_bounds(bbox, offset);
}

Vector Grid::normal(const Point&, const HitInfo&)
{
  cerr << "Error: Grid normal should not be called!\n";
  return Vector(0,0,0);
}

const int GRID_VERSION = 1;

void 
Grid::io(SCIRun::Piostream &str)
{
  str.begin_class("Grid", GRID_VERSION);
  Object::io(str);
  SCIRun::Pio(str, obj);
  SCIRun::Pio(str, bbox);
  SCIRun::Pio(str, nx);
  SCIRun::Pio(str, ny);
  SCIRun::Pio(str, nz);
  int ngrid = nx*ny*nz;
  //Pio(str, grid);
  //Pio(str, counts);
  SCIRun::Pio(str, nsides);
  if (str.reading()) {
    set_matl(new LambertianMaterial(Color(1,0,0)));
    counts=new int[2*ngrid];
  }
  // Read in the counts...
  for(int i=0;i<ngrid*2;i++) {
    SCIRun::Pio(str, counts[i]);
  }
  int total=0;
  for(int i=0;i<ngrid;i++){
    total+=counts[i*2+1];
  }

  if (str.reading()) {
    grid=new Object*[total];
  }
  // Read in grid.
  for(int j=0;j<total;j++) {
    SCIRun::Pio(str, grid[j]);
  }
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::Grid*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Grid::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Grid*>(pobj);
  }
}
} // end namespace SCIRun
