/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  Tex.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <Dataflow/Modules/Render/Tex.h>

#include <Core/Malloc/Allocator.h>
#include <strings.h>

#include <Core/Geometry/BBox.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/NotFinished.h>
#include <iostream>
using std::cerr;
using std::ostream;
#include <GL/gl.h>

#define USELUMIALPH 1

namespace SCIRun {


// these are some functions for the cube primitive...

void oCube::Init(Point& pmin, Point& pmax)
{
  pts[0] = pmin;                              // ---
  pts[1] = Point(pmin.x(),pmin.y(),pmax.z()); // --+
  pts[2] = Point(pmin.x(),pmax.y(),pmax.z()); // -++
  pts[3] = Point(pmin.x(),pmax.y(),pmin.z()); // -+-
  pts[4] = Point(pmax.x(),pmax.y(),pmin.z()); // ++-
  pts[5] = Point(pmax.x(),pmin.y(),pmax.z()); // +-+
  pts[6] = Point(pmax.x(),pmin.y(),pmin.z()); // +--
  pts[7] = pmax;                              // +++

  bbox.reset(); // reset the bounding box...

  bbox.extend(pmin);
  bbox.extend(pmax);

  // now do the faces...

  faces[0].pts[0] = 0;
  faces[0].pts[1] = 1;
  faces[0].pts[2] = 2;
  faces[0].pts[3] = 3;

  faces[1].pts[0] = 1;
  faces[1].pts[1] = 2;
  faces[1].pts[2] = 5;
  faces[1].pts[3] = 7;

  faces[2].pts[0] = 5;
  faces[2].pts[1] = 7;
  faces[2].pts[2] = 4;
  faces[2].pts[3] = 6;

  faces[3].pts[0] = 0;
  faces[3].pts[1] = 3;
  faces[3].pts[2] = 4;
  faces[3].pts[3] = 6;

  faces[4].pts[0] = 1;
  faces[4].pts[1] = 5;
  faces[4].pts[2] = 6;
  faces[4].pts[3] = 0;

  faces[5].pts[0] = 3;
  faces[5].pts[1] = 2;
  faces[5].pts[2] = 7;
  faces[5].pts[3] = 4;

  int i;
  for(i=0;i<12;i++) { // init everything to 0...
    edges[i].nodes[0] = edges[i].nodes[1] = 
      edges[i].faces[0] = edges[i].faces[1] = -1;
    edges[i].id = i;
  }

  for(i=0;i<6;i++) {
    for(int j=0;j<4;j++) {
      faces[i].edges[j] = faces[i].nbrs[j] = -1;
    }
    faces[i].id = i;
  }

  FixConnect();

  // now build the edge equations

  for(i=0;i<12;i++) {
    edges[i].p0 = pts[edges[i].nodes[0]];
    edges[i].v = (pts[edges[i].nodes[1]] - pts[edges[i].nodes[0]]);

    if (edges[i].nodes[0] == -1) {
      cerr << i << " Bad Edge!\n";
    }

  }
}

void oCubeEdge::InitView(Vector& view, double d)
{
  double np0 = Dot(view,p0);
  start = np0 + d;

  double nv = Dot(view,v);
  flip=0;
  if (Abs(nv) < 0.0001) { // are the perpendicular?
    end = start;
    fac = 1.0; // doesn't matter...
//    cerr << id << " " << start << " " << end << endl;
    return;
  } else {
    end = start + nv;
   }

  if (start > end) {
    double tmp = end;
    end = start;
    start = tmp;
    flip=1;
  }
  
  fac = 1.0/(end-start);
//  cerr << id << " " << start << " " << end << endl;
}

void oCube::SetView(Vector& view, Point& eye)
{
  double d = -Dot(view,eye);

  int i;
  for(i=0;i<12;i++)
    edges[i].InitView(view,d);

  for(i=0;i<6;i++)
    faces[i].generation = 0;
  
  curgeneration = 1; 
}

// this builds the edge and face connectivty from face info...

void oCube::FixConnect()
{
  // run through each face trying to build edge stuff...

  for(int i=0;i<6;i++) {
    faces[i].Connect(this);
  }
}

void oCubeFace::Connect(oCube *oCube)
{
  for(int i=0;i<4;i++) {
    for(int j=0;j<4;j++) {
      if ((j != i)) {
	oCube->BuildEdge(id,pts[i],pts[j]);
      }
    }
  }
}

void oCube::BuildEdge(int startf, int v0, int v1)
{
  // first make sure this edge doesn't already exist...
  int i;
  for(i=0;i<12;i++) {
    if (edges[i].IsEdge(v0,v1))
      return; // it already is there
  }

  // now find a face besides startf that has both of these vertices

  for(i=0;i<6;i++) {
    if (i != startf) {
      int nmatch=0;
      for(int j=0;j<4;j++) {
	if ((faces[i].pts[j] == v0) ||
	    (faces[i].pts[j] == v1)) {
	  nmatch++;
	}
      }

      if (nmatch == 2) { // this is legit...
	int starti=0,endi=0;
	while(faces[startf].edges[starti] != -1)
	  starti++;
	while(faces[i].edges[endi] != -1)
	  endi++;

	faces[startf].nbrs[starti] = i;
	faces[i].nbrs[endi] = startf;

	// now find a empty edge
	for(int k=0;k<12;k++) {
	  if (edges[k].nodes[0] == -1) {
	    faces[startf].edges[starti] = k;
	    faces[i].edges[endi] = k;

	    edges[k].nodes[0] = v0;
	    edges[k].nodes[1] = v1;
	    
	    edges[k].faces[0] = startf;
	    edges[k].faces[1] = i;
	    return;
	  }
	}

	cerr << "Error! should have found a empty edge!\n";

      }

    }
  }
}

Persistent* make_GeomTexVolRender()
{
  Point p0(0,0,0),p1(1,1,1);
    return scinew GeomTexVolRender(p0,p1);
}

PersistentTypeID GeomTexVolRender::type_id("GeomTexVolRender", "GeomObj",
					   make_GeomTexVolRender);
					   

GeomTexVolRender::GeomTexVolRender(Point /*pts*/[8])
  : nslice(64),
    s_alpha(0.5),
    vol3d(0),
    id(0),
    id2(0),
    usemip(1),
    other(0),
    doOther(0),
    map1d(0),
    map2d(0),
    quantnvol(0),
    quantclrs(0),
    quantsz(0),
    qalwaysupdate(1),
    qupdate(1),
    rgbavol(0)
{
#if 0
  for(int i=0;i<8;i++)
    cube[i] = pts[i];
#endif

  cerr << "Error - this constructor is not defined!\n";
}

GeomTexVolRender::GeomTexVolRender(Point& pmin, Point& pmax)
  : nslice(256),
    s_alpha(1.0),
    vol3d(0),
    id(0),
    id2(0),
    usemip(1),
    other(0),
    doOther(0),
    map1d(0),
    map2d(0),
    quantnvol(0),
    quantclrs(0),
    quantsz(0),
    qalwaysupdate(1),
    qupdate(1),
    rgbavol(0),
    ambC(0.2),
    difC(0.8),
    specC(0.3),
    specP(5),
    np(8)
{
  myCube.Init(pmin,pmax);

  Vector diff = pmax-pmin;
  myCube.centroid = (pmin + diff*0.5);

  vvec = Vector(0,0,0); // init...

  EyeLightVec = Vector(1,0,1);
  EyeLightVec.normalize();

}

GeomTexVolRender::~GeomTexVolRender()
{

}

GeomObj* GeomTexVolRender::clone()
{
    cerr << "GeomTexVolRender::clone not implemented!\n";
    return 0;
}

const int MAXSIZE = 1<<(2*9);

void GeomTexVolRender::SetQuantStuff(vector< Vector > &vecs, int *v, int sz)
{
  quantsz = sz;
  if (quantnrms.size() == 0) { // init stuff...
    quantnrms.resize(MAXSIZE);
#ifndef USELUMIALPH
    quantclrs = scinew unsigned char[MAXSIZE*4];
    rgbavol = scinew unsigned char[nx*ny*nz*4];
#else
    quantclrs = scinew unsigned char[MAXSIZE*2];
    rgbavol = scinew unsigned char[nx*ny*nz*2];
#endif
  }

  for(unsigned int i=0;i<vecs.size();i++) {
    quantnrms[i] = vecs[i];
  }

  quantnvol = v; // that should be it...
}
void GeomTexVolRender::get_bounds(BBox& bound)
{	
  for(int i=0;i<8;i++)
    bound.extend(myCube.pts[i]);
}

void GeomTexVolRender::io(Piostream&)
{
  // do nothing
}

void GeomTexVolRender::CreateTexMatrix3D(void) {
  //  glMatrixMode(GL_TEXTURE);
  //  glLoadIdentity();

  // you have to translate and scale 3 space points 
  
  Point min,max;
  Vector diag;

  //tex->sf->get_bounds(min,max);  // this is the extent of the volume...

  min = myCube.pts[0];
  max = myCube.pts[7];

  diag = max-min;

  // do the translate, then the scale...

  //  glScaled(1.0/diag.x(),1.0/diag.y(),1.0/diag.z()); // this should scale everything...
  //  glTranslated(-min.x(),-min.y(),-min.z());  // now min goes to 0...

  // use texGen instead...

  glTexGend(GL_S,GL_TEXTURE_GEN_MODE,GL_OBJECT_LINEAR);
  glTexGend(GL_T,GL_TEXTURE_GEN_MODE,GL_OBJECT_LINEAR);
  glTexGend(GL_R,GL_TEXTURE_GEN_MODE,GL_OBJECT_LINEAR);
  glTexGend(GL_Q,GL_TEXTURE_GEN_MODE,GL_OBJECT_LINEAR);

  double splane[4]={0,0,0,0};

  splane[0] = 1.0/diag.x()*(sx*1.0/nx);
  splane[3] = -min.x()/diag.x();

  double tplane[4]={0,0,0,0};

  tplane[1] = 1.0/diag.y()*(sy*1.0/ny);
  tplane[3] = -min.y()/diag.y();

  double rplane[4]={0,0,0,0};

  rplane[2] = 1.0/diag.z()*(sz*1.0/nz);
  rplane[3] = -min.z()/diag.z();

  double qplane[4]={0,0,0,1};

  glTexGendv(GL_S,GL_OBJECT_PLANE,splane);
  glTexGendv(GL_T,GL_OBJECT_PLANE,tplane);
  glTexGendv(GL_R,GL_OBJECT_PLANE,rplane);
  glTexGendv(GL_Q,GL_OBJECT_PLANE,qplane);
}


void oCubeEdge::StartPts(double dist, oCube* owner, int first)
{
  Point p;
  if (flip) {
    p = p0 + (v*(end-dist))*fac;
  } else {
    p = p0 + (v*(dist-start))*fac;
  }

  //glTexCoord3f(p.x(),p.y(),p.z());
  glVertex3f(p.x(),p.y(),p.z());

  if (owner->faces[faces[0]].generation != owner->curgeneration) {
    owner->faces[faces[0]].generation = owner->curgeneration;
    if (first && 
	(owner->faces[faces[1]].generation != owner->curgeneration)) {
      owner->faces[faces[1]].generation = owner->curgeneration;
    }
    owner->faces[faces[0]].Recurse(id,dist,owner);
    return;  
  }

  if (owner->faces[faces[1]].generation != owner->curgeneration) {
    owner->faces[faces[1]].generation = owner->curgeneration;
    if (first && 
	(owner->faces[faces[0]].generation != owner->curgeneration)) {
      owner->faces[faces[0]].generation = owner->curgeneration;
    }
    owner->faces[faces[1]].Recurse(id,dist,owner);
    return;
  }
}

void oCubeFace::Recurse(int eid, double dist, oCube *owner)
{
  for(int i=0;i<4;i++) {
    if ((edges[i] != eid) &&
	owner->edges[edges[i]].is_visible &&
	(owner->faces[nbrs[i]].generation != owner->curgeneration)) {
      owner->edges[edges[i]].StartPts(dist,owner);
      return;
    }
  }
}

void oCube::EmitStuff(double dist)
{
  int i;
  for(i=0;i<12;i++) {
    edges[i].Classify(dist);
  }

  // now just find the first valid one...
  for(i=0;i<12;i++) {
    if (edges[i].is_visible) {
      edges[i].StartPts(dist, this);
      i = 14;
    }
  }

  curgeneration++;
}

#define DO_SMART_SLICE 1

//const double LIGHTCOMP=0.99939083; // 2 degrees...

void GeomTexVolRender::draw(DrawInfoOpenGL* di, Material *m, double time)
{
  if (usemip == 3) return;	// this is the "don't draw" option!
  pre_draw(di, m, 0);
#ifdef __sgi
  glEnable(GL_TEXTURE_3D_EXT);
#endif
  glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,
	    GL_MODULATE);
  glColor4f(1,1,1,1); // set to all white for modulation
  
  if (map1d && !quantnvol) {
#ifdef __sgi
//    cerr << "Using Lookup!\n";
    glEnable(GL_TEXTURE_COLOR_TABLE_SGI);
    glColorTableSGI(GL_TEXTURE_COLOR_TABLE_SGI,
		    GL_RGBA,
		    256, // try larger sizes?
		    GL_RGBA,  // need an alpha value...
		    GL_UNSIGNED_BYTE, // try shorts...
		    map1d);
#endif
  }

  Vector vx,vy,vz;
  
  Point origin,o2;
  
  double model_mat[16];
  
  glGetDoublev(GL_MODELVIEW_MATRIX,model_mat);  

  vx = Vector(model_mat[0*4+0],model_mat[1*4+0],model_mat[2*4+0]);
  vy = Vector(model_mat[0*4+1],model_mat[1*4+1],model_mat[2*4+1]);
  vz = Vector(model_mat[0*4+2],model_mat[1*4+2],model_mat[2*4+2]);
 
  origin = Point(model_mat[0*4+3],model_mat[1*4+3],model_mat[2*4+3]);
  o2 = Point(model_mat[3*4+0],model_mat[3*4+1],model_mat[3*4+2]);

  origin = -o2; // is this right???

  Vector vs[3];

  vs[0] = vx.normal();
  vs[1] = vy.normal();
  vs[2] = -vz.normal();

  double min[3],max[3];
  if (!id || map2d) {
    if (!id) {
      glGenTextures(1,&id);
    }
      
    cerr << "Loading texture...\n";

#ifdef __sgi
    glBindTexture(GL_TEXTURE_3D_EXT,id);
    // assume environment is outside!
    glTexParameterf(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_R_EXT, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    
    glTexImage3DEXT(GL_TEXTURE_3D_EXT,0,
		    GL_INTENSITY8_EXT,
		    nx,ny,nz,0,
		    GL_RED,GL_UNSIGNED_BYTE,vol3d);
#endif
    
    
    map2d = 0; // clear it out...
  } else { // jost load texture object...
#ifdef __sgi
    glBindTexture(GL_TEXTURE_3D_EXT,id);
#endif
  }

  glEnable(GL_TEXTURE_GEN_S);
  glEnable(GL_TEXTURE_GEN_T);
  glEnable(GL_TEXTURE_GEN_R);
  glEnable(GL_TEXTURE_GEN_Q);
  CreateTexMatrix3D();
  
  if (other && doOther) {

      other->draw(di,m,time);
  } else {
      // now do volume rendering...
  
      int i;
      for(i=0;i<3;i++) {
	  min[i] = 1000000;
	  max[i] = -1000000;
      }

      for(i=0;i<8;i++) {
	  for(int j=0;j<3;j++) {
	      double val = Dot((myCube.pts[i]-origin),vs[j]);
	      if (val < min[j])
		  min[j] = val;
	      if (val > max[j])
		  max[j] = val;
	  }
      }
  
      // now just whip out the planes...
  
      glEnable(GL_BLEND);
      //  if (quantnvol)
      if (usemip != 2)
	  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
      else {
#ifdef __sgi
	  glBlendEquationEXT(GL_MAX_EXT);
#endif
	  glBlendFunc(GL_ONE,GL_ONE);
      }
      //  else 
      //glBlendFunc(GL_ONE,GL_ONE_MINUS_SRC_ALPHA);

      glColor4f(1,1,1,s_alpha);
  
      myCube.SetView(vs[2],origin);
  
      float offset = min[2] + (max[2]-min[2])/(nslice*2);
      float scale = 1.0/(nslice-1.0)*(max[2]-min[2])/nslice*(nslice-1.0);
  
      if (usemip) {
    
	  for(i=0;i<nslice;i++) {
	      //    double cz = ((nslice-1)-i)/(nslice-1.0)*(max[2]-min[2]) + min[2];
	      double cz = ((nslice-1)-i)*scale + offset;
      
	      glBegin(GL_POLYGON);
	      myCube.EmitStuff(cz);
	      glEnd();
	  }
    
	  glDepthMask(GL_TRUE);
#ifdef __sgi
	  glBlendEquationEXT(GL_FUNC_ADD_EXT);
#endif
      }
      //  glMatrixMode(GL_TEXTURE);
      //glLoadIdentity();
  }
  glMatrixMode(GL_MODELVIEW);
  
#if 1
#ifdef __sgi
  glDisable(GL_TEXTURE_3D_EXT);
#endif

  glDisable(GL_TEXTURE_GEN_S);
  glDisable(GL_TEXTURE_GEN_T);
  glDisable(GL_TEXTURE_GEN_R);
  glDisable(GL_TEXTURE_GEN_Q);
#endif
  glDisable(GL_BLEND);

#ifdef __sgi
  if (map1d && !quantnvol)
    glDisable(GL_TEXTURE_COLOR_TABLE_SGI);
#endif
}

bool GeomTexVolRender::saveobj(ostream&, const string&, GeomSave*)
{
    NOT_FINISHED("GeomTexVolRender::savObj");
  return 0;
}

void GeomTexVolRender::Clear()
{
  bzero(vol3d,sizeof(char)*nx*ny*nz);
  cerr << "Cleared...\n";
  map2d = vol3d;
}

} // End namespace SCIRun
