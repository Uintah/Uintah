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

#ifndef SCI_Geom_Tex_h
#define SCI_Geom_Tex_h 1

/*
 * This file contains code for managing texture stuff...
 * Peter-Pike Sloan
 */

#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/BBox.h>
#include <Core/Containers/Handle.h>
#include <Core/Persistent/Persistent.h>

namespace SCIRun {


struct oCube;

struct oCubeEdge {
  int id;
  int nodes[2];
  int faces[2];

  double start,end; // when edge is visible...

  int is_visible;
  int flip;

  Vector v;
  Point p0;

  double fac; // equate steps on view with steps for edge

  void Classify(double val) { is_visible = (((val >=start)&&(val<end))); }


  int IsEdge(int v0, int v1) 
    {
      return ( ( (nodes[0] == v0) && (nodes[1] == v1)) ||
	      ( (nodes[1] == v0) && (nodes[0] == v1)) );
    }

  void InitView(Vector& view, double d); // plane equation

  void StartPts(double dist, oCube*, int first=0);
};

struct oCubeFace {
  int id;
  int edges[4];   // edges that are connected to this face
  int nbrs[4];    // corresponding neighbors
  int pts[4];     // pts that are connected to this edge...

  int generation; // used for traversing this face...

  void Connect(oCube*);

  void Recurse(int eid, double dist, oCube*);
};

struct oCube {
  Point pts[8]; // 8 pts of the oCube
  
  Point centroid; // center of the oCube...

  BBox bbox; // bounding box...

  oCubeEdge edges[12];
  oCubeFace faces[6];

  void Init(Point &p1, Point& p2);

  void FixConnect();

  void BuildEdge(int startf, int v0, int v1);

  int curgeneration; // current generation

  void SetView(Vector& view, Point& eye);

  void EmitStuff(double dist); // distance along given plane...
};

// this should only be the child of a 3D texture thingy

// this is for tiled rendering...

struct VolChunk {
  oCube      cube; // might not be square if on boundary...

  unsigned char *rawVol; // 1 chan, real voxels...

  void CreateTexInfo(void); // use glTexGen...

  int flags;    // info on chunk...
  int nx,ny,nz; // valid voxel dims - could be subset...
};

class GeomTexVolRender: public GeomObj {
  oCube         myCube;  // this is the global one...

  int nslice;
  float s_alpha;

  void CreateTexMatrix3D(void); // creates texture matrix...
  
  unsigned char *vol3d;
  int nx,ny,nz; // size of this volume...
  int sx,sy,sz;
  unsigned int id;  // for 1d maps - so you can cache the volume...
  unsigned int id2; // rgba volume...

  int usemip;

  GeomObj *other;

  int doOther;

public:
    GeomTexVolRender(Point pts[8]); // 8 pts that define a cube

    GeomTexVolRender(Point& p1, Point& p2);

    void SetNumSlices(int ns) { nslice = ns; };

  void SetMode(int mode) { usemip = mode; };

  void Clear();

  void SetAlpha(double v) { s_alpha = v; };

  void SetVol(unsigned char *vol, int rnx, int rny, int rnz) {
    sx = nx = rnx; sy = ny = rny; sz = nz = rnz;
    vol3d = vol;
    map2d = vol; // force a rebuild...
  }

  void SetOther(GeomObj *o) { other = o;};
    
  void SetDoOther(int v) { doOther = v; }

  void SubVol(int x, int y, int z) { sx = x; sy = y; sz = z; };

  unsigned char *map1d; // use it if it is set...
  unsigned char *map2d; // if this is set, reload the volume and clear

  void SetQuantStuff(vector< Vector > &, int *, int sz);

  int *quantnvol;       // quantized normal volume (if neccesary)

  vector< Vector >  quantnrms;   // quantized normals

  vector< float >   LUT;         // for specular part of lighting model

  unsigned char     *quantclrs;  // lit quantized normals...

  int               quantsz;

  int               qalwaysupdate;
  int               qupdate;
  
  unsigned char *rgbavol; // 4 channel - for illumination

  Vector EyeLightVec;     // light vector in eyespace...

  Vector vvec;            // view vector for illumination
  Vector lvec;            // light vector
  Vector hvec;            // halfway vector

  float  ambC,difC,specC; // coefs for illumination (amb/dif/spec)

  int specP;              // specular power for illumination

  int np;                 // procs for shading...

  virtual ~GeomTexVolRender();
  
  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);
  
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  virtual void io(Piostream&);
  static PersistentTypeID type_id;  
  virtual bool saveobj(std::ostream&, const string& format, GeomSave*);
};

} // End namespace SCIRun


#endif

