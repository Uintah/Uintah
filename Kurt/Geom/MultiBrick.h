#ifndef MULTIBRICK_H
#define MULTIBRICK_H

#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Ray.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Transform.h>
#include <SCICore/Datatypes/ScalarFieldRGuchar.h>
#include <SCICore/Geom/GeomObj.h>
#include <GL/gl.h>
#include <GL/glu.h>

#include "SliceTable.h"
#include "Brick.h"
#include "VolumeOctree.h"

namespace SCICore {
namespace GeomSpace  {


using namespace SCICore::Geometry;
using namespace SCICore::Datatypes;
using Kurt::GeomSpace::VolumeOctree;
using Kurt::GeomSpace::Brick;
  using Kurt::GeomSpace::SliceTable;

class MultiBrick : public GeomObj
{
public:

  MultiBrick(int id, int slices, double alpha,
	     int maxdim, Point min, Point max,
	     int mode, bool debug,
	     int X, int Y, int Z,
	     const ScalarFieldRGuchar* tex,
	     const GLvoid* cmap);
  void SetMaxBrickSize(int x,int y,int z);
  void SetNSlices(int s) { slices = s; }
  void SetDrawLevel(int l){ drawLevel = l;}
  void SetPlaneIntersection(const Point& p){ widgetPoint = p; }
  void SetMode(int mode) { this->mode = mode;}
  void Reload() { reload = (unsigned char *)1;}
  void SetVol( const ScalarFieldRGuchar *tex );
  void SetColorMap( const GLvoid* map){ cmap = map;}
  void SetDebug( bool db){ debug = db;}
  void SetAlpha( double alpha){ this->alpha = alpha;}
  int getMaxLevel(){ return treeDepth; }
  int getMaxSize(){ return ((X < Y && X < Z)? X:((Y < Z)? Y:Z)); }

  MultiBrick(const MultiBrick&);
  ~MultiBrick();

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);

private:
  int X,Y,Z; // tex size
  const ScalarFieldRGuchar *tex;
  unsigned char *reload;
  const GLvoid *cmap;
  int mode;
  bool debug;
  int nodeId;
  int drawLevel;
  int slices;
  int treeDepth;
  int xmax, ymax, zmax;  // max brick sizes.
  Point min, max;
  Point widgetPoint;
  double alpha;
  double dx, dy, dz;
  bool drawWireFrame;
  
  VolumeOctree<Brick*>*  octree;
  void computeTreeDepth();
  VolumeOctree<Brick*>* buildOctree(Point min, Point max,
				       int xoff, int yoff, int zoff,
				       int xsize, int ysize, int zsize,
				       int level);
  VolumeOctree<Brick*>* buildBonTree(Point min, Point max,
				       int xoff, int yoff, int zoff,
				       int xsize, int ysize, int zsize,
				       int level, int id);

  static int traversalTable[27][8];
  void drawSlices();
  void drawOctree( const VolumeOctree<Brick*>* node,
		   const Ray&  viewRay);
  void drawBonTree(const VolumeOctree<Brick*>* node,
		   const Ray&  viewRay, const SliceTable& st);
  void drawTree( const VolumeOctree<Brick*>* node, bool useLevel,
		   const Ray&  viewRay, const SliceTable& st);
  void makeBrickData(int x, int y, int z,
		     int xsize, int ysize, int zsize,
		     int xoff, int yoff, int zoff,
		     Array3<unsigned char>*& bd);

  void makeLowResBrickData(int x, int y, int z,
		     int xsize, int ysize, int zsize,
		     int xoff, int yoff, int zoff,
		     int level, int& padx, int& pady, int& padz,
			   Array3<unsigned char>*& bd);
  void MultiBrick::BuildChild(int i, int id, Point min, Point mid, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level,  VolumeOctree<Brick*>* node);
};
}  // namespace GeomSpace
} // namespace SCICore


#endif
