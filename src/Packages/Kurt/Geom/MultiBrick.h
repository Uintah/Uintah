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

#include "Brick.h"
#include "VolumeOctree.h"

namespace SCICore {
namespace GeomSpace  {


using namespace SCICore::Geometry;
using namespace SCICore::Datatypes;
using Kurt::GeomSpace::VolumeOctree;
using Kurt::GeomSpace::Brick;

class MultiBrick : public GeomObj
{
public:

  MultiBrick(int id, int slices, double alpha,
	     int maxdim, Point min, Point max,
	     bool drawMIP,
	     int X, int Y, int Z,
	     const ScalarFieldRGuchar* tex,
	     const GLvoid* cmap);
  void SetMaxBrickSize(int x,int y,int z);
  void SetNSlices(int s) { slices = s; }

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
  const GLvoid *cmap;
  bool drawMIP;
  int slices;
  int xmax, ymax, zmax;  // max brick sizes.
  Point min, max;
  double alpha;
  VolumeOctree<Brick*>*  octree;
  VolumeOctree<Brick*>* buildOctree(Point min, Point max,
				       int xoff, int yoff, int zoff,
				       int xsize, int ysize, int zsize,
				       int level);
  VolumeOctree<Brick*>* buildBonTree(Point min, Point max,
				       int xoff, int yoff, int zoff,
				       int xsize, int ysize, int zsize,
				       int level);
  VolumeOctree<Brick*>* BonXYZ(Point min, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level);
  VolumeOctree<Brick*>* BonXY(Point min, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level);
  VolumeOctree<Brick*>* BonXZ(Point min, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level);
  VolumeOctree<Brick*>* BonYZ(Point min, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level);
  VolumeOctree<Brick*>* BonX(Point min, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level);
  VolumeOctree<Brick*>* BonY(Point min, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level);
  VolumeOctree<Brick*>* BonZ(Point min, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level);

  static int traversalTable[27][8];
  void drawSlices();
  void drawWireFrame();
  void drawOctree( const VolumeOctree<Brick*>* node,
		   const Ray&  viewRay);
  void drawBonTree( const VolumeOctree<Brick*>* node,
		   const Ray&  viewRay);
  void makeBrickData(int xsize, int ysize, int zsize,
		     int xoff, int yoff, int zoff,
		     Array3<unsigned char>*& bd);
  void BuildChild0(Point min, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level,  VolumeOctree<Brick*>* node);
  void  BuildChild1(Point min, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level,  VolumeOctree<Brick*>* node);
  void  BuildChild2(Point min, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level,  VolumeOctree<Brick*>* node);
  void  BuildChild3(Point min, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level,  VolumeOctree<Brick*>* node);
  void  BuildChild4(Point min, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level,  VolumeOctree<Brick*>* node);
  void  BuildChild5(Point min, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level,  VolumeOctree<Brick*>* node);
  void  BuildChild6(Point min, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level,  VolumeOctree<Brick*>* node);
  void  BuildChild7(Point min, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level,  VolumeOctree<Brick*>* node);
};



inline void MultiBrick::BuildChild0(Point min, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level,  VolumeOctree<Brick*>* node)
{
  Point pmax = min + Vector( (max.x() - min.x())* (double)X2/xsize, 
			  (max.y() - min.y())* (double)Y2/ysize,
			  (max.z() - min.z())* (double)Z2/zsize);
  Point pmin = min;
  node->SetChild(0, buildBonTree(pmin, pmax, xoff, yoff, zoff,
				X2, Y2, Z2, level+1));
}
inline void MultiBrick::BuildChild1(Point min, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level,  VolumeOctree<Brick*>* node)
{
  Point pmax = min + Vector( (max.x() - min.x())* (double)X2/xsize, 
			  (max.y() - min.y())* (double)Y2/ysize,
			  (max.z() - min.z())* (double)Z2/zsize);
  Point pmin = min;
  pmin.z(min.z() + (max.z() - min.z())* (double)(Z2-1)/zsize);
  pmax.z(max.z());
  node->SetChild(1, buildBonTree(pmin, pmax,
				 xoff, yoff, zoff + Z2 -1,
				 X2, Y2, zsize-Z2+1, level+1));
}
inline void MultiBrick::BuildChild2(Point min, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level,  VolumeOctree<Brick*>* node)
{
  Point pmax = min + Vector( (max.x() - min.x())* (double)X2/xsize, 
			  (max.y() - min.y())* (double)Y2/ysize,
			  (max.z() - min.z())* (double)Z2/zsize);
  Point pmin = min;
  pmin.y(min.y() + (max.y() - min.y())* (double)(Y2-1)/ysize);
  pmax.y(max.y());

  node->SetChild(2, buildBonTree(pmin, pmax,
				xoff, yoff + Y2 - 1, zoff,
				X2, ysize - Y2 + 1, Z2, level+1));

}
inline void MultiBrick::BuildChild3(Point min, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level,  VolumeOctree<Brick*>* node)
{
  Point pmax = min + Vector( (max.x() - min.x())* (double)X2/xsize, 
			  (max.y() - min.y())* (double)Y2/ysize,
			  (max.z() - min.z())* (double)Z2/zsize);
  Point pmin = min;
  pmin.y(min.y() + (max.y() - min.y())* (double)(Y2-1)/ysize);
  pmax.y(max.y());
  pmin.z(min.z() + (max.z() - min.z())* (double)(Z2-1)/zsize);
  pmax.z(max.z());

  node->SetChild(3, buildBonTree(pmin, pmax,
				xoff, yoff + Y2 - 1 , zoff + Z2 - 1,
				X2, ysize - Y2 + 1, zsize - Z2 + 1, level+1));

}
inline void MultiBrick::BuildChild4(Point min, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level,  VolumeOctree<Brick*>* node)
{
  Point pmax = min + Vector( (max.x() - min.x())* (double)X2/xsize, 
			  (max.y() - min.y())* (double)Y2/ysize,
			  (max.z() - min.z())* (double)Z2/zsize);
  Point pmin = min;
  pmin.x(min.x() + (max.x() - min.x())* (double)(X2-1)/xsize);
  pmax.x(max.x());

  node->SetChild(4, buildBonTree(pmin, pmax,
				xoff + X2 - 1, yoff, zoff,
				xsize - X2 + 1, Y2, Z2, level+1));

}
inline void MultiBrick::BuildChild5(Point min, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level,  VolumeOctree<Brick*>* node)
{
  Point pmax = min + Vector( (max.x() - min.x())* (double)X2/xsize, 
			  (max.y() - min.y())* (double)Y2/ysize,
			  (max.z() - min.z())* (double)Z2/zsize);
  Point pmin = min;
  pmin.x(min.x() + (max.x() - min.x())* (double)(X2-1)/xsize);
  pmax.x(max.x());
  pmin.z(min.z() + (max.z() - min.z())* (double)(Z2-1)/zsize);
  pmax.z(max.z());
  node->SetChild(5, buildBonTree(pmin, pmax,
				 xoff + X2 - 1, yoff, zoff +  Z2 - 1,
				 xsize - X2 + 1, Y2, zsize - Z2 + 1, level+1));


}
inline void MultiBrick::BuildChild6(Point min, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level,  VolumeOctree<Brick*>* node)
{
  Point pmax = min + Vector( (max.x() - min.x())* (double)X2/xsize, 
			  (max.y() - min.y())* (double)Y2/ysize,
			  (max.z() - min.z())* (double)Z2/zsize);
  Point pmin = min;
  pmin.x(min.x() + (max.x() - min.x())* (double)(X2-1)/xsize);
  pmax.x(max.x());
  pmin.y(min.y() + (max.y() - min.y())* (double)(Y2-1)/ysize);
  pmax.y(max.y());
  node->SetChild(6, buildBonTree(pmin, pmax,
				 xoff + X2 - 1, yoff + Y2 - 1, zoff,
				 xsize - X2 + 1, ysize - Y2 + 1, Z2, level+1));
}
inline void MultiBrick::BuildChild7(Point min, Point max,
			       int xoff, int yoff, int zoff,
			       int xsize, int ysize, int zsize,
			       int X2, int Y2, int Z2,
			       int level,  VolumeOctree<Brick*>* node)
{
  Point pmin = min + Vector( (max.x() - min.x())* (double)(X2-1)/xsize, 
			  (max.y() - min.y())* (double)(Y2-1)/ysize,
			  (max.z() - min.z())* (double)(Z2-1)/zsize);
  Point pmax = max;
  node->SetChild(7, buildBonTree(pmin, pmax,  xoff + X2 - 1,
				 yoff + Y2 - 1, zoff +  Z2 - 1,
				 xsize - X2 + 1, ysize - Y2 + 1,
				 zsize - Z2 + 1, level+1));
}
}  // namespace GeomSpace
} // namespace SCICore


#endif
