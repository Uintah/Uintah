\
#include <SCICore/Geom/GeomOpenGL.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Geometry/Ray.h>
#include <SCICore/Util/NotFinished.h>
#include "MultiBrick.h"
#include "stdlib.h"
#include "VolumeUtils.h"
#include <iostream>

namespace SCICore {
namespace GeomSpace {

using namespace SCICore::Geometry;
using namespace Kurt::GeomSpace; 
using std::cerr;
 using std::endl;

int MultiBrick::traversalTable[27][8] = { {7,3,5,6,1,2,4,0},
					  {6,7,2,3,4,5,0,1},
					  {6,2,4,7,0,3,5,1},
					  {5,7,1,3,4,6,0,2},
					  {4,5,6,7,0,1,2,3},
					  {4,6,0,2,5,7,1,3},
					  {5,1,4,7,0,3,6,2},
					  {4,5,0,1,6,7,2,3},
					  {4,0,5,6,1,2,7,3},
					  {3,7,1,2,5,6,0,4},
					  {2,3,6,7,0,1,4,5},
					  {2,6,0,3,4,7,1,5},
					  {1,3,5,7,0,2,4,6},
					  {0,1,2,3,4,5,6,7},
					  {0,2,4,6,1,3,5,7},
					  {1,5,0,3,4,7,2,6},
					  {0,1,4,5,2,3,6,7},
					  {0,4,1,2,5,6,3,7},
					  {3,1,2,7,0,5,6,4},
					  {2,3,0,1,6,7,4,5},
					  {2,0,3,6,1,4,7,5},
					  {1,3,0,2,5,7,4,6},
					  {0,1,2,3,4,5,6,7},
					  {0,2,1,3,4,6,5,7},
					  {1,0,3,5,2,4,7,6},
					  {0,1,2,3,4,5,6,7},
					  {0,1,2,4,3,5,6,7}};



  
MultiBrick::MultiBrick(int id, int slices, double alpha,
		       int maxdim, Point min, Point max,
		       bool drawMIP,
		       int X, int Y, int Z,
		       const ScalarFieldRGuchar* tex,
		       const GLvoid* cmap) :
  GeomObj(id), alpha(alpha),  slices(slices),
  tex( tex ), cmap(cmap), min(min), max(max),
  X(X), Y(Y), Z(Z), drawMIP(drawMIP),
  xmax(maxdim),ymax(maxdim),zmax(maxdim)
{
  
  /* The cube is numbered in the following way 
     
          2________6        y
         /|        |        |  
        / |       /|        |
       /  |      / |        |
      /   0_____/__4        |
     3---------7   /        |_________ x
     |  /      |  /         /
     | /       | /         /
     |/        |/         /
     1_________5         /
                        z  
  */

#ifdef SCI_OPENGL
  //  octree = buildOctree(min, max, 0, 0, 0, X, Y, Z, 0);
  octree = buildBonTree(min, max, 0, 0, 0, X, Y, Z, 0);
#endif

}
 
MultiBrick::~MultiBrick()
{
#ifdef SCI_OPENGL
  //glDeleteTextures(1, &texName );
#endif
}

void MultiBrick::SetMaxBrickSize(int x,int y,int z)
{
  GLint xtex, ytex, ztex;
  glTexImage3DEXT(GL_PROXY_TEXTURE_3D_EXT, 0, 4, x, y, z, 0,
                    GL_RGBA, GL_UNSIGNED_BYTE, tex);
  glGetTexLevelParameteriv( GL_PROXY_TEXTURE_3D_EXT, 0,
                            GL_TEXTURE_WIDTH, &xtex);
  glGetTexLevelParameteriv( GL_PROXY_TEXTURE_3D_EXT, 0,
                            GL_TEXTURE_HEIGHT, &ytex);
  glGetTexLevelParameteriv( GL_PROXY_TEXTURE_3D_EXT, 0,
                            GL_TEXTURE_DEPTH_EXT, &ztex);

  if( xtex && ytex && ztex) { // we can accommodate
    xmax = x; ymax = y; zmax = z;
  }
}

VolumeOctree<Brick*>*
MultiBrick::buildBonTree(Point min, Point max,
            int xoff, int yoff, int zoff,
            int xsize, int ysize, int zsize,
	    int level)
{
  
    /* The cube is numbered in the following way 
     
          2________6        y
         /|        |        |  
        / |       /|        |
       /  |      / |        |
      /   0_____/__4        |
     3---------7   /        |_________ x
     |  /      |  /         /
     | /       | /         /
     |/        |/         /
     1_________5         /
                        z  
  */

  VolumeOctree<Brick *> *node;
  Brick* brick;
  Array3<unsigned char> *brickData;
  // Check to make sure that we can accommodate the requested texture
  GLint xtex =0 , ytex = 0 , ztex = 0;
/*   glTexImage3DEXT(GL_PROXY_TEXTURE_3D_EXT, 0, 4, xsize, ysize, zsize, 0, */
/*                     GL_RGBA, GL_UNSIGNED_BYTE, tex); */
/*   glGetTexLevelParameteriv( GL_PROXY_TEXTURE_3D_EXT, 0, */
/*                             GL_TEXTURE_WIDTH, &xtex); */
/*   glGetTexLevelParameteriv( GL_PROXY_TEXTURE_3D_EXT, 0, */
/*                             GL_TEXTURE_HEIGHT, &ytex); */
/*   glGetTexLevelParameteriv( GL_PROXY_TEXTURE_3D_EXT, 0, */
/*                             GL_TEXTURE_DEPTH_EXT, &ztex); */
  if ( xsize <= xmax ) xtex = 1;
  if ( ysize <= ymax ) ytex = 1;
  if ( zsize <= zmax ) ztex = 1;

  if( xtex && ytex && ztex) { // we can accommodate
    brickData = new Array3<unsigned char>();
    makeBrickData(xsize,ysize,zsize, xoff,yoff,zoff, brickData);
    brick = new Brick(min, max,1.0/pow(2.0,level),true,  brickData);

    node = new VolumeOctree<Brick*>(min, max, brick,
                                     VolumeOctree<Brick *>::LEAF );
  } else { // we must subdivide
    //    brick = new Brick(min, max, 1.0/pow(2.0,level), true, brickData);
    int X2, Y2, Z2;
    X2 = largestPowerOf2( xsize -1 );
    Y2 = largestPowerOf2( ysize -1 );
    Z2 = largestPowerOf2( zsize -1);

    if( Z2 == Y2 && Y2 == X2 ){
      node = BonXYZ(min, max, xoff, yoff, zoff,
		    xsize, ysize, zsize, X2, Y2, Z2,level);
    } else if( Z2 > Y2 && Z2 > X2 ) {
      node = BonZ(min, max, xoff, yoff, zoff,
		  xsize, ysize, zsize,  xsize, ysize, Z2, level);
    } else  if( Y2 > Z2 && Y2 > X2 ) {
      node = BonY(min, max, xoff, yoff, zoff,
		  xsize, ysize, zsize,  xsize, Y2, zsize, level);
    } else  if( X2 > Z2 && X2 > Y2 ) {
      node = BonX(min, max, xoff, yoff, zoff,
		  xsize, ysize, zsize,  X2, ysize, zsize, level);
    } else if( Z2 == Y2 ){
      node = BonYZ(min, max, xoff, yoff, zoff,
		   xsize, ysize, zsize,  xsize, Y2, Z2, level);
    } else if( X2 == Y2 ){
      node = BonXY(min, max, xoff, yoff, zoff,
		   xsize, ysize, zsize,  X2, Y2, zsize, level);
    } else if( Z2 == X2 ){
      node = BonXZ(min, max, xoff, yoff, zoff,
		   xsize, ysize, zsize,  X2, ysize, Z2, level);
    }
  }
  return node;
}


VolumeOctree<Brick*>*
MultiBrick::BonXYZ(Point min, Point max,
		   int xoff, int yoff, int zoff,
		   int xsize, int ysize, int zsize,
		   int X2, int Y2, int Z2,
		   int level)
{
  VolumeOctree<Brick*> *node = new VolumeOctree<Brick*>(min, max, 0,
				  VolumeOctree<Brick *>::PARENT );
  BuildChild0(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  BuildChild1(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  BuildChild2(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  BuildChild3(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  BuildChild4(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  BuildChild5(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  BuildChild6(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  BuildChild7(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);

  return node;
}


VolumeOctree<Brick*>*
MultiBrick::BonZ(Point min, Point max,
		   int xoff, int yoff, int zoff,
		   int xsize, int ysize, int zsize,
		   int X2, int Y2, int Z2,
		   int level)
{
  VolumeOctree<Brick*> *node = new VolumeOctree<Brick*>(min, max, 0,
				  VolumeOctree<Brick *>::PARENT );
  BuildChild0(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  BuildChild1(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  return node;
}
VolumeOctree<Brick*>*
MultiBrick::BonY(Point min, Point max,
		   int xoff, int yoff, int zoff,
		   int xsize, int ysize, int zsize,
		   int X2, int Y2, int Z2,
		   int level)
{
  VolumeOctree<Brick*> *node = new VolumeOctree<Brick*>(min, max, 0,
				  VolumeOctree<Brick *>::PARENT );
  BuildChild0(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  BuildChild2(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  return node; 
}
VolumeOctree<Brick*>*
MultiBrick::BonX(Point min, Point max,
		   int xoff, int yoff, int zoff,
		   int xsize, int ysize, int zsize,
		   int X2, int Y2, int Z2,
		   int level)
{
  VolumeOctree<Brick*> *node = new VolumeOctree<Brick*>(min, max, 0,
				  VolumeOctree<Brick *>::PARENT );
  BuildChild0(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  BuildChild4(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  return node;
}
VolumeOctree<Brick*>*
MultiBrick::BonXY(Point min, Point max,
		   int xoff, int yoff, int zoff,
		   int xsize, int ysize, int zsize,
		   int X2, int Y2, int Z2,
		   int level)
{
  VolumeOctree<Brick*> *node = new VolumeOctree<Brick*>(min, max, 0,
				  VolumeOctree<Brick *>::PARENT );
  BuildChild0(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  BuildChild2(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  BuildChild4(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  BuildChild6(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  return node;
}
VolumeOctree<Brick*>*
MultiBrick::BonYZ(Point min, Point max,
		   int xoff, int yoff, int zoff,
		   int xsize, int ysize, int zsize,
		   int X2, int Y2, int Z2,
		   int level)
{
  VolumeOctree<Brick*> *node = new VolumeOctree<Brick*>(min, max, 0,
				  VolumeOctree<Brick *>::PARENT );
  BuildChild0(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  BuildChild1(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  BuildChild2(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  BuildChild3(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  return node;
}

VolumeOctree<Brick*>*
MultiBrick::BonXZ(Point min, Point max,
		   int xoff, int yoff, int zoff,
		   int xsize, int ysize, int zsize,
		   int X2, int Y2, int Z2,
		   int level)
{
  VolumeOctree<Brick*> *node = new VolumeOctree<Brick*>(min, max, 0,
				  VolumeOctree<Brick *>::PARENT );
  BuildChild0(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  BuildChild1(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  BuildChild4(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  BuildChild5(min, max, xoff, yoff, zoff,
	     xsize, ysize, zsize, X2, Y2, Z2, level, node);
  return node;
}



VolumeOctree<Brick*>*
MultiBrick::buildOctree(Point min, Point max,
            int xoff, int yoff, int zoff,
            int xsize, int ysize, int zsize,
	    int level)
{
  
    /* The cube is numbered in the following way 
     
          2________6        y
         /|        |        |  
        / |       /|        |
       /  |      / |        |
      /   0_____/__4        |
     3---------7   /        |_________ x
     |  /      |  /         /
     | /       | /         /
     |/        |/         /
     1_________5         /
                        z  
  */

  VolumeOctree<Brick *> *node;
  Brick* brick;
  Array3<unsigned char> *brickData;
  // Check to make sure that we can accommodate the requested texture
  GLint xtex =0 , ytex = 0 , ztex = 0;
/*   glTexImage3DEXT(GL_PROXY_TEXTURE_3D_EXT, 0, 4, xsize, ysize, zsize, 0, */
/*                     GL_RGBA, GL_UNSIGNED_BYTE, tex); */
/*   glGetTexLevelParameteriv( GL_PROXY_TEXTURE_3D_EXT, 0, */
/*                             GL_TEXTURE_WIDTH, &xtex); */
/*   glGetTexLevelParameteriv( GL_PROXY_TEXTURE_3D_EXT, 0, */
/*                             GL_TEXTURE_HEIGHT, &ytex); */
/*   glGetTexLevelParameteriv( GL_PROXY_TEXTURE_3D_EXT, 0, */
/*                             GL_TEXTURE_DEPTH_EXT, &ztex); */
  if ( xsize <= xmax ) xtex = 1;
  if ( ysize <= ymax ) ytex = 1;
  if ( zsize <= zmax ) ztex = 1;

  if( xtex && ytex && ztex) { // we can accommodate
    brickData = new Array3<unsigned char>();
    makeBrickData(xsize,ysize,zsize, xoff,yoff,zoff, brickData);
    brick = new Brick(min, max,1.0/pow(2.0,level),true,  brickData);

    node = new VolumeOctree<Brick*>(min, max, brick,
                                     VolumeOctree<Brick *>::LEAF );
  } else { // we must subdivide
    //    brick = new Brick(min, max, 1.0/pow(2.0,level), true, brickData);
    brick = 0;
    node = new VolumeOctree<Brick*>(min, max, brick,
				     VolumeOctree<Brick *>::PARENT );

    Point mid = min + (max - min)*0.5;
    
    node->SetChild(0, buildOctree(min, mid, xoff, yoff, zoff,
                                  xsize/2, ysize/2, zsize/2, level+1));
    node->SetChild(1, buildOctree(Point(min.x(), min.y(), mid.z()),
                                  Point(mid.x(), mid.y(), max.z()),
                                  xoff, yoff, zoff + zsize/2 -1,
                                  xsize/2, ysize/2, zsize/2, level+1));
    node->SetChild(2, buildOctree(Point(min.x(), mid.y(), min.z()),
                                  Point(mid.x(), max.y(), mid.z()),
                                  xoff, yoff + ysize/2 - 1, zoff,
                                  xsize/2, ysize/2, zsize/2, level+1));
    node->SetChild(3, buildOctree(Point(min.x(), mid.y(), mid.z()),
                                  Point(mid.x(), max.y(), max.z()),
                                  xoff, yoff + ysize/2 -1 , zoff + zsize/2 - 1,
                                  xsize/2, ysize/2, zsize/2, level+1));
    node->SetChild(4, buildOctree(Point(mid.x(), min.y(), min.z()),
                                  Point(max.x(), mid.y(), mid.z()),
                                  xoff + xsize/2 - 1, yoff, zoff,
                                  xsize/2, ysize/2, zsize/2, level+1));
    node->SetChild(5, buildOctree(Point(mid.x(), min.y(), mid.z()),
                                  Point(max.x(), mid.y(), max.z()),
                                  xoff + xsize/2 - 1, yoff, zoff +  zsize/2 - 1,
                                  xsize/2, ysize/2, zsize/2, level+1));
    node->SetChild(6, buildOctree(Point(mid.x(), mid.y(), min.z()),
                                  Point(max.x(), max.y(), mid.z()),
                                  xoff + xsize/2 - 1, yoff + ysize/2 - 1, zoff,
                                  xsize/2, ysize/2, zsize/2, level+1));
    node->SetChild(7, buildOctree(mid, max,  xoff + xsize/2 - 1,
                                  yoff + ysize/2 - 1, zoff + zsize/2 -1,
                                  xsize/2, ysize/2, zsize/2, level+1));
 }

    return node;
}

void MultiBrick::makeBrickData(int xsize, int ysize, int zsize,
			       int xoff, int yoff, int zoff,
			       Array3<unsigned char>*& bd)
{
  int i,j,k,ii,jj,kk;

  bd->newsize( zsize, ysize, xsize );
  for(kk = 0, k = zoff; kk < zsize; kk++, k++)
    for(jj = 0, j = yoff; jj < ysize; jj++, j++)
      for(ii = 0, i = xoff; ii < xsize; ii++, i++){
	(*bd)(kk,jj,ii) = tex->grid(k,j,i);
  }

}
						





#ifdef SCI_OPENGL
void 
MultiBrick::draw(DrawInfoOpenGL* di, Material* mat, double time)
{
  if( !pre_draw(di, mat, 0) ) return;

  if ( di->get_drawtype() == DrawInfoOpenGL::WireFrame ) {
    drawWireFrame();
  } else {
    drawSlices();
  }
}  
#endif

MultiBrick::MultiBrick(const MultiBrick& copy)
: GeomObj(copy.id), slices(copy.slices)
{
}


GeomObj* MultiBrick::clone()
{
    return new MultiBrick(*this);
}

void MultiBrick::get_bounds(BBox& bb)
{
  bb.extend( min );
  bb.extend( max );
}
  

void
MultiBrick::drawWireFrame()
{ // Draw the bounding box of the brick
  NOT_FINISHED("MultiBrick::drawWireFrame()");
}


void 
MultiBrick::drawSlices()
{


  double mvmat[16];
  Transform mat;
  Vector view;
  Point viewPt;
  Ray viewRay;
      
  glGetDoublev( GL_MODELVIEW_MATRIX, mvmat);
  /* remember that the glmatrix is stored as
       0  4  8 12
       1  5  9 13
       2  6 10 14
       3  7 11 15 */
 
  view = Vector(mvmat[12], mvmat[13], mvmat[14]);
  view.normalize();
  viewPt = Point(-mvmat[12], -mvmat[13], -mvmat[14]);
    
  /* set the translation to zero */
  mvmat[12] = mvmat[13] = mvmat[14] = 0;
  /* Because of the order of the glmatrix we are storing as a transpose.
       if there is not use of scale then the transpose is the  inverse */
  mat.set( mvmat );
    
  /* project view info into object space */
  view = mat.project( view );
  viewPt = mat.project( viewPt );
  viewRay = Ray(viewPt, view);

  // Slice the volume---use GL_TEXTURE_GEN to generate texture coords.
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_TEXTURE_3D_EXT);
  glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,
	    GL_MODULATE);

#ifdef __sgi
  //cerr << "Using Lookup!\n";
  glEnable(GL_TEXTURE_COLOR_TABLE_SGI);
  glColorTableSGI(GL_TEXTURE_COLOR_TABLE_SGI,
		  GL_RGBA,
		  256, // try larger sizes?
		  GL_RGBA,  // need an alpha value...
		  GL_UNSIGNED_BYTE, // try shorts...
		  cmap);
#endif
  
  glColor4f(1,1,1,1); // set to all white for modulation
  

  glEnable(GL_BLEND);
  // Maximum Intensity Projections
  if( drawMIP ){
    glBlendEquationEXT(GL_MAX_EXT);
    glBlendFunc(GL_ONE, GL_ZERO);  //glBlendFunc(GL_ONE, GL_ONE);
  } else {
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
  }
  // This combo works
  //glBlendEquationEXT(GL_FUNC_ADD_EXT);
  //glBlendFunc(GL_SRC_ALPHA, GL_ONE);
  //  drawOctree( octree, viewRay);
  drawBonTree( octree, viewRay);
  // NOT_FINISHED("MultiBrick::drawSlices()");

  glDisable(GL_BLEND);
#ifdef __sgi
  glDisable(GL_TEXTURE_COLOR_TABLE_SGI);
#endif
  glDisable(GL_TEXTURE_3D_EXT);
  glEnable(GL_DEPTH_TEST);  
}

void MultiBrick::drawBonTree( const VolumeOctree<Brick*>* node,
			     const Ray&  viewRay )
{
  int i;
  double  ts[8];

  if ( node == NULL ) return;
  Brick* brick = (*node)(); // get the contents of the node

  if( node->getType() == VolumeOctree<Brick*>::LEAF ) {
    for( i = 0; i < 8; i++)
      ts[i] = intersectParam(-viewRay.direction(),
			     brick->getCorner(i), viewRay);
    sortParameters(ts,8);
    brick->draw( viewRay, alpha, ts[7], ts[0], (ts[0]-ts[7])/slices );

  } else {
    int *traversal;
    int traversalIndex, x, y, z;
    Point min, max, mid;
    const VolumeOctree<Brick*>* child;
    child = node->child(0);
    
    min = brick->getCorner(0);
    max = brick->getCorner(7);
    mid = (*child)()->getCorner(7);

    if( viewRay.origin().x() < mid.x()) x = 0;
    else if( viewRay.origin().x() == mid.x()) x = 1;
    else x = 2;
    if( viewRay.origin().y() < mid.y()) y = 0;
    else if( viewRay.origin().y() == mid.y()) y = 1;
    else y = 2;
    if( viewRay.origin().z() < mid.z()) z = 0;
    else if( viewRay.origin().z() == mid.z()) z = 1;
    else z = 2;
    
    traversalIndex = 9*x + 3*y + z;
    traversal = traversalTable[ traversalIndex ];

    for( i = 0; i < 8; i++){
      drawOctree( node->child( traversal[i] ), viewRay);
    }
  }
}

void MultiBrick::drawOctree( const VolumeOctree<Brick*>* node,
			     const Ray&  viewRay )
{
  int i;
  double  ts[8];

  if ( node == NULL ) return;
  Brick* brick = (*node)(); // get the contents of the node

  if( node->getType() == VolumeOctree<Brick*>::LEAF ) {
    for( i = 0; i < 8; i++)
      ts[i] = intersectParam(-viewRay.direction(),
			     brick->getCorner(i), viewRay);
    sortParameters(ts,8);
    brick->draw( viewRay, alpha, ts[7], ts[0], (ts[0]-ts[7])/slices );

  } else {
    int *traversal;
    int traversalIndex, x, y, z;
    Point min, max, mid;
    min = brick->getCorner(0);
    max = brick->getCorner(7);
    mid = min + (max - min) * 0.5;

    if( viewRay.origin().x() < mid.x()) x = 0;
    else if( viewRay.origin().x() == mid.x()) x = 1;
    else x = 2;
    if( viewRay.origin().y() < mid.y()) y = 0;
    else if( viewRay.origin().y() == mid.y()) y = 1;
    else y = 2;
    if( viewRay.origin().z() < mid.z()) z = 0;
    else if( viewRay.origin().z() == mid.z()) z = 1;
    else z = 2;
    
    traversalIndex = 9*x + 3*y + z;
    traversal = traversalTable[ traversalIndex ];

    for( i = 0; i < 8; i++){
      drawOctree( node->child( traversal[i] ), viewRay);
    }
  }
}


#define MULTIBRICK_VERSION 1

void MultiBrick::io(Piostream&)
{
    // Nothing for now...
  NOT_FINISHED("MultiBrick::io");
}

bool
MultiBrick::saveobj(std::ostream&, const clString& format, GeomSave*)
{
   NOT_FINISHED("MultiBrick::saveobj");
    return false;
}

  
} // namespace SCICore
} // namespace GeomSpace
