#include <SCICore/Geom/GeomOpenGL.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Geometry/Ray.h>
#include <SCICore/Util/NotFinished.h>
#include "TexBrick.h"
#include "stdlib.h"
#include <iostream.h>

namespace SCICore {
namespace GeomSpace {

using namespace SCICore::Geometry;
  
TexBrick::TexBrick(int id, int slices, double alpha,
		   bool hasNeighbor,
		   Point min, Point max,
		   int xoff, int yoff, int zoff,
		   int texX, int texY, int texZ,
		   int padx, int pady, int padz,
		   const GLvoid* tex,
		   const GLvoid* cmap) :
  GeomObj(id), alpha(alpha), X(texX), Y(texY), Z( texZ ),
  padx( padx ), pady( pady ), padz( padz ),
  xoff( xoff ), yoff( yoff ), zoff(zoff),
  slices(slices), hasNeighbor( hasNeighbor ),
  tex( tex ), cmap(cmap)
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

  // set up vertices
  corner[0] = min;
  corner[1] = Point(min.x(), min.y(), max.z());
  corner[2] = Point(min.x(), max.y(), min.z());
  corner[3] = Point(min.x(), max.y(), max.z());
  corner[4] = Point(max.x(), min.y(), min.z());
  corner[5] = Point(max.x(), min.y(), max.z());
  corner[6] = Point(max.x(), max.y(), min.z());
  corner[7] = max;

  // set up edges
  edge[0] = Ray(corner[0], corner[2] - corner[0]);
  edge[1] = Ray(corner[2], corner[6] - corner[2]);
  edge[2] = Ray(corner[6], corner[4] - corner[6]);
  edge[3] = Ray(corner[4], corner[0] - corner[4]);
  edge[4] = Ray(corner[1], corner[3] - corner[1]);
  edge[5] = Ray(corner[3], corner[7] - corner[3]);
  edge[6] = Ray(corner[7], corner[5] - corner[7]);
  edge[7] = Ray(corner[5], corner[1] - corner[5]);
  edge[8] = Ray(corner[0], corner[1] - corner[0]);
  edge[9] = Ray(corner[2], corner[3] - corner[2]);
  edge[10] = Ray(corner[6], corner[7] - corner[6]);
  edge[11] = Ray(corner[4], corner[5] - corner[4]);


  // These will be used to create the texture Matrix
  aX = ( 1.0/X == 1.0 ) ? 2.0 : 1.0/X;
  aY = ( 1.0/Y == 1.0 ) ? 2.0 : 1.0/Y;
  aZ = ( 1.0/Z == 1.0 ) ? 2.0 : 1.0/Z;

  


}
 
TexBrick::~TexBrick()
{
#ifdef SCI_OPENGL
   glDeleteTextures(1, &texName );
#endif
}

#ifdef SCI_OPENGL
void 
TexBrick::draw(DrawInfoOpenGL* di, Material* mat, double time)
{
  if( !pre_draw(di, mat, 0) ) return;

  if ( di->get_drawtype() == DrawInfoOpenGL::WireFrame ) {
    drawWireFrame();
  } else {
    drawSlices();
  }
}  
#endif

TexBrick::TexBrick(const TexBrick& copy)
: GeomObj(copy.id), X(copy.X), Y(copy.Y), Z( copy.Z ),
  slices(copy.slices), hasNeighbor(copy.hasNeighbor )
{
}


GeomObj* TexBrick::clone()
{
    return new TexBrick(*this);
}

void TexBrick::get_bounds(BBox& bb)
{
  bb.extend( corner[0] );
  bb.extend( corner[7] );
}
  

void
TexBrick::drawWireFrame()
{ // Draw the bounding box of the brick

  int i;
  
  glPushMatrix();
  glBegin(GL_LINES);
  for(i = 0; i < 4; i++){
    glVertex3d(corner[i].x(), corner[i].y(), corner[i].z());
    glVertex3d(corner[i+4].x(), corner[i+4].y(), corner[i+4].z());
  }
  glEnd();

  glBegin(GL_LINE_LOOP);
  glVertex3d(corner[0].x(), corner[0].y(), corner[0].z());
  glVertex3d(corner[1].x(), corner[1].y(), corner[1].z());
  glVertex3d(corner[3].x(), corner[3].y(), corner[3].z());
  glVertex3d(corner[2].x(), corner[2].y(), corner[2].z());
  glEnd();

  glBegin(GL_LINE_LOOP);
  glVertex3d(corner[4].x(), corner[4].y(), corner[4].z());
  glVertex3d(corner[5].x(), corner[5].y(), corner[5].z());
  glVertex3d(corner[7].x(), corner[7].y(), corner[7].z());
  glVertex3d(corner[6].x(), corner[6].y(), corner[6].z());
  glEnd();
  glPopMatrix();
}


void 
TexBrick::drawSlices()
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
  cerr << "Using Lookup!\n";
  glEnable(GL_TEXTURE_COLOR_TABLE_SGI);
  glColorTableSGI(GL_TEXTURE_COLOR_TABLE_SGI,
		  GL_RGBA,
		  256, // try larger sizes?
		  GL_RGBA,  // need an alpha value...
		  GL_UNSIGNED_BYTE, // try shorts...
		  cmap);
#endif
  
  glColor4f(1,1,1,1); // set to all white for modulation
  
  if( !texName ){


    glGenTextures(1, &texName);
    glBindTexture(GL_TEXTURE_3D_EXT, texName);

    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_R_EXT, GL_CLAMP);

      // set the pixel strides before creation.  
      // GL_UNPACK_ALIGNMENT defaults to 1.
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, xoff);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, yoff);
    glPixelStorei(GL_UNPACK_SKIP_IMAGES_EXT, zoff);

    // set up the texture
    glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0,
		    GL_INTENSITY8_EXT,
		    X, Y, Z, 0,
		    GL_RED, GL_UNSIGNED_BYTE, tex);
  } else {
    glBindTexture(GL_TEXTURE_3D_EXT, texName);
  }
  
  glEnable(GL_TEXTURE_GEN_S);
  glEnable(GL_TEXTURE_GEN_T);
  glEnable(GL_TEXTURE_GEN_R);
  glEnable(GL_TEXTURE_GEN_Q);
  makeTextureMatrix();

  glEnable(GL_BLEND);
  //glBlendEquationEXT(GL_MAX_EXT);
  //  glBlendFunc(GL_ONE, GL_ZERO);
  //glBlendFunc(GL_ONE, GL_ONE);
  
  // This combo  works
  //glBlendEquationEXT(GL_FUNC_ADD_EXT);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
  // This combo works
  //glBlendEquationEXT(GL_FUNC_ADD_EXT);
  //glBlendFunc(GL_SRC_ALPHA, GL_ONE);


  glColor4f(1,1,1,alpha);

  float ts[8];  // a list of ray parameters.
  // The plane perpendicular to the view vector intersecting each
  // vertex is computed.
  computeParameters(-viewRay.direction(), viewRay, ts , 8);
  // now sort the parameters from furthest to nearest
  sortParameters( ts, 8 );
  // now Compute the intersection of the slices and the TexBrick
  // and draw the intersection.
  ComputeAndDrawPolys( viewRay, ts[7], ts[0], (ts[0] - ts[7])/slices, ts );

  glDisable(GL_TEXTURE_GEN_S);
  glDisable(GL_TEXTURE_GEN_T);
  glDisable(GL_TEXTURE_GEN_R);
  glDisable(GL_TEXTURE_GEN_Q);
  glDisable(GL_BLEND);
  glDisable(GL_TEXTURE_3D_EXT);
  glDisable(GL_TEXTURE_COLOR_TABLE_SGI);
  glEnable(GL_DEPTH_TEST);  

}

void 
TexBrick::makeTextureMatrix()
{

  double splane[4]={0,0,0,0};
  double tplane[4]={0,0,0,0};
  double rplane[4]={0,0,0,0};
  double qplane[4]={0,0,0,1};


  Vector diag;

  
  /* The cube is numbered in the following way 
      
         2________ 6        y
        /|       /|         |  
       / |      / |         |
      /  |     /  |         |
    3/__0|____/7__|4        |_________ x
     |   /    |   /         /
     |  /     |  /         /
     | /      | /         /
    1|/_______|/5        /
                        z  
  */



  diag = corner[7] - corner[0];

  glTexGend(GL_S,GL_TEXTURE_GEN_MODE,GL_OBJECT_LINEAR);
  glTexGend(GL_T,GL_TEXTURE_GEN_MODE,GL_OBJECT_LINEAR);
  glTexGend(GL_R,GL_TEXTURE_GEN_MODE,GL_OBJECT_LINEAR);
  glTexGend(GL_Q,GL_TEXTURE_GEN_MODE,GL_OBJECT_LINEAR);


  if( hasNeighbor ){
    //  This code is for render overlapping bricks.  The plane equations
    //  for s are  (Nx * Pxmin) + d = aX  and  (Nx * Pxmax) + d = 1 - aX where
    //  Nx is the x component of the normal,  Pxmin and Pxmax are the x 
    //  components of the min and max points on the TexCube, and  aX is one
    //  texel width.  Solving for Nx and d we get
    //  Nx = (1 - 2 * aX)/(Pxmax - Pxmin) and
    //  d = aX - (Pxmin *(1 - 2 * aX))/(Pxmax - Pxmin)

    splane[0] = (1 - 2 * aX)/diag.x();
    splane[3] = aX - (corner[0].x() * (1 - 2 * aX)/diag.x());
    tplane[1] = (1 - 2 * aY)/diag.y();
    tplane[3] = aY - (corner[0].y() * (1 - 2 * aY)/diag.y());
    rplane[2] = (1 - 2 * aZ)/diag.z();
    rplane[3] = aZ - (corner[0].z() * (1 - 2 * aZ)/diag.z());

  } else {
    //  This code is for rendering a single non overlapping brick
    splane[0] = 1.0/diag.x(); splane[3] = - corner[0].x()/diag.x();
    tplane[1] = 1.0/diag.y(); tplane[3] = - corner[0].y()/diag.y();
    rplane[2] = 1.0/diag.z(); rplane[3] = - corner[0].z()/diag.z();
  }
  
  glTexGendv(GL_S,GL_OBJECT_PLANE,splane);
  glTexGendv(GL_T,GL_OBJECT_PLANE,tplane);
  glTexGendv(GL_R,GL_OBJECT_PLANE,rplane);
  glTexGendv(GL_Q,GL_OBJECT_PLANE,qplane);
}

void
TexBrick::computeParameters(const Vector& N, const Ray&  R, float *t, int len_t )
{
  // compute a plane parameter for each corner of the texture cube
  int i;
  for( i = 0; i < len_t; i++)
    t[i] = computeParameter(N, corner[i], R);
}

float
TexBrick::computeParameter(const Vector& N, const Point& P, const Ray& R)
{
  // Computes the ray parameter t at which the ray R will
  // intersect the plane specified by the normal N and the 
  // point P

  /*  Dot(N, ((O + t0*V) - P)) = 0   solve for t0 */

  Point O = R.origin();
  Vector V = R.direction();
  double D = -(N.x()*P.x() + N.y()*P.y() + N.z()*P.z());
  double NO = (N.x()*O.x() + N.y()*O.y() + N.z()*O.z());

  double NV = Dot(N,V);
  if( NV == 0 ) {  /* No Intersection, plane is parallel */
    return -1.0;
  } else {
    return -(D + NO)/NV;
  }
}

void
TexBrick::sortParameters( float *t, int len_t )
{
  // sorts ray parameters from largest to smallest
  int i,j;
  float tmp;
  for(j = 0; j < len_t; j++){
    for(i = j+1; i < len_t; i++){
      if( t[j] < t[i] ){
	tmp = t[i];
	t[i] = t[j];
	t[j] = tmp;
      }
    }
  }
}

float
TexBrick::ComputeAndDrawPolys(Ray r, float tmin, float tmax,
			    float dt, float* ts)
{
  // For a series of planes defined by r and tmin + n*dt,
  // compute the polygon that intersects the texture cube.
  // ts is a list of parameters that correspond the the planes defined
  // by -r.direction and the corners of the texture cube.
  // ts are used for optimization.

  double t = tmax; 
  double t0, t1;
  int i,j,k, tIndex = 1;
  Point p0, p1;

  Ray edgeList[6];
  Point intersects[6];
  Vector view = r.direction();
  bool buildEdgeList = true;
  RayStep dts[6];
  int nIntersects = 0;

  // dt is positive, but we compute polys back to front.
  // use a negative dt
  while( t >= ts[0] ) t -= dt;

  while( t > ts[7] && t > tmin ){

    while( t < ts[tIndex] ){
      /* printf("t = %f\n", t); */
      buildEdgeList = true;
      tIndex++;
    }

    if(buildEdgeList  || !nIntersects){
      nIntersects = 0;
      buildEdgeList = false;
      p0 = r.parameter(t);
      p1 = r.parameter(t-dt);
      for( j = 0; j < 12; j++) {
	t0 = computeParameter(-view, p0, edge[j] );
	t1 = computeParameter(-view, p1, edge[j] );
	if(t0 > 0.0 && t0 < 1.0 ) {
	  intersects[nIntersects] = edge[j].parameter(t0);
	  edgeList[nIntersects] = edge[j];
	  dts[nIntersects].base = t0;
	  dts[nIntersects++].step = t1 - t0;
	}
      }
      if(nIntersects > 3) {
	OrderIntersects( intersects, edgeList, dts, nIntersects );
      }

    } else {
      for( j = 0; j < nIntersects; j++ ){
	dts[j].base += dts[j].step;
	intersects[j] = edgeList[j].parameter(dts[j].base);
	if (dts[j].base < 0.0 ||  dts[j].base > 1.0)
	  buildEdgeList = true;
      }
    }
    drawPolys(intersects, nIntersects);
    t -= dt;

  }
  return t;
}

void
TexBrick::OrderIntersects(Point *p, Ray *r, RayStep *dt, int n)
{ 
  // We have a series of points, p, that intesect the edges, r, of the 
  // texture cube.  We know that these points will make a convex hull
  // so lets sort the points, rays, and raysteps so that when the points
  // are connected they create and convex polygon.
  Point sorted[6]; 
  Ray sortedE[6];
  RayStep sortedDt[6];
  Vector v0, v1;
  int nSorted = 3;
  int i, j, k;
  double cosTheta, maxCosTheta;
  int i0, i1;

  for(i = 0; i < nSorted;  i++){
      sorted[i] = p[i];
      sortedE[i] = r[i];
      sortedDt[i] = dt[i];
  }

  for(k = nSorted; k < n; k++){
    maxCosTheta = 1.0;
    /* find the neighboring points by finding the maximum angle formed
       by the vectors p[k] to any two points. */ 
    for(j = 0; j < nSorted; j++){
      for(i = j + 1; i < nSorted; i++) {
	v0 = sorted[j] - p[k];
	v1 = sorted[i] - p[k];
	v0.normalize();
	v1.normalize();
	cosTheta = Dot( v0, v1 );
	if( cosTheta < maxCosTheta ){
	  i0 = j;
	  i1 = i;
	  maxCosTheta = cosTheta;
	}
      }
    }
    /* if the neigbors = the 1st and last point, tag new point to the end
       of the sorted list. */
    if( i0 == 0 && i1 == nSorted - 1 ){
      sorted[nSorted] = p[k];
      sortedE[nSorted] = r[k];
      sortedDt[nSorted] = dt[k];
      nSorted++;
    } else { /* move everything to the right of i0 and insert p[k] at i1. */
      for( i = nSorted; i > i0; i-- ){
	sorted[i] = sorted[i-1];
	sortedE[i] = sortedE[i-1];
	sortedDt[i] = sortedDt[i-1];
      }
      sorted[i1] = p[k];
      sortedE[i1] = r[k];
      sortedDt[i1] = dt[k];
      nSorted++;
    }
  }
  /* put the sorted points back in p */
  for(i = 0; i < n; i++){
    p[i] = sorted[i];
    r[i] = sortedE[i];
    dt[i] = sortedDt[i];    
  }
}
  
void
TexBrick::drawPolys(Point *intersects, int nIntersects)
{
  int k;
  switch (nIntersects) {
    case 1:
      glBegin(GL_POINTS);
        glVertex3f(intersects[0].x(),intersects[0].y(),intersects[0].z());
      glEnd();
      break;
    case 2:
      glBegin(GL_LINES);
        glVertex3f(intersects[0].x(),intersects[0].y(),intersects[0].z());
	glVertex3f(intersects[1].x(),intersects[1].y(),intersects[1].z());
      glEnd();
      break;
    case 3:
      glBegin(GL_TRIANGLES);
        glVertex3f(intersects[0].x(),intersects[0].y(),intersects[0].z());
        glVertex3f(intersects[1].x(),intersects[1].y(),intersects[1].z());
        glVertex3f(intersects[2].x(),intersects[2].y(),intersects[2].z());
      glEnd();
      break;
    case 4:
    case 5:
    case 6:
      glBegin(GL_POLYGON);
        for(k =0; k < nIntersects; k++){
	  glVertex3f(intersects[k].x(),intersects[k].y(),intersects[k].z());
	}
      glEnd();
      break;
    }
}
#define TEXBRICK_VERSION 1

void TexBrick::io(Piostream&)
{
    // Nothing for now...
     NOT_FINISHED("TexBrick::io");
}

bool
TexBrick::saveobj(std::ostream&, const clString& format, GeomSave*)
{
   NOT_FINISHED("TexBrick::saveobj");
    return false;
}

  
} // namespace SCICore
} // namespace GeomSpace
