#include <SCICore/Geom/GeomOpenGL.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Geometry/Ray.h>
#include <SCICore/Util/NotFinished.h>
#include "VolumeUtils.h"
#include "Brick.h"
#include "stdlib.h"
#include <iostream>
#include <string>
#include <unistd.h>
#include <stdlib.h>
using std::cerr;
using std::endl;
using std::string;
using std::cin;

namespace Kurt {
namespace GeomSpace {

using namespace SCICore::Geometry;

void glPrintError(const string& word){
  GLenum errCode;
  const GLubyte *errString;

  if((errCode = glGetError()) != GL_NO_ERROR) {
    errString = gluErrorString(errCode);
    cerr<<"OpenGL Error at "<<word.c_str()<<": "<< errString<<endl;
  }
}
  
 Brick::Brick(Point min, Point max,
	      int padx, int pady, int padz,
	      int level, 
	      double alphaScale,
	      bool hasNeighbor, bool debug,
	      const Array3<unsigned char>* tex) :
  padx(padx), pady(pady), padz(padz), alphaScale(alphaScale), tex( tex ),
   hasNeighbor(hasNeighbor), debug(debug), texName(0), level(level)
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

  Vector diag = corner[7] - corner[0];
  // These will be used to create the texture Matrix
  aX = ( 1.0/tex->dim1() == 1.0 ) ? 2.0 : 1.0/tex->dim1();
  aY = ( 1.0/tex->dim2() == 1.0 ) ? 2.0 : 1.0/tex->dim2();
  aZ = ( 1.0/tex->dim3() == 1.0 ) ? 2.0 : 1.0/tex->dim3();
  
}
 
Brick::~Brick()
{

  delete tex;
#ifdef SCI_OPENGL
  glDeleteTextures(1, &texName );
#endif
}


void Brick::get_bounds(BBox& bb)
{
  bb.extend( corner[0] );
  bb.extend( corner[7] );
}
  

void 
Brick::draw(Ray viewRay, double alpha, bool drawWireFrame, bool reload,
		  double tmin, double tmax, double dt )
{
  if( drawWireFrame ){
    this->drawWireFrame();
  } else {
   //cerr<<"BBox = ("<<corner[0]<<", "<<corner[7]<<")"<<endl;
    drawSlices(viewRay, alpha, reload, tmin, tmax, dt );
  }
}  

void
Brick::draw(Ray viewRay, bool drawWireFrame, bool reload,
	    const Point& planeIntersection)
{
  Point view = viewRay.origin();
  Point mid = corner[0] + (corner[7] - corner[0])*0.5;
  
  Point xOrigin(corner[0].x(),mid.y(),mid.z());
  Point yOrigin(mid.x(), corner[0].y(),mid.z());
  Point zOrigin(mid.x(),mid.y(),corner[0].z());
  Vector xDirection(planeIntersection.x() - xOrigin.x(),0,0);
  Vector yDirection(0,planeIntersection.y() - yOrigin.y(),0);
  Vector zDirection(0,0,planeIntersection.z() - zOrigin.z());


  if (planeIntersection.x() > corner[0].x() &&
      planeIntersection.x() < corner[7].x()) {
    if ( view.x() > planeIntersection.x()){
      xOrigin.x( corner[7].x());
      xDirection.x(planeIntersection.x() - xOrigin.x());
    }
    draw( Ray( xOrigin, xDirection),
	  1.0, drawWireFrame, reload, 1.0, 1.0, 1.0);
  }
  if (planeIntersection.y() > corner[0].y() &&
      planeIntersection.y() < corner[7].y()) {
    if ( view.y() > planeIntersection.y()){
      yOrigin.y( corner[7].y() );
      yDirection.y(planeIntersection.y() - yOrigin.y());
    }
    draw( Ray( yOrigin, yDirection),
	  1.0, drawWireFrame, reload, 1.0, 1.0, 1.0);
  }
  
  if (planeIntersection.z() > corner[0].z() &&
      planeIntersection.z() < corner[7].z()) {
    if ( view.z() > planeIntersection.z()){
      zOrigin.z( corner[7].z());
      zDirection.z(planeIntersection.z() - zOrigin.z());
    }
    draw( Ray( zOrigin, zDirection),
	  1.0, drawWireFrame, reload, 1.0, 1.0, 1.0);
  }
}
  

void
Brick::drawWireFrame()
{ // Draw the bounding box of the brick

  int i;
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_TEXTURE_3D_EXT);
  if(debug){
    double rand = drand48();
    if( rand < 0.3333 ){
      glColor4f(0.6667 + rand, 0, 0, 1);
    } else if (rand > 0.6667) {
      glColor4f(0,rand,0,1);
    } else {
      glColor4f(0,0,0.3333 + rand,1);
    }
  } else {
    glColor4f(1,1,1,1);
  }
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
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_TEXTURE_3D_EXT);

  if(debug){
    glFlush();
    cin.get();
  }

}


void 
Brick::drawSlices(Ray viewRay, double alpha, bool reload,
		  double tmin, double tmax, double dt  )
{
  int i;
   if( !texName || reload ) {
     if( !texName )
       glGenTextures(1, &texName);

    glPrintError("glGenTextures");
    glBindTexture(GL_TEXTURE_3D_EXT, texName);
    glPrintError("glBindTexture");
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    //   glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    //   glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_S,
		    GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_T,
		    GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_R_EXT,
		    GL_CLAMP);
    glPrintError("glTexParameter GL_CLAMP_TO_BORDER_SGIS");

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    //    cerr<<"X = "<<tex->dim1()<<", Y= "<<tex->dim2()
    //	<< ", Z= "<<tex->dim3()<<endl;
    // set up the texture
    glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0,
		    GL_INTENSITY8_EXT,
		    tex->dim1(), tex->dim2(), tex->dim3(), 0,
		    GL_RED, GL_UNSIGNED_BYTE,
		    &(*tex)(0,0,0));
    glPrintError("glTexImage3");
   } else {
     glBindTexture(GL_TEXTURE_3D_EXT, texName);
    glPrintError("glBindTexture2");
   }

  makeTextureMatrix();

  glEnable(GL_TEXTURE_GEN_S);
  glPrintError("glEnable(GL_TEXTURE_GEN_S)");
  glEnable(GL_TEXTURE_GEN_T);
  glPrintError("glEnable(GL_TEXTURE_GEN_T)");
  glEnable(GL_TEXTURE_GEN_R);
  glPrintError("glEnable(GL_TEXTURE_GEN_R)");
  glEnable(GL_TEXTURE_GEN_Q);
  glPrintError("glEnable(GL_TEXTURE_GEN_Q)");

  glColor4f(1,1,1,alpha*alphaScale);

  double ts[8];  // a list of ray parameters.
  // The plane perpendicular to the view vector intersecting each
  // vertex is computed.
  for( i = 0; i < 8; i++)
      ts[i] = intersectParam(-viewRay.direction(), getCorner(i), viewRay);
  // now sort the parameters from furthest to nearest
  sortParameters( ts, 8 );
  // now Compute the intersection of the slices and the Brick
  // and draw the intersection.
  ComputeAndDrawPolys( viewRay, tmin, tmax, dt,  ts );

  glDisable(GL_TEXTURE_GEN_S);
  glDisable(GL_TEXTURE_GEN_T);
  glDisable(GL_TEXTURE_GEN_R);
  glDisable(GL_TEXTURE_GEN_Q);
}

void 
Brick::makeTextureMatrix()
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
  glPrintError("glTexGend(GL_S,GL_TEXTURE_GEN_MODE,GL_OBJECT_LINEAR)");
  glTexGend(GL_T,GL_TEXTURE_GEN_MODE,GL_OBJECT_LINEAR);
  glPrintError("glTexGend(GL_T,GL_TEXTURE_GEN_MODE,GL_OBJECT_LINEAR)");
  glTexGend(GL_R,GL_TEXTURE_GEN_MODE,GL_OBJECT_LINEAR);
  glPrintError("glTexGend(GL_R,GL_TEXTURE_GEN_MODE,GL_OBJECT_LINEAR)");
  glTexGend(GL_Q,GL_TEXTURE_GEN_MODE,GL_OBJECT_LINEAR);
  glPrintError("glTexGend(GL_Q,GL_TEXTURE_GEN_MODE,GL_OBJECT_LINEAR");


  if( hasNeighbor ){
    //  This code is for render overlapping bricks.  The plane equations
    //  for s are  (Nx * Pxmin) + d = aX/2  and
    //  (Nx * Pxmax) + d = 1 - aX/2 where
    //  Nx is the x component of the normal,  Pxmin and Pxmax are the x 
    //  components of the min and max points on the TexCube, and  aX is one
    //  texel width.  Solving for Nx and d we get
    //  Nx = (1 - aX)/(Pxmax - Pxmin) and
    //  d = aX/2 - (Pxmin *(1 - aX))/(Pxmax - Pxmin)

/*     splane[0] = (1 - 2 * aX)/diag.x(); */
/*     splane[3] = aX - (corner[0].x() * (1 - 2 * aX)/diag.x()); */
/*     tplane[1] = (1 - 2 * aY)/diag.y(); */
/*     tplane[3] = aY - (corner[0].y() * (1 - 2 * aY)/diag.y()); */
/*     rplane[2] = (1 - 2 * aZ)/diag.z(); */
/*     rplane[3] = aZ - (corner[0].z() * (1 - 2 * aZ)/diag.z()); */
    splane[0] = (1 - aX * (padx + 1))/diag.x();
    splane[3] = aX * 0.5 - (corner[0].x() * (1 - aX * (padx+1))/diag.x());
    tplane[1] = (1 - aY * (pady + 1))/diag.y();
    tplane[3] = aY * 0.5 - (corner[0].y() * (1 - aY * (pady+1))/diag.y());
    rplane[2] = (1 - aZ * (padz + 1))/diag.z();
    rplane[3] = aZ * 0.5 - (corner[0].z() * (1 - aZ * (padz+1))/diag.z());

  } else {
    //  This code is for rendering a single non overlapping brick
    splane[0] = 1.0/diag.x(); splane[3] = - corner[0].x()/diag.x();
    tplane[1] = 1.0/diag.y(); tplane[3] = - corner[0].y()/diag.y();
    rplane[2] = 1.0/diag.z(); rplane[3] = - corner[0].z()/diag.z();
  }
  
  glTexGendv(GL_S,GL_OBJECT_PLANE,splane);
  glPrintError("gl_s");
  glTexGendv(GL_T,GL_OBJECT_PLANE,tplane);
  glPrintError("gl_t");
  glTexGendv(GL_R,GL_OBJECT_PLANE,rplane);
  glPrintError("gl_r");
  glTexGendv(GL_Q,GL_OBJECT_PLANE,qplane);
}





double
Brick::ComputeAndDrawPolys(Ray r, double tmin, double tmax,
			    double dt, double* ts)
{
  // For a series of planes defined by r and tmin + n*dt,
  // compute the polygon that intersects the texture cube.
  // ts is a list of parameters that correspond the the planes defined
  // by -r.direction and the corners of the texture cube.
  // ts are used for optimization.
  //cerr<<"tmin, tmax, dt = "<<tmin<<", "<<tmax<<", "<<dt<<endl;

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
  //while( t >= ts[0] ) t -= dt;

  while( t > ts[7] && t >= tmin ){

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
	t0 = intersectParam(-view, p0, edge[j] );
	t1 = intersectParam(-view, p1, edge[j] );
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
    //cin.get();
    //cerr<< "t = "<<t<<endl;
    drawPolys(intersects, nIntersects);
    t -= dt;

  }
  return t;
}

void
Brick::OrderIntersects(Point *p, Ray *r, RayStep *dt, int n)
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
Brick::drawPolys(Point *intersects, int nIntersects)
{
  //  glDisable(GL_TEXTURE_3D_EXT);
  int k;
  switch (nIntersects) {
    case 1:
      //glColor4f(1.0,1.0,1.0,1.0);
      glBegin(GL_POINTS);
        glVertex3f(intersects[0].x(),intersects[0].y(),intersects[0].z());
	//cerr<<intersects[0]<<endl;
      glEnd();
      break;
    case 2:
      //glColor4f(1.0,1.0,1.0,1.0);
      glBegin(GL_LINES);
        glVertex3f(intersects[0].x(),intersects[0].y(),intersects[0].z());
	glVertex3f(intersects[1].x(),intersects[1].y(),intersects[1].z());
	//cerr<<intersects[0]<<endl;
	//cerr<<intersects[1]<<endl;
      glEnd();
      break;
    case 3:
      //glColor4f(1.0,1.0,0.0,1.0);
      glBegin(GL_TRIANGLES);
        glVertex3f(intersects[0].x(),intersects[0].y(),intersects[0].z());
        glVertex3f(intersects[1].x(),intersects[1].y(),intersects[1].z());
        glVertex3f(intersects[2].x(),intersects[2].y(),intersects[2].z());
	//cerr<<intersects[0]<<endl;
	//cerr<<intersects[1]<<endl;
	//cerr<<intersects[2]<<endl;
      glEnd();
      break;
    case 4:
    case 5:
    case 6:
      glBegin(GL_POLYGON);
        for(k =0; k < nIntersects; k++){
	  glVertex3f(intersects[k].x(),intersects[k].y(),intersects[k].z());
	  //cerr<<intersects[k]<<endl;
	}
      glEnd();
      break;
    }
  //cerr<<endl;
  // glFlush();
  //  glEnable(GL_TEXTURE_3D_EXT);
}
#define BRICK_VERSION 1

  
} // namespace GeomSpace
} // namespace Kurt
