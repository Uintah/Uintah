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

#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GLVolumeRenderer/VolumeUtils.h>
#include <Core/GLVolumeRenderer/Brick.h>
#include <stdlib.h>
#include <iostream>
#include <string>
using std::cerr;
using std::endl;
using std::string;
using std::cin;
using std::vector;

namespace SCIRun {


Brick::Brick() :
  tex(0),padx(0), pady(0), padz(0), lev(0),
    name(0)
{
}

Brick::Brick(const Point& min, const Point& max,
	      int padx, int pady, int padz,
	      int level,
	      Array3<unsigned char>* tex) :
  tex(tex),padx(padx), pady(pady), padz(padz), lev(level),
   name(0)
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

  // These will be used to create the texture Matrix
  aX = ( 1.0/tex->dim1() == 1.0 ) ? 2.0 : 1.0/tex->dim1();
  aY = ( 1.0/tex->dim2() == 1.0 ) ? 2.0 : 1.0/tex->dim2();
  aZ = ( 1.0/tex->dim3() == 1.0 ) ? 2.0 : 1.0/tex->dim3();
  
  // set up vertices
  corner[0] = min;
  corner[1] = Point(min.x(), min.y(), max.z());
  corner[2] = Point(min.x(), max.y(), min.z());
  corner[3] = Point(min.x(), max.y(), max.z());
  corner[4] = Point(max.x(), min.y(), min.z());
  corner[5] = Point(max.x(), min.y(), max.z());
  corner[6] = Point(max.x(), max.y(), min.z());
  corner[7] = max;

  Point texture[8];
  // set up texture vertices
  texture[0] = Point(0 + 0.5*aX, 0 + 0.5*aY, 0 + 0.5*aZ);
  texture[1] = Point(0 + 0.5*aX, 0 + 0.5*aY, 1 - 0.5*aZ);
  texture[2] = Point(0 + 0.5*aX, 1 - 0.5*aY, 0 + 0.5*aZ);
  texture[3] = Point(0 + 0.5*aX, 1 - 0.5*aY, 1 - 0.5*aZ);
  texture[4] = Point(1 - 0.5*aX, 0 + 0.5*aY, 0 + 0.5*aZ);
  texture[5] = Point(1 - 0.5*aX, 0 + 0.5*aY, 1 - 0.5*aZ);
  texture[6] = Point(1 - 0.5*aX, 1 - 0.5*aY, 0 + 0.5*aZ);
  texture[7] = Point(1 - 0.5*aX, 1 - 0.5*aY, 1 - 0.5*aZ);

  

  // set up edges
  edge[0] = Ray(corner[0], corner[2] - corner[0]);
  edge[1] = Ray(corner[2], corner[6] - corner[2]);
  edge[2] = Ray(corner[4], corner[6] - corner[4]);
  edge[3] = Ray(corner[0], corner[4] - corner[0]);
  edge[4] = Ray(corner[1], corner[3] - corner[1]);
  edge[5] = Ray(corner[3], corner[7] - corner[3]);
  edge[6] = Ray(corner[5], corner[7] - corner[5]);
  edge[7] = Ray(corner[1], corner[5] - corner[1]);
  edge[8] = Ray(corner[0], corner[1] - corner[0]);
  edge[9] = Ray(corner[2], corner[3] - corner[2]);
  edge[10] = Ray(corner[6], corner[7] - corner[6]);
  edge[11] = Ray(corner[4], corner[5] - corner[4]);

  // set up texture coordinate edges
  texEdge[0] = Ray(texture[0], texture[2] - texture[0]);
  texEdge[1] = Ray(texture[2], texture[6] - texture[2]);
  texEdge[2] = Ray(texture[4], texture[6] - texture[4]);
  texEdge[3] = Ray(texture[0], texture[4] - texture[0]);
  texEdge[4] = Ray(texture[1], texture[3] - texture[1]);
  texEdge[5] = Ray(texture[3], texture[7] - texture[3]);
  texEdge[6] = Ray(texture[5], texture[7] - texture[5]);
  texEdge[7] = Ray(texture[1], texture[5] - texture[1]);
  texEdge[8] = Ray(texture[0], texture[1] - texture[0]);
  texEdge[9] = Ray(texture[2], texture[3] - texture[2]);
  texEdge[10] = Ray(texture[6], texture[7] - texture[6]);
  texEdge[11] = Ray(texture[4], texture[5] - texture[4]);
  


}
 
Brick::~Brick()
{
  delete tex;
}


BBox
Brick::bbox() const
{
  BBox bb;
  bb.extend( corner[0] );
  bb.extend( corner[7] );
  return bb;
}

void Brick::ComputePoly(Ray r, double t, Polygon*& p) const
{
  double t0, t1;
  //int i,j,k, tIndex = 1;
  int j;
  Point p0, p1;
  Ray edgeList[6];
  Ray texEdgeList[6];
  Point intersects[6];
  Point texcoords[6];
  Vector view = r.direction();
  //bool buildEdgeList = true;
  RayStep dts[6];
  int nIntersects = 0;
  p0 = r.parameter(t);
  p1 = r.parameter(t);
  for( j = 0; j < 12; j++) {
    t0 = intersectParam(-view, p0, edge[j] );
    t1 = intersectParam(-view, p1, edge[j] );
    if(t0 > 0.0 && t0 < 1.0 ) {
      intersects[nIntersects] = edge[j].parameter(t0);
      texcoords[nIntersects] = texEdge[j].parameter(t0);
      edgeList[nIntersects] = edge[j];
      texEdgeList[nIntersects] = texEdge[j];
      dts[nIntersects].base = t0;
      dts[nIntersects++].step = t1 - t0;
    }
  }
  if(nIntersects > 3) {
    OrderIntersects( intersects, texcoords, edgeList, texEdgeList,
		     dts, nIntersects );
  }
  
  p = scinew Polygon( intersects, texcoords, nIntersects );

}

void
Brick::ComputePolys(Ray r, double tmin, double tmax,
		    double dt, double* ts, vector<Polygon*>& polys ) const
{
  // For a series of planes defined by r and tmin + n*dt,
  // compute the polygon that intersects the texture cube.
  // ts is a list of parameters that correspond the the planes defined
  // by -r.direction and the corners of the texture cube.
  // ts are used for optimization.
  //cerr<<"tmin, tmax, dt = "<<tmin<<", "<<tmax<<", "<<dt<<endl;

  double t = tmax; 
  double t0, t1;
  //int i,j,k, tIndex = 1;
  int j, tIndex = 1;
  Point p0, p1;

  Ray edgeList[6];
  Ray texEdgeList[6];
  Point intersects[6];
  Point texcoords[6];
  Vector view = r.direction();
  bool buildEdgeList = true;
  RayStep dts[6];
  int nIntersects = 0;

  // dt is positive, but we compute polys back to front.
  // use a negative dt

  while( t > ts[7] && t >= tmin ){

    while( t < ts[tIndex] ){
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
	  texcoords[nIntersects] = texEdge[j].parameter(t0);
	  edgeList[nIntersects] = edge[j];
	  texEdgeList[nIntersects] = texEdge[j];
	  dts[nIntersects].base = t0;
	  dts[nIntersects++].step = t1 - t0;
	}
      }
      if(nIntersects > 3) {
	OrderIntersects( intersects, texcoords, edgeList, texEdgeList,
			 dts, nIntersects );
      }

    } else {
      for( j = 0; j < nIntersects; j++ ){
	dts[j].base += dts[j].step;
	intersects[j] = edgeList[j].parameter(dts[j].base);
	texcoords[j] = texEdgeList[j].parameter(dts[j].base);
	if (dts[j].base < 0.0 ||  dts[j].base > 1.0)
	  buildEdgeList = true;
      }
    }
    Polygon *poly = scinew Polygon( intersects, texcoords, nIntersects );
    polys.push_back( poly );
    t -= dt;

  }
}

void
Brick::OrderIntersects(Point *p, Point *t, Ray *r, Ray *te,
		       RayStep *dt, int n) const
{ 
  // We have a series of points, p, that intesect the edges, r, of the 
  // texture cube.  We know that these points will make a convex hull
  // so lets sort the points, rays, and raysteps so that when the points
  // are connected they create and convex polygon.
  Point sorted[6]; 
  Point sortedT[6];
  Ray sortedE[6];
  Ray sortedTE[6];
  RayStep sortedDt[6];
  Vector v0, v1;
  int nSorted = 3;
  int i, j, k;
  double cosTheta, maxCosTheta;
  int i0=0, i1=0;

  for(i = 0; i < nSorted;  i++){
      sorted[i] = p[i];
      sortedT[i] = t[i];
      sortedTE[i] = te[i];      
      sortedE[i] = r[i];
      sortedDt[i] = dt[i];
  }

  for(k = nSorted; k < n; k++){
    maxCosTheta = 1.0;
    // find the neighboring points by finding the maximum angle formed
    //   by the vectors p[k] to any two points. // 
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
    // if the neigbors = the 1st and last point, tag new point to the end
    // of the sorted list. 
    if( i0 == 0 && i1 == nSorted - 1 ){
      sorted[nSorted] = p[k];
      sortedT[nSorted] = t[k];
      sortedTE[nSorted] = te[k];
      sortedE[nSorted] = r[k];
      sortedDt[nSorted] = dt[k];
      nSorted++;
    } else { // move everything to the right of i0 and insert p[k] at i1. 
      for( i = nSorted; i > i0; i-- ){
	sorted[i] = sorted[i-1];
	sortedT[i] = sortedT[i-1];
	sortedTE[i] = sortedTE[i-1];
	sortedE[i] = sortedE[i-1];
	sortedDt[i] = sortedDt[i-1];
      }
      sorted[i1] = p[k];
      sortedT[i1] = t[k];
      sortedTE[i1] = te[k];
      sortedE[i1] = r[k];
      sortedDt[i1] = dt[k];
      nSorted++;
    }
  }
  // put the sorted points back in p
  for(i = 0; i < n; i++){
    p[i] = sorted[i];
    t[i] = sortedT[i];
    te[i] = sortedTE[i];
    r[i] = sortedE[i];
    dt[i] = sortedDt[i];    
  }
}
#define BRICK_VERSION 1

  
} // End namespace SCIRun
