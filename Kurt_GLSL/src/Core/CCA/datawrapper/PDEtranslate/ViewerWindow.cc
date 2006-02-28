/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


/*
 *  ViewerWindow.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */


#include "ViewerWindow.h"
#include <iostream>




using namespace std;


ViewerWindow::ViewerWindow( const std::vector<double> nodes1d,
                            const std::vector<int> triangles,
                            const std::vector<double> solution,
                            int width, int height)
 
:width(width), height(height) 
{
  this->nodes1d=nodes1d;
  this->solution=solution;
  this->triangles=triangles;
 
  int N=nodes1d.size()/2;
  
  if(N==0)return;
  minx=maxx=nodes1d[0];
  miny=maxy=nodes1d[1];


  for(int i=0; i<N; i++){
    if(minx>nodes1d[i*2]) minx=nodes1d[i*2];
    if(maxx<nodes1d[i*2]) maxx=nodes1d[i*2];    
    if(miny>nodes1d[i*2+1]) miny=nodes1d[i*2+1];
    if(maxy<nodes1d[i*2+1]) maxy=nodes1d[i*2+1];
  }

  double minVal=solution[0];
  double maxVal=solution[0];
  for(int i=0; i<N; i++){
    if(minVal>solution[i]) minVal=solution[i];
    if(maxVal<solution[i]) maxVal=solution[i];
  }
}

QPoint ViewerWindow::toViewport(vector2d v)
{
  int x=int((v.x-minx)*width/(maxx-minx)); 
  int y=int(height-(v.y-miny)*height/(maxy-miny)); 
  return QPoint(x,y);
}

vector2d ViewerWindow::toField(QPoint p)
{
  double x=minx+(p.x())*(maxx-minx)/width; 
  double y=miny+(height-(p.y()))*(maxy-miny)/height; 
  return vector2d(x,y);
}
/*
void ViewerWindow::convert(SSIDL::array2<double>& image)
{
  int nTri=triangles.size()/3;
  if(nTri==0) return;
  //assign values all over the image
  for(int ix=0; ix<width; ix++){
    for(int iy=0; iy<height; iy++){    
      image[ix][iy]=0;
      for(int t=0; t<nTri; t++){
	double x[3];
	double y[3];
	double a[3];
	double b[3];
	double c[3];
	double u[3];

	for(int i=0;i<3;i++){
	  x[i]=nodes1d[triangles[t*3+i]*2 ];
	  y[i]=nodes1d[triangles[t*3+i]*2+1 ];
	  u[i]=solution[triangles[t*3+i]];
	}

	vector2d vp=toField(QPoint(ix, iy));

	if(nodeInTriangle(vp, vector2d(x[0],y[0]),
			  vector2d(x[1],y[1]), vector2d(x[2],y[2]) ) ){

	  for(int i=0;i<3;i++){
	    int i2=(i+1)%3;
	    int i3=(i+2)%3;
	    a[i]=x[i2]*y[i3]-x[i3]*y[i2];
	    b[i]=y[i2]-y[i3];
	    c[i]=x[i3]-x[i2];
	  }
	  double A2=a[0]+b[0]*x[0]+c[0]*y[0];

	  double val=0;
	  for(int i=0;i<3;i++){
	    val+=(a[i]+b[i]*vp.x+c[i]*vp.y)*u[i]/A2;
	  }

	  image[ix][iy] = val;
	  break; //only one triangle can contain this point

	}
      }
    }
  }  
}
*/


void ViewerWindow::convert(SSIDL::array2<double>& image)
{
  /*
  int nTri=triangles.size()/3;
  if(nTri==0) return;
  //assign values all over the image
  for(int ix=0; ix<width; ix++){
    for(int iy=0; iy<height; iy++){    
      image[ix][iy]=0;
      for(int t=0; t<nTri; t++){
	double x[3];
	double y[3];
	double a[3];
	double b[3];
	double c[3];
	double u[3];

	for(int i=0;i<3;i++){
	  x[i]=nodes1d[triangles[t*3+i]*2 ];
	  y[i]=nodes1d[triangles[t*3+i]*2+1 ];
	  u[i]=solution[triangles[t*3+i]];
	}

	vector2d vp=toField(QPoint(ix, iy));

	if(nodeInTriangle(vp, vector2d(x[0],y[0]),
			  vector2d(x[1],y[1]), vector2d(x[2],y[2]) ) ){

	  for(int i=0;i<3;i++){
	    int i2=(i+1)%3;
	    int i3=(i+2)%3;
	    a[i]=x[i2]*y[i3]-x[i3]*y[i2];
	    b[i]=y[i2]-y[i3];
	    c[i]=x[i3]-x[i2];
	  }
	  double A2=a[0]+b[0]*x[0]+c[0]*y[0];

	  double val=0;
	  for(int i=0;i<3;i++){
	    val+=(a[i]+b[i]*vp.x+c[i]*vp.y)*u[i]/A2;
	  }

	  image[ix][iy] = val;
	  break; //only one triangle can contain this point

	}
      }
    }
  }  
  */


  int nTri=triangles.size()/3;
  if(nTri==0) return;
  for(int t=0; t<nTri; t++){
    double x[3];
    double y[3];
    double a[3];
    double b[3];
    double c[3];
    double u[3];

    for(int i=0;i<3;i++){
      x[i]=nodes1d[triangles[t*3+i]*2 ];
      y[i]=nodes1d[triangles[t*3+i]*2+1 ];
      u[i]=solution[triangles[t*3+i]];
    }
    
    for(int i=0;i<3;i++){
      int i2=(i+1)%3;
      int i3=(i+2)%3;
      a[i]=x[i2]*y[i3]-x[i3]*y[i2];
      b[i]=y[i2]-y[i3];
      c[i]=x[i3]-x[i2];
    }
    double A2=a[0]+b[0]*x[0]+c[0]*y[0];

    double minx=x[0];
    double miny=y[0];
    double maxx=x[0];
    double maxy=y[0];
    for(int i=1;i<3;i++){
      if(minx>x[i]) minx=x[i];
      if(miny>y[i]) miny=y[i];
      if(maxx<x[i]) maxx=x[i];
      if(maxy<y[i]) maxy=y[i];
    }

    //decide x,y here
    QPoint vmin=toViewport( vector2d(minx, miny));
    QPoint vmax=toViewport( vector2d(maxx, maxy));

    for(int wx=vmin.x()-1; wx<=vmax.x()+1; wx++){
      for(int wy=vmin.y()+1; wy>=vmax.y()-1; wy--){
	vector2d vp=toField(QPoint(wx,wy));

	//test if (xp,yp) is in the triangle
	if(!nodeInTriangle(vp, vector2d(x[0],y[0]),
			   vector2d(x[1],y[1]), vector2d(x[2],y[2]) ) )
	   continue;

	double val=0;
	for(int i=0;i<3;i++){
	  val+=(a[i]+b[i]*vp.x+c[i]*vp.y)*u[i]/A2;
	}
	image[wx][wy] = val;
      }
    }
  }


}


bool ViewerWindow::nodeInTriangle(vector2d vp, vector2d v1, vector2d v2, vector2d v3)
{

  vector2d a=v2-v1;
  vector2d b=v3-v1;
  vector2d c=vp-v1;

  double A=a%a;
  double C=a%b;
  double B=b%b;
  double C1=a%c;
  double C2=b%c;
	
  double p=(C*C1-A*C2)/(C*C-B*A);
  double q=(C*C2-B*C1)/(C*C-B*A);

  double eps=0.001;

  return p+eps>=0 && q+eps>=0 && p+q-eps<=1;		
}
  






 




