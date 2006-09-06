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


#include <CCA/Components/Viewer/ViewerWindow.h>

namespace Viewer {

using namespace SCIRun;

BEGIN_EVENT_TABLE(ViewerWindow, wxPanel)
  EVT_LEFT_UP(ViewerWindow::OnLeftClick)
  EVT_RIGHT_UP(ViewerWindow::OnRightClick)
  EVT_PAINT(ViewerWindow::OnPaint)
END_EVENT_TABLE()

ViewerWindow::ViewerWindow(wxWindow *parent,
                           Colormap *cmap,
                           const std::vector<double> nodes1d,
                           const std::vector<int> triangles,
                           const std::vector<double> solution)
  : wxPanel(parent, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSUNKEN_BORDER|wxDOUBLE_BORDER),
    cmap(cmap),
    nodes1d(nodes1d),
    triangles(triangles),
    solution(solution),
    showMesh(false),
    showCoordinates(false),
    borderX(12*5),
    borderY(20)
{
//   setSizePolicy(QSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding, 100,200));
//   setBackground("Gray");
//   setLineWidth(2);

  const int N = nodes1d.size() / 2;

  if (N == 0) {
    return;
  }
  minx = maxx = nodes1d[0];
  miny = maxy = nodes1d[1];

  for (int i = 0; i < N; i++) {
    if (minx > nodes1d[i*2]) {
      minx = nodes1d[i*2];
    }
    if (maxx < nodes1d[i*2]) {
      maxx = nodes1d[i*2];
    }
    if (miny > nodes1d[i*2+1]) {
      miny = nodes1d[i*2+1];
    }
    if (maxy < nodes1d[i*2+1]) {
      maxy = nodes1d[i*2+1];
    }
  }

  double minVal = solution[0];
  double maxVal = solution[0];
  for (int i = 0; i < N; i++) {
    if (minVal > solution[i]) {
      minVal = solution[i];
    }
    if (maxVal < solution[i]) {
      maxVal = solution[i];
    }
  }
  cmap->setValues(minVal, maxVal);
}

void ViewerWindow::refresh(const wxString &type)
{
  cmap->setType(type);
  setBackground(type);
  Refresh();
}

void ViewerWindow::setBackground(const wxString &type)
{
//   if(type=="Gray")  setPaletteBackgroundColor(QColor(0,127,0));
//   if(type=="Color") setPaletteBackgroundColor(QColor(255,255,255));
}

void ViewerWindow::OnToggleMesh(wxCommandEvent& event)
{
  showMesh = !showMesh;
  Refresh();
}

void ViewerWindow::OnToggleCoordinates(wxCommandEvent& event)
{
  showCoordinates = !showCoordinates;
  Refresh();
}

wxPoint ViewerWindow::toViewport(vector2d v)
{
  int w, h;
  GetSize(&w, &h);

  int x = int(borderX + (v.getX() - minx) * w / (maxx - minx));
  int y = int(borderY + h - (v.getY() - miny) * h / (maxy - miny));
  return wxPoint(x, y);
}

vector2d ViewerWindow::toField(wxPoint p)
{
  int w, h;
  GetSize(&w, &h);
  double x = minx + (p.x - borderX) * (maxx - minx) / w;
  double y = miny + (h - (p.y - borderY)) * (maxy - miny) / h;
  return vector2d(x, y);
}

//define a smaller window height
int ViewerWindow::height()
{
  int w, h;
  GetSize(&w, &h);
  return h - borderY * 2;
}

//define a smaller window width
int ViewerWindow::width()
{
  int w, h;
  GetSize(&w, &h);
  return w - borderX * 2;
}

void ViewerWindow::OnPaint(wxPaintEvent& event)
{
//   QFrame::paintEvent(e);
//   QPainter p(this);

//   //cerr<<"solution size="<<solution.size()<<endl;
//   //cerr<<"nodes1d size="<<nodes1d.size()<<endl;

//   p.setPen(NoPen);

//   int nTri=triangles.size()/3;
//   if(nTri==0) return;
//   for(int t=0; t<nTri; t++){
//     double x[3];
//     double y[3];
//     double a[3];
//     double b[3];
//     double c[3];
//     double u[3];

//     for(int i=0;i<3;i++){
//       x[i]=nodes1d[triangles[t*3+i]*2 ];
//       y[i]=nodes1d[triangles[t*3+i]*2+1 ];
//       u[i]=solution[triangles[t*3+i]];
//     }


//     for(int i=0;i<3;i++){
//       int i2=(i+1)%3;
//       int i3=(i+2)%3;
//       a[i] = x[i2]*y[i3]-x[i3]*y[i2];
//       b[i] = y[i2]-y[i3];
//       c[i] = x[i3]-x[i2];
//     }
//     double A2=a[0]+b[0]*x[0]+c[0]*y[0];


//     double minx=x[0];
//     double miny=y[0];
//     double maxx=x[0];
//     double maxy=y[0];
//     for(int i=1;i<3;i++){
//       if(minx>x[i]) minx = x[i];
//       if(miny>y[i]) miny = y[i];
//       if(maxx<x[i]) maxx = x[i];
//       if(maxy<y[i]) maxy = y[i];
//     }

//     //decide x,y here
//     QPoint vmin=toViewport( vector2d(minx, miny));
//     QPoint vmax=toViewport( vector2d(maxx, maxy));

//     for(int wx=vmin.x()-1; wx<=vmax.x()+1; wx++){
//       for(int wy=vmin.y()+1; wy>=vmax.y()-1; wy--){
//   vector2d vp=toField(QPoint(wx,wy));

//   //test if (xp,yp) is in the triangle
//   if(!nodeInTriangle(vp, vector2d(x[0],y[0]),
//          vector2d(x[1],y[1]), vector2d(x[2],y[2]) ) )
//      continue;

//   double val=0;
//   for(int i=0;i<3;i++){
//     val+=(a[i]+b[i]*vp.x+c[i]*vp.y)*u[i]/A2;
//   }
//   p.setPen(cmap->getColor(val));
//   p.drawPoint(QPoint(wx,wy));
//       }
//     }
//   }

//   if(showMesh){
//     p.setPen(QColor(255,0,255));
//     for(int i=0;i<nTri;i++){
//       double x1= nodes1d[triangles[i*3]*2];
//       double y1= nodes1d[triangles[i*3]*2+1];
//       double x2= nodes1d[triangles[i*3+1]*2];
//       double y2= nodes1d[triangles[i*3+1]*2+1];
//       double x3= nodes1d[triangles[i*3+2]*2];
//       double y3= nodes1d[triangles[i*3+2]*2+1];
//       QPoint p1=toViewport(vector2d(x1,y1));
//       QPoint p2=toViewport(vector2d(x2,y2));
//       QPoint p3=toViewport(vector2d(x3,y3));

//       p.drawLine(p1,p2);
//       p.drawLine(p2,p3);
//       p.drawLine(p3,p1);
//     }


//     p.setBrush(black);
//     int r=3;
//     for(unsigned i=0;i<nodes1d.size();i+=2){
//       QPoint vp=toViewport(vector2d(nodes1d[i],nodes1d[i+1]));
//       p.drawEllipse(vp.x()-r, vp.y()-r, r+r, r+r);
//       //char s[5];
//       //sprintf(s,"%d",i%2);
//       //p.drawText(x+r,y+r,s);
//     }
//   }


//   //display corrdinates
//   if(showCoordinates){
//     int nx=5;
//     int ny=5;

//     p.setPen(blue);
//     p.setBrush(NoBrush);
//     p.drawRect(borderX,borderY,width(),height());
//     p.setBrush(blue);


//     for(int ix=0; ix<=nx; ix++){
//       for(int iy=0; iy<=ny; iy+=ny){
//   vector2d vp(minx+ix*(maxx-minx)/nx, miny+iy*(maxy-miny)/ny);
//   char s[10];
//   if( fabs(vp.x)>0.01 && fabs(vp.x)<100){
//     sprintf(s,"%6.4lf",vp.x);
//   }
//   else{
//     sprintf(s,"%7.2lf",vp.x);
//   }

//   QPoint wp=toViewport(vp);
//   int r=2;

//   p.drawEllipse(wp.x()-r, wp.y()-r, 2*r, 2*r);
//   int dx=-12*2;
//   int dy= iy==0 ? 12+3: (-3);

//   p.drawText(wp+QPoint(dx, dy), s);
//       }
//     }

//     for(int iy=0; iy<=ny; iy++){
//       for(int ix=0; ix<=nx; ix+=nx){
//   vector2d vp(minx+ix*(maxx-minx)/nx, miny+iy*(maxy-miny)/ny);
//   char s[10];
//   if( fabs(vp.y)>0.01 && fabs(vp.y)<100){
//     sprintf(s,"%6.4lf",vp.y);
//   }
//   else{
//     sprintf(s,"%7.2lf",vp.y);
//   }

//   QPoint wp=toViewport(vp);
//   int r=2;

//   p.drawEllipse(wp.x()-r, wp.y()-r, 2*r, 2*r);
//   int dy=12/2;
//   int dx= ix==0 ? -12*5 : 3;

//   p.drawText(wp+QPoint(dx, dy), s);
//       }
//     }
//   }

}

bool ViewerWindow::nodeInTriangle(vector2d vp, vector2d v1, vector2d v2, vector2d v3)
{

  vector2d a = v2 - v1;
  vector2d b = v3 - v1;
  vector2d c = vp - v1;

  double A = a % a;
  double C = a % b;
  double B = b % b;
  double C1 = a % c;
  double C2 = b % c;

  double p = (C * C1 - A * C2) / (C * C - B * A);
  double q = (C * C2 - B * C1) / (C * C - B * A);

  //std::cerr << "p,q=" << p << " " << q << std::endl;
  return p >= 0 && q >= 0 && p + q <= 1;
}


//left button to add one node
//right button to toggle the option: showCircles
// void ViewerWindow::mousePressEvent( QMouseEvent * )
// {
  /*QPoint p=e->pos();
    if(e->button()==LeftButton){
    vector2d v=vector2d(2.0*(p.x()-border)/width(),2-2.0*(p.y()-border)/height());
    mesh.addNode(v);
    mesh.triangulation();
    update();
  }
  else if(e->button()==RightButton){
    showCircles=!showCircles;
    update();
  }
*/
// }

void ViewerWindow::OnLeftClick(wxMouseEvent& event)
{
}

void ViewerWindow::OnRightClick(wxMouseEvent& event)
{
}

}
