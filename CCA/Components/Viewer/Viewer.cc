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
 *  Viewer.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#include <CCA/Components/Viewer/Viewer.h>
#include <iostream>
#include <CCA/Components/Builder/QtUtils.h>

#include <qapplication.h>
#include <qpushbutton.h>
#include <qmainwindow.h>
#include <qmessagebox.h>
#include <qapplication.h>
#include <qframe.h>
#include <qpainter.h>
#include <qcolor.h>
#include <qdialog.h>

#include <math.h>
#include <stdlib.h>
#include <iostream.h>
#include <fstream.h>


using namespace std;
using namespace SCIRun;

extern "C" gov::cca::Component::pointer make_SCIRun_Viewer()
{
  return gov::cca::Component::pointer(new Viewer());
}







class Matrix: public gov::cca::Matrix{
public:
  Matrix();
  Matrix(int nRow, int nCol);
  ~Matrix();
  double getElement(int row, int col);
  void setElement(int row, int col, double val);
  int numOfRows();
  int numOfCols();
  Matrix & operator=(const Matrix &m);
private:
  void copy(const Matrix &m);
  int nRow;
  int nCol;
  double *data;
};

Matrix::Matrix()
{
  nRow=nCol=0;
  data=0;
}

Matrix::Matrix(int nRow, int nCol)
{
  this->nRow=nRow;
  this->nCol=nCol;
  data=new double[nRow*nCol];
}  

Matrix::~Matrix()
{
  if(data!=0) delete []data;
}

void Matrix::copy(const Matrix &m)
{
  if(data!=0) delete data;
  nRow=m.nRow;
  nCol=m.nCol;
  data=new double[nRow*nCol];
  memcpy(data, m.data, nRow*nCol*sizeof(double));
}

Matrix & Matrix::operator=(const Matrix &m)
{
  copy(m);
  return *this;
}

double Matrix::getElement(int row, int col)
{
  return data[row*nCol+col];
}

void Matrix::setElement(int row, int col, double val)
{
  data[row*nCol+col]=val;
}


int Matrix::numOfRows()
{
  return nRow;
}

int Matrix::numOfCols()
{
  return nCol;
}


class ViewerWindow:public QDialog
{
public:
    ViewerWindow( QWidget *parent,  const gov::cca::Matrix::pointer &m);
		int height();
		int width();
		int w, h;
protected:
		void	paintEvent(QPaintEvent*e);
		void	ViewerWindow::mousePressEvent(QMouseEvent* e);
		int border;
		bool showCircles;
  gov::cca::Matrix::pointer matrix;
};

ViewerWindow::ViewerWindow( QWidget *parent, 
			    const gov::cca::Matrix::pointer &m)
  : QDialog( parent )
{

  matrix=m;
  w=h=500;
  border=20;
  setGeometry(QRect(200,200,w,h));
  showCircles=true;
}

//define a smaller window height
int ViewerWindow::height()
{
  return QDialog::height()-border*2;
}

//define a smaller window width
int ViewerWindow::width()
{
  return QDialog::width()-border*2;
}

void ViewerWindow::paintEvent(QPaintEvent* e)
{
  QDialog::paintEvent(e);
  QPainter p(this);
  int R=5;
  for(int r=0; r<matrix->numOfRows(); r++){
     double val=matrix->getElement(r,2);
     int c=int(val*255);
     p.setBrush(QColor(c,c,c));
     int x=int(matrix->getElement(r,0)*width());
     int y=int(matrix->getElement(r,1)*height());
     p.drawEllipse(border+x-R, border+y-R, R+R, R+R);
  }
}

//left button to add one node
//right button to toggle the option: showCircles
void ViewerWindow::mousePressEvent( QMouseEvent * )
{
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
}




Viewer::Viewer()
{
  uiPort.setParent(this);
}

Viewer::~Viewer()
{
  cerr << "called ~Viewer()\n";
}

void Viewer::setServices(const gov::cca::Services::pointer& svc)
{
  services=svc;
  //register provides ports here ...  

  gov::cca::TypeMap::pointer props = svc->createTypeMap();
  myUIPort::pointer uip(&uiPort);
  svc->addProvidesPort(uip,"ui","gov.cca.UIPort", props);
  svc->registerUsesPort("dataPort", "gov.cca.Field2DPort",props);
  // Remember that if the PortInfo is created but not used in a call to the svc object
  // then it must be freed.
  // Actually - the ref counting will take care of that automatically - Steve
}

void myUIPort::ui() 
{
  gov::cca::Port::pointer pp=com->getServices()->getPort("dataPort");	
  if(pp.isNull()){
    QMessageBox::warning(0, "Viewer", "dataPort is not available!");
    return;
  }  
  gov::cca::ports::Field2DPort::pointer fport=
    pidl_cast<gov::cca::ports::Field2DPort::pointer>(pp);
  gov::cca::Matrix::pointer m=fport->getField();	
  /*    int nRow=20;
    int nCol=3;
    Matrix m(nRow, nCol);
    for(int r=0; r<nRow; r++){
      m.setElement(r,0,drand48());
      m.setElement(r,1,drand48());
      m.setElement(r,2,drand48());      
    }
  */
    (new ViewerWindow(0, m))->show();
}


 


