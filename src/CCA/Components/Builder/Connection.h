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

#ifndef CONNECTION_H
#define CONNECTION_H

#include <qapplication.h>
#include <qpushbutton.h>
#include <qslider.h>
#include <qlcdnumber.h>
#include <qfont.h>
#include <qvbox.h>
#include <qgrid.h>
#include <qlabel.h>
#include <qdialog.h>
#include <qscrollview.h>
#include <qmessagebox.h>
#include <qcanvas.h>
#include <qwmatrix.h>
#include <qcursor.h>
#include <vector>
#include <qlayout.h>
#include <qpainter.h>
#include <Core/CCA/spec/cca_sidl.h>
#include "Module.h"

using namespace SCIRun;

class Connection:public QCanvasPolygon
{
public:
  Connection(Module*, const std::string&, Module*, const std::string&, const sci::cca::ConnectionID::pointer &connID,  QCanvasView *cv);
	void resetPoints();
	bool isConnectedTo(Module *);
  sci::cca::ConnectionID::pointer getConnectionID();
	void highlight();
	void setDefault();
	Module * getUsesModule();
	Module * getProvidesModule();
	std::string getUsesPortName();
	std::string getProvidesPortName();
	// TEK 
	sci::cca::ConnectionID::pointer connID;
	// TEK
        virtual std::string getConnectionType();
protected:
	void drawShape ( QPainter & );
	QCanvasView *cv;
	Module *pUse, *pProvide;
	std::string portname1, portname2;
	QColor color;
};

#endif



