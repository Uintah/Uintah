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
protected:
	void drawShape ( QPainter & );
	QCanvasView *cv;
	Module *pUse, *pProvide;
	std::string portname1, portname2;
	QColor color;
};

#endif



