#ifndef CONNECTION_H
#define CONNECTION_H

#include <iostream.h>
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
#include "Module.h"

class Connection:public QCanvasPolygon
{
public:
  Connection(Module*, Module*, QCanvasView *cv);
	void resetPoints();
	bool isConnectedTo(Module *);
protected:
	void drawShape ( QPainter & );
	QCanvasView *cv;
	Module *pUse, *pProvide;

};

#endif
