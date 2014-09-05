#include <vector>
#include <iostream>
#include <stdlib.h>
#include <qwidget.h>
#include <qpainter.h>
#include <qpixmap.h>

#include "ZGraph.h"


ZGraph::ZGraph( QWidget *parent, const char *name )
       : QWidget( parent, name )
{
	style=false;
}

void ZGraph::refresh()
{
   repaint();
}

void ZGraph::setData(const double *val, int size)
{
	std::vector<double> v;
	for(int i=0; i<size; i++)
		v.push_back(val[i]);
	this->val=v;
}

void ZGraph::setStyle(bool style)
{
	this->style=style;
	refresh();
}

void ZGraph::paintEvent(QPaintEvent *)
{
	int np=val.size();
	if(np==0 ) return;

	QPainter p( this );
	int dx=width()/10;
	int dy=height()/10;
        int w=width()-dx*2;
        int h=height()-dy*2;
	const int r=2;
	QPixmap pix( size() );

	QPainter tmp( &pix );
	
        tmp.setBrush(white);
        tmp.drawRect(rect());
       
	tmp.setPen(blue);
	tmp.setBrush(red);
	
	double min, max;
	min=max=val[0];
	for(int i=1; i<np; i++){
		if(max<val[i]) max=val[i];
		if(min>val[i]) min=val[i];
	}
	
	double span=max-min;
	int x1=0, y1=0, x2, y2;
	
        for(int i  = 0; i<np; i++)
        {
	        if(span==0){
			y2=dy+h/2;
		}
		else{
			y2=dy+h-int((val[i]-min)*h/span);
		}
		x2=dx+int(i*w/(np-1));
		if(i>0 && style) tmp.drawLine(x1, y1, x2, y2);
		x1=x2;
		y1=y2;
		tmp.drawEllipse(x1-r,y1-r,r+r,r+r);
    	}
	
  	p.drawPixmap( rect().topLeft(), pix );
}

