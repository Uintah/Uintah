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
 *  NetworkCanvasView.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <CCA/Components/Builder/NetworkCanvasView.h>
#include <qwmatrix.h>
using namespace SCIRun;

NetworkCanvasView::NetworkCanvasView(QCanvas* canvas, QWidget* parent,
				     const char* name, WFlags f)
  : QCanvasView(canvas, parent, name, f)
{
}

NetworkCanvasView::~NetworkCanvasView()
{
}

void NetworkCanvasView::contentsMousePressEvent(QMouseEvent* e)
{
  QPoint p = inverseWorldMatrix().map(e->pos());
  QCanvasItemList l=canvas()->collisions(p);
  for (QCanvasItemList::Iterator it=l.begin(); it!=l.end(); ++it) {
#if 0
    if ( (*it)->rtti() == imageRTTI ) {
      ImageItem *item= (ImageItem*)(*it);
      if ( !item->hit( p ) )
	continue;
    }
#endif
    moving = *it;
    moving_start = p;
    return;
  }
  moving = 0;
}

void NetworkCanvasView::contentsMouseMoveEvent(QMouseEvent* e)
{
  if ( moving ) {
    QPoint p = inverseWorldMatrix().map(e->pos());
    moving->moveBy(p.x() - moving_start.x(),
		   p.y() - moving_start.y());
    moving_start = p;
    canvas()->update();
  }
}
