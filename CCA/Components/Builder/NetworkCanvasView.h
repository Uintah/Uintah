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
 *  NetworkCanvas.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_NetworkCanvas_h
#define SCIRun_Framework_NetworkCanvas_h

#include <qcanvas.h>

namespace SCIRun {
  class NetworkCanvasView : public QCanvasView {
    Q_OBJECT
  public:
    NetworkCanvasView(QCanvas* canvas, QWidget* parent=0, const char* name=0,
		  WFlags f=0);
    virtual ~NetworkCanvasView();

  protected:
    void contentsMousePressEvent(QMouseEvent*);
    void contentsMouseMoveEvent(QMouseEvent*);

  private:
    QCanvasItem* moving;
    QPoint moving_start;

    NetworkCanvasView(const NetworkCanvasView&);
    NetworkCanvasView& operator=(const NetworkCanvasView&);
  };
}

#endif
