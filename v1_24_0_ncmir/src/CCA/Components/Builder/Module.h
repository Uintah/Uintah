/*   For more information, please see: http://software.sci.utah.edu

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
 *  Module.h:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef MODULE_H
#define MODULE_H

#include <qframe.h>
#include <qpoint.h>
#include <qrect.h>
#include <qpopmenu.h>
#include <qprogressbar.h>
#include "Core/CCA/spec/cca_sidl.h"
#include "Core/CCA/PIDL/PIDL.h"
class NetworkCanvasView;

class Module:public QFrame
{
  Q_OBJECT

public:
  enum PortType{USES, PROVIDES}; 
  Module(NetworkCanvasView *parent, const std::string& name,
	 SSIDL::array1<std::string> & up, SSIDL::array1<std::string> &pp,
	 const sci::cca::Services::pointer& services,
	 const sci::cca::ComponentID::pointer &cid);
  QPoint usePortPoint(int num);
  QPoint posInCanvas();
  QPoint usePortPoint(const std::string &portname);
	
  QPoint providePortPoint(int num);
  QPoint providePortPoint(const std::string &portname);
  std::string providesPortName(int num);
  std::string usesPortName(int num);	
  QRect portRect(int portnum, PortType porttype);
  bool clickedPort(QPoint localpos, PortType &porttype,
		   std::string &portname);

  std::string moduleName;
  std::string displayName;
public slots:
  void go();
  void stop();
  void destroy();
  void ui();
signals:
  void destroyModule(Module *);

protected:
  void paintEvent(QPaintEvent *e);
  void mousePressEvent(QMouseEvent*);
  QRect nameRect;
  //  std::string moduleName;
  std::string instanceName;
  SSIDL::array1<std::string> up, pp;
private:
  sci::cca::Services::pointer services;
  QPopupMenu *menu;
  int pd; //distance between two ports
  int pw; //port width
  int ph; //prot height
  bool hasGoPort;
  bool hasUIPort;
  QProgressBar *progress;
  NetworkCanvasView * viewWindow;
public:
  sci::cca::ComponentID::pointer cid;
};

#endif
