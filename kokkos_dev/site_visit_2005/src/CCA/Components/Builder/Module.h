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

#include <Core/CCA/spec/cca_sidl.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <CCA/Components/Builder/PortIcon.h>

#include <qframe.h>
#include <qrect.h>
#include <qpopmenu.h>
#include <qprogressbar.h>

class NetworkCanvasView;
class PortIcon;

class QPoint;
class QPushButton;

class Module : public QFrame
{
    Q_OBJECT

public:
    Module(NetworkCanvasView *parent,
           const std::string& name,
           SSIDL::array1<std::string> &up,
           SSIDL::array1<std::string> &pp,
           const sci::cca::Services::pointer &services,
           const sci::cca::ComponentID::pointer &cid);
    virtual ~Module();
    QPoint posInCanvas();
    PortIcon* clickedPort(QPoint localpos);
    QPoint usesPortPoint(int num);
    QPoint providesPortPoint(int num);
    PortIcon* getPort(const std::string &name, PortIcon::PortType type);
    NetworkCanvasView* parent() const;

    std::string moduleName() const;
    std::string displayName() const;
    sci::cca::ComponentID::pointer componentID() const;

    // defaults -- replace with framework properties?
    static const int PORT_DIST = 10;
    static const int PORT_W = 4;
    static const int PORT_H = 10;

public slots:
    void go();
    void stop();
    void destroy();
    void ui();
    void desc();

signals:
    void destroyModule(Module *);

protected:
    virtual void paintEvent(QPaintEvent *);
    virtual void mousePressEvent(QMouseEvent *);
    virtual void timerEvent(QTimerEvent *);

    QRect nameRect;
    std::string instanceName;
//     SSIDL::array1<std::string> up, pp;

private:
  PortIcon* port(int portnum, const std::string& model,
                 const std::string& type, const std::string& portname,
                 PortIcon::PortType porttype);

    std::string mName;
    std::string dName;
    std::string mDesc;
    std::string uiPortName;
    std::string goPortName;
    std::string iconName;

    sci::cca::Services::pointer services;
    sci::cca::ComponentID::pointer cid;

    QPushButton *uiButton;
    //QPushButton *statusButton;
    QPopupMenu *menu;
    QProgressBar *progress;
    NetworkCanvasView *viewWindow;
    std::vector<PortIcon*> ports;

    int pd; //distance between two ports
    int pw; //port width
    int ph; //port height
    int steps;
    bool hasGoPort;
    bool hasUIPort;
    bool hasComponentIcon;
};

inline NetworkCanvasView* Module::parent() const
{
    return viewWindow;
}

inline sci::cca::ComponentID::pointer Module::componentID() const
{
    return cid;
}

inline std::string Module::moduleName() const
{
    return mName;
}

inline std::string Module::displayName() const
{
    return dName;
}


#endif
