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

#include <Core/CCA/spec/cca_sidl.h>
#include <CCA/Components/Builder/PortIcon.h>

#include <qcanvas.h>
#include <qcursor.h>
#include <qpainter.h>

#include <vector>

using namespace SCIRun;

class Module;

/**
 * \class Connection
 *
 * \brief Represents a connection between components in the Builder GUI.
 */

class Connection : public QCanvasPolygon
{
public:
    Connection(PortIcon *pU, PortIcon* pP, const sci::cca::ConnectionID::pointer &connID, QCanvasView *cv);
    void resetPoints();
    bool isConnectedTo(Module *who);
    sci::cca::ConnectionID::pointer getConnectionID() const;
    void highlight();
    void setDefault();
    PortIcon* usesPort() const;
    PortIcon* providesPort() const;

    // TEK
    virtual std::string getConnectionType();

    static void drawConnection(QPainter& p,
                               const QPointArray& points,
                               QPointArray& drawPoints);
    virtual int rtti() const;

    static int RTTI;
    static const int Rtti_Connection = 2001;
    static const unsigned int NUM_POINTS = 12;
    static const unsigned int NUM_DRAW_POINTS = 6;

protected:
    /** Override QCanvasPolygon::drawShape to draw a polyline not a polygon. */
    virtual void drawShape(QPainter &);
    QColor color;

private:
    QCanvasView *cv;
    PortIcon *pUse, *pProvide;
    std::string uPortName, pPortName;
    sci::cca::ConnectionID::pointer connID;
};


inline sci::cca::ConnectionID::pointer Connection::getConnectionID() const
{
  return connID;
}

// Distinguish between QCanvasItems on a QCanvas.
inline int Connection::rtti() const
{
    return RTTI;
}

inline void Connection:: setDefault()
{
  color = pUse->color();
}

inline void Connection:: highlight()
{
  color = pUse->highlight();
}

inline PortIcon* Connection::usesPort() const
{
  return pUse;
}

inline PortIcon* Connection::providesPort() const
{
  return pProvide;
}

inline std::string Connection::getConnectionType()
{
  return "Connection";
}

inline void Connection::drawConnection(QPainter& p,
            const QPointArray& points, QPointArray& drawPoints)
{
    for (unsigned int i = 0; i < NUM_DRAW_POINTS; i++) {
        drawPoints[i] = ( points[i] + points[NUM_POINTS - 1 - i] ) / 2;
    }
}

/**
 * \class MiniConnection
 *
 * \brief Draw a connection on the BuilderWindow's miniCanvas.
 */
class MiniConnection : public QCanvasPolygon
{
public:
    MiniConnection(QCanvasView *cv, QPointArray pa, double scaleX, double scaleY);

protected:
    /** Override QCanvasPolygon::drawShape to draw a polyline not a polygon. */
    virtual void drawShape(QPainter&);

private:
    void scalePoints(QPointArray *pa, double scaleX, double scaleY);
};

#endif



