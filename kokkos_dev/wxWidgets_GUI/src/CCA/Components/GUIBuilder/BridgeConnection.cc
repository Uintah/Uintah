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

#include <CCA/Components/GUIBuilder/BridgeConnection.h>

namespace GUIBuilder {

BridgeConnection::BridgeConnection(PortIcon* pU, PortIcon* pP, const sci::cca::ConnectionID::pointer& connID, bool possibleConnection) : Connection(pU, pP, connID, possibleConnection)
{
}


void BridgeConnection::OnDraw(wxDC& dc)
{
  ResetPoints();
  setConnection();
  if (IsHighlighted()) {
    wxPen* pen = wxThePenList->FindOrCreatePen(hColor, 2, wxLONG_DASH);
    dc.SetPen(*pen);
  } else {
    wxPen* pen = wxThePenList->FindOrCreatePen(color, 2, wxLONG_DASH);
    dc.SetPen(*pen);
  }
  dc.SetBrush(*wxTRANSPARENT_BRUSH);
  dc.DrawLines(NUM_DRAW_POINTS, drawPoints, 0, 0);
}

}

#if 0

// void BridgeConnection::drawShape(QPainter& p)
// {
//   QPointArray par(Connection::NUM_DRAW_POINTS);
//   Connection::drawConnection(p, points(), par);

//   QPen pen(color,4);
//   pen.setStyle(DotLine);
//   p.setPen(pen);
//   p.setBrush(blue);
//   p.drawPolyline(par);
// }

// std::string BridgeConnection::getConnectionType()
// {
//   return "BridgeConnection";
// }

#endif
