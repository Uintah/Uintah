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

/*
 * Connection.h
 *
 * Written by:
 *  Keming Zhang
 *  SCI Institute
 *  April 2002
 *
 * Ported to wxWidgets by:
 *  Ayla Khan
 *  January 2006
 */

#ifndef CCA_Components_GUIBuilder_Connection_h
#define CCA_Components_GUIBuilder_Connection_h

#include <Core/CCA/spec/cca_sidl.h>
#include <CCA/Components/GUIBuilder/GUIBuilder.h>
// #include <wx/event.h>

class wxPoint;
class wxDC;
class wxColor;

namespace GUIBuilder {

class PortIcon;

class Connection {
public:
  Connection(PortIcon* pU,
             PortIcon* pP,
             const sci::cca::ConnectionID::pointer& connID,
             bool possibleConnection = false);
  virtual ~Connection();

  void ResetPoints();
  virtual void OnDraw(wxDC& dc);

  bool IsMouseOver(const wxPoint& position);
  void Highlight() { highlight = true; }
  void Unhighlight() { highlight = false; }
  bool IsHighlighted() const { return highlight; }

  const sci::cca::ConnectionID::pointer GetConnectionID() const { return connectionID; }
  PortIcon* GetProvidesPortIcon() const { return pProvides; }
  PortIcon* GetUsesPortIcon() const { return pUses; }
  static const int GetDrawingPointsSize() { return NUM_DRAW_POINTS; }
  void GetDrawingPoints(wxPoint **pa, const int size);

  const static int PEN_WIDTH = 4;

protected:
  void setConnection(wxDC& dc);

  static const int NUM_POINTS;
  static const int NUM_DRAW_POINTS;
  wxColor color;
  wxColor hColor;
  wxPoint* drawPoints;

private:
  wxPoint* points;
  PortIcon* pUses;
  PortIcon* pProvides;
  bool possibleConnection;
  bool highlight;
  sci::cca::ConnectionID::pointer connectionID;
};

}

#endif
