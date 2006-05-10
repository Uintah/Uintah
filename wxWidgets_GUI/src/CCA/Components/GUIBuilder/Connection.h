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

#ifndef Connection_h
#define Connection_h

#include <Core/CCA/spec/cca_sidl.h>
#include <CCA/Components/GUIBuilder/GUIBuilder.h>
// #include <wx/event.h>

class wxPoint;
class wxDC;
class wxColor;

// should probably check for wxUSE_THREADS

namespace GUIBuilder {

class PortIcon;

class Connection /*: public wxEvtHandler*/ {
public:
  Connection(PortIcon* pU, PortIcon* pP, const sci::cca::ConnectionID::pointer& connID,
             bool possibleConnection = false);
  virtual ~Connection();

  void ResetPoints();
  void OnDraw(wxDC& dc);

  bool IsMouseOver(const wxPoint& position);
  void HighlightConnection() { highlight = true; }
  void UnhighlightConnection() { highlight = false; }
  bool IsHighlighted() const { return highlight; }

  const sci::cca::ConnectionID::pointer GetConnectionID() const { return connectionID; }
  PortIcon* GetProvidesPortIcon() const { return pProvides; }
  PortIcon* GetUsesPortIcon() const { return pUses; }
  static const int GetDrawingPointsSize() { return NUM_DRAW_POINTS; }
  void GetDrawingPoints(wxPoint **pa, const int size);

protected:
  wxColor color;
  wxColor hColor;
  void setConnection();

private:
  static const int NUM_POINTS;
  static const int NUM_DRAW_POINTS;
  wxPoint* points;
  wxPoint* drawPoints;
  PortIcon* pUses;
  PortIcon* pProvides;
  bool possibleConnection;
  bool highlight;
  sci::cca::ConnectionID::pointer connectionID;

//   DECLARE_EVENT_TABLE()
};

}

#endif
