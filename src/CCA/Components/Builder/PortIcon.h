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
 * PortIcon.h
 *
 */

#ifndef PortIcon_h
#define PortIcon_h

#include <Core/CCA/spec/cca_sidl.h>
#include <CCA/Components/Builder/Builder.h>

class wxWindow;

namespace GUIBuilder {

class NetworkCanvas;
class ComponentIcon;

class PortIcon : public wxWindow {
public:
  PortIcon(ComponentIcon *parent, wxWindowID id, Builder::PortType pt, const std::string& name);
  virtual ~PortIcon();
  bool Create(wxWindow *parent, wxWindowID id, const wxString &name);
  void OnLeftDown(wxMouseEvent& event);
  void OnLeftUp(wxMouseEvent& event);
  void OnMouseMove(wxMouseEvent& event);
  void OnRightClick(wxMouseEvent& event);

  wxColour GetPortColour() { return pColour; }
  const std::string GetPortName() const { return name; }
  Builder::PortType GetPortType() const { return type; }

  ComponentIcon* GetParent() const { return parent; }

  //void PortIcon::OnDraw(wxDC& dc)

  const static int PORT_WIDTH = 7;
  const static int PORT_HEIGHT = 10;
  const static int HIGHLIGHT_WIDTH = 2;
  //const static int PORT_DISTANCE = 10;

protected:
  PortIcon();
  void Init();

private:
  PortIcon(const PortIcon&);
  PortIcon& operator=(const PortIcon&);

  ComponentIcon* parent;
  Builder::PortType type;
  std::string name;
  bool connecting;

  wxRect hRect;
  //wxRegion region;

  wxColour hColour;
  wxColour pColour;

  const int ID_MENU_POPUP;

  DECLARE_DYNAMIC_CLASS(PortIcon)
  DECLARE_EVENT_TABLE()
};

}

#endif
