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


#ifndef ComponentIcon_h
#define ComponentIcon_h

#include <Core/CCA/spec/cca_sidl.h>

//#include <map>
#include <vector>

//class wxRegion;

class wxWindow;
class wxPanel;
class wxGridBagSizer;
class wxButton;
class wxGauge;

namespace GUIBuilder {

class NetworkCanvas;
class PortIcon;

// to replace button?
// class MessageControl : public wxWindow {
// public:
//   MessageControl();
//   ~MessageControl();

// private:
// };

//typedef std::map<std::string, PortIcon*> PortMap;
typedef std::vector<PortIcon*> PortList;

class ComponentIcon : public wxPanel {
public:
  enum {
    ID_MENU_GO = wxID_HIGHEST,
    ID_MENU_INFO,
    ID_MENU_DELETE,
    ID_MENU_POPUP,
    ID_BUTTON_UI,
    ID_BUTTON_STATUS,
    ID_PROGRESS,
  };

  // deal with wxValidator?
  ComponentIcon(const sci::cca::GUIBuilder::pointer& bc, wxWindowID winid, NetworkCanvas* parent, const sci::cca::ComponentID::pointer& compID, int x, int y);
  virtual ~ComponentIcon();

  // draw own border?
  bool Create(wxWindow* parent, wxWindowID winid, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxNO_BORDER|wxCLIP_CHILDREN /*wxDOUBLE_BORDER|wxRAISED_BORDER|wxCLIP_CHILDREN*/);

  void OnPaint(wxPaintEvent& event);
  // Empty implementation, to prevent flicker - from wxWidgets book
  //void OnEraseBackground(wxEraseEvent& event) {}
  void OnLeftDown(wxMouseEvent& event);
  void OnLeftUp(wxMouseEvent& event);
  void OnMouseMove(wxMouseEvent& event);
  void OnRightClick(wxMouseEvent& event);
  void OnGo(wxCommandEvent& event);
  void OnDelete(wxCommandEvent& event);

  //const wxSize& GetBorderSize() const { return borderSize; }
  //void DrawPorts(wxDC& dc);

  const sci::cca::ComponentID::pointer GetComponentInstance() const { return cid; }
  const std::string GetComponentInstanceName() { return cid->getInstanceName(); }
  //PortIcon* GetPortIcon(const std::string& portName) { return ports[portName]; }
  PortIcon* GetPortIcon(const std::string& portName);

  const PortList& GetProvidesPorts() const { return providesPorts; }
  const PortList& GetUsesPorts() const { return usesPorts; }

  void GetCanvasPosition(wxPoint& p);
  NetworkCanvas* GetCanvas() const { return canvas; }

  // possible to set shape w/ bitmap region?

protected:
  ComponentIcon();
  void Init();
  void SetLayout();
  void SetPortIcons();

  NetworkCanvas *canvas;

  wxGridBagSizer* gridBagSizer;
  wxStaticText* label;
  wxStaticText* timeLabel;
  wxButton* uiButton;
  wxButton* msgButton;
  wxGauge* progressGauge;
  //wxSize borderSize;

  wxMenu *popupMenu;
  //wxMenu* goMenu;

  bool hasUIPort;
  bool hasGoPort;
  //bool isSciPort;
  bool isMoving;

  sci::cca::ComponentID::pointer cid;
  sci::cca::GUIBuilder::pointer builder;
  std::string goPortName;
  std::string uiPortName;

  //PortMap ports;
  PortList usesPorts;
  PortList providesPorts;
  wxPoint movingStart;

  static const int PROV_PORT_COL = 0;
  static const int USES_PORT_COL = 5;
  static const int GAP_SIZE = 1;
  static const int BORDER_SIZE = 4;
  static const int PORT_BORDER_SIZE = 10;

private:
  ComponentIcon(const ComponentIcon&);
  ComponentIcon& operator=(const ComponentIcon&);

  DECLARE_EVENT_TABLE()
  DECLARE_DYNAMIC_CLASS(ComponentIcon)
  //DECLARE_NO_COPY_CLASS(ComponentIcon)
};

}

#endif
