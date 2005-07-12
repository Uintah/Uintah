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
 * PortIcon.cc
 *
 */

#include <CCA/Components/Builder/PortIcon.h>
#include <CCA/Components/Builder/Module.h>
#include <CCA/Components/Builder/NetworkCanvasView.h>
#include <CCA/Components/Builder/BuilderWindow.h>
#include <SCIRun/TypeMap.h>

#include <qmenudata.h>
#include <qpainter.h>
#include <qpopupmenu.h>
#include <iostream>

using namespace SCIRun;

// TODO: filter menu for type, find a better way to show port type
PortIcon::PortIcon(Module *module, const std::string& model,
                   const std::string& type, const std::string &name,
                   PortType pType, const QRect &r, const int num,
                   const sci::cca::Services::pointer &services)
    : mod(module), pModel(model), tName(type), pType(pType), pName(name),
      num(num), pRect(r), services(services)
{
  //    std::cerr << "PortIcon::PortIcon " << name << " model=" << model << std::endl;
    portColorMap();
    if (pType == USES) {
        iColor = Qt::green;
        iRect = QRect(r.x() - Module::PORT_W, r.top(), 3, r.height());
    } else { // PROVIDES
        iColor = Qt::red;
        iRect = QRect(r.x() + Module::PORT_W, r.top(), 3, r.height());
    }

    unsigned int i;
    std::string t;
    if ((i = type.find_last_of(".")) != std::string::npos) {
        t = type.substr(i + 1);
    } else {
        t = type;
    }

    std::string color = colorMap->getString(t, "");
    if (color.empty()) {
        color = colorMap->getString(std::string("default"), "");
    }
    pColor.setNamedColor(color.c_str());

    //TODO: find a better way to detect the component model
    pMenu = new QPopupMenu((QWidget *) module);
    std::string s;
    if ("cca" == model || "CCA" == model) {
        s = "CCA";
    } else if ("babel" == model) {
        s = "babel";
    } else if ("Vtk" == model || "vtk" == model) {
        s = "Vtk";
    } else if ("Dataflow" == model || "dataflow" == model) {
        s = "Dataflow";
    }

    if (! s.empty() ) {
        MenuMap *menus = module->parent()->p2BuilderWindow->packageMenus();
        (*menus)[s]->populateMenu(pMenu);
    } else {
        pMenu->hide();
    }
}

PortIcon::~PortIcon()
{
}

void PortIcon::drawPort(QPainter &p)
{
    p.setPen(pColor);
    p.setBrush(pColor);
    p.drawRect(pRect);

    p.setPen(iColor);
    p.setBrush(iColor);
    p.drawRect(iRect);
}

QPoint PortIcon::portPoint()
{
    if (pType == USES) {
        return mod->usesPortPoint(num);
    } else {
        return mod->providesPortPoint(num);
    }
}

// better way to figure out how to map color to port type?
void PortIcon::portColorMap()
{
    colorMap = sci::cca::TypeMap::pointer(new TypeMap);

    // using named colors
    // Qt can use these for either Unix or Windows systems
    // see Qt documentation for QColor
    colorMap->putString(std::string("default"),
                        std::string("yellow"));
    colorMap->putString(std::string("highlight"),
                        std::string("white"));
    colorMap->putString(std::string("StringPort"),
                        std::string("cadetblue1"));
    colorMap->putString(std::string("ZListPort"),
                        std::string("goldenrod4"));
    colorMap->putString(std::string("LinSolverPort"),
                        std::string("darkorange"));
    colorMap->putString(std::string("PDEdescriptionPort"),
                        std::string("tomato"));
    colorMap->putString(std::string("MeshPort"),
                        std::string("magenta"));
    colorMap->putString(std::string("ViewPort"),
                        std::string("darkseagreen"));
    colorMap->putString(std::string("FEMmatrixPort"),
                        std::string("gray65"));
}
