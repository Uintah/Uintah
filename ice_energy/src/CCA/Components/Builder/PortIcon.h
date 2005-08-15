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

#include <qcolor.h>
#include <qrect.h>
#include <qpoint.h>
#include <string>

class Module;
class QPainter;
class QPopupMenu;
class QRect;

class PortIcon
{
public:
    enum PortType{ USES, PROVIDES };

    PortIcon(Module *module, const std::string& model,
             const std::string& type, const std::string &name,
             PortType pType, const QRect &r, const int num,
             const sci::cca::Services::pointer &services);
    ~PortIcon();
    void drawPort(QPainter &p);

    QPopupMenu* menu() const;
    QPoint portPoint();
    int number() const;
    Module* module() const;
    std::string typeName() const;
    std::string name() const;
    PortType type() const;
    QColor color() const;
    QColor highlight() const;
    QRect rect() const;

private:
    void portColorMap();

    Module *mod;
    std::string pModel;
    std::string tName;
    PortType pType;
    std::string pName;
    // allows modules to keep track of their ports & used to calc. posn.
    int num;

    QPopupMenu *pMenu;
    QColor pColor;
    QColor iColor;
    QRect pRect;
    QRect iRect;
    sci::cca::TypeMap::pointer colorMap;
    sci::cca::Services::pointer services;
};

inline QPopupMenu* PortIcon::menu() const
{
    return pMenu;
}

inline int PortIcon::number() const
{
    return num;
}

inline Module* PortIcon::module() const
{
    return mod;
}

inline std::string PortIcon::typeName() const
{
    return tName;
}

inline std::string PortIcon::name() const
{
    return pName;
}

inline PortIcon::PortType PortIcon::type() const
{
    return pType;
}

inline QColor PortIcon::color() const
{
    return pColor;
}

inline QColor PortIcon::highlight() const
{
    std::string highlight =
        colorMap->getString(std::string("highlight"), "");
    return QColor(highlight.c_str());
}

inline QRect PortIcon::rect() const
{
    return pRect;
}

#endif
