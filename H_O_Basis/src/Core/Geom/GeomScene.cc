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
 *  GeomScene.cc: ?
 *
 *  Written by:
 *   Author?
 *   Department of Computer Science
 *   University of Utah
 *   Date?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <Core/Geom/GeomScene.h>

#include <Core/Geom/Lighting.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/GeomSave.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/Trig.h>

#include <iostream>
using std::cerr;
using std::endl;
using std::ostream;
#include <fstream>
using std::ofstream;

namespace SCIRun {


GeomScene::GeomScene()
{
    lighting=new Lighting;
}

GeomScene::GeomScene(const Color& bgcolor, const View& view,
		   Lighting* lighting, GeomObj* top)
: bgcolor(bgcolor), view(view), lighting(lighting), top(top)
{
}

#define GEOMSCENE_VERSION 1

void GeomScene::io(Piostream& stream)
{

    stream.begin_class("GeomScene", GEOMSCENE_VERSION);
    Pio(stream, bgcolor);
    Pio(stream, view);
    Pio(stream, *lighting);
    Pio(stream, top);
    stream.end_class();
}

void GeomSave::indent()
{
    nindent+=4;
}

void GeomSave::unindent()
{
    nindent-=4;
}

void GeomSave::indent(ostream& out)
{
    for(int i=0;i<nindent;i++){
	out << ' ';
    }
}

// These guys are for VRML.
void GeomSave::start_sep(ostream& out)
{
    indent(out);
    out << "Separator {\n";
    indent();
}

void GeomSave::end_sep(ostream& out)
{
    unindent();
    indent(out);
    out << "}\n";
}

void GeomSave::start_tsep(ostream& out)
{
    indent(out);
    out << "TransformSeparator {\n";
    indent();
}

void GeomSave::end_tsep(ostream& out)
{
    unindent();
    indent(out);
    out << "}\n";
}

// These guys are for RIB.
void GeomSave::start_attr(ostream& out)
{
    indent(out);
    out << "AttributeBegin\n";
    indent();
}

void GeomSave::end_attr(ostream& out)
{
    unindent();
    indent(out);
    out << "AttributeEnd\n";
}

void GeomSave::start_trn(ostream& out)
{
    indent(out);
    out << "TransformBegin\n";
    indent();
}

void GeomSave::end_trn(ostream& out)
{
    unindent();
    indent(out);
    out << "TransformEnd\n";
}

void GeomSave::start_node(ostream& out, char* name)
{
    indent(out);
    out << name << " {\n";
    indent();
}

void GeomSave::end_node(ostream& out)
{
    unindent();
    indent(out);
    out << "}\n";
}

void GeomSave::translate(ostream& out, const Point& p)
{
    start_node(out, "Translation");
    indent(out);
    out << "translation " << -p.x() << " " << -p.y() << " " << -p.z() << "\n";
    end_node(out);
}

void GeomSave::rotateup(ostream& out, const Vector& up, const Vector& /*new_up*/)
{
    Vector axis(Cross(Vector(0,1,0), up.normal()));
    if(axis.length2() > 1.e-6){
	double l=axis.normalize();
	double angle=Asin(l);
	start_node(out, "Rotation");
	indent(out);
	out << "rotation " << axis.x() << " " << axis.y() << " " << axis.z() << " " << -angle << "\n";
	end_node(out);
    } else if(Dot(up, Vector(0,1,0)) > 0){
	double angle=Pi;
	start_node(out, "Rotation");
	indent(out);
	out << "rotation 1 0 0 " << angle << "\n";
	end_node(out);
    }
}

void GeomSave::orient(ostream& out, const Point& center, const Vector& up,
		      const Vector& /*new_up&*/)
{
    start_node(out, "Transform");
    indent(out);
    out << "translation " << center.x() << " " << center.y() << " " << center.z() << "\n";
    indent(out);
    Vector axis(Cross(Vector(0,1,0), up.normal()));
    if(axis.length2() > 1.e-6){
	double l=axis.normalize();
	double angle=Asin(l);
	indent(out);
	out << "rotation " << axis.x() << " " << axis.y() << " " << axis.z() << " " << -angle << "\n";
    } else if(Dot(up, Vector(0,1,0)) > 0){
	double angle=Pi;
	indent(out);
	out << "rotation 1 0 0 " << angle << "\n";
    }
    end_node(out);
}

void GeomSave::rib_orient(ostream& out, const Point& center, const Vector& up,
		      const Vector& /*new_up*/)
{
    indent(out);

    Vector upn(up.normal());

    Vector axis(Cross(upn, Vector(0,0,1)));
    if(axis.length2() > 1.e-6) {
      Vector NewX = axis.normal();

      Vector NewY1(Cross(NewX, upn));
      Vector NewY = NewY1.normal();

      out << "ConcatTransform [ "
	  << -NewX.x() << " " << -NewX.y() << " " << -NewX.z() << " 0  "
	  << -NewY.x() << " " << -NewY.y() << " " << -NewY.z() << " 0  "
	  << upn.x() << " " << upn.y() << " " << upn.z() << " 0  "
	  << center.x() << " " << center.y() << " " << center.z() << " 1 ]\n";
    } else if(Dot(up, Vector(0,0, 1)) > 0){
      out << "ConcatTransform [ "
	  << "1 0 0 0\n"
	  << "0 1 0 0\n"
	  << "0 0 1 0\n"
	  << center.x() << " " << center.y() << " " << center.z() << " 1 ]\n";
    }      
}

} // End namespace SCIRun

