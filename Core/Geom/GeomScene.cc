/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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

bool GeomScene::save(const string& filename, const string& format)
{
    ofstream out(filename.c_str());
    if(!out)
	return false;
    if(format == "vrml" || format == "iv"){
	if(format=="iv")
	    out << "#Inventor V2.1 ascii\n";
	else
	    out << "#VRML V1.0 ascii\n";
	out << "Separator {\n";
	GeomSave saveinfo;
	saveinfo.nindent=4;
	// Do lights...
	//lighting->saveobj(out, format, &saveinfo);
	saveinfo.start_node(out, "ShapeHints");
	saveinfo.indent(out);
	out << "vertexOrdering COUNTERCLOCKWISE\n";
	saveinfo.end_node(out);

	// Do view...
	saveinfo.start_tsep(out);
	saveinfo.orient(out, view.eyep(), view.eyep()-view.lookat(), Vector(0,0,1));
	saveinfo.indent(out);
	out << "PerspectiveCamera {\n";
	saveinfo.indent();
	saveinfo.indent(out);
	Point eyep(view.eyep());
	//out << "position " << eyep.x() << " " << eyep.y() << " " << eyep.z() << "\n";
	saveinfo.indent(out);
	Vector up(view.up());
	up.normalize();
	Vector axis(Cross(Vector(0,0,1), up));
	double l=axis.normalize();
	double angle=Asin(l);
	out << "orientation " << axis.x() << " " << axis.y() << " " << axis.z() << " " << angle << "\n";
	saveinfo.indent(out);
	Point lookat(view.lookat());
	double dist=(lookat-eyep).length();
	out << "focalDistance " << dist << "\n";
	saveinfo.indent(out);
	out << "heightAngle " << DtoR(view.fov()) << "\n";
	saveinfo.unindent();
	saveinfo.indent(out);
	out << "}\n";
	saveinfo.end_tsep(out);
	// Do objects...
	bool status=top->saveobj(out, format, &saveinfo);
	if(!status)
	    return false;
	saveinfo.unindent();
	out << "}\n";
    } else if(format == "rib"){
	//////////////////////////////
	// RIB Output
	out << "##RenderMan RIB-Structure 1.0\n";
	out << "version 3.03\n\n";
	GeomSave saveinfo;
	saveinfo.nindent=0;

	// Do rendering info.
	out << "PixelVariance 0.01\n";
	out << "Format 600 500 1.0\n";
	out << "Display \"" << filename << ".tif\" \"file\" \"rgb\"\n\n";

	// Do lighting.

	out << "LightSource \"distantlight\" 1  \"intensity\" 0.75  \"from\"  [ 1 -3 -5 ]  \"to\"  [ 0 0 0 ]  \"lightcolor\"  [ 1 1 1 ]\n";
	out << "LightSource \"spotlight\" 1  \"intensity\" 0.5  \"from\"  [ -2 0 5 ] \"to\"  [ .22 0 0 ]  \"lightcolor\"  [ 0.5 1 1 ] \"coneangle\" 0.8\n";
	out << "LightSource \"ambientlight\" 2  \"intensity\" 0.5\n";

	// Do view.
	Point eyep(view.eyep());
	out << "\nProjection \"perspective\" \"fov\" " << view.fov() << "\n";

	/* Z vector */
	Vector z = view.lookat() - view.eyep();

	z.normalize();
	
	/* Y vector */
	Vector y = -view.up();
	
	/* X vector = Z cross Y */
	Vector x = Cross(z,y);

	x.normalize();
	
	/* Recompute Y = X cross Z */
	y = Cross(x, z);

	// Not technically necessary.
	y.normalize();

	out << "ConcatTransform [ "
#if 1
	    << -x.x() << " " << -y.x() << " " << z.x() << " 0  "
	    << -x.y() << " " << -y.y() << " " << z.y() << " 0  "
	    << -x.z() << " " << -y.z() << " " << z.z() << " 0  "
#else
	    << x.x() << " " << x.y() << " " << x.z() << " 0  "
	    << y.x() << " " << y.y() << " " << y.z() << " 0  "
	    << z.x() << " " << z.y() << " " << z.z() << " 0  "
#endif
	    << "0 0 0 1 ]\n";
	out << "Translate " << -eyep.x() << " " << -eyep.y() << " " << -eyep.z() << "\n";

	saveinfo.indent(out);
	out << "\nWorldBegin\n";
	saveinfo.indent();
	{
	    // Do objects...
	    bool status=top->saveobj(out, format, &saveinfo);
	    if(!status)
		return false;
	}
	saveinfo.unindent();
	out << "WorldEnd\n";

    } else {
	cerr << "Format not supported: " << format << endl;
	return false;
    }
    return true;
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

