
#include <Geom/Scene.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <iostream.h>
#include <Geom/Lighting.h>
#include <Geom/Geom.h>
#include <Geom/Save.h>
#include <Math/Trig.h>
#include <fstream.h>

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

bool GeomScene::save(const clString& filename, const clString& format)
{
    ofstream out(filename());
    if(!out)
	return false;
    if(format == "vrml"){
	out << "#VRML V1.0 ascii\n";
	out << "Separator {\n";
	GeomSave saveinfo;
	saveinfo.nindent=4;
	// Do lights...
	//lighting->saveobj(out, format, &saveinfo);

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

void GeomSave::rotateup(ostream& out, const Vector& up, const Vector& new_up)
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
		      const Vector& new_up)
{
    start_node(out, "Transform");
    indent(out);
    out << "translation " << -center.x() << " " << -center.y() << " " << -center.z() << "\n";
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
