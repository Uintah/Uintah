
/*
 *  BasicSurfaces.cc: Cylinders and stuff
 *
 *  Written by:
 *   Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifdef _WIN32
#pragma warning(disable:4291) // quiet the visual C++ compiler
#endif

#include <Core/Datatypes/BasicSurfaces.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geom/GeomCylinder.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Pt.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Trig.h>
#include <Core/Math/TrigTable.h>
#include <Core/GuiInterface/TCL.h>
#include <iostream>
using std::cerr;
using std::endl;
#include <stdio.h>

namespace SCIRun {

static Persistent* make_CylinderSurface()
{
  return scinew CylinderSurface(Point(0,0,0), Point(0,0,1),1,10,10,10);
}

PersistentTypeID CylinderSurface::type_id("CylinderSurface", "Surface",
					  make_CylinderSurface);

CylinderSurface::CylinderSurface(const Point& p1, const Point& p2,
				 double radius, int nu, int nv, int ndiscu)
  : Surface(RepOther, 1),
    p1(p1), p2(p2), radius(radius), nu(nu), nv(nv), ndiscu(ndiscu)
{
  axis=p2-p1;
  rad2=radius*radius;
  if (axis.length2() > 1.e-6)
  {
    height=axis.normalize();
  }
  else
  {
    // Degenerate cylinder
    height=0;
    axis=Vector(0,0,1);
  }
  axis.find_orthogonal(u, v);
}


CylinderSurface::~CylinderSurface()
{
}


CylinderSurface::CylinderSurface(const CylinderSurface& copy)
  : Surface(copy), p1(copy.p1), p2(copy.p2), radius(copy.radius),
    nu(copy.nu), nv(copy.nv), ndiscu(copy.ndiscu)
{
}


Surface* CylinderSurface::clone()
{
  return scinew CylinderSurface(*this);
}


bool
CylinderSurface::inside(const Point& p)
{
  const double l=Dot(p-p1, axis);
  if (l<0)
  {
    return false;
  }
  if (l>height)
  {
    return false;
  }
  const Point c(p1+axis*l);
  const double dl2=(p-c).length2();
  if (dl2 > rad2)
  {
    return false;
  }
  return true;
}


#if 0
void
CylinderSurface::add_node(Array1<NodeHandle>& nodes,
			  char* id, const Point& p, double r, double rn,
			  double theta, double h, double hn)
{
  Node* node=new Node(p);
  if (boundary_type == DirichletExpression)
  {
    char str [200];
    /*
      ostrstream s(str, 200);
      s << id << " " << p.x() << " " << p.y() << " " << p.z()
      << " " << r << " " << rn << " " << theta << " " << h
      << " " << hn << '\0';
    */
    sprintf(str,"%s %f %f %f %f %f %f %f %f",id,p.x(),p.y(),p.z(),r,rn,theta,h,hn);
    clString retval;
    int err=TCL::eval(str, retval);
    if (err)
    {
      cerr << "Error evaluating boundary value" << endl;
      boundary_type = BdryNone;
      return;
    }
    double value;
    if (!retval.get_double(value))
    {
      cerr << "Bad result from boundary value" << endl;
      boundary_type = BdryNone;
      return;
    }
    node->bc=scinew DirichletBC(this, value);
  }
  nodes.add(node);
}
#endif



#define CYLINDERSURFACE_VERSION 1

void
CylinderSurface::io(Piostream& stream)
{
  stream.begin_class("CylinderSurface", CYLINDERSURFACE_VERSION);
  Surface::io(stream);
  Pio(stream, p1);
  Pio(stream, p2);
  Pio(stream, radius);
  Pio(stream, nu);
  Pio(stream, nv);
  Pio(stream, ndiscu);
  stream.end_class();
}


void
CylinderSurface::construct_grid(int, int, int, const Point&, double)
{
  NOT_FINISHED("CylinderSurface::construct_grid");
}

void
CylinderSurface::construct_grid()
{
  NOT_FINISHED("CylinderSurface::construct_grid");
}

GeomObj *
CylinderSurface::get_geom(const ColorMapHandle& /*cmap*/)
{
  if (boundary_type == BdryNone)
    return scinew GeomCappedCylinder(p1, p2, radius);

  NOT_FINISHED("CylinderSurface::get_geom");
  return NULL;
}



static Persistent *
make_SphereSurface()
{
  return scinew SphereSurface(Point(0,0,0),1,Vector(0,0,1),10,10);
}


PersistentTypeID SphereSurface::type_id("SphereSurface", "Surface",
					make_SphereSurface);


SphereSurface::SphereSurface(const Point& cen, double radius,
			     const Vector& p,
			     int nu, int nv)
  : Surface(RepOther, 1),
    cen(cen), radius(radius), pole(p), nu(nu), nv(nv)
{
  rad2=radius*radius;
  if (pole.length2() > 1.e-6)
  {
    pole.normalize();
  }
  else
  {
    // Degenerate sphere
    pole=Vector(0,0,1);
  }
  pole.find_orthogonal(u, v);
}


SphereSurface::~SphereSurface()
{
}


SphereSurface::SphereSurface(const SphereSurface& copy)
  : Surface(copy), cen(copy.cen), radius(copy.radius),
    pole(copy.pole),
    nu(copy.nu), nv(copy.nv)
{
}


Surface*
SphereSurface::clone()
{
  return scinew SphereSurface(*this);
}

bool
SphereSurface::inside(const Point& p)
{
  double dl2=(p-cen).length2();
  if (dl2 > rad2)
  {
    return false;
  }
  return true;
}


#if 0
void
SphereSurface::add_node(Array1<NodeHandle>& nodes,
			char* id, const Point& p, double r,
			double theta, double phi)
{
  Node* node=new Node(p);
  if (boundary_type == DirichletExpression)
  {
    char str [200];
    /*
      ostrstream s(str, 200);
      s << id << " " << p.x() << " " << p.y() << " " << p.z()
      << " " << r << " " << theta << " " << phi << '\0';
    */
    sprintf(str,"%s %f %f %f %f %f %f",id,p.x(),p.y(),p.z(),r,theta,phi);
    clString retval;
    int err=TCL::eval(str, retval);
    if (err)
    {
      cerr << "Error evaluating boundary value" << endl;
      boundary_type = BdryNone;
      return;
    }
    double value;
    if (!retval.get_double(value))
    {
      cerr << "Bad result from boundary value" << endl;
      boundary_type = BdryNone;
      return;
    }
    node->bc=scinew DirichletBC(this, value);
  }
  nodes.add(node);
}
#endif


#define SPHERESURFACE_VERSION 1

void
SphereSurface::io(Piostream& stream)
{

  /*int version=*/stream.begin_class("SphereSurface", SPHERESURFACE_VERSION);
  Surface::io(stream);
  Pio(stream, cen);
  Pio(stream, radius);
  Pio(stream, nu);
  Pio(stream, nv);
  stream.end_class();
}


void
SphereSurface::construct_grid(int, int, int, const Point&, double)
{
  NOT_FINISHED("SphereSurface::construct_grid");
}


void
SphereSurface::construct_grid()
{
  NOT_FINISHED("SphereSurface::construct_grid");
}


GeomObj *
SphereSurface::get_geom(const ColorMapHandle& /*cmap */)
{
  if (boundary_type == BdryNone)
    return scinew GeomSphere(cen, radius);

  NOT_FINISHED("SphereSurface::get_geom");
  return NULL;
}



static
Persistent* make_PointSurface()
{
  return scinew PointSurface(Point(0,0,0));
}


PersistentTypeID PointSurface::type_id("PointSurface", "Surface",
				       make_PointSurface);


PointSurface::PointSurface(const Point& pos)
  : Surface(RepOther, 0), pos(pos)
{
}


PointSurface::~PointSurface()
{
}


PointSurface::PointSurface(const PointSurface& copy)
  : Surface(copy), pos(copy.pos)
{
}


Surface* 
PointSurface::clone()
{
  return scinew PointSurface(*this);
}


bool
PointSurface::inside(const Point&)
{
  return false;
}


#if 0
void
PointSurface::add_node(Array1<NodeHandle>& nodes,
		       char* id, const Point& p)
{
  Node* node=new Node(p);
  if (boundary_type == DirichletExpression)
  {
    char str [200];
    /*
      ostrstream s(str, 200);
      s << id << " " << p.x() << " " << p.y() << " " << p.z() << '\0';
    */
    sprintf(str,"%s %f %f %f",id,p.x(),p.y(),p.z());
    clString retval;
    int err=TCL::eval(str, retval);
    if (err)
    {
      cerr << "Error evaluating boundary value" << endl;
      return;
    }
    double value;
    if (!retval.get_double(value))
    {
      cerr << "Bad result from boundary value" << endl;
      return;
    }
    node->bc=scinew DirichletBC(this, value);
  }
  nodes.add(node);
}
#endif


#define POINTSURFACE_VERSION 1

void
PointSurface::io(Piostream& stream)
{
  stream.begin_class("PointSurface", POINTSURFACE_VERSION);
  Surface::io(stream);
  Pio(stream, pos);
  stream.end_class();
}


void
PointSurface::construct_grid(int, int, int, const Point &, double)
{
  NOT_FINISHED("PointSurface::construct_grid");
}


void 
PointSurface::construct_grid()
{
  NOT_FINISHED("PointSurface::construct_grid");
}


GeomObj *
PointSurface::get_geom(const ColorMapHandle&  /*cmap */)
{
  NOT_FINISHED("PointSurface::get_geom");
  return NULL;
}


static Persistent* make_PointsSurface()
{
  return scinew PointsSurface();
}


PersistentTypeID PointsSurface::type_id("PointsSurface", "Surface",
					make_PointsSurface);


PointsSurface::PointsSurface(const Array1<Point>& pos, const Array1<double>& val)
  : Surface(PointsSurf, 0), pos(pos), val(val)

{
}


PointsSurface::PointsSurface()
  : Surface(PointsSurf, 0)
{
}


PointsSurface::~PointsSurface()
{
}


PointsSurface::PointsSurface(const PointsSurface& copy)
  : Surface(copy), pos(copy.pos)
{
}


Surface* PointsSurface::clone()
{
  return scinew PointsSurface(*this);
}


bool
PointsSurface::inside(const Point&)
{
  return false;
}


#define POINTSSURFACE_VERSION 1

void
PointsSurface::io(Piostream& stream)
{

  /*int version=*/stream.begin_class("PointsSurface", POINTSSURFACE_VERSION);
  Surface::io(stream);
  Pio(stream, pos);
  Pio(stream, val);		    
  stream.end_class();
}


void
PointsSurface::construct_grid(int, int, int, const Point &, double)
{
  NOT_FINISHED("PointsSurface::construct_grid");
}

void
PointsSurface::construct_grid()
{
  NOT_FINISHED("PointsSurface::construct_grid");
}


GeomObj*
PointsSurface::get_geom(const ColorMapHandle&)
{
  NOT_FINISHED("PointsSurface::get_geom");
  return NULL;
}


} // End namespace SCIRun

