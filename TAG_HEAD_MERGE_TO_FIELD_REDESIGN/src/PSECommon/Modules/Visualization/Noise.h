/*
 *  Noise.h
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Nov. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_project_Noise_h
#define SCI_project_Noise_h 

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ColorMapPort.h>      
#include <PSECore/Datatypes/GeometryPort.h>
#include <SCICore/Datatypes/Mesh.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Datatypes/ScalarFieldRGdouble.h>
#include <SCICore/Datatypes/ScalarFieldRGfloat.h>
#include <SCICore/Datatypes/ScalarFieldUG.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/SpanPort.h>
#include <PSECore/Datatypes/SpanTree.h>
#include <SCICore/Thread/Parallel.h>
#include <SCICore/Thread/Thread.h>
#include <SCICore/TclInterface/TCLvar.h>

#include <SCICore/Geom/Material.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Geometry;
using namespace SCICore::Math;
using namespace SCICore::Containers;

using namespace SCICore::Thread;


class Noise : public Module {
  // IO Ports

  //  static int number;
  SpanForestIPort* inforest;
  ScalarFieldIPort* incolorfield;
  ColorMapIPort* incolormap;
  
  GeometryOPort *ogeom;

  double v; // isovalue
  TCLdouble isoval;
  TCLdouble isoval_min, isoval_max;
  TCLdouble tcl_alpha;
  TCLint    tcl_bbox, tcl_trans;
  TCLint    tcl_np;
  int surface_id;
  MaterialHandle matl;
  //GeomGroup *group;
  GeomTrianglesP **triangles;
  GeomObj *surface;
  
  Value old_min;
  Value old_max;

  int counter;
  int init;
  
  ScalarField* field;
  SpanForestHandle forest;

  int forest_generation;
  int np;
  BBox field_bbox;
  double alpha;
  int trans;
  
  void (* interp)( ScalarField *, GeomTrianglesP *,int, double );
  char pad[128];
  int n_trees;
  char pad1[128];

  Mutex lock;
  int trees;

  TCLint tcl_map_type;
  int map_type;

  clString my_name;
  
public:
  Noise(const clString& id);
  Noise(const Noise&, int deep);
  virtual ~Noise();
  virtual void execute();

  void extract();
  void do_search( int );
  int count( SpanTree &); 
  void search( SpanTree &, GeomTrianglesP *);
  
private:
  void _search_min_max( SpanPoint [], int, GeomTrianglesP * );
  void _search_max_min( SpanPoint [], int, GeomTrianglesP * );
  void _search_min( SpanPoint [], int, GeomTrianglesP * );
  void _search_max( SpanPoint [], int, GeomTrianglesP * );
  void _collect( SpanPoint [], int, GeomTrianglesP * );

  void _count_min_max( SpanPoint [], int, int & );
  void _count_max_min( SpanPoint [], int, int & );
  void _count_min( SpanPoint [], int, int & );
  void _count_max( SpanPoint [], int, int & );
  void _count_collect( int, int & );
};

}
}

#endif
