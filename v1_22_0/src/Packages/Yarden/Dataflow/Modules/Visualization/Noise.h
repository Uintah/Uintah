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

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColorMapPort.h>      
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/ScalarField.h>
#include <Core/Datatypes/ScalarFieldRG.h>
#include <Core/Datatypes/ScalarFieldUG.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Thread.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Geom/Material.h>

#include <Packages/Yarden/Dataflow/Ports/SpanPort.h>
#include <Packages/Yarden/Core/Datatypes/SpanTree.h>

namespace Yarden {

using namespace SCIRun;


class Noise : public Module {
  // IO Ports

  //  static int number;
  SpanForestIPort* inforest;
  ScalarFieldIPort* incolorfield;
  ColorMapIPort* incolormap;
  
  GeometryOPort *ogeom;

  double v; // isovalue
  GuiDouble isoval;
  GuiDouble isoval_min, isoval_max;
  GuiDouble tcl_alpha;
  GuiInt    tcl_bbox, tcl_trans;
  GuiInt    tcl_np;
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

  GuiInt tcl_map_type;
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

} // End namespace Yarden

#endif
