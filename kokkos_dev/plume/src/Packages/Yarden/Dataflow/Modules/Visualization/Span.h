/*
 *  Span.h
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Sep. 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCI_project_Span_h
#define SCI_project_Span_h 1

namespace Yarden {

using namesapce SCIRun;

class Span : public Module {
public:
  ScalarFieldIPort* infield;
  
  SpanForestOPort *ospan;
  
  ScalarFieldHandle field;
  SpanForest *forest;
  int skiped[60];
  int field_generation;
  int nt;
  GuiInt tcl_split, tcl_maxsplit, tcl_z_max, tcl_z_from, tcl_z_size;
  int z_from, z_size;
    
  int forest_id;
  int forest_generation;
  
public:
  Span(const clString& id);
  virtual ~Span();
  virtual void execute();
  void execute_ext();
  void tcl_command(TCLArgs &, void *);
  void create_span();
  void do_create( int );
  void select_min( SpanPoint *, int );
  void select_max( SpanPoint *, int );
};


} // End namespace Yarden

#endif
