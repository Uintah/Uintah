//static char *id="@(#) $Id$";

/*
 *  TriangleReader.cc: Triangle Reader class
 *
 *  Written by:
 *   Kurt Zimmerman
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <SCICore/Util/NotFinished.h>
#include <SCICore/Geom/GeomTriangles.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Containers/String.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Malloc/Allocator.h>
#include "TriangleReader.h"
#include <fstream>
using std::ifstream;
#include <iostream>
using std::cerr;
using std::endl;
using std::ios;
using std::istream;
#include <iomanip>
using std::setw;
#include <sstream>
using std::ostringstream;

#include <ctype.h>
#include <unistd.h>


namespace Uintah {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;
using namespace SCICore::Containers; 
using namespace SCICore::Malloc; 


Module* make_TriangleReader(const clString& id)
{
    return scinew Uintah::Modules::TriangleReader(id);
}


TriangleReader::TriangleReader(const clString& id)
: Module("TriangleReader", id, Source), filename("filename", id, this),
  animate("animate", id, this),
  startFrame("startFrame", id, this), endFrame("endFrame", id, this),
  increment("increment", id, this),
  status("status",id,this)
{
    inport=scinew ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
    add_iport(inport);

    // Create the output data handle and port
    outport=scinew GeometryOPort(this, "Output Data", GeometryIPort::Atomic);
    add_oport(outport);
}


TriangleReader::~TriangleReader()
{
}

bool
TriangleReader::Read(istream& is, ColorMapHandle cmh, GeomGroup *tris)
{  // grap the color map from the input port
  double max = -1e30;
  double min = 1e30;
  int nPoints;
  Array1<Point> points;
  Array1<double> scalars;
  int nTriangles;
  MaterialHandle m1,m2,m3;
  
  clString in;
  
  is >> in;
  if( in != "HONGLAI_TRIANGLES"){
    cerr<<"Error: Unkown triangle file.\n";
    return false;
  } else {
    is >> nPoints;
    Point *p;
    double s;
    int i;
    for(i = 0; i < nPoints; i++){
      p = new Point();
      is>>s; p->x(s);
      is>>s; p->y(s);
      is>>s; p->z(s);
      is>>s;
      points.add( *p );
      scalars.add( s);
    }
    cerr<<endl;

  // default colormap--nobody has scaled it.
    if( !cmh->IsScaled()) {
      int i;
      for( i = 0; i < scalars.size(); i++ ) {
	max = ( scalars[i] > max ) ? scalars[i] : max;
	min = ( scalars[i] < min ) ? scalars[i] : min;
      }
      if (min == max) {
	min -= 0.001;
	max += 0.001;
      }
      cmh->Scale(min,max);
    }   

    cmh->Scale(0.0, 1.0);

    is >> nTriangles;
    int p0,p1,p2;
    GeomTriangles *tri = new GeomTriangles();
    for( i = 0; i < nTriangles; i++){
      is >> p0 >> p1 >> p2;
      
/*       GeomTri *tri = new GeomTri(points[p0], points[p1],points[p2], */
/* 				 (cmh->lookup(scalars[p0])), */
/* 				 (cmh->lookup(scalars[p1])), */
/* 				 (cmh->lookup(scalars[p2]))); */
      //      AuditAllocator(DefaultAllocator());
      tri->add(points[p0],cmh->lookup(scalars[p0]),
	       points[p1],cmh->lookup(scalars[p1]),
	       points[p2],cmh->lookup(scalars[p2]));
      
    }
    tris->add(tri);
    return true;
  }
}

bool TriangleReader::checkFile( clString filename)
{
  ifstream  is( filename() );
  clString in;
  
  is >> in;
  if( in != "HONGLAI_TRIANGLES"){
    cerr<<"Error: Unkown triangle file.\n";
    return false;
  } else {
    return true;
  }
}
  
  
void TriangleReader::execute()
{
  bool wasRead = 0;
  ColorMapHandle cmh;
  GeomGroup *tris = scinew GeomGroup;
  if( !(inport->get( cmh )) ){
    // create a default colormap
    Array1<Color> rgb;
    Array1<float> rgbT;
    Array1<float> alphas;
    Array1<float> alphaT;
    rgb.add( Color(1,0,0) );
    rgb.add( Color(0,0,1) );
    rgbT.add(0.0);
    rgbT.add(1.0);
    alphas.add(1.0);
    alphas.add(1.0);
    alphaT.add(1.0);
    alphaT.add(1.0);
      
    cmh  = new ColorMap(rgb,rgbT,alphas,alphaT,16);
  }

  if( !animate.get() && checkFile( filename.get() )){
     clString command( id + " activate");
     TCL::execute(command);
    status.set("Reading file");
    ifstream is(filename.get()(), ios::in);
    wasRead = Read( is, cmh, tris );
    if( wasRead ){
      outport->delAll();
      outport->addObj(tris, "Triangles");
    }
  } else if( animate.get() && checkFile( filename.get())){
    status.set("Animating");
    doAnimation( cmh );
  }
  status.set("Done");
}

void TriangleReader::doAnimation( ColorMapHandle cmh )
{
  bool wasRead = 0;
  clString file = basename( filename.get() );
  clString path = pathname( filename.get() );
  const char *p = file();
  char n[5];
  char root[ 80 ];
  int i;
  int j = 0;
  int k = 0;
  for( i= 0; i < file.len(); i++ )
    {
      if(isdigit(*p)) n[j++] = *p;
      else root[k++] = *p;
      p++;
    }
  root[k] = '\0';

  for(i = startFrame.get(); i <= endFrame.get(); i += increment.get() ){
    ostringstream ostr;
    ostr.fill('0');
    ostr << path << "/"<< root<< setw(4)<<i;
    cerr << ostr.str()<< endl;
    ifstream is( ostr.str().c_str(), ios::in);
    GeomGroup *tris = scinew GeomGroup;
    wasRead = Read(is, cmh, tris ); 
    if( wasRead ){
      outport->delAll();
      outport->addObj(tris, "Triangles");
      outport->flushViewsAndWait();
    }
    filename.set( ostr.str().c_str() );
    file = basename( filename.get() );
    reset_vars();
    status.set( file );
  }
  sleep(2);
  TCL::execute( id + " deselect");
}


} // End namespace Modules
} // End namespace Uintah

//
// $Log$
// Revision 1.8  1999/12/09 22:03:31  kuzimmer
// hardcoded colormap scaling for now.
//
// Revision 1.7  1999/10/07 22:42:38  kuzimmer
// fixed error in animation
//
// Revision 1.6  1999/10/07 02:08:31  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/10/05 16:40:36  kuzimmer
// added animation control to triangle file reader
//
// Revision 1.4  1999/09/21 16:12:27  kuzimmer
// changes made to support binary/ASCII file IO
//
// Revision 1.3  1999/08/25 03:49:05  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:40:14  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/08/02 20:00:39  kuzimmer
// checked in Triangle Reader for Honlai's Triangles.
//
// Revision 1.3  1999/07/07 21:10:26  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/04/27 22:57:53  dav
// updates in Modules for Datatypes
//
// Revision 1.1  1999/04/25 02:38:10  dav
// more things that should have been there but were not
//
//
