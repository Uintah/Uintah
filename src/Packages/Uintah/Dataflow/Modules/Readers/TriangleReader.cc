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

#include <Util/NotFinished.h>
#include <Dataflow/Module.h>
#include <CommonDatatypes/GeometryPort.h>
#include <Geom/GeomTriangles.h>
#include <Malloc/Allocator.h>
#include <TclInterface/TCLTask.h>
#include <TclInterface/TCLvar.h>
#include <Modules/Readers/TriangleReader.h>
#include <Malloc/Allocator.h>



namespace Uintah {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;
using namespace SCICore::Malloc; 


Module* make_TriangleReader(const clString& id)
{
    return scinew Uintah::Modules::TriangleReader(id);
}


TriangleReader::TriangleReader(const clString& id)
: Module("TriangleReader", id, Source), filename("filename", id, this)
{
    inport=scinew ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
    add_iport(inport);

    // Create the output data handle and port
    outport=scinew GeometryOPort(this, "Output Data", GeometryIPort::Atomic);
    add_oport(outport);
}

TriangleReader::TriangleReader(const TriangleReader& copy, int deep)
: Module(copy, deep), filename("filename", id, this)
{
    NOT_FINISHED("TriangleReader::TriangleReader");
}

TriangleReader::~TriangleReader()
{
}

Module* TriangleReader::clone(int deep)
{
    return scinew TriangleReader(*this, deep);
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
    cerr<<"We have "<<nPoints<<" points\n";
    Point *p;
    double s;
    int i;
    for(i = 0; i < nPoints; i++){
      p = new Point();
      is>>s; p->x(s);
      is>>s; p->y(s);
      is>>s; p->z(s);
      is>>s;
      cerr<<p<<" "<<*p<<" "<<s<<", ";
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

    is >> nTriangles;
    cerr<<"We have " << nTriangles << " Triangles.\n";
    int p0,p1,p2;
    GeomTriangles *tri = new GeomTriangles();
    for( i = 0; i < nTriangles; i++){
      is >> p0 >> p1 >> p2;
      
      cerr<<"Here "<< i+1<<" times.\n";
/*       GeomTri *tri = new GeomTri(points[p0], points[p1],points[p2], */
/* 				 (cmh->lookup(scalars[p0])), */
/* 				 (cmh->lookup(scalars[p1])), */
/* 				 (cmh->lookup(scalars[p2]))); */
      cerr<<p0<<" "<<p1<<" "<<p2<<endl;
      cerr<<points[p0]<<" "<<points[p1]<<" "<<points[p2]<<endl;
      cerr<<&(cmh->lookup(scalars[p0]))<<" "<<&(cmh->lookup(scalars[p1]))<<" "<<&(cmh->lookup(scalars[p2]))<<endl;
      cerr<<&(cmh->lookup(scalars[p0])->lock)<<" "<<&(cmh->lookup(scalars[p1])->lock)<<" "<<&(cmh->lookup(scalars[p2])->lock)<<endl;
      cerr<<(cmh->lookup(scalars[p0])->ref_cnt)<<" "<<(cmh->lookup(scalars[p1])->ref_cnt)<<" "<<(cmh->lookup(scalars[p2])->ref_cnt)<<endl;
      //      AuditAllocator(DefaultAllocator());
      cerr<<"\n Before tri-add\n";
      cerr<<*(void **)&(cmh->lookup(scalars[p0])->lock)<<" "<<*(void **)&(cmh->lookup(scalars[p1])->lock)<<" "<<*(void **)&(cmh->lookup(scalars[p2])->lock)<<endl;
      tri->add(points[p0],cmh->lookup(scalars[p0]),
	       points[p1],cmh->lookup(scalars[p1]),
	       points[p2],cmh->lookup(scalars[p2]));
      
      cerr<<"But here only "<<i+1<<" times\n";
      cerr<<"After tri-add\n";
      cerr<<*(void **)&(cmh->lookup(scalars[p0])->lock)<<" "<<*(void **)&(cmh->lookup(scalars[p1])->lock)<<" "<<*(void **)&(cmh->lookup(scalars[p2])->lock)<<endl<<endl;
    }
    tris->add(tri);
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

  /* clString fn(filename.get()); */
  /*   if( fn != old_filename){ */
  /*     old_filename=fn; */
  /*     ifstream is(filename.get()(), ios::in); */
  /*     if(!is){ */
  /*       error(clString("Error reading file: ")+filename.get()); */
  /*       return; // Can't open file... */
  /*     } */
  /*     // Read the file... */
  /*     wasRead = Read( is, cmh, tris ); */
  /*     is.close(); */
  /*   } else { */
  /*     wasRead = true; */
  /*   } */
  ifstream is(filename.get()(), ios::in);
  wasRead = Read( is, cmh, tris );

  if( wasRead )
    cerr<<"So what's going on?\n";
  outport->delAll();
  outport->addObj(tris, "Triangles");

}

} // End namespace Modules
} // End namespace Uintah

//
// $Log$
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
// updates in Modules for CoreDatatypes
//
// Revision 1.1  1999/04/25 02:38:10  dav
// more things that should have been there but were not
//
//
