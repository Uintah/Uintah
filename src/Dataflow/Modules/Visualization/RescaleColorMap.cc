//static char *id="@(#) $Id$";

/*
 *  RescaleColorMap.cc:  Generate Color maps
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */



#include <SCICore/Util/NotFinished.h>
#include <PSECore/CommonDatatypes/ColorMapPort.h>
#include <SCICore/CoreDatatypes/ColorMap.h>
#include <PSECore/CommonDatatypes/ScalarFieldPort.h>
#include <SCICore/CoreDatatypes/ScalarField.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

#include <PSECommon/Modules/Visualization/RescaleColorMap.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;


Module* make_RescaleColorMap(const clString& id) {
  return new RescaleColorMap(id);
}

RescaleColorMap::RescaleColorMap(const clString& id)
: Module("RescaleColorMap", id, Filter),
  isFixed("isFixed", id, this),
  min("min", id, this ),
  max("max", id, this)
{
    // Create the output port
    omap=scinew ColorMapOPort(this, "ColorMap", ColorMapIPort::Atomic);
    add_oport(omap);

    // Create the input ports
    imap=scinew ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
    add_iport(imap);
    ScalarFieldIPort* ifield=scinew ScalarFieldIPort(this, "ScalarField",
						     ScalarFieldIPort::Atomic);
    add_iport(ifield);
    fieldports.add(ifield);
}

RescaleColorMap::~RescaleColorMap()
{
}

void
RescaleColorMap::execute()
{
    ColorMapHandle cmap;
    if(!imap->get(cmap)) {
	return;
    }
    if( isFixed.get() ){
      cmap->min = min.get();
      cmap->max = max.get();
      cerr << "Rescale ColorMap " << min.get() << " " << max.get() << endl;
    } else {
      for(int i=0;i<fieldports.size()-1;i++){
        ScalarFieldHandle sfield;
        if(fieldports[i]->get(sfield)){
	  double min;
	  double max;
	  sfield->get_minmax(min, max);
	  //	    cmap.detach();
	  cmap->min=min;
	  cmap->max=max;
	  this->min.set( min );
	  this->max.set( max );
	  
	  cerr << "Rescale ColorMap " << min << " " << max << endl;
	}
      }
    }
    cerr << "Rescale: " << cmap.get_rep() << endl;
    cerr << cmap->colors.size() << " - Size\n";
    cerr << cmap->min << " - " << cmap->max << endl;
    
    omap->send(cmap);
}

void 
RescaleColorMap::connection(ConnectionMode mode, int which_port, int)
{
    if(which_port > 0){
        if(mode==Disconnected){
	    remove_iport(which_port);
	    fieldports.remove(which_port-1);
	} else {
	    ScalarFieldIPort* p=scinew ScalarFieldIPort(this, "Field",
							ScalarFieldIPort::Atomic);
	    fieldports.add(p);
	    add_iport(p);
	}
    }
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.3  1999/08/18 20:20:10  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:53  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:16  mcq
// Initial commit
//
// Revision 1.3  1999/06/02 14:58:34  kuzimmer
// added variables for a RescaleColorMap GUI interface
//
// Revision 1.2  1999/04/27 22:58:01  dav
// updates in Modules for CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:34  dav
// Import sources
//
//
