//static char *id="@(#) $Id$";

/*
 *  Histogram.cc: Histogram range widget
 *
 *  Written by:
 *   James T. Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Apr. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifdef _WIN32
#pragma warning(disable:4355) // quit complaining about 'this' in initializers
#endif

#include <SCICore/TclInterface/Histogram.h>
#include <SCICore/Thread/Mutex.h>

#include <string.h>

using SCICore::Thread::Mutex;

namespace SCICore {
namespace TclInterface {

using SCICore::Containers::to_string;

static clString widget_name("Histogram");

static clString make_id(const clString& name)
{
   static int next_widget_number=0;
   static Mutex idlock("Histogram id name lock");
   idlock.lock();
   clString id ( name+"_"+to_string(next_widget_number++) );
   idlock.unlock();
   return id;
}

Histogram::Histogram()
: id(make_id(widget_name)), numbuckets(200), freqs(1000),
  minfreq(0), maxfreq(0), minval(0), maxval(0),
  l("rangeleft", id, this), r("rangeright", id, this)
{
   init_tcl();
}

Histogram::~Histogram()
{
}

void
Histogram::init_tcl()
{
   TCL::add_command(id+"-c", this, 0);
   TCL::execute(widget_name+" "+id);
}

void
Histogram::tcl_command(TCLArgs& args, void*)
{
   if (args.count() < 2) {
      args.error("Histogram needs a minor command");
      return;
   } else if(args[1] == "buckets"){
      if (args.count() != 4) {
	 args.error("Histogram needs maxbuckets ratio");
	 return;
      }
      int maxbuckets;
      double ratio;
      if (!args[2].get_int(maxbuckets)) {
	 args.error("Histogram can't parse maxbuckets `"+args[2]+"'");
	 return;
      }
      if (!args[3].get_double(ratio)) {
	 args.error("Histogram can't parse ratio `"+args[3]+"'");
	 return;
      }
      if ((ratio < 0.0) || (ratio > 1.0)) {
	 args.error("Histogram ratio out of range `"+args[3]+"'");
	 return;
      }
      reset_vars();
      numbuckets = int(maxbuckets*ratio);
      if (numbuckets < 1) {
	 numbuckets = 1;
      }
      SetNumBuckets(numbuckets);
      TCL::execute(id+" ui");
   }
/*
   else if(args[1] == "left"){   
       double val;
       if (args.count() != 3) {
	   args.error("Histogram needs value");
	   return;
       }
       if (!args[2].get_double(val)) {
	   args.error("Histogram can't parse ratio `"+args[2]+"'");
	   return;
       }
// Tell module to execute
   }     
*/
}	


void
Histogram::SetData( const Array1<double> values )
{
   ASSERT(values.size() > 1);

   data = values;
   
   // Find minval/maxval.
   minval = maxval = values[0];
   for (int i=1; i<values.size(); i++) {
      if (values[i] < minval) {
	 minval = values[i];
      } else if (values[i] > maxval) {
	 maxval = values[i];
      }
   }

   FillBuckets();
}


void
Histogram::FillBuckets()
{
   double range(double(numbuckets-1)/(maxval-minval));
   
   initfreqs();
   int i;
   for (i=0; i<data.size(); i++) {
      freqs[int(0.5+(data[i]-minval)*range)]++;
   }
   
   minfreq = maxfreq = freqs[0];
   // C++ can calcminmax faster than tcl...
   for (i=1; i<numbuckets; i++) {
      if (freqs[i] < minfreq) {
	 minfreq = freqs[i];
      } else if (freqs[i] > maxfreq) {
	 maxfreq = freqs[i];
      }
   }

   Array1<clString> freqlist(numbuckets);

   for (i=0; i<numbuckets; i++) {
      freqlist[i] = to_string(freqs[i]);
   }

   TCL::execute(id+" config -minval "+to_string(minval)
		+" -maxval "+to_string(maxval)
		+" -freqs \""+TCLArgs::make_list(freqlist)+"\""
		+" -minfreq "+to_string(minfreq)
		+" -maxfreq "+to_string(maxfreq));
}


void
Histogram::initfreqs()
{
   for (int i=0; i<numbuckets; i++) {
      freqs[i] = 0;
   }
}


void
Histogram::ui() const
{
   TCL::execute(id+" ui");
}


void
Histogram::update() const
{
   ui();
}


void
Histogram::SetTitle( const clString& t ) const
{
   TCL::execute(id+" config -title \""+t+"\"");
}


void
Histogram::SetValueTitle( const clString& t ) const
{
   TCL::execute(id+" config -valtitle \""+t+"\"");
}


void
Histogram::SetFrequencyTitle( const clString& t ) const
{
   TCL::execute(id+" config -freqtitle \""+t+"\"");
}


void
Histogram::ShowGrid() const
{
   TCL::execute(id+" config -grid y");
}


void
Histogram::HideGrid() const
{
   TCL::execute(id+" config -grid n");
}


void
Histogram::ShowRange() const
{
   TCL::execute(id+" config -range y");
}


void
Histogram::HideRange() const
{
   TCL::execute(id+" config -range n");
}


void
Histogram::GetRange( double& left, double& right )
{
   reset_vars();
   left = l.get();
   right = r.get();
}

void
Histogram::GetMaxMin( double& left, double& right )
{
    left = minval;
    right = maxval;
}

void
Histogram::SetRange( const double left, const double right )
{
   l.set(left);
   r.set(right);
}


int
Histogram::GetNumBuckets()
{
   return numbuckets;
}


void
Histogram::SetNumBuckets( const int nb )
{
   freqs.resize(numbuckets=nb);
   FillBuckets();
}
   
} // End namespace TclInterface
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/08/28 17:54:52  sparker
// Integrated new Thread library
//
// Revision 1.2  1999/08/17 06:39:43  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:14  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:25  dav
// Import sources
//
//
