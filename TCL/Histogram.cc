
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

#include <TCL/Histogram.h>
#include <Multitask/ITC.h>

#include <string.h>
#include <iostream.h>

static clString widget_name("Histogram");

static clString make_id(const clString& name)
{
   static int next_widget_number=0;
   static Mutex idlock;
   idlock.lock();
   clString id ( name+"_"+to_string(next_widget_number++) );
   idlock.unlock();
   return id;
}

Histogram::Histogram()
: id(make_id(widget_name)), freqs(640), min(0), max(0),
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
   }
   
   if (args[1] == "getdata") {
      if (args.count() != 2) {
	 args.error("Histogram doesn't need a minor command");
	 return;
      }

      Array1<clString> freqlist(freqs.size());

      for (int i=0; i<freqs.size(); i++) {
	 freqlist[i] = to_string(freqs[i]);
      }

      args.result(args.make_list(to_string(min), to_string(max),
				 args.make_list(freqlist)));
   }
}


void
Histogram::SetData( const Array1<double> values )
{
   ASSERT(values.size() > 1);
   
   // Find min/max.
   min = max = values[0];
   for (int i=1; i<values.size(); i++) {
      if (values[i] < min) {
	 min = values[i];
      } else if (values[i] > max) {
	 max = values[i];
      }
   }

   double range(max-min);
   
   initfreqs();
   for (i=0; i<values.size(); i++) {
      freqs[int(639*(values[i]-min)/range)]++;
   }
}


void
Histogram::initfreqs()
{
   for (int i=0; i<freqs.size(); i++) {
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
Histogram::GetRange( double& left, double& right )
{
   reset_vars();
   left = l.get();
   right = r.get();
}


void
Histogram::SetRange( const double left, const double right )
{
   l.set(left);
   r.set(right);
}


