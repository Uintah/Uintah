
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
: id(make_id(widget_name)), freqs(640), freqlist(640),
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
   if (args.count() > 1) {
      args.error("Histogram needs no minor command");
      return;
   }
}


void
Histogram::SetData( const Array1<double> values )
{
   ASSERT(values.size() > 1);
   
   // Find minval/maxval.
   minval = maxval = values[0];
   for (int i=1; i<values.size(); i++) {
      if (values[i] < minval) {
	 minval = values[i];
      } else if (values[i] > maxval) {
	 maxval = values[i];
      }
   }

   double range(639.0/(maxval-minval));
   
   initfreqs();
   for (i=0; i<values.size(); i++) {
      freqs[int((values[i]-minval)*range)]++;
   }

   minfreq = maxfreq = freqs[0];
   // C++ can calcminmax faster than tcl...
   for (i=1; i<freqs.size(); i++) {
      if (freqs[i] < minfreq) {
	 minfreq = freqs[i];
      } else if (freqs[i] > maxfreq) {
	 maxfreq = freqs[i];
      }
   }

   for (i=0; i<freqs.size(); i++) {
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
Histogram::SetRange( const double left, const double right )
{
   l.set(left);
   r.set(right);
}


