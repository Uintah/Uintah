/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <Core/GuiInterface/Histogram.h>
#include <Core/Thread/Mutex.h>

#include <string.h>


namespace SCIRun {


static string widget_name("Histogram");

static string make_id(const string& name)
{
   static int next_widget_number=0;
   static Mutex idlock("Histogram id name lock");
   idlock.lock();
   string id ( name + "_" + to_string(next_widget_number++) );
   idlock.unlock();
   return id;
}

Histogram::Histogram()
: numbuckets(200), freqs(1000),
  minfreq(0), maxfreq(0), minval(0), maxval(0), id(make_id(widget_name)),
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
   TCL::add_command(id + "-c", this, 0);
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
      if (!string_to_int(args[2], maxbuckets)) {
	 args.error("Histogram can't parse maxbuckets `"+args[2]+"'");
	 return;
      }
      if (!string_to_double(args[3], ratio)) {
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
Histogram::SetData( const vector<double> &values )
{
   ASSERT(values.size() > 1);

   data = values;
   
   // Find minval/maxval.
   minval = maxval = values[0];
   for (unsigned int i=1; i<values.size(); i++) {
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
   unsigned int i;
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

   vector<string> freqlist(numbuckets);

   for (i=0; i<numbuckets; i++) {
      freqlist[i] = to_string(freqs[i]);
   }

   TCL::execute(id + " config -minval " + to_string(minval)
		+ " -maxval " + to_string(maxval)
		+ " -freqs \"" + TCLArgs::make_list(freqlist) + '"'
		+ " -minfreq " + to_string(minfreq)
		+ " -maxfreq " + to_string(maxfreq));
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
Histogram::SetTitle( const string& t ) const
{
   TCL::execute(id+" config -title \""+t+"\"");
}


void
Histogram::SetValueTitle( const string& t ) const
{
   TCL::execute(id+" config -valtitle \""+t+"\"");
}


void
Histogram::SetFrequencyTitle( const string& t ) const
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
   
} // End namespace SCIRun

