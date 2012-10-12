/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*
 *  SimpleProfiler.h: 
 *
 *  Written by:
 *   McKay Davis
 *   Scientific Computing and Imaging Instute
 *   University of Utah
 *   January, 2006
 *
 */


#include <Core/Util/SimpleProfiler.h>
#include <iostream>
using std::cerr;
using std::endl;

#if defined(__GNUC__) && defined(__linux)
#include <sys/time.h>
#endif


namespace SCIRun {

SimpleProfiler::SimpleProfiler (const string &name, bool enabled) 
  : name_(name),
    disabled_(!enabled),
    num_(0),
    cur_nest_(0)
{
}


void
SimpleProfiler::enter(const string &text) 
{
  if (disabled_) return;
  Nest &nest = nests_[num_++];
  nest.string_ = 0;
  nest.strings_[nest.string_++] = text;
  
  nest.time_ = 0;
  nest.times_[nest.time_++] = get_time();

  nest.nest_ = 0;

  nest.parent_ = cur_nest_;

  if (cur_nest_) {
    cur_nest_->nests_[cur_nest_->nest_++] = &nest;
    cur_nest_->times_[cur_nest_->time_++] = 0;
  } 
  cur_nest_ = &nest;

}


void
SimpleProfiler::leave() 
{
  if (disabled_) return;
  Nest &nest = *cur_nest_;

  nest.strings_[nest.string_++] = "leave " + nest.strings_[0];
  
  nest.times_[nest.time_++] = get_time();

  cur_nest_ = nest.parent_;
}


void
SimpleProfiler::operator()(const string &text)
{
  if (disabled_) return;
  Nest &nest = *cur_nest_;

  nest.strings_[nest.string_++] = text;
  
  nest.times_[nest.time_++] = get_time();
}



void
SimpleProfiler::print() 
{
  if (disabled_) return;
  cerr << "Timer: " << name_ << std::endl;
  print_nest(nests_[0], "  ");
  num_ = 0;
}




void
SimpleProfiler::print_nest(Nest &nest, string indent) 
{

  double total_time = nest.times_[nest.time_-1]-nest.times_[0];
  int s = 0;
  int nests = 0;
  double last_event_time = 0;
  for (int t = 0; t < nest.time_; ++t) 
  {
    if (t == 0) {
      cerr << indent << "-> " << nest.strings_[s++] << " total: " 
           << total_time << std::endl;
      last_event_time = nest.times_[t];
      continue;
    }
    if (nest.times_[t] == 0.0) {
      print_nest(*nest.nests_[nests++], indent+"  ");
      continue;
    }
    else {
      double event_time = nest.times_[t];
      double delta = event_time - last_event_time;
      cerr << indent << " * " << nest.strings_[s++] << ":  " 
           <<  delta
           << "  (" << (delta/total_time)*100.0 << "%)" 
           << std::endl;
      last_event_time = event_time;
      continue;
    }
  }
}

double 
SimpleProfiler::get_time()
{
  if (!disabled_) {
#if defined(__GNUC__) && defined(__linux)
    struct timeval tp;
    gettimeofday(&tp,0);
    return (tp.tv_sec + tp.tv_usec/1000000.0)*1000;
#endif
  }
  return 0.0;
}

}
