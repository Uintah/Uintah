/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
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

#ifndef SCIRun_Core_Util_SimpleProfiler_h
#define SCIRun_Core_Util_SimpleProfiler_h 1

#include <string>
using std::string;

#include <Core/Util/share.h>
namespace SCIRun {



const int SIMPLE_PROFILER_MAX_LEVEL = 64;


class SCISHARE SimpleProfiler {

public:
  SimpleProfiler(const string &, bool enabled=1);
  ~SimpleProfiler() {};
  void          enter(const string &);
  void          leave();
  void          operator()(const string &);
  void          print();
  void          disable() { disabled_ = true; }
  void          enable() { disabled_ = false; }


private:
  struct Nest {
    Nest *      parent_;
    string      strings_[SIMPLE_PROFILER_MAX_LEVEL];
    Nest *      nests_[SIMPLE_PROFILER_MAX_LEVEL];
    double      times_[SIMPLE_PROFILER_MAX_LEVEL];
    int         string_;
    int         nest_;
    int         time_;
  };
  string        name_;
  bool          disabled_;
  Nest          nests_[SIMPLE_PROFILER_MAX_LEVEL];
  int           num_;
  Nest *        cur_nest_;
  double        get_time();
  void          print_nest(Nest &nest, string indent);
};


}
#endif
