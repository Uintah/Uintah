/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



/*
 *  GraphPart.h
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_GraphPart_h
#define SCI_GraphPart_h 

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Core/Util/Signals.h>
#include <Core/Parts/Part.h>
#include <Core/Parts/PartInterface.h>
#include <Core/2d/DrawObj.h>

namespace SCIRun {
  
class GraphPart : public Part, public PartInterface  {
private:
  vector< vector<double> > data_;

public:
  GraphPart( PartInterface *parent = 0, const string &name="GraphPart",
	     bool=true);
  virtual ~GraphPart();

  void set_num_lines( int );

#ifdef CHRIS
  void add_values( unsigned , const vector<double> &);

  Signal1< const vector<DrawObj*> & > reset;
  Signal2< unsigned, const vector<double> & > new_values;
#else
  void add_values( const vector<double> &);

  Signal1< int > reset;
  Signal1< const vector<double> & > new_values;
#endif
};

} // namespace SCIRun

#endif // SCI_GraphPart_h
