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
 *  PartManager.cc
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Oct 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <iostream>
#include <typeinfo>
#include <Core/Parts/PartManager.h>
#include <vector>

namespace SCIRun {
  
using std::vector;
using std::cerr;
using std::endl;

PartManager::PartManager( PartInterface *parent, const string &name, 
		      bool initialize) : 
  Part( parent, name, this ), 
  PartInterface( this, parent, "PartManager", false )
{
    if ( initialize ) PartInterface::init();
}

PartManager::~PartManager()
{
}

int
PartManager::add_part( Part *part )
{
  int pos = parts_.size();
  parts_.push_back( part );
  if ( pos == 0 )
    current_ = part;

  has_part( part->name() );

  if ( part->interface() )
    part->interface()->set_parent( this );

  return pos;
};

void
PartManager::select_part( int i )
{
  if ( i < int(parts_.size()) ) {
    current_ = parts_[i];
    part_selected( i );
  }
}

// void
// PartManager::report_children( SlotBase1<PartInterface*> &slot )
// {
//   cerr << type_ << " report " << parts_.size() << " parts " << endl;

//   for (unsigned i=0; i<parts_.size(); i++)
//     slot.send(parts_[i]);
// }

} // namespace SCIRun


