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
 *  Interface.cc
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */


#include <Core/Persistent/Pstreams.h>
#include <Core/Parts/PartInterface.h>
#include <Core/Parts/GraphPart.h>
#include <iostream>

namespace SCIRun {

using namespace std;
  
PartInterface::PartInterface( Part *part, PartInterface *parent, 
			      const string &type, bool initialize )
  : type_(type), part_(part),  parent_(parent)
{
  if ( initialize ) init();
}
 
PartInterface::~PartInterface()
{
  for (unsigned i=0; i<children_.size(); i++)
    delete children_[i];
}

string
PartInterface::name()
{ 
  return part_->name(); 
}

void
PartInterface::init()
{
  if ( parent_ )
    parent_->add_child(this);
}

void
PartInterface::set_parent( PartInterface *parent )
{
  parent_ = parent;
  parent_->add_child( this );
}


void
PartInterface::add_child( PartInterface *child )
{
  children_.push_back(child);

  // signal you have a new child
  has_child( child  );
}


void
PartInterface::rem_child( PartInterface *child )
{
  vector<PartInterface *>::iterator i;
  for (i=children_.begin(); i!=children_.end(); i++)
    if ( *i == child ) {
      children_.erase( i );
      return;
    }
}

void
PartInterface::report_children( SlotBase1<PartInterface*> &slot )
{
  for (unsigned i=0; i<children_.size(); i++)
    slot.send(children_[i]);
}


} // namespace SCIRun

