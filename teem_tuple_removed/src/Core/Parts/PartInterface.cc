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

