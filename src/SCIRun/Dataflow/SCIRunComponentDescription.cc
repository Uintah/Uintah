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
 *  SCIRunComponentDescription.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/Dataflow/SCIRunComponentDescription.h>
#include <SCIRun/Dataflow/SCIRunComponentModel.h>
#include <iostream>
using namespace std;
using namespace SCIRun;

SCIRunComponentDescription::SCIRunComponentDescription(SCIRunComponentModel* model,
						       const DOM_Node& node,
						       const std::string& package)
  : model(model), node(node), is_valid(false)
{
  DOM_Node name = node.getAttributes().getNamedItem("name");
  if(name != 0){
    type = name.getNodeValue().transcode();
    DOM_Node category = node.getAttributes().getNamedItem("category");
    if(name != 0){
      type = string(category.getNodeValue().transcode())+"."+type;
    }
    type = package+"."+type;
    is_valid=true;
  } else {
    cerr << "Warning: could not find name of component\n";
  }
}

SCIRunComponentDescription::~SCIRunComponentDescription()
{
}

string SCIRunComponentDescription::getType() const
{
  return type;
}

const ComponentModel* SCIRunComponentDescription::getModel() const
{
  return model;
}

