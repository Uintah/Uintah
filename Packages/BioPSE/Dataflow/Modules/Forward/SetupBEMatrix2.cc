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
 *  SetupBEMatrix2.cc: TODO: Describe this module.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   May, 2003
 *   
 *   Copyright (C) 2003 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/TriSurfField.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <math.h>

#include <map>
#include <iostream>
#include <string>
#include <fstream>

namespace BioPSE {

using namespace SCIRun;


class SetupBEMatrix2 : public Module
{
private:
  
  bool compute_nesting(vector<int> &nesting, vector<TriSurfMeshHandle> meshes);
  
public:
  
  //! Constructor
  SetupBEMatrix2(GuiContext *context);
  
  //! Destructor  
  virtual ~SetupBEMatrix2();

  virtual void execute();
};


DECLARE_MAKER(SetupBEMatrix2)


SetupBEMatrix2::SetupBEMatrix2(GuiContext *context): 
  Module("SetupBEMatrix2", context, Source, "Forward", "BioPSE")
{
}


SetupBEMatrix2::~SetupBEMatrix2()
{
}


bool
SetupBEMatrix2::compute_nesting(vector<int> &nesting,
				vector<TriSurfMeshHandle> meshes)
{
  nesting.resize(meshes.size());
 
  // Test code only, compute a concentric nesting.
  nesting[0] = meshes.size(); // Put outside at END of list.
  for (int i = 1; i < meshes.size(); i++)
  {
    nesting[i] = i-1;
  }
  return true;
}



void
SetupBEMatrix2::execute()
{
  port_range_type range = get_iports("Surface");
  if (range.first == range.second)
  {
    remark("No surfaces connected.");
    return;
  }

  // Gather up the surfaces from the input ports.
  vector<FieldHandle> fields;
  vector<TriSurfMeshHandle> meshes;
  port_map_type::iterator pi = range.first;
  while (pi != range.second)
  {
    FieldIPort *fip = (FieldIPort *)get_iport(pi->second);
    if (!fip)
    {
      error("Unable to initialize iport '" + to_string(pi->second) + "'.");
      return;
    }
    FieldHandle field;
    if (fip->get(field))
    {
      if (field.get_rep() == 0)
      {
	error("Surface port '" + to_string(pi->second) +
	      "' contained no data.");
	return;
      }
      TriSurfMesh *mesh = 0;
      if (!(mesh = dynamic_cast<TriSurfMesh *>(field->mesh().get_rep())))
      {
	error("Surface port '" + to_string(pi->second) +
	      "' does not contain a TriSurfField");
	return;
      }
      fields.push_back(field);
      meshes.push_back(mesh);
    }
    ++pi;
  }

  // Compute the nesting tree for the input meshes.
  vector<int> nesting;
  if (!compute_nesting(nesting, meshes))
  {
    error("Unable to compute a valid nesting for this set of surfaces.");
  }

  

}

} // end namespace BioPSE
