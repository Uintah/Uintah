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
 *  SetupBEMatrix.cc: TODO: Describe this module.
 *
 *  Written by:
 *   Saeed Babaei Zadeh - Norteastern University
 *   Michael Callahan - Department of Computer Science - University of Utah
 *   May, 2003
 *
 *   Copyright (C) 2003 SCI Group
 */


#include <Packages/BioPSE/Dataflow/Modules/Forward/SetupBEMatrix.h>


namespace BioPSE {

using namespace SCIRun;

unsigned int first_time_run = 1;
vector<int> filed_generation_no_old, old_nesting;
vector<double> old_conductivities;

SetupBEMatrix::SetupBEMatrix(GuiContext *context):
  Module("SetupBEMatrix", context, Source, "Forward", "BioPSE")
{
}


SetupBEMatrix::~SetupBEMatrix()
{
}


void
SetupBEMatrix::execute()
{
    MatrixOPort* oportMatrix_ = (MatrixOPort *)get_oport("BEM Forward Matrix");

    if (!oportMatrix_) {
    error("Unable to initialize oport 'BEM Forward Matrix'.");
    return;
  }

  port_range_type range = get_iports("Surface");
  if (range.first == range.second)
  {
    remark("No surfaces connected.");
    return;
  }

  // Gather up the surfaces from the input ports.
  vector<FieldHandle> fields;
  vector<TriSurfMeshHandle> meshes;
  vector<double> conductivities;
  vector<int> filed_generation_no_new;
  string condStr; double condVal;
  port_map_type::iterator pi = range.first;
  int input=-1, output=-1;
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
	error("Surface port '" + to_string(pi->second) + "' contained no data.");
	return;
      }


      TriSurfMesh *mesh = 0;
      if (!(mesh = dynamic_cast<TriSurfMesh *>(field->mesh().get_rep())))
      {
	error("Surface port '" + to_string(pi->second) +
	      "' does not contain a TriSurfField");
	return;
      }

      if (!field->get_property("Inside Conductivity", condStr))
      {
	error("The 'Inside Conductivity' of the Surface port '" + to_string(pi->second) + "' was not set. It assumes to be zero!");
        condVal = 0;
      }
      else condVal = atof(condStr.c_str());

      if (field->get_property("in/out", condStr))
         if (condStr == "in")
           input = pi->second;
         else if  (condStr == "out")
           output = pi->second;

      fields.push_back(field);
      meshes.push_back(mesh);
      conductivities.push_back(abs(condVal));
      if(first_time_run)
       {
        filed_generation_no_old.push_back(-1);
        old_conductivities.push_back(-1);
        old_nesting.push_back(-1);
       }
      filed_generation_no_new.push_back(field->generation);
    }
    ++pi;
  }

  first_time_run = 0;

   if (input==-1 || output==-1)
   {
     error(" You must define one source as the 'input' and another one as the 'output' ");
     return;
    }
   conductivities[input] = 0;
   // the conductivity inside the innermost surface does not matter in the equations, but for the program to work right, it should be zero.

  // Compute the nesting tree for the input meshes.
  vector<int> nesting;
  if (!compute_nesting(nesting, meshes))
  {
    error("Unable to compute a valid nesting for this set of surfaces.");
  }

   // Check to see if the input fields are new
   int new_fields = 0, new_nesting = 0;
   double new_conductivities = 0;
   int no_of_fields =  nesting.size();
   for (int i=0; i < no_of_fields; i++)
    {
      new_fields += abs( filed_generation_no_new[i] - filed_generation_no_old[i] );
      new_nesting += abs( nesting[i] - old_nesting[i] );
      new_conductivities += abs( conductivities[i] - old_conductivities[i] );
      filed_generation_no_old[i] = filed_generation_no_new[i];
      old_nesting[i] = nesting[i];
      old_conductivities[i] = conductivities[i];
    }

    if(new_fields>(no_of_fields+2) || new_nesting || new_conductivities )  // If the input fields are new
       build_Zoi(meshes, nesting, conductivities, input, output, hZoi_);
    else
       remark("Field inputs are old. Resending stored matrix.");

     // -- sending handles to cloned objects
   oportMatrix_->send(MatrixHandle(hZoi_->clone()));

   return;
}

} // end namespace BioPSE
