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
 *  SetupBEMatrix.cc: TODO: Describe this module.
 *
 *  Written by:
 *   Saeed Babaeizadeh - Northeastern University
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
    FieldHandle field;
    if (fip->get(field))
    {
      if (field.get_rep() == 0)
      {
	warning("Surface port '" + to_string(pi->second) + "' contained no data.");
	++pi;
	continue;
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
      conductivities.push_back(Abs(condVal));
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
      new_fields += Abs( filed_generation_no_new[i] - filed_generation_no_old[i] );
      new_nesting += Abs( nesting[i] - old_nesting[i] );
      new_conductivities += Abs( conductivities[i] - old_conductivities[i] );
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
