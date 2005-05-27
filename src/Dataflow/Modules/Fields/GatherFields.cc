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
 *  GatherFields.cc: Take in fields and append them into one field.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   July 2004
 *
 *  Copyright (C) 2004 SCI Group
 */

#include <Core/Datatypes/PointCloudField.h>
#include <Dataflow/Modules/Fields/GatherFields.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/Handle.h>
#include <iostream>

namespace SCIRun {

class GatherFields : public Module {
public:
  GatherFields(GuiContext* ctx);
  virtual ~GatherFields();
  virtual void execute();

private:
  GuiInt gui_force_pointcloud_;

  // Turn on across-execution accumulation of fields.
  GuiInt gui_accumulating_;
  GuiInt gui_clear_;
  GuiInt gui_precision_;

  int force_pointcloud_;
  int precision_;

  FieldHandle fHandle_;
  vector< int > fGeneration_;
  bool error_;
};

DECLARE_MAKER(GatherFields)
GatherFields::GatherFields(GuiContext* ctx)
  : Module("GatherFields", ctx, Filter, "FieldsCreate", "SCIRun"),
    gui_force_pointcloud_(ctx->subVar("force-pointcloud")),
    gui_accumulating_(ctx->subVar("accumulating")),
    gui_clear_(ctx->subVar("clear", false)),
    gui_precision_(ctx->subVar("precision")),
    force_pointcloud_(0),
    precision_(0),
    error_(0)
{
}


GatherFields::~GatherFields()
{
}


void
GatherFields::execute()
{
  bool update = false;
  unsigned int nFields = 0;
  std::vector<FieldHandle> fHandles;

  port_range_type range = get_iports("Field");
  if (range.first == range.second)
    return;

  if (gui_clear_.get())
  {
    gui_clear_.set(0);
    remark("Clearing accumulated fields.");
    fHandle_ = 0;

    // Send something.
    FieldOPort *ofield_port = (FieldOPort *) get_oport("Output Field");
#if 1
    // Sending 0 does not clear caches.
    FieldHandle empty =
      scinew PointCloudField<double>(scinew PointCloudMesh(), 0);
    ofield_port->send(empty);
#else
    ofield_port->send( fHandle_ );
#endif
    return;
  }

  if (gui_accumulating_.get() && fHandle_.get_rep()) // appending fields
  {
    fHandles.push_back(fHandle_);
    nFields++;
  }

  // Gather up all of the field handles.
  if (range.first != range.second)
  {
    port_map_type::iterator pi = range.first;
    
    while (pi != range.second)
    {
      FieldIPort *ifield = (FieldIPort *)get_iport(pi->second);

      // Increment here!  We do this because last one is always
      // empty so we can test for it before issuing empty warning.
      ++pi;

      FieldHandle fHandle;
      if (ifield->get(fHandle) && fHandle.get_rep())
      {

	fHandles.push_back(fHandle);

	if ( nFields == fGeneration_.size() )
        {
	  fGeneration_.push_back( fHandle->generation );
	  update = true;
	}
        else if ( fGeneration_[nFields] != fHandle->generation )
        {
	  fGeneration_[nFields] = fHandle->generation;
	  update = true;
	}

	nFields++;
      }
      else if (pi != range.second)
      {
	// Changed this to a warning because in the case of BioTensor, some
	// of the input connections get disabled based on what the user wants
	// to see
	warning("Input port " + to_string(nFields) + " contained no data.");
      }
    }
  }

  if (nFields == 0)
  {
    remark("No non-empty input fields.");
    return;
  }
  
  gui_precision_.reset();
  if(precision_ != gui_precision_.get())
  {
    precision_ = gui_precision_.get();
    update = true;
  }
  while( fGeneration_.size() > nFields )
  {
    update = true;
    fGeneration_.pop_back();
  }

  if( force_pointcloud_ != gui_force_pointcloud_.get() ||
      update ||
      error_ )
  {
    force_pointcloud_ = gui_force_pointcloud_.get();

    if (nFields == 1 && !force_pointcloud_)
    {
      fHandle_ = fHandles[0];
    }
    else
    {
      const TypeDescription *mtd0 =
	fHandles[0]->mesh()->get_type_description();
      const TypeDescription *ftd0 =
	fHandles[0]->get_type_description();
      const int loc0 = fHandles[0]->basis_order();
      bool same_field_kind = true;
      bool same_mesh_kind = true;
      bool same_data_location = true;
      for ( unsigned int i=1; i<nFields; i++)
      {
	if (fHandles[i]->mesh()->get_type_description()->get_name() !=
            mtd0->get_name())
        {
	  same_mesh_kind = false;
        }

	if (fHandles[i]->get_type_description()->get_name() !=
            ftd0->get_name())
        {
	  same_field_kind = false;
        }

	if (fHandles[i]->basis_order() != loc0)
        {
	  same_data_location = false;
        }
      }
	
      if (fHandles[0]->mesh()->is_editable() &&
	  (same_field_kind || same_mesh_kind) &&
	  !force_pointcloud_)
      {
	bool copy_data = same_data_location;
	
	if (!same_data_location)
        {
	  warning("Cannot copy data from meshes with different data locations.");
	}
        else if (!same_field_kind)
        {
	  warning("Copying data does not work for data of different kinds.");
	  copy_data = false;
	}
        const int new_basis = same_data_location?fHandles[0]->basis_order():1;
	CompileInfoHandle ci = GatherFieldsAlgo::get_compile_info(ftd0);
	Handle<GatherFieldsAlgo> algo;
	if (!module_dynamic_compile(ci, algo)) return;
	fHandle_ = algo->execute(fHandles, new_basis, copy_data, 
				 precision_);
      }
      else
      {
        if (!force_pointcloud_)
        {
          if (same_field_kind || same_mesh_kind)
          {
            warning("Non-editable meshes detected, try Unstructuring first, outputting PointCloudField.");
          }
          else
          {
            warning("Different mesh types detected, outputting PointCloudField.");
          }
        }

	PointCloudMeshHandle pc = scinew PointCloudMesh;
	
	for (unsigned int i=0; i<fHandles.size(); i++)
        {
	  const TypeDescription *mtd =
	    fHandles[i]->mesh()->get_type_description();
	  CompileInfoHandle ci = GatherPointsAlgo::get_compile_info(mtd);
	  Handle<GatherPointsAlgo> algo;
	  if (!module_dynamic_compile(ci, algo)) return;
	    algo->execute(fHandles[i]->mesh(), pc);
	}
	
	fHandle_ = scinew PointCloudField<double>(pc, 0);
      }
    }
  }

  // Send the data downstream
  if( fHandle_.get_rep() )
  {
    FieldOPort *ofield_port = (FieldOPort *) get_oport("Output Field");
    ofield_port->send( fHandle_ );
  }    
}


CompileInfoHandle
GatherPointsAlgo::get_compile_info(const TypeDescription *mesh_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("GatherPointsAlgoT");
  static const string base_class_name("GatherPointsAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       mesh_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       mesh_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  mesh_td->fill_compile_info(rval);
  return rval;
}


CompileInfoHandle
GatherFieldsAlgo::get_compile_info(const TypeDescription *field_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("GatherFieldsAlgoT");
  static const string base_class_name("GatherFieldsAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       field_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       field_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  field_td->fill_compile_info(rval);
  return rval;
}

} // End namespace SCIRun
