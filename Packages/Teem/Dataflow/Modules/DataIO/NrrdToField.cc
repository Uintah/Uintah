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
 *  NrrdToField.cc:  Convert a Nrrd to a Field
 *
 *  Written by:
 *   David Weinstein
 *   School of Computing
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Dataflow/Network/Module.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/MaskedLatVolField.h>
#include <Core/Malloc/Allocator.h>
#include <Teem/Dataflow/Modules/DataIO/ConvertToField.h>
#include <Core/Util/TypeDescription.h>
#include <iostream>

namespace SCITeem {

using namespace SCIRun;

class NrrdToField : public Module {
  NrrdIPort* inrrd;
  FieldOPort* ofield;
public:
  NrrdToField(GuiContext *ctx);
  virtual ~NrrdToField();
  virtual void execute();
};

} // end namespace SCITeem

using namespace SCITeem;
DECLARE_MAKER(NrrdToField)

NrrdToField::NrrdToField(GuiContext *ctx):
  Module("NrrdToField", ctx, Filter, "DataIO", "Teem")
{
}

NrrdToField::~NrrdToField()
{
}

#define COPY_INTO_FIELD_FROM_NRRD(type, dataat) \
    LatVolField<type> *f = \
      scinew LatVolField<type>(lvm, dataat); \
    type *p=(type *)n->data; \
    for (k=0; k<nz; k++) \
      for (j=0; j<ny; j++) \
	for(i=0; i<nx; i++) \
	  f->fdata()(k,j,i) = *p++; \
    fieldH = f


bool convert_to_field(Mesh*, Nrrd*, FieldHandle) {
  return false;
}

FieldHandle create_scanline_field(NrrdDataHandle) {
  return FieldHandle(0);
}

FieldHandle create_image_field(NrrdDataHandle) {
  return FieldHandle(0);
}

FieldHandle create_latvol_field(NrrdDataHandle) {
#if 0

  //FIX_ME get vector tensor from the sink label name
  if (n->dim == 4) {  // vector or tensor data
    if (n->type != nrrdTypeFloat) {
      error("Tensor nrrd's must be floats.");
      return;
    }
    // matrix, x, y, z
    if (n->axis[0].size == 7) {
      if (n->type != nrrdTypeFloat) {
	error("Can only convert float data for tensors.");
	return;
      }
      // the Nrrd assumed samples at nodes and gave min/max accordingly
      // but we want to think of those samples as centers of cells, so
      // we need a different mesh
      Point minP(0,0,0);
      Point maxP((n->axis[1].size+1)*n->axis[1].spacing, 
		 (n->axis[2].size+1)*n->axis[2].spacing,
		 (n->axis[3].size+1)*n->axis[3].spacing);
      int nx = n->axis[1].size;
      int ny = n->axis[2].size;
      int nz = n->axis[3].size;
      LatVolMesh *lvm = scinew LatVolMesh(nx+1, ny+1, nz+1, minP, maxP);
      MaskedLatVolField<Tensor> *f =
	scinew MaskedLatVolField<Tensor>(lvm, Field::CELL);
      float *p=(float *)n->data;
      vector<double> tens(6);
      for (k=0; k<nz; k++)
	for (j=0; j<ny; j++)
	  for (i=0; i<nx; i++) {
	    if (*p++ > .5) f->mask()(k,j,i)=1;
	    else f->mask()(k,j,i)=0;
	    for (int ch=0; ch<6; ch++) tens[ch]=*p++;
	    f->fdata()(k,j,i)=Tensor(tens);
	  }
      fieldH = f;      
      ofield->send(fieldH);
    } else {
      error("4D nrrd must have 7 entries per channel (1st dim).");
    }
    return;
  }

  if (n->dim != 3) {
    msgStream_ << "NrrdToField error: nrrd->dim="<<n->dim<<"\n";
    error("Can only deal with 3-dimensional scalar fields.");
    return;
  }
  int nx = n->axis[0].size;
  int ny = n->axis[1].size;
  int nz = n->axis[2].size;
  
  for (i=0; i<3; i++)
    if (!(AIR_EXISTS(n->axis[i].min) && AIR_EXISTS(n->axis[i].max)))
      nrrdAxisMinMaxSet(n, i);

  Point minP(n->axis[0].min, n->axis[1].min, n->axis[2].min);
  Point maxP(n->axis[0].max, n->axis[1].max, n->axis[2].max);

  LatVolMesh *lvm = scinew LatVolMesh(nx, ny, nz, minP, maxP);
  LatVolMeshHandle lvmH(lvm);
  SCIRun::Field::data_location dataat = Field::NODE;
  if (n->axis[0].center == nrrdCenterCell) dataat = Field::CELL;

  if (n->type == nrrdTypeChar) {
    COPY_INTO_FIELD_FROM_NRRD(char, dataat);
  } else if (n->type == nrrdTypeUChar) {
    COPY_INTO_FIELD_FROM_NRRD(unsigned char, dataat);
  } else if (n->type == nrrdTypeShort) {
    COPY_INTO_FIELD_FROM_NRRD(short, dataat);
  } else if (n->type == nrrdTypeUShort) {
    COPY_INTO_FIELD_FROM_NRRD(unsigned short, dataat);
  } else if (n->type == nrrdTypeInt) {
    COPY_INTO_FIELD_FROM_NRRD(int, dataat);
  } else if (n->type == nrrdTypeUInt) {
    COPY_INTO_FIELD_FROM_NRRD(unsigned int, dataat);
  } else if (n->type == nrrdTypeFloat) {
    COPY_INTO_FIELD_FROM_NRRD(float, dataat);
  } else if (n->type == nrrdTypeDouble) {
    COPY_INTO_FIELD_FROM_NRRD(double, dataat);
  } else {
    error("Unrecognized nrrd type.");
    return;
  }

#endif
  return FieldHandle(0);
}

const TypeDescription *
get_new_td(int t) {

  switch (t) {
  case nrrdTypeChar :  
    return get_type_description((char*)0);
    break;
  case nrrdTypeUChar : 
    return get_type_description((unsigned char*)0);
    break;
  case nrrdTypeShort : 
    return get_type_description((short*)0);
    break;
  case nrrdTypeUShort :
    return get_type_description((unsigned short*)0);
    break;
  case nrrdTypeInt :   
    return get_type_description((int*)0);
    break;
  case nrrdTypeUInt :  
    return get_type_description((unsigned int*)0);
    break;
  case nrrdTypeLLong : 
    //return get_type_description((long long*)0);
    break;
  case nrrdTypeULLong :
    //return get_type_description((unsigned long long*)0);
    break;
  case nrrdTypeFloat : 
    return get_type_description((float*)0);
    break;
  case nrrdTypeDouble :
    return get_type_description((double*)0);
    break;
  }
  return 0;
}

void NrrdToField::execute()
{
  NrrdDataHandle ninH;
  inrrd = (NrrdIPort *)get_iport("Nrrd");
  ofield = (FieldOPort *)get_oport("Field");

  if (!inrrd) {
    error("Unable to initialize iport 'Nrrd'.");
    return;
  }
  if (!ofield) {
    error("Unable to initialize oport 'Field'.");
    return;
  }

  if(!inrrd->get(ninH))
    return;

  Nrrd *n = ninH->nrrd;
  bool dim_based_convert = true;
  FieldHandle ofield_handle;

  if (ninH->is_sci_nrrd()) {
    // the NrrdData has a stored MeshHandle which from the originating field.
    FieldHandle fh = ninH->get_orig_field();
    const TypeDescription *td = fh->get_type_description();
    // manipilate the type to match the nrrd.
    const TypeDescription *sub = get_new_td(n->type);

    TypeDescription::td_vec *v = td->get_sub_type();
    v->clear();
    v->push_back(sub);

    CompileInfoHandle ci = ConvertToFieldBase::get_compile_info(td);
    Handle<ConvertToFieldBase> algo;
    if ((module_dynamic_compile(ci, algo)) && 
	(algo->convert_to_field(fh, ninH->nrrd, ofield_handle))) 
    {
      remark("Creating a Field from original mesh in input nrrd");
      dim_based_convert = false;
    }
    // if compilation fails or the algo cant match the data to the mesh,
    // do a standard dimemsion based convert.
  }

  if (dim_based_convert) {

    int dim = n->dim;
    if (ninH->is_sci_nrrd()) {
      // have always dim + 1 axes 
      --dim;
    }
    switch (dim) {

    case 1:
      {
	//get data from x axis and stuff into a Scanline
	remark("Creating a ScanlineField from input nrrd");
	ofield_handle = create_scanline_field(ninH);
      }
      break;
    case 2:
      {
	//get data from x,y axes and stuff into an Image
	remark("Creating a ImageField from input nrrd");
	ofield_handle = create_image_field(ninH);
      }
      break;
    case 3:
      {
	//get data from x,y,z axes and stuff into a LatVol
	remark("Creating a LatVolField from input nrrd");
	ofield_handle = create_latvol_field(ninH);
      }
      break;
    default:
      error("Cannot convert > 3 dimesional data to a SCIRun Field.");
      return;
    }
  }
 
  ofield->send(ofield_handle);
}
