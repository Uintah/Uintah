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
#include <Core/Datatypes/ScanlineField.h>
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

FieldHandle 
create_scanline_field(NrrdDataHandle &nrd) 
{
  Nrrd *n = nrd->nrrd;
  for (int a = 0; a < 2; a++) {
    if (!(AIR_EXISTS(n->axis[a].min) && AIR_EXISTS(n->axis[a].max)))
      nrrdAxisMinMaxSet(n, a);
  }
  Point min(n->axis[1].min, .0, .0);
  Point max(n->axis[1].max, .0, .0);
  ScanlineMesh *m = new ScanlineMesh(n->axis[1].size, min, max);
  ScanlineMeshHandle mh(m);
  ScanlineMesh::Node::iterator iter, end;
  mh->begin(iter);
  mh->end(end);
  FieldHandle fh;

  int mn_idx, mx_idx;
  nrd->get_tuple_index_info(0, 0, mn_idx, mx_idx);
  
  switch (mx_idx) {
  case 0:
    switch (n->type) {
    case nrrdTypeChar :  
      fh = new ScanlineField<char>(mh, Field::NODE);
      fill_data((ScanlineField<char>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeUChar : 
      fh = new ScanlineField<unsigned char>(mh, Field::NODE);
      fill_data((ScanlineField<unsigned char>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeShort : 
      fh = new ScanlineField<short>(mh, Field::NODE);
      fill_data((ScanlineField<short>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeUShort :
      fh = new ScanlineField<unsigned short>(mh, Field::NODE);
      fill_data((ScanlineField<unsigned short>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeInt : 
      fh = new ScanlineField<int>(mh, Field::NODE);
      fill_data((ScanlineField<int>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeUInt :  
      fh = new ScanlineField<unsigned int>(mh, Field::NODE);
      fill_data((ScanlineField<unsigned int>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeLLong : 
      //fh = new ScanlineField<long long>(mh, Field::NODE);
      //fill_data((ScanlineField<long long>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeULLong :
      //fh = new ScanlineField<unsigned long long>(mh, Field::NODE);
      //fill_data((ScanlineField<unsigned long long>*)fh.get_rep(), n,iter, end);
      break;
    case nrrdTypeFloat :
      fh = new ScanlineField<float>(mh, Field::NODE);
      fill_data((ScanlineField<float>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeDouble :
      fh = new ScanlineField<double>(mh, Field::NODE);
      fill_data((ScanlineField<double>*)fh.get_rep(), n, iter, end);
      break;
    }
    break;
  case 2: // Vector
    fh = new ScanlineField<Vector>(mh, Field::NODE);
    fill_data((ScanlineField<Vector>*)fh.get_rep(), n, iter, end);
    break;
  case 5: // Tensor
    fh = new ScanlineField<Tensor>(mh, Field::NODE);
    fill_data((ScanlineField<Tensor>*)fh.get_rep(), n, iter, end);
    break;
  default:
    cerr << "unknown index offset: " << mx_idx << endl;
    ASSERTFAIL("Unknown data size");
    break;
  }
  return fh;
}

FieldHandle 
create_image_field(NrrdDataHandle &nrd) 
{
  Nrrd *n = nrd->nrrd;
  for (int a = 0; a < 3; a++) {
    if (!(AIR_EXISTS(n->axis[a].min) && AIR_EXISTS(n->axis[a].max)))
      nrrdAxisMinMaxSet(n, a);
  }
  Point min(n->axis[1].min, n->axis[2].min, .0);
  Point max(n->axis[1].max, n->axis[2].max, .0);
  ImageMesh *m = new ImageMesh(n->axis[1].size, n->axis[2].size,
			       min, max);
  ImageMeshHandle mh(m);
  ImageMesh::Node::iterator iter, end;
  mh->begin(iter);
  mh->end(end);
  FieldHandle fh;
  int mn_idx, mx_idx;
  nrd->get_tuple_index_info(0, 0, mn_idx, mx_idx);
  
  switch (mx_idx) {
  case 0:
    switch (n->type) {
    case nrrdTypeChar :  
      fh = new ImageField<char>(mh, Field::NODE);
      fill_data((ImageField<char>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeUChar : 
      fh = new ImageField<unsigned char>(mh, Field::NODE);
      fill_data((ImageField<unsigned char>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeShort : 
      fh = new ImageField<short>(mh, Field::NODE);
      fill_data((ImageField<short>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeUShort :
      fh = new ImageField<unsigned short>(mh, Field::NODE);
      fill_data((ImageField<unsigned short>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeInt : 
      fh = new ImageField<int>(mh, Field::NODE);
      fill_data((ImageField<int>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeUInt :  
      fh = new ImageField<unsigned int>(mh, Field::NODE);
      fill_data((ImageField<unsigned int>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeLLong : 
      //fh = new ImageField<long long>(mh, Field::NODE);
      //fill_data((ImageField<long long>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeULLong :
      //fh = new ImageField<unsigned long long>(mh, Field::NODE);
      //fill_data((ImageField<unsigned long long>*)fh.get_rep(), n,iter, end);
      break;
    case nrrdTypeFloat :
      fh = new ImageField<float>(mh, Field::NODE);
      fill_data((ImageField<float>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeDouble :
      fh = new ImageField<double>(mh, Field::NODE);
      fill_data((ImageField<double>*)fh.get_rep(), n, iter, end);
      break;
    }
    break;
  case 2: // Vector
    fh = new ImageField<Vector>(mh, Field::NODE);
    fill_data((ImageField<Vector>*)fh.get_rep(), n, iter, end);
    break;
  case 5: // Tensor
    fh = new ImageField<Tensor>(mh, Field::NODE);
    fill_data((ImageField<Tensor>*)fh.get_rep(), n, iter, end);
    break;
  default:
    cerr << "unknown index offset: " << mx_idx << endl;
    ASSERTFAIL("Unknown data size");
    break;
  }
  return fh;
}

FieldHandle 
create_latvol_field(NrrdDataHandle &nrd) 
{
  Nrrd *n = nrd->nrrd;

  for (int a = 0; a < 4; a++) {
    if (!(AIR_EXISTS(n->axis[a].min) && AIR_EXISTS(n->axis[a].max)))
      nrrdAxisMinMaxSet(n, a);
  }
  Point min(n->axis[1].min, n->axis[2].min, n->axis[3].min);
  Point max(n->axis[1].max, n->axis[2].max, n->axis[3].max);
  LatVolMesh *m = new LatVolMesh(n->axis[1].size, n->axis[2].size, 
				 n->axis[3].size, min, max);
  LatVolMeshHandle mh(m);
  LatVolMesh::Node::iterator iter, end;
  mh->begin(iter);
  mh->end(end);
  FieldHandle fh;
  
  int mn_idx, mx_idx;
  nrd->get_tuple_index_info(0, 0, mn_idx, mx_idx);
  
  switch (mx_idx) {
  case 0:
    switch (n->type) {
    case nrrdTypeChar :  
      fh = new LatVolField<char>(mh, Field::NODE);
      fill_data((LatVolField<char>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeUChar : 
      fh = new LatVolField<unsigned char>(mh, Field::NODE);
      fill_data((LatVolField<unsigned char>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeShort : 
      fh = new LatVolField<short>(mh, Field::NODE);
      fill_data((LatVolField<short>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeUShort :
      fh = new LatVolField<unsigned short>(mh, Field::NODE);
      fill_data((LatVolField<unsigned short>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeInt : 
      fh = new LatVolField<int>(mh, Field::NODE);
      fill_data((LatVolField<int>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeUInt :  
      fh = new LatVolField<unsigned int>(mh, Field::NODE);
      fill_data((LatVolField<unsigned int>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeLLong : 
      //fh = new LatVolField<long long>(mh, Field::NODE);
      //fill_data((LatVolField<long long>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeULLong :
      //fh = new LatVolField<unsigned long long>(mh, Field::NODE);
      //fill_data((LatVolField<unsigned long long>*)fh.get_rep(), n,iter, end);
      break;
    case nrrdTypeFloat :
      fh = new LatVolField<float>(mh, Field::NODE);
      fill_data((LatVolField<float>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeDouble :
      fh = new LatVolField<double>(mh, Field::NODE);
      fill_data((LatVolField<double>*)fh.get_rep(), n, iter, end);
      break;
    }
    break;
  case 2: // Vector
    fh = new LatVolField<Vector>(mh, Field::NODE);
    fill_data((LatVolField<Vector>*)fh.get_rep(), n, iter, end);
    break;
  case 5: // Tensor
    fh = new LatVolField<Tensor>(mh, Field::NODE);
    fill_data((LatVolField<Tensor>*)fh.get_rep(), n, iter, end);
    break;
  default:
    cerr << "unknown index offset: " << mx_idx << endl;
    ASSERTFAIL("Unknown data size");
    break;
  }
  return fh;
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
    cout << "nrrd has field handle? " << fh.get_rep() << endl;
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
    // have always dim + 1 axes 
    --dim;

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
