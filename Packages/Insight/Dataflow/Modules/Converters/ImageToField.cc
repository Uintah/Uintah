/*
 *  ImageToField.cc:
 *
 *  Written by:
 *   darbyb
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/ImageField.h>

#include <Core/Datatypes/ImageMesh.h>

#include <Packages/Insight/share/share.h>

namespace Insight {

using namespace SCIRun;

class InsightSHARE ImageToField : public Module {  
public:
  ITKDatatypeIPort* inrrd;
  FieldOPort* ofield;

public:
  ImageToField(GuiContext*);

  virtual ~ImageToField();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  FieldHandle create_image_field(ITKDatatypeHandle &im);
  FieldHandle create_latvol_field(ITKDatatypeHandle &im);
  
  template<class Fld, class Iter>
  void fill_data(Fld* fld, itk::Image<unsigned char, 2>* inrrd, Iter &iter, Iter &end);

  template<class Val>
  void get_val_and_inc_imgptr(Val &val, void* &ptr, unsigned);
};


DECLARE_MAKER(ImageToField)
ImageToField::ImageToField(GuiContext* ctx)
  : Module("ImageToField", ctx, Source, "Converters", "Insight")
{
}

ImageToField::~ImageToField(){
}

void ImageToField::execute(){
  ITKDatatypeHandle ninH;
  inrrd = (ITKDatatypeIPort *)get_iport("InputImage");
  ofield = (FieldOPort *)get_oport("OutputImage");

  if (!inrrd) {
    error("Unable to initialize iport 'InputImage'.");
    return;
  }
  if (!ofield) {
    error("Unable to initialize oport 'OutputImage'.");
    return;
  }

  if(!inrrd->get(ninH))
    return;

  itk::Image<unsigned char, 2> *n = dynamic_cast< itk::Image<unsigned char, 2> * >( ninH.get_rep()->data_.GetPointer() );
  bool dim_based_convert = true;
  FieldHandle ofield_handle;
  
  // do a standard dimension based convert
  if(dim_based_convert) {
    int dim = n->GetImageDimension();
    
    switch(dim) {
      
    case 2:
      ofield_handle = create_image_field(ninH);
      break;

    case 3:
      ofield_handle = create_latvol_field(ninH);
      break;
    default:
      error("Cannot convert data that is not 2D or 3D to a SCIRun Field.");
      return;
    }

  }
  ofield->send(ofield_handle);
}

void ImageToField::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

template<class Val>
void ImageToField::get_val_and_inc_imgptr(Val &v, void* &ptr, unsigned) {
  Val *&p = (Val*&)ptr;
  v = *p;
  ++p;
}


template <class Fld, class Iter>
void ImageToField::fill_data(Fld* fld, itk::Image<unsigned char, 2>* inrrd, Iter &iter, Iter &end) {
  typedef typename Fld::value_type val_t;
  void* p = inrrd; // need pointer to image data

  //for(int col=0; col<

  //  while(iter != end) {
  //val_t tmp;
  //get_val_and_inc_imgptr(tmp, p, 1); // don't understand 1 param
  //fld->set_value(tmp, *iter);
  //++iter;
  //}
}

FieldHandle ImageToField::create_image_field(ITKDatatypeHandle &nrd) {
  itk::Image<unsigned char, 2> *n = dynamic_cast< itk::Image<unsigned char, 2> * >( nrd.get_rep()->data_.GetPointer() );

  double spc[2];
  double data_center = n->GetOrigin()[0];
  unsigned int size_x = 100;
  unsigned int size_y = 100; 

  Point min(0., 0., 0.);
  Point max(size_x, size_y, 0.);

  ImageMesh* m = new ImageMesh(size_x, size_y, min, max);
  ImageMeshHandle mh(m);

  FieldHandle fh;
  int mn_idx, mx_idx;
  
  // assume data type is unsigned char
  fh = new ImageField<unsigned char>(mh, Field::NODE); 
  ImageMesh::Node::iterator iter, end;
  mh->begin(iter);
  mh->end(end);

  // fill data
  itk::Image<unsigned char, 2>::IndexType pixelIndex;
  typedef ImageField<unsigned char>::value_type val_t;
  val_t tmp;
  ImageField<unsigned char>* fld = (ImageField<unsigned char>*)fh.get_rep();

  
  for(int row=0; row < size_y; row++) {
    for(int col=0; col < size_x; col++) {
      if(iter == end) {
	return fh;
      }
      pixelIndex[0] = col;
      pixelIndex[1] = row;

      tmp = n->GetPixel(pixelIndex);
      fld->set_value(tmp, *iter);
      ++iter;
    }
  }

  return fh;
}

FieldHandle ImageToField::create_latvol_field(ITKDatatypeHandle &im) {
  FieldHandle blah;
  return blah;
}
} // End namespace Insight


