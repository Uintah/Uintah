#include "Color.h"
#include "Array1.cc"
#include "Material.h"
#include "VolumeDpy.h"
#include "HVolumeMaterial.h"
#include <Core/Math/MinMax.h>
#ifdef USE_STF
#include "ScalarTransform1D.h"
#endif

using namespace rtrt;
using namespace SCIRun;

#ifndef USE_STF
Material* HVolumeTransferFunct::index(const float val) {
  int i = (int) ((val - datamin) * scale * (nmatls-1));
  //  cout << ", val = "<<val<<", i = "<<i;
  if (i < 0)
    i = 0;
  else if(i >= nmatls)
    i = nmatls - 1;
  //  cout << ", i = "<<i<<endl;
  return matls[i];
}

void HVolumeTransferFunct::add(HVolumeMaterial *hvcolor) {
  colors.add(hvcolor);
}

// this must be called after all the HVolumeMaterial's have been added
// and before rendering starts.
void HVolumeTransferFunct::compute_min_max() {
  // get the min and max for all the data values
  cout << "HVTF::datamin = "<<datamin<<", datamax = "<<datamax<<endl;
  for(unsigned int i = 0; i < colors.size(); i++) {
    colors[i]->get_min_max(datamin, datamax);
    cout << "after color["<<i<<"], HVTF::datamin = "<<datamin<<", datamax = "<<datamax<<endl;
  }
  // now make sure that we have good values
  if (datamin != datamax)
    scale = 1.0 / (datamax - datamin);
  else
    scale = 0;
  cout << "HVTF::scale = "<< scale << endl;
}
  
HVolumeMaterial::HVolumeMaterial(VolumeDpy *dpy, Array1<float> indata,
				 float datamin, float datamax,
				 HVolumeTransferFunct *trans):
  vdpy(dpy), data(indata), datamin(datamin), datamax(datamax),
  transfer(trans) 
{
  if (datamin != datamax)
    scale = 1.0 / (datamax - datamin);
  else
    scale = 0;
  size_minus_1 = data.size() - 1;
  transfer->add(this);
  cout << "datamin = "<<datamin<<", datamax = "<<datamax<<endl;
  cout << "scale = "<< scale << endl;
  cout << "size_minus_1 = "<<size_minus_1<<endl;
}

void HVolumeMaterial::shade(Color& result, const Ray& ray,
			    const HitInfo& hit, int depth,
			    double atten, const Color& accumcolor,
			    Context* cx) {
  
  // get the current value from vdpy
  float isoval = vdpy->isoval;
  //  cout << "isoval = " << isoval;
  // use this value to get a value out of our data array
  int i = (int) ((isoval - datamin) * scale * size_minus_1);
  //  cout <<", i = "<<i;
  if (i < 0)
    i = 0;
  else if(i > size_minus_1)
    i = size_minus_1;
  //  cout <<", i = "<<i;
  // use the data value to index into the transfer funtion
  // this will return a Material that we can use to shade with
  //  cout <<", data[i] = "<<data[i];
  Material *matl = transfer->index(data[i]);
  matl->shade(result, ray, hit, depth, atten, accumcolor, cx);
}

void HVolumeMaterial::get_min_max(float &in_min, float &in_max) {
  if (data.size() == 0) return;
  float min, max;
  min = max = data[0];
  for(unsigned int i = 0; i < data.size(); i++) {
    min = Min(min, data[i]);
    max = Max(max, data[i]);
  }
  in_min = Min(in_min,min);
  in_max = Max(in_max,max);
}
#endif // ifdef USE_STF

HVolumeMaterial::HVolumeMaterial(VolumeDpy *dpy,
				 ScalarTransform1D<float,float> *f1_to_f2,
				 ScalarTransform1D<float,Material*> *f2_to_material):
  vdpy(dpy), f1_to_f2(f1_to_f2), f2_to_material(f2_to_material) {

}

void HVolumeMaterial::shade(Color& result, const Ray& ray,
			    const HitInfo& hit, int depth,
			    double atten, const Color& accumcolor,
			    Context* cx) {
  // get the current value from vdpy
  float isoval = vdpy->isoval;
  // lookup into f1_to_f2
  float f2 = f1_to_f2->lookup_bound(isoval);
  // use this value to get the material
  Material *matl = f2_to_material->lookup(f2);
  matl->shade(result, ray, hit, depth, atten, accumcolor, cx);
}

#if 0
void HVolumeMaterial::get_min_max(float &in_min, float &in_max) {
  //  cerr << "f1_to_f2->size() = "<<f1_to_f2->size()<<endl;
  if (f1_to_f2->size() == 0) return;
  float min, max;
  min = max = (*f1_to_f2)[0];
  for(unsigned int i = 0; i < f1_to_f2->size(); i++) {
    //    cerr << "f1_to_f2[i="<<i<<"] = "<<(*f1_to_f2)[i]<<endl;
    min = Min(min, (*f1_to_f2)[i]);
    max = Max(max, (*f1_to_f2)[i]);
  }
  in_min = Min(in_min,min);
  in_max = Max(in_max,max);
}
#endif
