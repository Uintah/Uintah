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
 *  MRITissueClassifier.h
 *
 *  Written by:
 *   Tolga Tasdizen and McKay Davis
 *   Scientific Computing and Imaging Institute
 *   University of Utah
 *   February 2004
 *
 *  Copyright (C) 2004 SCI Institute
 */


#ifndef _MRITissueClassifier_h_
#define _MRITissueClassifier_h_

#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Dataflow/Ports/NrrdPort.h>


namespace SCITeem {
using namespace SCIRun;


class MRITissueClassifier : public Module
{
public:
  MRITissueClassifier(GuiContext *ctx);
  virtual ~MRITissueClassifier();

  virtual void execute();
  //  virtual void tcl_command(GuiArgs &, void *);

  void EyeDetection ();
  void BackgroundDetection ();
  void FatDetection ()
  {
    if (m_FatSat) FatDetection_FATSAT ();
    else FatDetection_NOFATSAT (4.0);
  }
  void ComputeForegroundCenter ();
  void BrainDetection (float maskr, float maskr2);
  void BrainClassification ();
  void ScalpClassification ();
  void BoneDetection (float maskr);
  
protected:
  void FatDetection_FATSAT ();
  void FatDetection_NOFATSAT (float maskr);
  void BoneDetection_PDthreshold (float *th_low, float *th_high);
  void BoneDetection_TopSection (float maskr, float th_low, float th_high);
  void BoneDetection_SecondSection (float th);
  void BoneDetection_ThirdSection (float th);
  void FindTemporalSlice ();
  void FindTemporalLobes ();
  float BackgroundNoiseLevel ();
  void AirDetection (float maskr);
  
  float Compute2ClassThreshold (float *p, float *v, int n);
  void Clear1DHistogram (int*, int);
  void Accumulate1DHistogram (float *upperlims, int *hist, int n,
                              NrrdDataHandle data,
                              int lx, int ly, int lz, int hx, int hy, int hz);
  void Accumulate1DHistogramWithLabel (float *upperlims, int *hist, int n,
                                       NrrdDataHandle data,
                                       int lx, int ly, int lz, int hx, int hy, int hz, int label);
  void Accumulate1DHistogramWithLabel  (float *upperlims, int *hist, int n, int m,
                                        int lx, int ly, int lz, int hx, int hy, int hz, int label);
  
  float *EM_CSF_GM_WM (ColumnMatrix &CSF_mean, DenseMatrix &CSF_cov, float *CSF_prior,
                       ColumnMatrix &GM_mean, DenseMatrix &GM_cov, float *GM_prior,
                       ColumnMatrix &WM_mean, DenseMatrix &WM_cov, float *WM_prior,
                       int label);
  float *EM_Muscle_Fat (ColumnMatrix &Muscle_mean, DenseMatrix &Muscle_cov, float *Muscle_prior,
                        ColumnMatrix &Fat_mean, DenseMatrix &Fat_cov, float *Fat_prior,
                        int l, int dim);
  float* EM1DWeighted (float *data, float *weight, int n,
                       int c, float *mean, float *prior, float *var);

  float* SmoothProbabilities (float *prob, int c, int l, int n, float sigma);


  //VisPack replacement functions
  void  write_flat_volume(NrrdDataHandle data, string filename);
  void  write_image(NrrdDataHandle data, string filename);
  NrrdDataHandle	dilate(NrrdDataHandle, NrrdDataHandle);
  NrrdDataHandle	erode(NrrdDataHandle, NrrdDataHandle);
  NrrdDataHandle	opening(NrrdDataHandle, NrrdDataHandle);
  NrrdDataHandle	closing(NrrdDataHandle, NrrdDataHandle);
  NrrdDataHandle	gaussian(const NrrdDataHandle &data, double sigma);
  void	floodFill(NrrdDataHandle, 
		  int, int, unsigned int, unsigned int, unsigned int);
  void	floodFill(NrrdDataHandle,
		  int, int, unsigned int, unsigned int);


  // 3d nrrds
  NrrdDataHandle create_nrrd_of_ints(int, int, int);
  NrrdDataHandle create_nrrd_of_floats(int, int, int);  
  bool nrrd_check_bounds(NrrdDataHandle, int, int, int);
  NrrdDataHandle extract_nrrd_slice_int(NrrdDataHandle, int);

  // 2d nrrds
  NrrdDataHandle create_nrrd_of_ints(int, int);
  NrrdDataHandle create_nrrd_of_floats(int, int);
  bool nrrd_check_bounds(NrrdDataHandle, int, int);

  
  inline int get_nrrd_int(const NrrdDataHandle &data, int x, int y, int z) {
    Nrrd &nrrd = *data.get_rep()->nrrd;
    int *idata = (int*)nrrd.data;
    return idata[x + nrrd.axis[0].size*(y + nrrd.axis[1].size*z)];
  }

  inline float get_nrrd_float(const NrrdDataHandle &data, int x, int y, int z) {
    Nrrd &nrrd = *data.get_rep()->nrrd;
    float *idata = (float*)nrrd.data;
    return idata[x + nrrd.axis[0].size*(y + nrrd.axis[1].size*z)];
  }

  inline void set_nrrd_int(const NrrdDataHandle &data, int val, int x, int y, int z) {
    Nrrd &nrrd = *data.get_rep()->nrrd;
    int *idata = (int*)nrrd.data;
    idata[x + nrrd.axis[0].size*(y + nrrd.axis[1].size*z)] = val;
  }

  inline void set_nrrd_float(const NrrdDataHandle &data, float val, int x, int y, int z){
    Nrrd &nrrd = *data.get_rep()->nrrd;
    float *idata = (float*)nrrd.data;
    idata[x + nrrd.axis[0].size*(y + nrrd.axis[1].size*z)] = val;
  }

  inline int get_nrrd_int2(const NrrdDataHandle &data, int x, int y) {
    Nrrd &nrrd = *data.get_rep()->nrrd;
    int *idata = (int*)nrrd.data;
    return idata[x + nrrd.axis[0].size*y];
  }

  inline float get_nrrd_float2(const NrrdDataHandle &data, int x, int y) {
    Nrrd &nrrd = *data.get_rep()->nrrd;
    float *idata = (float*)nrrd.data;
    return idata[x + nrrd.axis[0].size*y];
  }

  inline void set_nrrd_int2(const NrrdDataHandle &data, int val, int x, int y) {
    Nrrd &nrrd = *data.get_rep()->nrrd;
    int *idata = (int*)nrrd.data;
    idata[x + nrrd.axis[0].size*y] = val;
  }

  inline void set_nrrd_float2(const NrrdDataHandle &data, float val, int x, int y){
    Nrrd &nrrd = *data.get_rep()->nrrd;
    float *idata = (float*)nrrd.data;
    idata[x + nrrd.axis[0].size*y] = val;
  }

  
  ColumnMatrix get_m_Data(int, int, int);

  NrrdDataHandle	m_T1_Data;
  NrrdDataHandle	m_T2_Data;
  NrrdDataHandle	m_PD_Data;
  NrrdDataHandle	m_FATSAT_Data;
  
  NrrdDataHandle	m_Label;		// volume of ints
  NrrdDataHandle	m_DistBG;		// volume of floats
  NrrdDataHandle	m_DistBG2D;		// volume of floats
  NrrdDataHandle	m_Energy;		// volume of floats
  NrrdDataHandle	m_CSF_Energy;		// volume of floats
  NrrdDataHandle	m_Fat_Energy;		// volume of floats

  GuiInt		gui_max_iter_;
  GuiDouble		gui_min_change_;
  GuiDouble		gui_pixel_dim_;
  GuiDouble		gui_slice_thickness_;
  GuiInt		gui_top_;
  GuiInt		gui_anterior_;  
  GuiInt		gui_eyes_visible_;

  int m_width, m_height, m_depth; // data volume dimensions
  int m_FatSat;                   // 1 if we have a fatsat image, 0 otherwise
  char m_OutputPrefix[100];       // prefix to output file names



  int m_MaxIteration;             // max iterations for EM algorithms
  float m_MinChange;              // min change to decide convergence for EM
  int m_Top;                      // top of the head in axial (z) direction (0
                                  // if z=0, 1 if z=depth-1)
  int m_Anterior;                 // front of head in coronal (y) direction (0
                                  // if y=0, 1 if y=height-1)
  int m_EyesVisible;              // 1 if the scan goes down to the level of
                                  // the eyes, 0 otherwise
  float m_PixelDim;               // axialplane pixel dimensions (in mm)
  float m_SliceThickness;         // axial slice thickness in mm
  float m_center[2];              // foreground center of weight position in
                                  // axial slices
  float *m_slicecenter[2];        // per slice centers
  float m_FatThreshold;
  float m_BoneThreshold;
  int m_TopOfBrain, m_TopOfSkull, m_TopOfHead;
  int m_EyeSlice, m_TopOfEyeSlice, m_BottomOfEyeSlice;
  int m_TopTemporalSlice;
  float m_EyePosition[2][3];
  int *m_brainlim;


  vector<int>	generation_;

  //  GuiInt	m_MaxIteration;
  //GuiDouble	m_MinChange;
  

};


DECLARE_MAKER(MRITissueClassifier)

} // namespace SCIRun

#endif

