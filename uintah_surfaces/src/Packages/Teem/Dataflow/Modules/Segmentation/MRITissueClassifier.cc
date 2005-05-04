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
 *  MRITissueClassifier.cc
 *
 *  Written by:
 *   Tolga Tasdizen and McKay Davis
 *   Scientific Computing and Imaging Institute
 *   University of Utah
 *   February 2004
 *
 *  Copyright (C) 2004 SCI Institute
 */


#include <Teem/Dataflow/Modules/Segmentation/MRITissueClassifier.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/Assert.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>
#include <list>
#include <math.h>
#include <stdlib.h>
#include <set>

namespace SCITeem {

using namespace SCIRun;


#define TINY (1.0e-12f)
#define NBINMAX 500 // max number of bins to use in 1D histograms

// tissue labels
#define T_BACKGROUND 0
#define T_CSF 1
#define T_GM 2
#define T_WM 3
#define T_BONE 4
#define T_MARROW 5
#define T_MUSCLE 6
#define T_FAT 7
#define T_SINUS 8
#define T_EYE 9
#define T_EYE_LENS 10
#define T_AIR 11

// temporary labels
#define T_FOREGROUND 21
#define T_INTRACRANIAL 22


#define X_AXIS 0
#define Y_AXIS 1
#define Z_AXIS 2



void
bubble_sort (float *list, int n)
{
  float tmp;
  int i, j;
  
  for (i=0; i<n-1; i++)
    {
    for (j=0; j<n-1-i; j++)
      if (list[j+1] < list[j])
        {  
        tmp = list[j];     
        list[j] = list[j+1];
        list[j+1] = tmp;
        }
    }
}

int
distmap_4ssed( int *map, int n[2] )
{
  /*=======================================================================
    
  Program:  Distance Transforms
  Language: C
   
  Author:   Olivier Cuisenaire, http://ltswww.epfl.ch/~cuisenai
             
  Descr.: distmap_4ssed(int *map, int n[2])
 
   Takes an image (map) of integers of size n[0]*n[1] as input. It
   computes for all pixels the square of the Euclidean distance to
   the nearest zero pixel using the 4SSED algorithm. Note that this is
   only an approximation of the Euclidean DT.
 
  Reference for the algorithm:
   
   P.E. Danielsson, "Euclidean distance mapping", Computer Graphics and
   Image Processing, 14, pp. 227-248, 1980
 
   F. Leymarie and M.L. Levine, "Fast raster-scan distance propagation on
   the discrete rectangular lattice", CVGIP: Image Understanding, 55, pp.
   85-94, 1992.
    
  Reference for the implementation:
 
   Chapter 2 of "Distance transformations: fast algorithms and applications
   to medical image processing", Olivier Cuisenaire's Ph.D. Thesis, October
   1999, Universit? catholique de Louvain, Belgium.
     
  Terms of Use:
 
   You can use/modify this program for any use you wish, provided you cite
   the above references in any publication that uses it.
     
  Disclaimer:
                                                                                                 
   In no event shall the authors or distributors be liable to any party for
   direct, indirect, special, incidental, or consequential damages arising out
   of the use of this software, its documentation, or any derivatives thereof,
   even if the authors have been advised of the possibility of such damage.
    
   The authors and distributors specifically disclaim any warranties, including,
   but not limited to, the implied warranties of merchantability, fitness for a
   particular purpose, and non-infringement.  this software is provided on an
   "as is" basis, and the authors and distributors have no obligation to provide
   maintenance, support, updates, enhancements, or modifications.
    
   =========================================================================*/
  
  /***********************
    
    IMPORTANT NOTICE: at first we generate a signed EDT, under the form 
    of a array n[0]xn[1] of vectors (dx,dy) coded as two shorts in the
    memory case of an integer in the input image. 

    Later, these two shorts are replaced by the integer value dx*dx+dy*dy

  **********************/

  int *sq,*dummy;

  int i;
  int nmax,maxi;
  int maxx=n[0]-1; int maxy=n[1]-1;

  int x,y,*pt,*tpt;
  short *dx,*dy,*tdx,*tdy;

  short *ts;
  int *ti;

 /* initialisation */
  nmax=n[0]; if(nmax<n[1]) nmax=n[1];
  dummy=(int*)calloc(4*nmax+1,sizeof(int)); sq=dummy+2*nmax;
  for(i=2*nmax;i>=0;i--) sq[-i]=sq[i]=i*i;

  ti=&maxi; ts=(short*)ti; *ts=nmax; *(ts+1)=nmax;
  maxi=nmax+256*256*nmax;

  for(y=0,pt=map;y<=maxy;y++)
    for(x=0;x<=maxx;x++,pt++) 
      if(*pt!=0) *pt=maxi;
  
  /* compute simple signed DT */

  /* first raster scan */

  for(y=0;y<=maxy;y++)
    {
      pt=map+y*n[0];
      for(x=0;x<=maxx;x++,pt++) 
	{
	  dx=(short *)pt;
	  dy=dx+1;
	  
	  if(y>0)
	    {
	      tpt=pt-n[0];
	      tdx=(short *)tpt;
	      tdy=tdx+1;
	      if(sq[*dx]+sq[*dy]>sq[*tdx]+sq[*tdy+1]) { *dx=*tdx; *dy=*tdy+1; }
	    }

	  if(x>0)
	    {
	      tpt=pt-1;
	      tdx=(short *)tpt;
	      tdy=tdx+1;
	      if(sq[*dx]+sq[*dy]>sq[*tdx+1]+sq[*tdy]) { *dx=*tdx+1; *dy=*tdy; }
	    }
	}
      pt=map+y*n[0]+maxx-1;
      for(x=maxx-1;x>=0;x--,pt--) 
	{
	  dx=(short *)pt;
	  dy=dx+1;
	  
	  tpt=pt+1;
	  tdx=(short *)tpt;
	  tdy=tdx+1;
	  if(sq[*dx]+sq[*dy]>sq[*tdx-1]+sq[*tdy]) { *dx=*tdx-1; *dy=*tdy; }
	}
    }

  /* second raster scan */

  for(y=maxy,pt=map+n[0]*n[1]-1;y>=0;y--)
    {
      pt=map+n[0]*y+maxx;
      for(x=maxx;x>=0;x--,pt--) 
	{
	  dx=(short *)pt;
	  dy=dx+1;
	  
	  if(y<maxy)
	    {
	      tpt=pt+n[0];
	      tdx=(short *)tpt;
	      tdy=tdx+1;
	      if(sq[*dx]+sq[*dy]>sq[*tdx]+sq[*tdy-1]) { *dx=*tdx; *dy=*tdy-1; }
	    }

	  if(x<maxx)
	    {
	      tpt=pt+1;
	      tdx=(short *)tpt;
	      tdy=tdx+1;
	      if(sq[*dx]+sq[*dy]>sq[*tdx-1]+sq[*tdy]) { *dx=*tdx-1; *dy=*tdy; }
	    }
	}
      pt=map+n[0]*y+1;
      for(x=1;x<=maxx;x++,pt++) 
	{
	  dx=(short *)pt;
	  dy=dx+1;
	  
	  tpt=pt-1;
	  tdx=(short *)tpt;
	  tdy=tdx+1;
	  if(sq[*dx]+sq[*dy]>sq[*tdx+1]+sq[*tdy]) { *dx=*tdx+1; *dy=*tdy; }
	}
    }
     
  for(y=0,dx=(short *)map,pt=(int *)map;y<=maxy;y++)
    for(x=0;x<=maxx;x++,pt++,dx+=2) 
	*pt=sq[*dx]+sq[*(dx+1)];

  return 1;
}



/*
 * We use ParameterFile classin vispack to read arguments. This is passed 
 * on to the constructor for the class MRITissueClassifier. The 
 * constructor reads the data and other parameters. It also aligns the 
 * data into a standard coordinate frame, top-down and front-back 
 * alignment.
 * The rest of the function names are self descriptive and I have 
 * included comments at the top of the functions in the class files which 
 * I will send next. Some functions are better documented than others for 
 * now.

 * Tolga
 */

void
MRITissueClassifier::execute()
{
  update_state(Module::Executing);
  NrrdIPort *T1port = (NrrdIPort*)get_iport("T1"); 
  NrrdIPort *T2port = (NrrdIPort*)get_iport("T2"); 
  NrrdIPort *PDport = (NrrdIPort*)get_iport("PD"); 
  NrrdIPort *FSport = (NrrdIPort*)get_iport("FATSAT");
  NrrdOPort *OutPort = (NrrdOPort *)get_oport("Tissue");

  update_state(Module::NeedData);
  if (!T1port->get(m_T1_Data) || !m_T1_Data.get_rep() ||
      !T2port->get(m_T2_Data) || !m_T2_Data.get_rep() ||
      !PDport->get(m_PD_Data) || !m_PD_Data.get_rep())
  {
    return;
  }

  m_FatSat = (FSport->get(m_FATSAT_Data) && m_FATSAT_Data.get_rep())?1:0;

  if (generation_[0] != m_T1_Data.get_rep()->generation ||
      generation_[1] != m_T2_Data.get_rep()->generation ||
      generation_[2] != m_PD_Data.get_rep()->generation ||
      (m_FatSat &&
       generation_[3] != m_FATSAT_Data.get_rep()->generation)) 
  {
    generation_[0] = m_T1_Data.get_rep()->generation;
    generation_[1] = m_T2_Data.get_rep()->generation;
    generation_[2] = m_PD_Data.get_rep()->generation;
    if (m_FatSat)
      generation_[3] = m_FATSAT_Data.get_rep()->generation;

    // Get the dimensions of the volume
    m_width = m_T1_Data->nrrd->axis[X_AXIS].size;
    m_height = m_T1_Data->nrrd->axis[Y_AXIS].size;
    m_depth = m_T1_Data->nrrd->axis[Z_AXIS].size;

    // Get the GUI Variables
    m_MaxIteration = gui_max_iter_.get();
    m_MinChange = gui_min_change_.get();
    m_Top = gui_top_.get();
    m_Anterior = gui_anterior_.get();
    m_EyesVisible = gui_eyes_visible_.get();
    m_PixelDim = gui_pixel_dim_.get();
    m_SliceThickness = gui_slice_thickness_.get();
    
    NrrdDataHandle temp;

    update_state(Module::Executing);
    if (!m_Top) {
      temp = create_nrrd_of_floats(m_width, m_height, m_depth);
      nrrdFlip(temp->nrrd, m_T1_Data->nrrd, 2);
      m_T1_Data = temp;

      temp = create_nrrd_of_floats(m_width, m_height, m_depth);
      nrrdFlip(temp->nrrd, m_T2_Data->nrrd, 2);
      m_T2_Data = temp;

      temp = create_nrrd_of_floats(m_width, m_height, m_depth);
      nrrdFlip(temp->nrrd, m_PD_Data->nrrd, 2);
      m_PD_Data = temp;
      
      if (m_FatSat) {
	nrrdFlip(temp->nrrd, m_FATSAT_Data->nrrd, 2);
	m_FATSAT_Data = temp;
      }
    }
    update_state(Module::Executing);

    if (!m_Anterior) {
      temp = create_nrrd_of_floats(m_width, m_height, m_depth);
      nrrdFlip(temp->nrrd, m_T1_Data->nrrd, 1);
      m_T1_Data = temp;

      temp = create_nrrd_of_floats(m_width, m_height, m_depth);
      nrrdFlip(temp->nrrd, m_T2_Data->nrrd, 1);
      m_T2_Data = temp;

      temp = create_nrrd_of_floats(m_width, m_height, m_depth);
      nrrdFlip(temp->nrrd, m_PD_Data->nrrd, 1);
      m_PD_Data = temp;
      
      if (m_FatSat) {
	nrrdFlip(temp->nrrd, m_FATSAT_Data->nrrd, 1);
	m_FATSAT_Data = temp;
      }
    }
    update_state(Module::Executing);
  
    m_Label		= create_nrrd_of_ints(m_width, m_height, m_depth);
    m_DistBG		= create_nrrd_of_floats(m_width, m_height, m_depth);
    m_Energy		= create_nrrd_of_floats(m_width, m_height, m_depth);
    m_CSF_Energy	= create_nrrd_of_floats(m_width, m_height, m_depth);

    memset(m_Label->nrrd->data, T_BACKGROUND, m_width*m_height*m_depth*sizeof(int));

    update_state(Module::Executing);
    for (int x=0;x<m_width;x++)
      for (int y=0;y<m_height;y++)
	for (int z=0;z<m_depth;z++)
	{	 
	  float temp = get_nrrd_float(m_T2_Data, x, y, z) - get_nrrd_float(m_T1_Data, x, y, z);
	  set_nrrd_float(m_CSF_Energy, temp ,x,y,z); // T2 - T1
	}
    if (m_EyesVisible)
      EyeDetection();

    else
    {
      m_TopOfEyeSlice = -1;
      m_EyeSlice = -1;
    }
    update_state(Module::Executing);
    BackgroundDetection();
    update_state(Module::Executing);
    ComputeForegroundCenter();
    update_state(Module::Executing);
    FatDetection();
    update_state(Module::Executing);
    BrainDetection(4.0,10.0);
    update_state(Module::Executing);
    BoneDetection(10.0);
    update_state(Module::Executing);
    ScalpClassification();
    update_state(Module::Executing);
    BrainClassification();

    if (!m_Top) {
      temp = create_nrrd_of_ints(m_width, m_height, m_depth);
      nrrdFlip(temp->nrrd, m_Label->nrrd, 2);
      m_Label = temp;
    }
    update_state(Module::Executing);

    if (!m_Anterior) {
      temp = create_nrrd_of_ints(m_width, m_height, m_depth);
      nrrdFlip(temp->nrrd, m_Label->nrrd, 1);
      m_Label = temp;
    }
  }
  OutPort->send(m_Label);

  update_state(Module::Completed);
}


MRITissueClassifier::~MRITissueClassifier()
{
}


MRITissueClassifier::MRITissueClassifier (GuiContext *ctx) :
  Module("MRITissueClassifier",ctx,Filter,"Segmentation","Teem"),
  gui_max_iter_(ctx->subVar("maxIter")),
  gui_min_change_(ctx->subVar("minChange")),
  gui_pixel_dim_(ctx->subVar("pixelDim")),
  gui_slice_thickness_(ctx->subVar("sliceThickness")),
  gui_top_(ctx->subVar("top")),
  gui_anterior_(ctx->subVar("anterior")),
  gui_eyes_visible_(ctx->subVar("eyesVisible")),
  generation_(4)
    
{
}



void
MRITissueClassifier::EyeDetection()
{
  // this function should be very reliable

  // possible mode of failure: to speed up eye detection we restrict
  // the search space to where we think eyes should be in the volume -- these
  // assumptions are commented below in the code.

  // possible user feedback: if eyes are missed user can either instruct
  // algorithm to enlarge search space
  std::cout<<"EYE DETECTION\n";
  float rad = 12.5/m_PixelDim; // approximate diameter for eye = 25mm
  float radl = rad - 1.0;
  float radh = rad + 1.0;
  int mc = (int)ceil(radh);
  int ms = 2*mc +1;
  float radz = 12.5/m_SliceThickness;
  int mzc = (int)ceil(radz + 1.0);
  int mzs = 2*mzc +1;
  int depth = m_depth/2; 
  
  NrrdDataHandle mask = create_nrrd_of_floats(ms, ms, mzs);

  int x, y, z, w, h, d, x2, y2, z2, x3, y3, z3, xl, yl, xm1, xm2, xh, yl2, yh2, yh, zl, zh, zlow, zhigh, zm=0, *xpl, *xpr, *ypl, *ypr;
  float xo, yo, zo, r, sum, radhz, *maxl, *maxr, max=0.0, th, temp;

  xpl = (int*)calloc(depth,sizeof(int));
  xpr = (int*)calloc(depth,sizeof(int));
  ypl = (int*)calloc(depth,sizeof(int));
  ypr = (int*)calloc(depth,sizeof(int));
  maxl = (float*)calloc(depth,sizeof(float));
  maxr = (float*)calloc(depth,sizeof(float));

  // prepare search mask
  sum = 0.0;
  for (x=0;x<ms;x++)
    for (y=0;y<ms;y++)
      for (z=0;z<mzs;z++)
      {
        xo = x-(float)mc;
        yo = y-(float)mc;
        zo = (z-(float)mzc)*(m_SliceThickness/m_PixelDim);
        r = sqrt(xo*xo+yo*yo+zo*zo);

        if (r<radl) 
	  set_nrrd_float(mask,1.0,x,y,z);
        else if (r<radh)
	  set_nrrd_float(mask,1.0+0.5*(radl-r),x,y,z);
        else 
	  set_nrrd_float(mask,0.0,x,y,z);
	sum += get_nrrd_float(mask, x, y, z);
      }

  // mask/=mask.sum();
  for (x=0;x<ms;x++)
    for (y=0;y<ms;y++)
      for (z=0;z<mzs;z++)
	set_nrrd_float(mask, get_nrrd_float(mask,x,y,z)/sum, x,y,z);

  // we first do a slice by slice 2D search -- 2D convolutions faster than 3D
  // define search space -- assumptions about eye location in axial slices
  yl = (int)floor(0.66*(float)m_height);
  yh = m_height-mc;
  xl = (int)floor(0.25*(float)m_width);
  xm1 = (int)rint(0.5*(float)m_width)-mc;
  xm2 = xm1+2*mc;
  xh = (int)ceil(0.75*(float)m_width);
  std::cout<<"2d search in lower z-half of volume\n";
  std::cout<<"left eye search space = ("<<xl<<","<<yl<<") - ("<<xm1<<","<<yh<<")\n";
  std::cout<<"right eye search space = ("<<xm2<<","<<yl<<") - ("<<xh<<","<<yh<<")\n";

  for (z=0;z<depth;z++)
    // depth  = m_depth/2, we assume eyes are in the first half of the axial
    // slices. this asumption could fail if the data acquisition goes well
    // beyond the lower end of the brain towards the neck 
  {
    maxl[z]=maxr[z]=-100000.0;
    for (y=yl;y<yh;y+=2)
    {
      for (x=xl;x<=xm1;x+=2)
      {
        sum = 0.0;
        for (x2=0;x2<ms;x2++)
          for (y2=0;y2<ms;y2++)
            {
            x3 = x + x2 - mc;
            y3 = y + y2 - mc;
            if ((x3>=0)&&(x3<m_width)&&(y3>=0)&&(y3<m_height))	      
	      sum += ( get_nrrd_float(m_CSF_Energy,x3,y3,z) * get_nrrd_float(mask,x2,y2,mzc) );
            }
        if (sum>maxl[z])
          {
          maxl[z]=sum;
          xpl[z]=x;
          ypl[z]=y;
          }
        } // eye #1

      for (x=xm2;x<=xh;x+=2)
        {
        sum = 0.0;
        for (x2=0;x2<ms;x2++)
          for (y2=0;y2<ms;y2++)
            {
            x3 = x + x2 - mc;
            y3 = y + y2 - mc;
            if ((x3>=0)&&(x3<m_width)&&(y3>=0)&&(y3<m_height))
	      sum += ( get_nrrd_float(m_CSF_Energy,x3,y3,z) * get_nrrd_float(mask,x2,y2,mzc) );
            }
        if (sum>maxr[z])
          {
          maxr[z]=sum;
          xpr[z]=x;
          ypr[z]=y;
          }
        } // eye #2
      } // y-loop

    if (z==0)
      {
      max = Max(maxl[0],maxr[0]);
      zm = 0;
      }
    else
      {
      temp = Max(maxl[z],maxr[z]);
      if (temp>max)
        {
        max = temp;
        zm = z;
        }
      }
    } // z-loop

  // find maximum signal z-slice
  th = 0.75*max;
  for (z=zm-2;z>=0;z--)
    {
    temp = Max(maxl[z],maxr[z]);
    if (temp<th) break;
    }
  zlow = Max(z,0);
  for (z=zm+2;z<depth;z++)
    {
    temp = Max(maxl[z],maxr[z]);
    if (temp<th) break;
    }
  zhigh = Min(z,depth-1);

  // now use 2D results to define a narrow 3D search space
  // this part shouldn't fail given that the above 2D search worked properly 
  xl = xpl[zm]-(int)ceil(6.0/m_PixelDim);
  xm1 = xpl[zm]+(int)ceil(6.0/m_PixelDim);
  xm2 = xpr[zm]-(int)ceil(6.0/m_PixelDim);
  xh = xpr[zm]+(int)ceil(6.0/m_PixelDim);
  yl = ypl[zm]-(int)ceil(6.0/m_PixelDim);
  yh = ypl[zm]+(int)ceil(6.0/m_PixelDim);
  yl2 = ypr[zm]-(int)ceil(6.0/m_PixelDim);
  yh2 = ypr[zm]+(int)ceil(6.0/m_PixelDim);
  std::cout<<"3d search\n";
  std::cout<<"left eye search space = ("<<xl<<","<<yl<<") - ("<<xm1<<","<<yh<<")\n";
  std::cout<<"right eye search space = ("<<xm2<<","<<yl2<<") - ("<<xh<<","<<yh2<<")\n";
  
  for (z=zlow;z<=zhigh;z++)
    {
    maxl[z]=maxr[z]=-100000.0;
    for (y=yl;y<yh;y++)
      {
      for (x=xl;x<=xm1;x++)
        {
        sum = 0.0;
        for (x2=0;x2<ms;x2++)
          for (y2=0;y2<ms;y2++)
            for (z2=0;z2<mzs;z2++)
              {
              x3 = x + x2 - mc;
              y3 = y + y2 - mc;
              z3 = z + z2 - mzc;
              if ((x3>=0)&&(x3<m_width)&&(y3>=0)&&(y3<m_height)&&(z3>=0)&&(z3<m_depth))
                sum += ( get_nrrd_float(m_CSF_Energy,x3,y3,z3) * get_nrrd_float(mask,x2,y2,z2) );
              }
        if (sum>maxl[z])
          {
          maxl[z]=sum;
          xpl[z]=x;
          ypl[z]=y;
          }
        } // eye #1
      }
    for (y=yl2;y<yh2;y++)
      {
      for (x=xm2;x<=xh;x++)
        {
        sum = 0.0;
        for (x2=0;x2<ms;x2++)
          for (y2=0;y2<ms;y2++)
            for (z2=0;z2<mzs;z2++)
              {
              x3 = x + x2 - mc;
              y3 = y + y2 - mc;
              z3 = z + z2 - mzc;
              if ((x3>=0)&&(x3<m_width)&&(y3>=0)&&(y3<m_height)&&(z3>=0)&&(z3<m_depth))
                sum += ( get_nrrd_float(m_CSF_Energy,x3,y3,z3) * get_nrrd_float(mask,x2,y2,z2) );
              }
        if (sum>maxr[z])
          {
          maxr[z]=sum;
          xpr[z]=x;
          ypr[z]=y;
          }
        } // eye #2
      } 
    } //  z-loop

  // now that we located approximate centers of the eyes
  // do floodfills to extract eyes and eye lenses -- this can be simplified if
  // we don't want the eye-lens category
  float a, b, zp;
  NrrdDataHandle tempV; //   VISVolume<int> tempV;
  NrrdDataHandle ff2;   //   VISImage<int> ff2;
  
  max = maxl[zlow+1];
  zm = zlow+1;
  for (z=zlow+2;z<zhigh;z++)
    if (maxl[z]>max)
    {
      max=maxl[z];
      zm=z;
    }
  if ((zm==(zlow+1))&&(max<maxl[zlow]))
  {
    m_EyePosition[0][0]=(float)xpl[zlow];
    m_EyePosition[0][1]=(float)ypl[zlow];
    m_EyePosition[0][2]=(float)zlow;
  }
  else if ((zm==(zhigh-1))&&(max<maxl[zhigh]))
  {
    m_EyePosition[0][0]=(float)xpl[zhigh];
    m_EyePosition[0][1]=(float)ypl[zhigh];
    m_EyePosition[0][2]=(float)zhigh;
  }
  else
  {
    a = 0.5*(maxl[zm+1]+maxl[zm-1])-max;
    b = 0.5*(maxl[zm+1]-maxl[zm-1]);
    zp = -0.5*b/a;
    sum = maxl[zm-1]+max+maxl[zm+1];
    m_EyePosition[0][0]=((a-b+max)*(float)xpl[zm-1]+max*(float)xpl[zm]+(a+b+max)*(float)xpl[zm+1])/sum;
    m_EyePosition[0][1]=((a-b+max)*(float)ypl[zm-1]+max*(float)ypl[zm]+(a+b+max)*(float)ypl[zm+1])/sum;
    m_EyePosition[0][2]=zp+(float)zm;
  }
  th = 0.5*max;
  radh = 1.5*rad/m_PixelDim;
  // we work on a subset of the volume to speed things up
  xl = Max((int)floor(m_EyePosition[0][0]-radh),0);
  xh = Min((int)ceil(m_EyePosition[0][0]+radh),m_width-1);
  yl = Max((int)floor(m_EyePosition[0][1]-radh),0);
  yh = Min((int)ceil(m_EyePosition[0][1]+radh),m_height-1);
  radhz = 1.5*rad/m_SliceThickness;
  zl = Max((int)floor(m_EyePosition[0][2]-radhz),0);
  zh =  Min((int)ceil(m_EyePosition[0][2]+radhz),m_depth-1);
  w = xh-xl+1;
  h = yh-yl+1;
  d = zh-zl+1;
  tempV = create_nrrd_of_ints(w,h,d);
  for (x=0;x<w;x++)
    for (y=0;y<h;y++)
      for (z=0;z<d;z++)
        if (get_nrrd_float(m_CSF_Energy,xl+x,yl+y,zl+z)>=th) 
	  set_nrrd_int(tempV,1,x,y,z);
        else 
	  set_nrrd_int(tempV,0,x,y,z);

  floodFill(tempV, 1,2,Round(radh), Round(radh), Round(radhz));

  for (x=0;x<w;x++)
    for (y=0;y<h;y++)
      for (z=0;z<d;z++)
        if (get_nrrd_int(tempV,x,y,z)==1) 
	  set_nrrd_int(tempV,0,x,y,z);

  
  m_TopOfEyeSlice=-1;
  m_BottomOfEyeSlice=-1;
  for (z=0;z<d;z++)
  {
    // flood fill from outside (4 corners) to discriminate eye lens which is enclosed
    // within the eye tissue
    ff2 = extract_nrrd_slice_int(tempV, z);
    floodFill(ff2,0,3,0,0);
    floodFill(ff2,0,3,w-1,0);
    floodFill(ff2,0,3,0,h-1);
    floodFill(ff2,0,3,w-1,h-1);
    
    for (x=0;x<w;x++)
      for (y=0;y<h;y++)
        if (get_nrrd_int(tempV,x,y,z)==2)
	{
          set_nrrd_int(m_Label,T_EYE,x+xl,y+yl,z+zl);
          if ((z+zl)>m_TopOfEyeSlice) 
	    m_TopOfEyeSlice=z+zl;
          if (m_BottomOfEyeSlice<0) 
	    m_BottomOfEyeSlice=z+zl;
          }
        else if (get_nrrd_int2(ff2,x,y)!=3) 
	  set_nrrd_int(m_Label,T_EYE_LENS,x+xl,y+yl,z+zl);
    }
  std::cout<<"First eye position = ("<<m_EyePosition[0][0]<<","<<m_EyePosition[0][1]<<","<<m_EyePosition[0][2]<<")\n";

  // do same things for the other eye
  max = maxr[zlow+1];
  zm = zlow+1;
  for (z=zlow+2;z<zhigh;z++)
    if (maxr[z]>max)
      {
      max=maxr[z];
      zm=z;
      }
  if ((zm==(zlow+1))&&(max<maxr[zlow]))
    {
    m_EyePosition[1][0]=(float)xpr[zlow];
    m_EyePosition[1][1]=(float)ypr[zlow];
    m_EyePosition[1][2]=(float)zlow;
    }
  else if ((zm==(zhigh-1))&&(max<maxr[zhigh]))
    {
    m_EyePosition[1][0]=(float)xpr[zhigh];
    m_EyePosition[1][1]=(float)ypr[zhigh];
    m_EyePosition[1][2]=(float)zhigh;
    }
  else
    {
    a = 0.5*(maxr[zm+1]+maxr[zm-1])-max;
    b = 0.5*(maxr[zm+1]-maxr[zm-1]);
    zp = -0.5*b/a;
    sum = maxr[zm-1]+max+maxr[zm+1];
    m_EyePosition[1][0]=((a-b+max)*(float)xpr[zm-1]+max*(float)xpr[zm]+(a+b+max)*(float)xpr[zm+1])/sum;
    m_EyePosition[1][1]=((a-b+max)*(float)ypr[zm-1]+max*(float)ypr[zm]+(a+b+max)*(float)ypr[zm+1])/sum;
    m_EyePosition[1][2]=zp+(float)zm;
    }
  th = 0.5*max;
  radh = 1.5*rad/m_PixelDim;
  xl = Max((int)floor(m_EyePosition[1][0]-radh),0);
  xh = Min((int)ceil(m_EyePosition[1][0]+radh),m_width-1);
  yl = Max((int)floor(m_EyePosition[1][1]-radh),0);
  yh = Min((int)ceil(m_EyePosition[1][1]+radh),m_height-1);
  radhz = 1.5*rad/m_SliceThickness;
  zl = Max((int)floor(m_EyePosition[1][2]-radhz),0);
  zh =  Min((int)ceil(m_EyePosition[1][2]+radhz),m_depth-1);

  w = xh-xl+1;
  h = yh-yl+1;
  d = zh-zl+1;
  tempV= create_nrrd_of_ints(w,h,d);

  for (x=0;x<w;x++)
    for (y=0;y<h;y++)
      for (z=0;z<d;z++)
        if (get_nrrd_float(m_CSF_Energy,xl+x,yl+y,zl+z)>=th) 
	  set_nrrd_int(tempV,1,x,y,z);
        else 
	  set_nrrd_int(tempV,0,x,y,z);

  floodFill(tempV,1,2,Round(radh),Round(radh),Round(radhz));
  
  for (x=0;x<w;x++)
    for (y=0;y<h;y++)
      for (z=0;z<d;z++)
        if (get_nrrd_int(tempV,x,y,z)==1) 
	  set_nrrd_int(tempV,0,x,y,z);
  
  for (z=0;z<d;z++)
  {
    ff2 = extract_nrrd_slice_int(tempV, z);
    floodFill(ff2,0,3,0,0);
    floodFill(ff2,0,3,w-1,0);
    floodFill(ff2,0,3,0,h-1);
    floodFill(ff2,0,3,w-1,h-1);
    
    for (x=0;x<w;x++)
      for (y=0;y<h;y++)
        if (get_nrrd_int(tempV,x,y,z)==2)
	{
          set_nrrd_int(m_Label,T_EYE,x+xl,y+yl,z+zl);
          if ((z+zl)>m_TopOfEyeSlice) m_TopOfEyeSlice=z+zl;
          if (m_BottomOfEyeSlice<0) m_BottomOfEyeSlice=z+zl;
	}
        else if (get_nrrd_int2(ff2,x,y)!=3) 
	  set_nrrd_int(m_Label,T_EYE_LENS,x+xl,y+yl,z+zl);
  }
  std::cout<<"Second eye position = ("<<m_EyePosition[1][0]<<","<<m_EyePosition[1][1]<<","<<m_EyePosition[1][2]<<")\n";
  
  m_EyeSlice = Max((int)rint(0.5*(m_EyePosition[0][2]+m_EyePosition[1][2])),0);
  std::cout<<"Eye bottom slice = "<<m_BottomOfEyeSlice<<" , top slice = "<<m_TopOfEyeSlice<<std::endl;

  free (xpl);
  free (xpr);
  free (ypl);
  free (ypr);
  free (maxl);
  free (maxr);
}

float
MRITissueClassifier::Compute2ClassThreshold (float *p, float *v, int n)
{
  int i;
  float diffl, diffh, x;
  for (i = n-1; i>=0; i--)       // we go backwards because foreground has
                                 // larger variance and can have larger
                                 // probability on either side of the background peak
    if (p[i*2+1]<=p[i*2]) break; // is background probability larger?

  if (i<=0) return v[0];
  else
    {
    if (i>=(n-1)) return v[n-1];
    diffl=p[i*2+1]-p[i*2];
    diffh=p[(i+1)*2+1]-p[(i+1)*2];
    x = diffl/(diffl-diffh);
    return v[i] + x * (v[i+1]-v[i]);
    }
}

void
MRITissueClassifier::Clear1DHistogram (int *hist, int n)
{
  for (int i = 0; i<n; i++) hist[i]=0;
}



// data is assumed to be a 3d nrrd of floats
void
MRITissueClassifier::Accumulate1DHistogram (float *upperlims, int *hist,
					    int n,
					    NrrdDataHandle data,
					    int lx, int ly, int lz,
					    int hx, int hy, int hz)
{
  // n is the size of hist
  // (lx,ly,lz) and (hx,hy,hz) determine subvolume to be used
  int x, y, z, i;

  for (x=lx;x<=hx;x++)
    for (y=ly;y<=hy;y++)
      for (z=lz;z<=hz;z++)
      {
        for (i=0;i<n-1;i++) 
	  if (get_nrrd_float(data,x,y,z)<=upperlims[i]) break;
        hist[i]++;
      }
}

void
MRITissueClassifier::Accumulate1DHistogramWithLabel (float *upperlims, int *hist,
						     int n, int m,
						     int lx, int ly, int lz,
						     int hx, int hy, int hz, int label)
{
  ASSERT(0);
  // n is the size of hist
  // m is the data component selection
  // (lx,ly,lz) and (hx,hy,hz) determine subvolume to be used
  int x, y, z, i;
  ColumnMatrix data;
  for (x=lx;x<=hx;x++)
    for (y=ly;y<=hy;y++)
      for (z=lz;z<=hz;z++)
        if (get_nrrd_int(m_Label,x,y,z)==label)
	{
          data = get_m_Data(x,y,z);
          for (i=0;i<n-1;i++) 
	    if (data.get(m)<=upperlims[i]) break;
          hist[i]++;
	}
}


// data is assumed to be a 3d nrrd of floats
void
MRITissueClassifier::Accumulate1DHistogramWithLabel (float *upperlims, int *hist,
						     int n,
						     NrrdDataHandle data,
						     int lx, int ly, int lz,
						     int hx, int hy, int hz,
						     int label)
{
  // n is the size of hist
  // (lx,ly,lz) and (hx,hy,hz) determine subvolume to be used
  int x, y, z, i;
  for (x=lx;x<=hx;x++)
    for (y=ly;y<=hy;y++)
      for (z=lz;z<=hz;z++)
        if (get_nrrd_int(m_Label,x,y,z)==label)
	{
          for (i=0;i<n-1;i++) 
	    if (get_nrrd_float(data,x,y,z)<=upperlims[i]) break;
          hist[i]++;
	}
}

float*
MRITissueClassifier::EM1DWeighted (float *data, float *weight, int n,
				   int c, float *mean, float *prior, float *var)
{
  // n is length of data
  // c is number of classes
  // mean, prior, var should be initialized before calling this function

  bool flag = true;
  int i, j, k, it = 0;
  float diff, sum, sum2, distsq, dist, w;
  float *prob = (float*)calloc(n*c,sizeof(float));
  float *meanp= (float*)calloc(c,sizeof(float));
  float *istd = (float*)calloc(c,sizeof(float));
  
  while (flag&&(it<m_MaxIteration))
  {
    update_state(Module::Executing);
    for (j=0;j<c;j++) meanp[j]=mean[j];
    for (j=0;j<c;j++) istd[j]=1.0f/sqrt(var[j]);

    // evaluate probabilities
    for (i=0;i<n;i++)
      {
      sum = 0.0f;
      k=i*c;
      for (j=0;j<c;j++)
        {
        diff=data[i]-mean[j];
        distsq=(diff*diff)/var[j];
        prob[k+j]=exp(-0.5*distsq)*prior[j]*istd[j];
        sum+=prob[k+j];
        }
      for (j=0;j<c;j++) prob[k+j]/=Max(sum,TINY);
      }

    // evaluate statistics
    sum2=dist=0.0f;
    for (j=0;j<c;j++)
      {
      sum=0.0f;
      mean[j]=var[j]=0.0f;

      k=j;
      for (i=0;i<n;i++)
        {
        w = prob[k]*weight[i];
        mean[j]+=w*data[i];
        sum+=w;
        k+=c;
        }
      sum2+=sum;
      prior[j]=sum;
      mean[j]/=Max(sum,TINY);

      k=j;
      for (i=0;i<n;i++)
        {
        w = prob[k]*weight[i];
        diff=data[i]-mean[j];
        var[j]+=w*diff*diff;
        k+=c;
        }
      var[j]/=Max(sum,TINY);
      
      // compute amount of change
      if (it>0) dist+=fabs(mean[j]-meanp[j])/Max(istd[j],TINY);
      meanp[j]=mean[j];
      }
    for (j=0;j<c;j++) prior[j]/=Max(sum2,TINY);

    // check convergence
    if (it>0)
      {
      dist/=(float)c;
      if (dist<m_MinChange) flag=0;
      }
    it++;
    }
  std::cout<<it<<" EM iterations performed.\n";
  
  // evaluate final probabilities
  for (j=0;j<c;j++) istd[j]=1.0f/sqrt(var[j]);
  for (i=0;i<n;i++)
    {
    sum = 0.0f;
    k=i*c;
    for (j=0;j<c;j++)
      {
      diff=data[i]-mean[j];
      distsq=(diff*diff)/var[j];
      prob[k+j]=exp(-0.5*distsq)*prior[j]*istd[j];
      sum+=prob[k+j];
      }
    for (j=0;j<c;j++) prob[k+j]/=Max(sum,TINY);
    }
  
  free(istd);
  free (meanp);
  
  return prob;
}

float *MRITissueClassifier::EM_CSF_GM_WM (ColumnMatrix &CSF_mean, DenseMatrix &CSF_cov, float *CSF_prior,
					  ColumnMatrix &GM_mean, DenseMatrix &GM_cov, float *GM_prior,
					  ColumnMatrix &WM_mean, DenseMatrix &WM_cov, float *WM_prior,
					  int label)
{
  int x, y, z, n, flag, it, j, k, flops, memrefs;    
  ColumnMatrix CSF_meanp, GM_meanp, WM_meanp;
  float sum, sum2, dist, CSF_const, GM_const, WM_const;
  ColumnMatrix feature(3), diff(3), temp(3);
  DenseMatrix CSF_icov = CSF_cov, GM_icov = GM_cov, WM_icov = WM_cov;
  DenseMatrix temp_dense(3,3);
  
  n = 0;
  for (z=0;z<m_depth;z++)
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if (get_nrrd_int(m_Label,x,y,z)==label) n++;

  float *prob = (float*)calloc(n*3,sizeof(float));

  flag = 1;
  it = 0;
  while (flag&&(it<25))
  {
    update_state(Module::Executing);
    CSF_meanp = CSF_mean;GM_meanp = GM_mean;WM_meanp = WM_mean;

    // evaluate probabilities
    // compute matrix inverse
    CSF_icov = CSF_cov; GM_icov = GM_cov; WM_icov = WM_cov;
    if (!CSF_icov.invert()) error("Cannot invert CSF_icov"); 
    if (!GM_icov.invert()) error("Cannot invert GM_icov");
    if (!WM_icov.invert()) error("Cannot invert WM_icov");

    CSF_const = 1.0/sqrt(Max((float)(CSF_cov.get(0,0)*(CSF_cov.get(1,1)*CSF_cov.get(2,2)-CSF_cov.get(1,2)*CSF_cov.get(2,1))
				     -CSF_cov.get(0,1)*(CSF_cov.get(1,0)*CSF_cov.get(2,2)-CSF_cov.get(2,0)*CSF_cov.get(1,2))
				     +CSF_cov.get(0,2)*(CSF_cov.get(1,0)*CSF_cov.get(2,1)-CSF_cov.get(2,0)*CSF_cov.get(1,1))),TINY));
    GM_const = 1.0/sqrt(Max((float)(GM_cov.get(0,0)*(GM_cov.get(1,1)*GM_cov.get(2,2)-GM_cov.get(1,2)*GM_cov.get(2,1))
				    -GM_cov.get(0,1)*(GM_cov.get(1,0)*GM_cov.get(2,2)-GM_cov.get(2,0)*GM_cov.get(1,2))
				    +GM_cov.get(0,2)*(GM_cov.get(1,0)*GM_cov.get(2,1)-GM_cov.get(2,0)*GM_cov.get(1,1))),TINY));
    WM_const = 1.0/sqrt(Max((float)(WM_cov.get(0,0)*(WM_cov.get(1,1)*WM_cov.get(2,2)-WM_cov.get(1,2)*WM_cov.get(2,1))
				    -WM_cov.get(0,1)*(WM_cov.get(1,0)*WM_cov.get(2,2)-WM_cov.get(2,0)*WM_cov.get(1,2))
				    +WM_cov.get(0,2)*(WM_cov.get(1,0)*WM_cov.get(2,1)-WM_cov.get(2,0)*WM_cov.get(1,1))),TINY));
    
    k=0;
    for (z=0;z<m_depth;z++)
      for (y=0;y<m_height;y++)
        for (x=0;x<m_width;x++)
          if (get_nrrd_int(m_Label,x,y,z)==label)
            {
	      feature = get_m_Data(x,y,z);
	      if (feature.nrows() != 3) feature.resize(3);
            
	      Sub(diff, feature, CSF_mean); //  diff=feature-CSF_mean;
	      CSF_icov.mult(diff, temp, flops, memrefs); // CSF_icov*diff = temp
	      dist=Max(Dot(diff, temp),0.0);
	      prob[k]=exp(-0.5*dist)*(*CSF_prior)*CSF_const;
	      sum=prob[k];
	      
	      Sub(diff, feature, GM_mean); // diff=feature-GM_mean;
	      GM_icov.mult(diff, temp, flops, memrefs);    
	      dist=Max(Dot(diff,temp),0.0);// dist=Max(diff.dot(GM_icov*diff),0.0);            
	      prob[k+1]=exp(-0.5*dist)*(*GM_prior)*GM_const;
	      sum+=prob[k+1];
	      
	      Sub(diff, feature, WM_mean); // diff=feature-WM_mean;
	      WM_icov.mult(diff, temp, flops, memrefs);    
	      dist=Max(Dot(diff,temp),0.0);// dist=Max(diff.dot(GM_icov*diff),0.0);            
	      prob[k+2]=exp(-0.5*dist)*(*WM_prior)*WM_const;
	      sum+=prob[k+2];
	      
	      for (j=0;j<3;j++) prob[k+j]/=Max(sum,TINY);
	      k+=3;
            }

    // compute new statistics
    dist=0.0f;
    sum2=0.0f;
    CSF_mean.zero(); GM_mean.zero();  WM_mean.zero();
    CSF_cov.zero(); GM_cov.zero(); WM_cov.zero();
    (*CSF_prior) = 0.0; (*GM_prior) = 0.0;(*WM_prior) = 0.0;
    
    k = 0;
    for (z=0;z<m_depth;z++)
      for (y=0;y<m_height;y++)
        for (x=0;x<m_width;x++)
          if (get_nrrd_int(m_Label,x,y,z)==label)
	  {
	    feature = get_m_Data(x,y,z);
	    if (feature.nrows() != 3) feature.resize(3);
	    
	    temp = feature;
	    temp.scalar_multiply(prob[k]);
            Add(CSF_mean, CSF_mean, temp);
	    
	    temp = feature;
	    temp.scalar_multiply(prob[k+1]);
            Add(GM_mean, GM_mean, temp);

	    temp = feature;
	    temp.scalar_multiply(prob[k+2]);
            Add(WM_mean, WM_mean, temp);

            (*CSF_prior) += prob[k];
            (*GM_prior) += prob[k+1];
            (*WM_prior) += prob[k+2];

            k+=3;
	  }

    // CSF_mean /= Max((*CSF_prior), TINY);
    CSF_mean.scalar_multiply(1.0 / Max((*CSF_prior), TINY));
    // GM_mean /= Max((*GM_prior), TINY);
    GM_mean.scalar_multiply(1.0 / Max((*GM_prior), TINY));
    // WM_mean /= Max((*WM_prior), TINY);
    WM_mean.scalar_multiply(1.0 / Max((*WM_prior), TINY));

    k = 0;
    for (z=0;z<m_depth;z++)
      for (y=0;y<m_height;y++)
        for (x=0;x<m_width;x++)
          if (get_nrrd_int(m_Label,x,y,z)==label)
	  {
            feature = get_m_Data(x,y,z);
	    if (feature.nrows() != 3) feature.resize(3);
            
            Sub(diff,feature,CSF_mean);
	    temp_dense = diff.exterior(diff);
	    temp_dense.scalar_multiply(prob[k]);
	    Add(CSF_cov, CSF_cov, temp_dense);

            Sub(diff,feature,GM_mean); // diff = feature - GM_mean;
	    temp_dense = diff.exterior(diff);
	    temp_dense.scalar_multiply(prob[k+1]);
	    Add(GM_cov, GM_cov, temp_dense);

            Sub(diff,feature,WM_mean); // diff = feature - WM_mean;
	    temp_dense = diff.exterior(diff);
	    temp_dense.scalar_multiply(prob[k+2]);
	    Add(WM_cov, WM_cov, temp_dense);

            k+=3;
	  }
    // CSF_cov /= Max((*CSF_prior), TINY);
    CSF_cov.scalar_multiply (1.0 / Max((*CSF_prior), TINY));
    // GM_cov /= Max((*GM_prior), TINY);
    GM_cov.scalar_multiply (1.0 / Max((*GM_prior), TINY));
    // WM_cov /= Max((*WM_prior), TINY);
    WM_cov.scalar_multiply (1.0 / Max((*WM_prior), TINY));

    sum2 = (*CSF_prior)+(*GM_prior)+(*WM_prior);
    (*CSF_prior)/=sum2;
    (*GM_prior)/=sum2;
    (*WM_prior)/=sum2;

    // compute amount of change
    dist = 0.0;
    if (it!=0)
    {
      // diff=CSF_mean - CSF_meanp;
      Sub(diff, CSF_mean, CSF_meanp);
      dist+=(diff.vector_norm()/(TINY+CSF_meanp.vector_norm()));
      
      // diff=GM_mean - GM_meanp;
      Sub(diff, GM_mean, GM_meanp);
      dist+=(diff.vector_norm()/(TINY+GM_meanp.vector_norm()));
      
      // diff=WM_mean - WM_meanp;
      Sub(diff, WM_mean, WM_meanp);
      dist+=(diff.vector_norm()/(TINY+WM_meanp.vector_norm()));
      
      dist=sqrt(dist/3.0);
      if (dist<m_MinChange) flag=0;
    }
    it++;
  } // end while

  std::cout<<it<<" iterations performed\n";
  std::cout<<"EM final\n";
  std::cout<<"WM  mean = ("<<WM_mean.get(0)<<","<<WM_mean.get(1)<<","<<WM_mean.get(2)<<") prior = "<<(*WM_prior)<<"\n";
  std::cout<<"    covariance "; WM_cov.print(std::cout);//<<WM_cov<<"\n";
  std::cout<<"GM  mean = ("<<GM_mean.get(0)<<","<<GM_mean.get(1)<<","<<GM_mean.get(2)<<") prior = "<<(*GM_prior)<<"\n";
  std::cout<<"    covariance "; GM_cov.print(std::cout);//<<GM_cov<<"\n";
  std::cout<<"CSF mean = ("<<CSF_mean.get(0)<<","<<CSF_mean.get(1)<<","<<CSF_mean.get(2)<<") prior = "<<(*CSF_prior)<<"\n";
  std::cout<<"    covariance "; CSF_cov.print(std::cout);//<<CSF_cov<<"\n";
  
  return prob;
}

float *
MRITissueClassifier::EM_Muscle_Fat (ColumnMatrix &Muscle_mean, DenseMatrix &Muscle_cov, float *Muscle_prior,
				    ColumnMatrix &Fat_mean, DenseMatrix &Fat_cov, float *Fat_prior,
				    int label, int dim)
{
  int x, y, z, n, flag, it, j, k, flops, memrefs;    
  ColumnMatrix Muscle_meanp(dim), Fat_meanp(dim);
  float sum, sum2, dist, Muscle_const, Fat_const;
  ColumnMatrix diff(dim), data(dim), temp(dim);
  DenseMatrix Muscle_icov = Muscle_cov, Fat_icov = Fat_cov, tempDense(dim,dim);

  n = 0;
  for (z=0;z<m_depth;z++)
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if (get_nrrd_int(m_Label,x,y,z)==label) n++;

  float *prob = (float*)calloc(n*2,sizeof(float));

  flag = 1;
  it = 0;
  while (flag&&(it<25))
  {
    update_state(Module::Executing);
    Muscle_meanp = Muscle_mean;Fat_meanp = Fat_mean;

    // evaluate probabilities
    // compute matrix inverse
    Muscle_icov = Muscle_cov;
    Fat_icov = Fat_cov;
    if (!Muscle_icov.invert()) error ("cannot invert muslce_iconv");
    if (!Fat_icov.invert()) error ("cannot invert fat_cov");

    if (dim==3)
    {
      Muscle_const = 1.0/sqrt(Max((float)(Muscle_cov.get(0,0)*(Muscle_cov.get(1,1)*Muscle_cov.get(2,2)-Muscle_cov.get(1,2)*Muscle_cov.get(2,1))
					  -Muscle_cov.get(0,1)*(Muscle_cov.get(1,0)*Muscle_cov.get(2,2)-Muscle_cov.get(2,0)*Muscle_cov.get(1,2))
					  +Muscle_cov.get(0,2)*(Muscle_cov.get(1,0)*Muscle_cov.get(2,1)-Muscle_cov.get(2,0)*Muscle_cov.get(1,1))),TINY));
      Fat_const = 1.0/sqrt(Max((float)(Fat_cov.get(0,0)*(Fat_cov.get(1,1)*Fat_cov.get(2,2)-Fat_cov.get(1,2)*Fat_cov.get(2,1))
					 -Fat_cov.get(0,1)*(Fat_cov.get(1,0)*Fat_cov.get(2,2)-Fat_cov.get(2,0)*Fat_cov.get(1,2))
				       +Fat_cov.get(0,2)*(Fat_cov.get(1,0)*Fat_cov.get(2,1)-Fat_cov.get(2,0)*Fat_cov.get(1,1))),TINY));
    }
    else
    {
      Muscle_const = 1.0/sqrt(Max((float)fabs(Muscle_cov.determinant()),TINY));  
      Fat_const = 1.0/sqrt(Max((float)fabs(Fat_cov.determinant()),TINY));
    }
    
    k=0;
    for (z=0;z<m_depth;z++)
      for (y=0;y<m_height;y++)
        for (x=0;x<m_width;x++)
          if (get_nrrd_int(m_Label,x,y,z)==label)
	  {
	      
            data = get_m_Data(x,y,z);
            
	    Sub(diff, data, Muscle_mean); //  diff=data-Muscle_mean;
	    Muscle_icov.mult(diff, temp, flops, memrefs); // Muscle_icov*diff = temp
	    dist = Max(Dot(diff, temp), 0.0);
            prob[k]=exp(-0.5*dist)*(*Muscle_prior)*Muscle_const;
            sum=prob[k];

            Sub(diff, data, Fat_mean); //diff=data-Fat_mean;
	    Fat_icov.mult(diff, temp, flops, memrefs); // Fat_icov*diff = temp
	    dist = Max(Dot(diff, temp), 0.0);
            prob[k+1]=exp(-0.5*dist)*(*Fat_prior)*Fat_const;
            sum+=prob[k+1];

#if defined(__APPLE__) && !defined(isnan)
// On the mac, the isnan define (from math.h) gets screwed up if
// iostream is included... go figure... this is a hack to fix that.
#  define isnan(x) __isnanf(x)
#endif
            for (j=0;j<2;j++) if (isnan(prob[k+j]))
	    {
              std::cout<<"NaN prob. in EM1DVolumetric "<<k<<"\n";
              exit(-1);
	    }
            
            for (j=0;j<2;j++) prob[k+j]/=Max(sum,TINY);
            k+=2;
	  }

    // compute new statistics
    dist=0.0f;
    sum2=0.0f;
    Muscle_mean.zero();
    Fat_mean.zero();
    Muscle_cov.zero();
    Fat_cov.zero();
    (*Muscle_prior) = 0.0; (*Fat_prior) = 0.0;
    
    k = 0;
    for (z=0;z<m_depth;z++)
      for (y=0;y<m_height;y++)
        for (x=0;x<m_width;x++)
          if (get_nrrd_int(m_Label,x,y,z)==label)
	  {
            data = get_m_Data(x,y,z);
	    data.scalar_multiply(prob[k]);
	    Add(Muscle_mean, Muscle_mean, data);

            data = get_m_Data(x,y,z);
	    data.scalar_multiply(prob[k+1]);
            Add(Fat_mean, Fat_mean, data);

            (*Muscle_prior) += prob[k];
            (*Fat_prior) += prob[k+1];

            k+=2;
	  }
    
    Muscle_mean.scalar_multiply(1.0 / Max((*Muscle_prior), TINY));
    Fat_mean.scalar_multiply(1.0 / Max((*Fat_prior), TINY));

    k = 0;
    for (z=0;z<m_depth;z++)
      for (y=0;y<m_height;y++)
        for (x=0;x<m_width;x++)
          if (get_nrrd_int(m_Label,x,y,z)==label)
	  {
            data = get_m_Data(x,y,z);

            Sub(diff, data, Muscle_mean);
	    tempDense = diff.exterior(diff);
	    tempDense.scalar_multiply(prob[k]);
	    Add(Muscle_cov, Muscle_cov, tempDense);

            Sub(diff, data, Fat_mean);
	    tempDense = diff.exterior(diff);
	    tempDense.scalar_multiply(prob[k+1]);
	    Add(Fat_cov, Fat_cov, tempDense);

            k+=2;
	  }
    Muscle_cov.scalar_multiply(1.0 / Max((*Muscle_prior), TINY));
    Fat_cov.scalar_multiply(1.0 / Max((*Fat_prior), TINY));

    sum2 = (*Muscle_prior)+(*Fat_prior);

    (*Muscle_prior)/=sum2;
    (*Fat_prior)/=sum2;

    // compute amount of change
    dist = 0.0;
    if (it!=0)
    {
      Sub(diff, Muscle_mean, Muscle_meanp); // diff=Muscle_mean - Muscle_meanp;
      dist+=(diff.vector_norm()/(TINY+Muscle_meanp.vector_norm()));
      Sub (diff, Fat_mean, Fat_meanp);
      dist+=(diff.vector_norm()/(TINY+Fat_meanp.vector_norm()));
      dist=sqrt(dist/2.0);
      if (dist<m_MinChange) flag=0;
      }
    it++;
  } // end while

  std::cout<<it<<" iterations performed\n";
  std::cout<<"EM final\n";
  if (dim==3)
  {
    std::cout<<"Fat    mean = ("<<Fat_mean.get(0)<<","<<Fat_mean.get(1)<<","<<Fat_mean.get(2)<<") prior = "<<(*Fat_prior)<<"\n";
    std::cout<<"       covariance "; Fat_cov.print(std::cout); //<<Fat_cov<<"\n";
    std::cout<<"Muscle mean = ("<<Muscle_mean.get(0)<<","<<Muscle_mean.get(1)<<","<<Muscle_mean.get(2)<<") prior = "<<(*Muscle_prior)<<"\n";
    std::cout<<"       covariance "; Muscle_cov.print(std::cout); //<<Muscle_cov<<"\n";
  }
  else
  {
    std::cout<<"Fat    mean = ("<<Fat_mean.get(0)<<","<<Fat_mean.get(1)<<","
	     <<Fat_mean.get(2)<<","<<Fat_mean.get(3)<<") prior = "<<(*Fat_prior)<<"\n";
    std::cout<<"       covariance "; Fat_cov.print(std::cout); //<<Fat_cov<<"\n";
    std::cout<<"Muscle mean = ("<<Muscle_mean.get(0)<<","<<Muscle_mean.get(1)<<","
	     <<Muscle_mean.get(2)<<","<<Muscle_mean.get(3)<<") prior = "<<(*Muscle_prior)<<"\n";
    std::cout<<"       covariance "; Muscle_cov.print(std::cout); //<<Muscle_cov<<"\n";
  }
  
  return prob;
}

void
MRITissueClassifier::BackgroundDetection ()
{
  // ths function should be reliable
  // possible mode of failure: background can leak into bone through ear canal
  // we treat the volume in 2 parts divided at the axial slice through the
  // centers of the eyes to avoid this
  // possible user interaction: if leakage still occurs, the user can instruct
  // the algoirth to decrease th_low
  
  std::cout<<"FOREGROUND/BACKGROUND DETECTION\n";
  int x, y, z, i, flag;
  ColumnMatrix vec;
  float e, stepsize, w, mean[2], prior[2], var[2], temp, th_low, th_high, min=0.0, max=0.0;
  float *pdfest;
   
  for (z=0;z<m_depth;z++)
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
      {
        e=0.0;
        vec=get_m_Data(x,y,z);
        for (i=0;i<3;i++) e+=vec.get(i);
        set_nrrd_float(m_Energy, e, x, y, z);
        if (!(x||y||z)) min=max=e;
        else
	{
          if (e>max) max=e;
          else if (e<min) min=e;
	}
      }
  w = 1.0/((float)(m_width*m_height*m_depth));
  std::cout<<"Energy min = "<<min<<" , max = "<<max<<std::endl;
  // compute 1d histogram
  stepsize = ceil((max-min)/(float)NBINMAX);
  int nbin = (int)rint((max-min)/stepsize);
  int   *hist      = (int*)calloc(nbin, sizeof(int));
  float *pdf       = (float*)calloc(nbin, sizeof(int));
  float *centers   = (float*)calloc(nbin, sizeof(float));
  float *upperlims = (float*)calloc(nbin-1, sizeof(float));
  std::cout<<"Using "<<nbin<<" bins for histogram\n";
  
  upperlims[0]=min+stepsize;
  centers[0]=min;
  for (i=1;i<nbin-1;i++)
    {
    upperlims[i]=upperlims[i-1]+stepsize;
    centers[i]=centers[i-1]+stepsize;
    }
  centers[nbin-1]=max-stepsize;
  Clear1DHistogram(hist,nbin);
  Accumulate1DHistogram (upperlims,hist,nbin,m_Energy,
                         0,0,0,m_width-1,m_height-1,m_depth-1);
  
  // initialize gaussian parameters for EM algorithm
  for (i=0;i<nbin;i++) pdf[i]=w*(float)hist[i];
  mean[0] = centers[0];       // mean for background
  mean[1] = centers[(int)ceil(0.5*(float)nbin)];   // mean for foreground is the middle bin
  temp  = (mean[1]-mean[0])/4.0;
  var[1] = temp*temp;
  temp/=15.0; 
  var[0]  = temp*temp;
  // the algorithm is relatively insensitive to initialization of variances
  prior[0]=prior[1]=0.5f;
 
  std::cout<<"Initialization for EM :\n";
  std::cout<<"Background energy mean = "<<mean[0]<<" , var = "<<var[0]<<std::endl;
  std::cout<<"Foreground energy mean = "<<mean[1]<<" , var = "<<var[1]<<std::endl;
  pdfest=EM1DWeighted (centers, pdf, nbin, 2, mean, prior, var);
  std::cout<<"After EM:\n";
  std::cout<<"Background energy mean = "<<mean[0]<<" , var = "<<var[0]<<std::endl;
  std::cout<<"Foreground energy mean = "<<mean[1]<<" , var = "<<var[1]<<std::endl;

  // compute foreground-background threshold
  th_low = Compute2ClassThreshold (pdfest,centers,nbin);
  th_high = (mean[0]+mean[1])/2.0;
  std::cout<<"Foreground/Background low  threshold = "<<th_low<<std::endl;
  std::cout<<"                      high threshold = "<<th_high<<std::endl;
  
  free (pdfest);
  free (hist);
  free (upperlims);
  free (pdf);
  free (centers);
  
  int xa, ya, k;
  NrrdDataHandle slice = create_nrrd_of_ints(m_width, m_height);
  int *slicebg = (int*)calloc(m_width*m_height,sizeof(int));
  int slicesize[2];
  slicesize[0]=m_width;
  slicesize[1]=m_height;

  m_TopOfHead = m_depth;
  int zlim = Max(m_EyeSlice,0);
  for (z=m_depth-1;z>=zlim;z--)
    {
    // this part of the volume (part of head above the eyes) should be very
    // robust, we only use the high threshold here
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if (get_nrrd_float(m_Energy,x,y,z)>=th_high) 
	  set_nrrd_int2(slice,1,x,y); 
	else 
	  set_nrrd_int2(slice,0,x,y);
    // floodfill background from all boundary pixels -- could also just use the
    // 4 corners
    for (xa=0;xa<m_width;xa++)
      {
      if (get_nrrd_int2(slice,xa,0)==0) 
	floodFill(slice,0,2,xa,0);      
      if (get_nrrd_int2(slice,xa,m_height-1)==0) 
	floodFill(slice,0,2,xa,m_height-1);
      }
    for (ya=0;ya<m_height;ya++)
      {
      if (get_nrrd_int2(slice,0,ya)==0) 
	floodFill(slice,0,2,0,ya);
      if (get_nrrd_int2(slice,m_width-1,ya)==0) 
	floodFill(slice,0,2,m_width-1,ya);
      }
    // mark the pixels in m_Labels image
    flag=0;
    k=0;
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if ((get_nrrd_int(m_Label,x,y,z)==T_BACKGROUND)&&(get_nrrd_int2(slice,x,y)!=2))
          {
          flag=1;
          set_nrrd_int(m_Label,T_FOREGROUND,x,y,z);
          slicebg[k++]=1;
          }
        else if (get_nrrd_int(m_Label,x,y,z)==T_BACKGROUND) slicebg[k++]=0;
        else slicebg[k++]=1;

    // compute 2D distance transform the background pixels
    if (!flag)
      {
      m_TopOfHead=z;
      for (y=0;y<m_height;y++)
        for (x=0;x<m_width;x++)
          set_nrrd_float(m_DistBG,0.0,x,y,z);
      }
    else 
      {
      distmap_4ssed(slicebg,slicesize);
      k=0;
      for (y=0;y<m_height;y++)
        for (x=0;x<m_width;x++)
          set_nrrd_float(m_DistBG,m_PixelDim*sqrt((float)slicebg[k++]),x,y,z);
      }
    } // z-loop
  std::cout<<"Top of head slice = "<<m_TopOfHead<<std::endl;

  // slices below the eyes have to be dealt with differently because the
  // background can leak into bone through the ears
  for (;z>=0;z--)
    {
    // we use the low threshold here to avoid leaking
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if (get_nrrd_float(m_Energy,x,y,z)>=th_high) set_nrrd_int2(slice,1,x,y);
        else if (get_nrrd_float(m_Energy,x,y,z)>=th_low) set_nrrd_int2(slice,3,x,y);
        else set_nrrd_int2(slice,0,x,y);

    for (ya=0;ya<m_height;ya++)
      for (xa=0;xa<m_width;xa++)
        if ((get_nrrd_int2(slice,xa,ya)==3)&&(get_nrrd_int(m_Label,xa,ya,z+1)!=T_BACKGROUND)&&(get_nrrd_float(m_DistBG,xa,ya,z+1)>10.0))
          floodFill(slice,3,1,xa,ya);
    // floodfill pixels between the low and high thresholds from seed
    // locations 10mm deep in the head, by doing this we avoid some of
    // the noise outside the head that we could pick up as foreground if
    // we simply thresholded with the low threshold
    
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if (get_nrrd_int2(slice,x,y)==3) set_nrrd_int2(slice,0,x,y); // these are pixels outside the
                                              // head that fall between the 2 thresholds
    
    // floodfill background from all boundary pixels -- could also just use the
    // 4 corners
    for (xa=0;xa<m_width;xa++)
    {
      if (get_nrrd_int2(slice,xa,0)==0) 
	floodFill(slice,0,2,xa,0);
      if (get_nrrd_int2(slice,xa,m_height-1)==0) 
	floodFill(slice,0,2,xa,m_height-1);
    }
    for (ya=0;ya<m_height;ya++)
    {
      if (get_nrrd_int2(slice,0,ya)==0) 
	floodFill(slice,0,2,0,ya);
      if (get_nrrd_int2(slice,m_width-1,ya)==0) 
	floodFill(slice,0,2,m_width-1,ya);
    }
    k=0;
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if ((get_nrrd_int(m_Label,x,y,z)==T_BACKGROUND)&&(get_nrrd_int2(slice,x,y)!=2))
          {
	    set_nrrd_int(m_Label,T_FOREGROUND,x,y,z);
	    slicebg[k++]=1;
          }
        else if (get_nrrd_int(m_Label,x,y,z)==T_BACKGROUND) slicebg[k++]=0;
        else slicebg[k++]=1;

    // compute 2D distance transform the background pixels
    distmap_4ssed(slicebg,slicesize);
    k=0;
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        set_nrrd_float(m_DistBG,m_PixelDim*sqrt((float)slicebg[k++]),x,y,z);
    } // z-loop
  
  free(slicebg);

  // m_DistBG2D = m_DistBG; // save 2D distance transform
  m_DistBG2D = m_DistBG->clone();

  // fix 2D distance to approximate 3D distance  
  for (x=0;x<m_width;x++)
    for (y=0;y<m_height;y++)
      if (get_nrrd_float(m_DistBG,x,y,m_TopOfHead-1)>0.0)
	set_nrrd_float(m_DistBG,m_SliceThickness,x,y,m_TopOfHead-1);

  for (z=m_TopOfHead-2;z>=0;z--)
  {
    for (x=0;x<m_width;x++)
      for (y=0;y<m_height;y++)
        if (get_nrrd_float(m_DistBG,x,y,z)>0.0) {
	  const float val = Min (get_nrrd_float(m_DistBG,x,y,z),get_nrrd_float(m_DistBG,x,y,z+1)+m_SliceThickness);
	  set_nrrd_float(m_DistBG,val,x,y,z);
	}
  }
}

void
MRITissueClassifier::ComputeForegroundCenter ()
{
  // simple function to compute center of mass locations for foreground pixels
  // in each axial slice -- and a single global center as the median of these
  int x, y, z, k, k2;
  m_slicecenter[0]=(float*)calloc(m_depth,sizeof(float));
  m_slicecenter[1]=(float*)calloc(m_depth,sizeof(float));

  k2=0;
  for (z=0;z<m_TopOfHead;z++)
    {
    m_slicecenter[0][z]=0.0f;
    m_slicecenter[1][z]=0.0f;
    k=0;
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if (get_nrrd_int(m_Label,x,y,z)==T_FOREGROUND)
          {
          k++;
          m_slicecenter[0][z]+=(float)x;
          m_slicecenter[1][z]+=(float)y;
          }
    if (k>0)
      {
      m_slicecenter[0][z]/=(float)k;
      m_slicecenter[1][z]/=(float)k;
      }
    else
      {
      m_slicecenter[0][z]=-1.0f;
      m_slicecenter[1][z]=-1.0f;
      k2++;
      }
    }
  for (z=m_TopOfHead;z<m_depth;z++)
    {
    m_slicecenter[0][z]=-1.0f;
    m_slicecenter[1][z]=-1.0f;
    k2++;
    }

  float median[2];
  float *meanx = (float*)calloc(m_TopOfHead-k2,sizeof(float));
  float *meany = (float*)calloc(m_TopOfHead-k2,sizeof(float));

  k=0;
  for (z=0;z<m_TopOfHead;z++)
    if (m_slicecenter[0][z]>=0.0)
      {
      meanx[k]=m_slicecenter[0][z];
      meany[k++]=m_slicecenter[1][z];
      }

  bubble_sort(meanx,k);
  bubble_sort(meany,k);
  k=(int)rint(0.5*(float)k);
  median[0]=m_slicecenter[0][k];
  median[1]=m_slicecenter[1][k];
  m_center[0]=median[0];
  m_center[1]=median[1];

  free(meanx);
  free(meany);
}

void
MRITissueClassifier::FatDetection_NOFATSAT (float maskr)
{
  // this function serves 2 purposes: (i) to compute a threshold for fat using
  // the T1 signal, this threshold is later used to initialize the scalp
  // classification routine
  // (ii) use the threshold here to do a pre-labelling in certain slices to
  // aid in intra-extracranial separation -- this is needed in slices below the
  // eyes because the skull no longer forms a closed surface around the brain
  // and floodfill methods can leak -- we supplement the skull barrier with a
  // fat barrier in these regions and this seems to solve the proble,

  // I think this is the part of the algorithm most prone to failing
  // If it does fail, the user can tell the algorithm to lower the fat
  // threshold to increase the chances of succesfully blocking any leakage

  // Furthermore, in the absence of a fatsat sequence we use the T1 signal (in
  // which fat appears bright) but this is not totally reliable so we have to
  // do some morphology to clean things up -- this is not done in the other
  // FatDetection function which uses a T1 and a T1Fatsat sequence
  
  std::cout<<"FAT DETECTION (without fatsat)\n";
  int x, y, z, cnt, i;
  float min=0.0, max=0.0,  temp, mean[4], var[4], prior[4], *pdfest, th, fat_energy;
  ColumnMatrix data;

  cnt = 0;
  for (z=0;z<m_TopOfHead;z++)
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
      {
        fat_energy = get_nrrd_float(m_T1_Data, x, y, z);
        if (get_nrrd_int(m_Label,x,y,z)==T_FOREGROUND)
	{
          if (cnt==0) min=max=fat_energy;
          else
	  {
            if (fat_energy>max) max=fat_energy;
            else if (fat_energy<min) min=fat_energy;
	  } 
          cnt++;
	}
      }
  
  std::cout<<"T1 min = "<<min<<" , max = "<<max<<std::endl;
  // compute bins for the histogram
  float stepsize = ceil((max-min)/(float)NBINMAX);
  int nbin = (int)rint((max-min)/stepsize);
  std::cout<<"Using "<<nbin<<" bins for histogram\n";
  int   *hist      = (int*)calloc(nbin, sizeof(int));
  float *pdf       = (float*)calloc(nbin, sizeof(int));
  float *centers   = (float*)calloc(nbin, sizeof(float));
  float *upperlims = (float*)calloc(nbin-1, sizeof(float));
  float *cdf       = (float*)calloc(nbin, sizeof(int));
  
  upperlims[0]=min+stepsize;
  centers[0]=min;
  for (i=1;i<nbin-1;i++)
    {
    upperlims[i]=upperlims[i-1]+stepsize;
    centers[i]=centers[i-1]+stepsize;
    }
  centers[nbin-1]=max-stepsize;
  Clear1DHistogram(hist,nbin);
  Accumulate1DHistogramWithLabel (upperlims,hist,nbin,m_T1_Data,
                                  0,0,0,
				  m_width-1,m_height-1,m_TopOfHead-1,
                                  T_FOREGROUND); 
  
  // initialize gaussian parameters for EM algorithm -- algorithm is relatively
  // insensitive to this initialization
  for (i=0;i<nbin;i++) pdf[i]=(float)hist[i]/(float)cnt;
  cdf[0] = pdf[0];
  for (i=1;i<nbin;i++) cdf[i] = cdf[i-1] + pdf[i];

  for (i=0;i<nbin;i++) if (cdf[i]>=0.1) break;
  mean[0] = centers[i];
  for (;i<nbin;i++) if (cdf[i]>=0.3) break;
  mean[1] = centers[i];
  for (;i<nbin;i++) if (cdf[i]>=0.6) break;
  mean[2] = centers[i];
  for (;i<nbin;i++) if (cdf[i]>=0.9) break;
  mean[3] = centers[Min(i,nbin-1)];           // mean for fat
  prior[0]= 0.2;
  prior[1]= 0.2;
  prior[2]= 0.4;
  prior[3]= 0.2; 
  temp  = (mean[1]-mean[0])/10.0;
  var[0] = temp*temp;
  var[1] = var[0];
  var[2] = var[0];
  var[3] = var[0];
 
  std::cout<<"Initialization for EM:"<<std::endl;
  std::cout<<"non-fat1 mean = "<<mean[0]<<" variance = "<<var[0]<<" prior = "<<prior[0]<<std::endl;
  std::cout<<"non-fat2 mean = "<<mean[1]<<" variance = "<<var[1]<<" prior = "<<prior[1]<<std::endl;
  std::cout<<"non-fat3 mean = "<<mean[2]<<" variance = "<<var[2]<<" prior = "<<prior[2]<<std::endl;
  std::cout<<"fat      mean = "<<mean[3]<<" variance = "<<var[3]<<" prior = "<<prior[3]<<std::endl;
  pdfest=EM1DWeighted (centers, pdf, nbin, 4, mean, prior, var);
  std::cout<<"After EM:"<<std::endl;
  std::cout<<"non-fat1 mean = "<<mean[0]<<" variance = "<<var[0]<<" prior = "<<prior[0]<<std::endl;
  std::cout<<"non-fat2 mean = "<<mean[1]<<" variance = "<<var[1]<<" prior = "<<prior[1]<<std::endl;
  std::cout<<"non-fat3 mean = "<<mean[2]<<" variance = "<<var[2]<<" prior = "<<prior[2]<<std::endl;
  std::cout<<"fat      mean = "<<mean[3]<<" variance = "<<var[3]<<" prior = "<<prior[3]<<std::endl;
  
  float diffl, diffh, pos;
  for (i=nbin-1;i>=0;i--) if (pdfest[i*4+3]<pdfest[i*4+2]) break;
  if (i<=0) th = centers[0];
  else
    {
    if (i>=(nbin-1)) th = centers[nbin-1];
    diffl=pdfest[i*4+3]-pdfest[i*4+2];
    diffh=pdfest[(i+1)*4+3]-pdfest[(i+1)*4+2];
    pos = diffl/(diffl-diffh);
    th = centers[i] + pos * stepsize;
    }
  
  // free memory not needed any longer
  free (pdfest);
  free (hist);
  free (upperlims);
  free (pdf);
  free (centers);
  free (cdf);
  
  std::cout<<"Fat threshold = "<<th<<std::endl;
  m_FatThreshold = th;

  // labeling is done here in certain slices to aid in intra-extra cranial
  // separation, actual fat detection is done in scalp classification

  if (m_TopOfEyeSlice>=0)
    {
    int depth = 1 + m_TopOfEyeSlice + (int)ceil(6.0/m_SliceThickness);
    int ycenter = (int)rint(m_center[1]);
    
    for (z=0;z<depth;z++)
      for (y=ycenter;y<m_height;y++)
        for (x=0;x<m_width;x++)
          if (get_nrrd_int(m_Label,x,y,z)==T_FOREGROUND)
	  {
            fat_energy = get_nrrd_float(m_T1_Data, x, y, z);
            if (fat_energy>th)
              set_nrrd_int(m_Label, T_FAT, x,y,z);
	  }
    
    int rad = (int)floor(maskr);
    int len = 2*rad+1;
    
    float xo, yo;
    NrrdDataHandle tempI = create_nrrd_of_ints(m_width, m_height-ycenter);;
    NrrdDataHandle maskI = create_nrrd_of_ints(len, len);


    for (y=0;y<len;y++)
      for (x=0;x<len;x++)
        {
        xo = (float)( x - rad );
        yo = (float)( y - rad );
        if ( sqrt(xo*xo + yo*yo) <=maskr )
	  set_nrrd_int2(maskI,1,x,y);
        else
	  set_nrrd_int2(maskI,0,x,y);
        }

    for (z=0;z<depth;z++)
    {
      for (y=ycenter;y<m_height;y++)
        for (x=0;x<m_width;x++)
	{
	  const int label = get_nrrd_int(m_Label,x,y,z);
          if (label == T_FAT || label == T_EYE || label ==T_EYE_LENS)
            set_nrrd_int2(tempI,1,x,y-ycenter); 
          else
	    set_nrrd_int2(tempI,0,x,y-ycenter);
	}

      // we need to clean up noise because T1 signal by itself is not very
      // reliable for fat detection
      tempI = closing(tempI, maskI); // close holes
      tempI = opening(tempI, maskI); // remove outliers
      tempI = dilate (tempI,maskI);  // increase fat volume just to make sure
			    // brain doesn't leak out 
        
      for (y=ycenter;y<m_height;y++)
        for (x=0;x<m_width;x++)
	{
	  const int label = get_nrrd_int(m_Label, x, y, z);
          if ((label == T_FAT) && (get_nrrd_int2(tempI,x,y-ycenter) == 0))
	    set_nrrd_int(m_Label, T_FOREGROUND, x,y,z);
          else if ((get_nrrd_int2(tempI,x,y-ycenter) == 1) && 
		   (label == T_FOREGROUND)) 
	    set_nrrd_int(m_Label, T_FAT, x,y,z);
	}
      }
    }
}

void
MRITissueClassifier::FatDetection_FATSAT ()
{
  // this function serves 2 purposes: (i) to compute a threshold for fat using
  // the T1 signal, this threshold is later used to initialize the scalp
  // classification routine
  // (ii) use the threshold here to do a pre-labelling in certain slices to
  // aid in intra-extracranial separation -- this is needed in slices below the
  // eyes because the skull no longer forms a closed surface around the brain
  // and floodfill methods can leak -- we supplement the skull barrier with a
  // fat barrier in these regions and this seems to solve the proble,

  // I think this is the part of the algorithm most prone to failing
  // If it does fail, the user can tell the algorithm to lower the fat
  // threshold to increase the chances of succesfully blocking any leakage

  std::cout<<"FAT DETECTION (with fatsat)\n";
  NrrdDataHandle m_Fat_Energy = create_nrrd_of_floats(m_width, 
						      m_height, 
						      m_depth);

  int x, y, z, cnt, i;
  float min=0.0, max=0.0,  temp, mean[2], var[2], prior[2], *pdfest, th;
  ColumnMatrix data;

  cnt = 0;
  for (z=0;z<m_TopOfHead;z++)
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
      {
	const float energy = (get_nrrd_float(m_T1_Data,x,y,z) -
			      get_nrrd_float(m_FATSAT_Data,x,y,z));
	set_nrrd_float(m_Fat_Energy, energy, x,y,z);
        if (get_nrrd_int(m_Label,x,y,z)==T_FOREGROUND)
	{
	  if (cnt==0) min=max=energy;
          else
	  {	   
            if (energy>max) max=energy;
            else if (energy<min) min=energy;
	  } 
          cnt++;
	}
      }

  std::cout<<"Fat energy min = "<<min<<" , max = "<<max<<std::endl;
  // compute bins for the histogram
  float stepsize = ceil((max-min)/(float)NBINMAX);
  int nbin = (int)rint((max-min)/stepsize);
  std::cout<<"Using "<<nbin<<" bins for histogram\n";
  int   *hist      = (int*)calloc(nbin, sizeof(int));
  float *pdf       = (float*)calloc(nbin, sizeof(int));
  float *centers   = (float*)calloc(nbin, sizeof(float));
  float *upperlims = (float*)calloc(nbin-1, sizeof(float));
  float *cdf       = (float*)calloc(nbin, sizeof(int));
  
  upperlims[0]=min+stepsize;
  centers[0]=min;
  for (i=1;i<nbin-1;i++)
    {
    upperlims[i]=upperlims[i-1]+stepsize;
    centers[i]=centers[i-1]+stepsize;
    }
  centers[nbin-1]=max-stepsize;
  Clear1DHistogram(hist,nbin);
  Accumulate1DHistogramWithLabel (upperlims,hist,nbin,m_Fat_Energy,
                                  0,0,0,
				  m_width-1,m_height-1,m_TopOfHead-1,
				  T_FOREGROUND);
  
  // initialize gaussian parameters for EM algorithm
  for (i=0;i<nbin;i++) pdf[i]=(float)hist[i]/(float)cnt;
  cdf[0] = pdf[0];
  for (i=1;i<nbin;i++) cdf[i] = cdf[i-1] + pdf[i];

  for (i=0;i<nbin;i++) if (cdf[i]>=0.4) break;
  mean[0] = centers[i];                          
  for (;i<nbin;i++) if (cdf[i]>=0.9) break;
  mean[1] = centers[Min(i,nbin-1)];           // mean for fat
  prior[0]= 0.8; 
  prior[1]= 0.2; // prior for fat
  temp  = (mean[1]-mean[0])/10.0;
  var[0] = temp*temp;  // bone/air has the smallest variance
  var[1] = var[0]; // muscle has the largest
 
  std::cout<<"Initialization for EM:"<<std::endl;
  std::cout<<"non-fat    mean = "<<mean[0]<<" variance = "<<var[0]<<" prior = "<<prior[0]<<std::endl;
  std::cout<<"fat      mean = "<<mean[1]<<" variance = "<<var[1]<<" prior = "<<prior[1]<<std::endl;

  // Expectation-Maximization (EM) on the 1D histogram
  pdfest=EM1DWeighted (centers, pdf, nbin, 2, mean, prior, var);

  std::cout<<"After EM:"<<std::endl;
  std::cout<<"non-fat    mean = "<<mean[0]<<" variance = "<<var[0]<<" prior = "<<prior[0]<<std::endl;
  std::cout<<"fat      mean = "<<mean[1]<<" variance = "<<var[1]<<" prior = "<<prior[1]<<std::endl;

  //th = mean[1];//+2.0*sqrt(var[1]);
  th = Compute2ClassThreshold (pdfest,centers,nbin);
  m_FatThreshold = th;
  
  // free memory not needed any longer
  free (pdfest);
  free (hist);
  free (upperlims);
  free (pdf);
  free (centers);
  free (cdf);
  
  std::cout<<"Fat threshold = "<<th<<std::endl;

  if (m_TopOfEyeSlice>=0)
  {
    int depth = 1 + m_TopOfEyeSlice + (int)ceil(6.0/m_SliceThickness);
    for (z=0;z<depth;z++)
      for (y=0;y<m_height;y++)
        for (x=0;x<m_width;x++)
          if ((get_nrrd_int(m_Label,x,y,z)==T_FOREGROUND)&&(get_nrrd_float(m_Fat_Energy,x,y,z)>th))
	    set_nrrd_int(m_Label, T_FAT, x,y,z);
    }
}

void
MRITissueClassifier::BrainDetection (float maskr, float maskr2)
{
  // this function separates brain from non-brain by using a 3 class clustering
  // on the T1+T2+PD signal. Brain and skin are brightest in this signal but
  // are separated by skull only in slices higher than the eyes and by skull
  // and fat in other slices. So we do a floodfill from the area around the
  // foreground center using the threshold and the skull/fat barriers. The
  // fat barrier could fail (see comments in fat detection functions)
  
  // maskr size of mask to do opening closing to remove noise
  // maskr2 parameter should be related to the size of the sinus cavities 
  
  std::cout<<"INTRA/EXTRA-CRANIAL SEPARATION"<<std::endl;
  // compute new histogram without background pixels
  int x, y, z, i, j, cnt;
  float min=0.0, max=0.0, temp, mean[3], var[3], prior[3];
  float *pdfest;

  // find min-max of m_Energy
  cnt = 0;
  for (z=0;z<m_TopOfHead;z++)
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if (get_nrrd_int(m_Label,x,y,z)==T_FOREGROUND)
	{
	  const float energy = get_nrrd_float(m_Energy, x,y,z);
          if (cnt==0) min=max=energy;
          else
	  {
            if (energy>max) max=energy;
            else if (energy<min) min=energy;
	  }
          cnt++;
	}

  std::cout<<"Energy min = "<<min<<" , max = "<<max<<std::endl;
  // compute bins for the histogram
  float stepsize = ceil((max-min)/(float)NBINMAX);
  int nbin = (int)rint((max-min)/stepsize);
  std::cout<<"Using "<<nbin<<" bins for histogram\n";
  int   *hist      = (int*)calloc(nbin, sizeof(int));
  float *pdf       = (float*)calloc(nbin, sizeof(int));
  float *centers   = (float*)calloc(nbin, sizeof(float));
  float *upperlims = (float*)calloc(nbin-1, sizeof(float));
  float *cdf       = (float*)calloc(nbin, sizeof(int));
  
  upperlims[0]=min+stepsize;
  centers[0]=min;
  for (i=1;i<nbin-1;i++)
    {
    upperlims[i]=upperlims[i-1]+stepsize;
    centers[i]=centers[i-1]+stepsize;
    }
  centers[nbin-1]=max-stepsize;
  Clear1DHistogram(hist,nbin);
  Accumulate1DHistogramWithLabel (upperlims,hist,nbin,m_Energy,
                                  0,0,0,
				  m_width-1,m_height-1,m_TopOfHead-1,
				  T_FOREGROUND);

  // initialize gaussian parameters for EM algorithm
  for (i=0;i<nbin;i++) pdf[i]=(float)hist[i]/(float)cnt;
  cdf[0] = pdf[0];
  for (i=1;i<nbin;i++) cdf[i] = cdf[i-1] + pdf[i];
  mean[0] = centers[0];                          // mean for bone and air cavities
  for (i=1;i<nbin;i++) if (cdf[i]>=0.2) break;
  mean[1] = centers[i];                          // mean for muscle
  for (;i<nbin;i++) if (cdf[i]>=0.65) break;
  mean[2] = centers[Min(i,nbin-1)];           // mean for brain matter/fat/skin
  prior[0]= 0.1; // prior for bone/air
  prior[1]= 0.2; // prior for muscle
  prior[2]= 0.7; // prior for brain matter 
  temp  = 0.5*centers[(int)rint(0.05*(float)nbin)];
  var[0] = temp*temp;  // bone/air has the smallest variance
  var[1] = 9.0*var[0]; // muscle has the largest
  var[2] = 6.0*var[0]; 
 
  std::cout<<"Initialization for EM:"<<std::endl;
  std::cout<<"bone/air    mean = "<<mean[0]<<" variance = "<<var[0]<<" prior = "<<prior[0]<<std::endl;
  std::cout<<"muscle      mean = "<<mean[1]<<" variance = "<<var[1]<<" prior = "<<prior[1]<<std::endl;
  std::cout<<"brain/scalp mean = "<<mean[2]<<" variance = "<<var[2]<<" prior = "<<prior[2]<<std::endl;
  pdfest=EM1DWeighted (centers, pdf, nbin, 3, mean, prior, var);
  std::cout<<"After EM:"<<std::endl;
  std::cout<<"bone/air    mean = "<<mean[0]<<" variance = "<<var[0]<<" prior = "<<prior[0]<<std::endl;
  std::cout<<"muscle      mean = "<<mean[1]<<" variance = "<<var[1]<<" prior = "<<prior[1]<<std::endl;
  std::cout<<"brain/scalp mean = "<<mean[2]<<" variance = "<<var[2]<<" prior = "<<prior[2]<<std::endl;

  // compute threshold for intra/extra-cranial separation
  for (i=nbin-1;i>=0;i--) if (pdfest[i*3+1]<pdfest[i*3+2]) break;
  for (;i>=0;i--) if (pdfest[i*3+1]>=pdfest[i*3+2]) break;
  
  float diffl, diffh, pos, threshold;
  if (i<=0) threshold = centers[0];
  else
    {
    if (i>=(nbin-1)) threshold = centers[nbin-1];
    diffl=pdfest[i*3+2]-pdfest[i*3+1];
    diffh=pdfest[(i+1)*3+2]-pdfest[(i+1)*3+1];
    pos = diffl/(diffl-diffh);
    threshold = centers[i] + pos * stepsize;
    }
  std::cout<<"Intra/Extra-cranial separation threshold = "<<threshold<<"\n";

  // free memory not needed any longer
  free (pdfest);
  free (hist);
  free (upperlims);
  free (pdf);
  free (centers);
  free (cdf);

  // prepare noise removal mask
  int rad = (int)ceil(maskr);
  int len = 2*rad+1;
  float xo, yo;
  NrrdDataHandle tempI = create_nrrd_of_ints(m_width,m_height);
  NrrdDataHandle mask = create_nrrd_of_ints(len, len);

  for (x=0;x<len;x++)
    for (y=0;y<len;y++)
      {
      xo = (float)( x - rad );
      yo = (float)( y - rad );
      if ( sqrt(xo*xo + yo*yo) <=maskr ) set_nrrd_int2(mask,1,x,y);
      else set_nrrd_int2(mask,0,x,y);
      }

  // prepare sinus detection mask
  int rad2 = (int)ceil(maskr2);
  int len2 = 2*rad2+1;
  NrrdDataHandle mask2 = create_nrrd_of_ints(len2, len2);
  for (x=0;x<len2;x++)
    for (y=0;y<len2;y++)
      {
      xo = (float)( x - rad2 );
      yo = (float)( y - rad2 );
      if ( sqrt(xo*xo + yo*yo) <=maskr2 ) set_nrrd_int2(mask2,1,x,y);
      else set_nrrd_int2(mask2,0,x,y);
      }
  
  int yl=(int)rint(m_center[1])-10;
  int yh=yl+20;
  int xl=(int)rint(m_center[0])-10;
  int xh=xl+20;
  int xa, ya;
  int cnt2, cntp;
  int zlow = m_TopOfHead-(int)rint(15.0/m_SliceThickness);
  int ylow = (int)rint(Max(m_EyePosition[0][1],m_EyePosition[1][1])-10.0/m_PixelDim);
  int zt = m_TopOfEyeSlice+(int)ceil(9.0/m_SliceThickness);
  float r, rp;
  float distth = 15.0/m_PixelDim;  
  float dist, ndx, ndy, val, dl, dh;
  int flag, flag2, xp, yp, xn, yn, yl2, yh2, xl2, xh2;
  
  m_TopOfBrain = -1;
  // slices from eyes to top of head
  for (z=Max(m_TopOfEyeSlice,0);z<m_TopOfHead;z++)
  {
    for (x=0;x<m_width;x++)
      for (y=0;y<m_height;y++)
        if ((get_nrrd_int(m_Label,x,y,z)==T_FOREGROUND)&&
	    (get_nrrd_float(m_Energy,x,y,z)>threshold)) 
	  set_nrrd_int2(tempI,2,x,y);
	else
	  set_nrrd_int2(tempI,0,x,y);
    
    if (z<=zt)
    {
      // can't have any brain pixels within 15 mm of the background for these
      // slices , we do this to avoid leaking around the eyes
      for (x=0;x<m_width;x++)
        for (y=ylow;y<m_height;y++)
          if (get_nrrd_float(m_DistBG,x,y,z)<distth)
            set_nrrd_int2(tempI,0,x,y);
      }

    // floodfill from inside
    for (xa=xl;xa<=xh;xa++)
      for (ya=yl;ya<=yh;ya++)
        if (get_nrrd_int2(tempI,xa,ya)==2)
          floodFill(tempI,2,1,xa,ya);
    
    yl2 = m_height-1; yh2 = 0;
    for (x=0;x<m_width;x++)
      for (y=0;y<m_height;y++)
        if (get_nrrd_int2(tempI,x,y)==2) 
	  set_nrrd_int2(tempI,0,x,y);
        else if (get_nrrd_int2(tempI,x,y)==1)
	{
          if (y<yl2) yl2 = y;
          if (y>yh2) yh2 = y;
	}

    // remove holes in brain (low energy pixels enclosed in brain tissue such
    // as some sinuses)
    tempI = closing(tempI,mask);

    // find sinuses that are not totally enclosed in brain matter
    yl2 = Max(yl2-3,0);
    yh2 = Min(yh2+3,m_height-1);
    if (z<=zt) yh2 = (int)rint(m_slicecenter[1][z]);
    for (y=yl2;y<yh2;y++)
      {
      dist = Min(fabsf((float)y-m_slicecenter[1][z]),(float)15.0);
      xl2 = Max((int)floor(m_slicecenter[0][z]-dist),0);
      xh2 = Min((int)ceil(m_slicecenter[0][z]+dist),m_width-1);
      for (x=xl2;x<=xh2;x++)
        if (get_nrrd_int2(tempI,x,y)==0)
          {
          dist = get_nrrd_float(m_DistBG2D,x,y,z);
          ndx = (get_nrrd_float(m_DistBG2D,x+1,y,z) -
		 get_nrrd_float(m_DistBG2D,x-1,y,z));
          ndy = (get_nrrd_float(m_DistBG2D,x,y+1,z) -
		 get_nrrd_float(m_DistBG2D,x,y-1,z));
          val = sqrt(ndx*ndx+ndy*ndy);
          ndx /= val;
          ndy /= val;
          dl = dist - 1.5*m_PixelDim;
          dh = dist;
          flag = flag2 = 0;
          for (xp = -rad2; xp<=rad2; xp++)
            {
            for (yp = -rad2; yp<=rad2; yp++)
              if (((xp!=0)||(yp!=0))&&get_nrrd_int2(mask2,xp+rad2,yp+rad2))
                {
                xn = x + xp ;
                yn = y + yp;
                if ((xn>=0)&&(xn<m_width)&&(yn>=0)&&(yn<m_height)&&
                    (get_nrrd_float(m_DistBG2D,xn,yn,z)>=dl)&&
		    (get_nrrd_float(m_DistBG2D,xn,yn,z)<=dh))
                  {
                  if (((!flag)||(!flag2))&&(get_nrrd_int2(tempI,xn,yn)==1))
                    {
                    val = ndy*(float)xp - ndx*(float)yp;
                    if (val>0.0) flag = 1;
                    else if (val<0.0) flag2 = 1;
                    }
                  }
                }
            }
          if (flag&&flag2) set_nrrd_int2(tempI,1,x,y);
          }
      }

    floodFill(tempI,0,3,0,0);
    floodFill(tempI,0,3,m_width-1,0);
    floodFill(tempI,0,3,0,m_height-1);
    floodFill(tempI,0,3,m_width-1,m_height-1);

    for (x=0;x<m_width;x++)
      for (y=0;y<m_height;y++)
        if (get_nrrd_int2(tempI,x,y)==0) set_nrrd_int2(tempI,1,x,y);
        else if (get_nrrd_int2(tempI,x,y)==3) set_nrrd_int2(tempI,0,x,y);
    
    if (z>=zlow)
      {
      cnt = cnt2 = 0;
      for (x=0;x<m_width;x++)
        for (y=0;y<m_height;y++)
          if (get_nrrd_int(m_Label,x,y,z) == T_FOREGROUND)
            {
            if (get_nrrd_int2(tempI,x,y)==1) cnt++;
            cnt2++;
            }
      r = ((float)cnt)/((float)cnt2);
      if ((z>zlow)&&(cnt>cntp)&&(rp<0.01))
        {
        m_TopOfBrain = z-1;
        break;
        }
      rp = r;
      cntp = cnt;
      }

    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if ((get_nrrd_int(m_Label,x,y,z)==T_FOREGROUND)&&
	    (get_nrrd_int2(tempI,x,y)==1))
          set_nrrd_int(m_Label,T_INTRACRANIAL,x,y,z); // brain pixel
    }

  if (m_TopOfBrain == -1) m_TopOfBrain = m_TopOfHead;
  std::cout<<"Top of brain slice = "<<m_TopOfBrain<<"\n";

  // slices below eyes.. things are done slightly differently here
  zt = m_TopOfEyeSlice-(int)ceil(3.0/m_SliceThickness);
  if ((m_EyesVisible==1)&&(m_TopOfEyeSlice>0))
    {
    NrrdDataHandle tempV = create_nrrd_of_ints(m_width,
					       m_height,
					       m_TopOfEyeSlice);
    
    for (z=0;z<m_TopOfEyeSlice;z++)
      {
      for (x=0;x<m_width;x++)
        for (y=0;y<m_height;y++)
          if ((get_nrrd_int(m_Label,x,y,z)==T_FOREGROUND)&&(get_nrrd_float(m_Energy,x,y,z)>threshold))
	    set_nrrd_int2(tempI,1,x,y);
	  else
	    set_nrrd_int2(tempI,0,x,y);
      
      if (z>=zt)
        {
        for (x=0;x<m_width;x++)
          for (y=ylow;y<m_height;y++)
            set_nrrd_int2(tempI,0,x,y);
        }
      tempI = opening(tempI, mask);

      for (x=0;x<m_width;x++)
        for (y=0;y<m_height;y++)
          set_nrrd_int(tempV,get_nrrd_int2(tempI,x,y),x,y,z);
      }
    
    for (xa=xl;xa<=xh;xa++)
      for (ya=yl;ya<=yh;ya++)
        if (get_nrrd_int(tempV,xa,ya,m_TopOfEyeSlice-1)==1)
          floodFill(tempV,1,2,xa,ya,m_TopOfEyeSlice-1);
    
    for (z=0;z<m_TopOfEyeSlice;z++)
    {
      yl2 = m_height-1;
      for (x=0;x<m_width;x++)
        for (y=0;y<m_height;y++)
          if (get_nrrd_int(tempV,x,y,z)==2)
	  {
            set_nrrd_int2(tempI,1,x,y);
            if (y<yl2) yl2 = y;
	  }
	  else set_nrrd_int2(tempI,0,x,y);
      tempI = closing(tempI, mask);
      
      if (z>=m_BottomOfEyeSlice)
      {
        yl2 = Max(yl2-3,0);
        yh2 = (int)rint(m_slicecenter[1][z]);
        for (y=yl2;y<yh2;y++)
	{
          dist = Min(fabsf((float)y-m_slicecenter[1][z]),(float)15.0);
          xl2 = Max((int)floor(m_slicecenter[0][z]-dist),0);
          xh2 = Min((int)ceil(m_slicecenter[0][z]+dist),m_width-1);
          for (x=xl2;x<=xh2;x++)
            if (get_nrrd_int2(tempI,x,y)==0)
              {
              dist = get_nrrd_float(m_DistBG2D,x,y,z);
              ndx = get_nrrd_float(m_DistBG2D,x+1,y,z)-get_nrrd_float(m_DistBG2D,x-1,y,z);
              ndy = get_nrrd_float(m_DistBG2D,x,y+1,z)-get_nrrd_float(m_DistBG2D,x,y-1,z);
              val = sqrt(ndx*ndx+ndy*ndy);
              ndx /= val;
              ndy /= val;
              dl = dist - 1.5*m_PixelDim;
              dh = dist;
              flag = flag2 = 0;
              for (xp = -rad2; xp<=rad2; xp++)
                {
                for (yp = -rad2; yp<=rad2; yp++)
                  if (((xp!=0)||(yp!=0))&&get_nrrd_int2(mask2,xp+rad2,yp+rad2))
                    {
                    xn = x + xp ;
                    yn = y + yp;
                    if ((xn>=0)&&(xn<m_width)&&(yn>=0)&&(yn<m_height)&&
                        (get_nrrd_float(m_DistBG2D,xn,yn,z)>=dl)&&(get_nrrd_float(m_DistBG2D,xn,yn,z)<=dh))
                      {
                      if (((!flag)||(!flag2))&&(get_nrrd_int2(tempI,xn,yn)==1))
                        {
                        val = ndy*(float)xp - ndx*(float)yp;
                        if (val>0.0) flag = 1;
                        else if (val<0.0) flag2 = 1;
                        }
                      }
                    }
                }
              if (flag&&flag2) set_nrrd_int2(tempI,1,x,y);
              }
          }
        }
      floodFill(tempI,0,3,0,0);
      floodFill(tempI,0,3,m_width-1,0);
      floodFill(tempI,0,3,0,m_height-1);
      floodFill(tempI,0,3,m_width-1,m_height-1);
      
      for (x=0;x<m_width;x++)
        for (y=0;y<m_height;y++)
          if (get_nrrd_int2(tempI,x,y)==0) set_nrrd_int2(tempI,1,x,y);
          else if (get_nrrd_int2(tempI,x,y)==3) set_nrrd_int2(tempI,0,x,y);
      
      for (y=0;y<m_height;y++)
        for (x=0;x<m_width;x++)
          if ((get_nrrd_int(m_Label,x,y,z)==T_FOREGROUND)&&(get_nrrd_int2(tempI,x,y)==1))
            set_nrrd_int(m_Label,T_INTRACRANIAL,x,y,z); // brain pixel
      }
    }

  int d = 1 + m_TopOfEyeSlice + (int)ceil(6.0/m_SliceThickness);
  
  for (z=0;z<d;z++)
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if (get_nrrd_int(m_Label,x,y,z)==T_FAT)
          set_nrrd_int(m_Label,T_FOREGROUND,x,y,z); // remove any previous fat markings
                                            // in these slices -- they were
                                            // only needed for the brain floodfills

  if (m_EyesVisible) FindTemporalSlice();
  else m_TopTemporalSlice = -1;

  int N[8][2];
  N[0][0] = 1;N[0][1] = 0;
  N[1][0] = -1;N[1][1] = 0;
  N[2][0] = 0;N[2][1] = 1;
  N[3][0] = 0;N[3][1] = -1;
  N[4][0] = 1;N[4][1] = 1;
  N[5][0] = 1;N[5][1] = -1;
  N[6][0] = -1;N[6][1] = 1;
  N[7][0] = -1;N[7][1] = -1;

  list<pair<int, int> > ilist, rlist;

  // connected component analysis to fill bigger holes in brain if any
  unsigned int at_x, at_y, next_x, next_y;
  int ymin, yc;
  for (z=0;z<m_depth;z++)
  {
    tempI = extract_nrrd_slice_int(m_Label, z);
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if (get_nrrd_int2(tempI,x,y) == T_FOREGROUND)
	{
	  ilist.clear();
	  rlist.clear();

	  ilist.push_back(make_pair(x,y));
	  rlist.push_back(make_pair(x,y));
	  	    
          set_nrrd_int2(tempI,255,x,y);
	  
          flag=0;
          for (j=0;j<8;j++)
	  {
            xa = x + N[j][0];
            ya = y + N[j][0];
            if ((nrrd_check_bounds(tempI,xa,ya))&&
		(get_nrrd_int(m_Label,xa,ya,z)==T_BACKGROUND))
	    {
              flag=1;
              break;
	    }
	  }
          while (!ilist.empty())
	  {
	    at_x = ilist.front().first;
            at_y = ilist.front().second;
    
            for (i = 0; i < 4; i++)
	    {
              next_x = at_x + N[i][0];
              next_y = at_y + N[i][1];
              if (nrrd_check_bounds(tempI,next_x, next_y))
	      {
                if (get_nrrd_int2(tempI,next_x, next_y) == T_FOREGROUND)
		{
                  set_nrrd_int2(tempI,255,next_x, next_y);
		  ilist.push_back(make_pair(next_x, next_y));
		  rlist.push_back(make_pair(next_x, next_y));
                  if (!flag)
                    for (j=0;j<8;j++)
		    {
                      xa = next_x + N[j][0];
                      ya = next_y + N[j][0];
                      if ((nrrd_check_bounds(tempI,xa,ya))&&
			  (get_nrrd_int(m_Label,xa,ya,z)==T_BACKGROUND))
		      {
                        flag=1;
                        break;
		      }
		    }
		}
	      }
	    }
	    ilist.pop_front();
	  }

          if (!flag)
	  {
            while (!rlist.empty())
	    {
              at_x = rlist.front().first;
              at_y = rlist.front().second;
              set_nrrd_int(m_Label,T_INTRACRANIAL,at_x,at_y,z);
	      rlist.pop_front();
	    }
	  }
	}
    
    if (z<=m_TopTemporalSlice)
    {
      ymin = m_height;
      for (y=0;y<m_height;y++)
        for (x=0;x<m_width;x++)
          if ((get_nrrd_int(m_Label,x,y,z)==T_INTRACRANIAL)&&(y<ymin)) ymin = y;
      yc = ymin + (int)rint(40.0/m_PixelDim);
    }

    // separate sinus vs. brain
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
      {
        if ((get_nrrd_int(m_Label,x,y,z)==T_INTRACRANIAL)&&(get_nrrd_float(m_Energy,x,y,z)<threshold))
	{
          if (z>m_TopTemporalSlice) 
	    set_nrrd_int(m_Label,T_SINUS,x,y,z);
          else
	  {
            if (y<yc)
	      set_nrrd_int(m_Label,T_SINUS,x,y,z);
            else
	      set_nrrd_int(m_Label,T_FOREGROUND,x,y,z);
	  }
	}
      }
  }
  FindTemporalLobes(); 
}

void
MRITissueClassifier::FindTemporalSlice ()
{
  // find the slice where temporal lobes begin
  // this is used in later functions to break up the volume into parts that are
  // treated differently
  int x, y, z, xcl, xcr, maxl, maxr, maxc;

  int x1l = (int)floor(m_EyePosition[0][0]-10.0/m_PixelDim);
  int x1r = (int)ceil(m_EyePosition[0][0]+10.0/m_PixelDim);
  int x2l = (int)floor(m_EyePosition[1][0]-10.0/m_PixelDim);
  int x2r = (int)ceil(m_EyePosition[1][0]+10.0/m_PixelDim);
  int zt = m_TopOfEyeSlice + (int)rint(20.0/m_SliceThickness);
  m_brainlim = (int*)calloc(m_depth,sizeof(int));
  
  m_TopTemporalSlice = -1;
  
  for (z=zt;z>=0;z--)
    {
    xcl = (int)rint(m_slicecenter[0][z]-5.0/m_PixelDim);
    xcr = (int)rint(m_slicecenter[0][z]+5.0/m_PixelDim);
    
    maxl = maxr = maxc = 0;

    for (x = xcl; x<=xcr; x++)
      {
      for (y=m_height;y>=0;y--)
        if ((get_nrrd_int(m_Label,x,y,z)==T_INTRACRANIAL)||(get_nrrd_int(m_Label,x,y,z)==T_SINUS))
          break;
      if (y>maxc) maxc=y;
      }
    m_brainlim[z]=maxc;
    if (z<=m_TopOfEyeSlice)
      {
      for (x = x1l; x<=x1r; x++)
        {
        for (y=m_height;y>=0;y--)
          if ((get_nrrd_int(m_Label,x,y,z)==T_INTRACRANIAL)||(get_nrrd_int(m_Label,x,y,z)==T_SINUS))
            break;
        if (y>maxl) maxl=y;
        }
      for (x = x2l; x<=x2r; x++)
        {
        for (y=m_height;y>=0;y--)
          if ((get_nrrd_int(m_Label,x,y,z)==T_INTRACRANIAL)||(get_nrrd_int(m_Label,x,y,z)==T_SINUS))
            break;
        if (y>maxr) maxr=y;
        }
      
      if ((m_TopTemporalSlice==-1)&&(Min(maxl,maxr)>maxc))
        m_TopTemporalSlice = z;
      }
    }
}

void
MRITissueClassifier::FindTemporalLobes ()
{
  // this function removes leakage from temporal lobes into non-brain pixels
  int x, y, z, m, i, *pcnt, cnt, flag, maxy, max;
  int zt = m_TopTemporalSlice - (int)rint(21.0/m_SliceThickness);
  int zb = m_TopTemporalSlice - (int)rint(28.0/m_SliceThickness);
 
  if (zt>0) 
    {
    NrrdDataHandle slice;

    for (z=zt;z>=0;z--)
    {
      flag = 0;
      slice = extract_nrrd_slice_int(m_Label, z);
      for (x=0;x<m_width;x++)
        for (y=0;y<m_height;y++)
          if ((get_nrrd_int2(slice,x,y)==T_INTRACRANIAL)||(get_nrrd_int2(slice,x,y)==T_SINUS))
	  {
            set_nrrd_int2(slice,1,x,y);
            flag = 1;
	  }
          else 
	    set_nrrd_int2(slice,0,x,y);

      if (flag)
      {
        cnt = 2;
        while (flag)
	{
          flag = 0;
          for (x=0;x<m_width;x++)
            for (y=0;y<m_height;y++)
              if (get_nrrd_int2(slice,x,y)==1)
	      {
                floodFill(slice,1,cnt++,x,y);
                flag = 1;
	      }
	}
	
        if (cnt>2)
	{
          pcnt = (int*)calloc(cnt-1,sizeof(int));
          
          for (x=0;x<m_width;x++)
            for (y=0;y<m_height;y++)
              if (get_nrrd_int2(slice,x,y)>1) pcnt[get_nrrd_int2(slice,x,y)-1]++;
          max = 0;
          for (i=0;i<cnt-1;i++)
            if (pcnt[i]>max)
              {
              max = pcnt[i];
              m = i+1;
              }
          
          maxy = m_height-1;
          if (z<zb)
            {
            maxy = 0;
            for (y=m_height-1;y>=0;y--)
              {
              for (x=0;x<m_width;x++)
                if (get_nrrd_int2(slice,x,y)==m)
                  {
                  maxy = y;
                  break;
                  }
              if (maxy>0) break;
              }
            }
          for (x=0;x<m_width;x++)
            for (y=0;y<m_height;y++)
              if (((get_nrrd_int(m_Label,x,y,z)==T_INTRACRANIAL)||(get_nrrd_int(m_Label,x,y,z)==T_SINUS))&&(get_nrrd_int2(slice,x,y)!=m))
                {
                if (!((get_nrrd_int(m_Label,x,y,z+1)==T_INTRACRANIAL)||(get_nrrd_int(m_Label,x,y,z+1)==T_SINUS)))
                  {
                  set_nrrd_int(m_Label,T_FOREGROUND,x,y,z);
                  pcnt[get_nrrd_int2(slice,x,y)-1]--;
                  }
                else if (y>maxy)
                  {
                  set_nrrd_int(m_Label,T_FOREGROUND,x,y,z);
                  pcnt[get_nrrd_int2(slice,x,y)-1]--;
                  }
                }
          free(pcnt);
          } // if more than 1 2D connected component
        } // if brain pixel exists in slice
      } // z-loop
    }
}

void
MRITissueClassifier::BoneDetection_PDthreshold (float *th_low, float *th_high)
{
  // this is a reliable function that does a 2 class clustering (bone+air
  // vs. scalp) using the proton density signal. I found that pd works slightly
  // better than  t1+t2+pd for some reason
  int i, x, y, z, cnt;
  float min=0.0, max=0.0, val, mean[2], var[2], prior[2], temp, *pdfest;
  ColumnMatrix data;
  
  cnt = 0;
  for (z=0;z<m_TopOfHead;z++)
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if (get_nrrd_int(m_Label,x,y,z)==T_FOREGROUND)
	{
          val = get_nrrd_float(m_PD_Data, x, y, z);
          if (cnt==0) 
	    min=max=val;
          else
	  {
            if (val>max) 
	      max=val;
            else if (val<min) 
	      min=val;
	  }
          cnt++;
	}
  
  std::cout<<"PD min = "<<min<<" , max = "<<max<<std::endl;
  // compute bins for the histogram
  float stepsize = ceil((max-min)/(float)NBINMAX);
  int nbin = (int)rint((max-min)/stepsize);
  std::cout<<"Using "<<nbin<<" bins for histogram\n";
  int   *hist      = (int*)calloc(nbin, sizeof(int));
  float *pdf       = (float*)calloc(nbin, sizeof(int));
  float *centers   = (float*)calloc(nbin, sizeof(float));
  float *upperlims = (float*)calloc(nbin-1, sizeof(float));
  float *cdf       = (float*)calloc(nbin, sizeof(int));
  
  upperlims[0]=min+stepsize;
  centers[0]=min;
  for (i=1;i<nbin-1;i++)
    {
    upperlims[i]=upperlims[i-1]+stepsize;
    centers[i]=centers[i-1]+stepsize;
    }
  centers[nbin-1]=max-stepsize;
  Clear1DHistogram(hist,nbin);
  Accumulate1DHistogramWithLabel (upperlims,hist,nbin,m_PD_Data,
                                  0,0,0,
				  m_width-1,m_height-1,m_TopOfHead-1,
                                  T_FOREGROUND);


  for (i=0;i<nbin;i++) pdf[i]=(float)hist[i]/(float)cnt;
  cdf[0] = pdf[0];
  for (i=1;i<nbin;i++) cdf[i] = cdf[i-1] + pdf[i];

  for (i=1;i<nbin;i++) if (cdf[i]>=0.15) break;
  mean[0] = centers[i];                       
  for (;i<nbin;i++) if (cdf[i]>=0.65) break;
  mean[1] = centers[Min(i,nbin-1)];        
  prior[0]= 0.3; 
  prior[1]= 0.7; 
  temp  = 0.5*centers[(int)rint(0.05*(float)nbin)];
  var[0] = temp*temp;  // bone/air has the smallest variance
  var[1] = 4.0*var[0];
  
  std::cout<<"Initialization for EM:"<<std::endl;
  std::cout<<"bone/air    mean = "<<mean[0]<<" variance = "<<var[0]<<" prior = "<<prior[0]<<std::endl;
  std::cout<<"scalp      mean = "<<mean[1]<<" variance = "<<var[1]<<" prior = "<<prior[1]<<std::endl;

  pdfest=EM1DWeighted (centers, pdf, nbin, 2, mean, prior, var);

  std::cout<<"After EM:"<<std::endl;
  std::cout<<"bone/air    mean = "<<mean[0]<<" variance = "<<var[0]<<" prior = "<<prior[0]<<std::endl;
  std::cout<<"scalp      mean = "<<mean[1]<<" variance = "<<var[1]<<" prior = "<<prior[1]<<std::endl;

  // compute threshold for intra/extra-cranial separation
  *th_low = 0.5*(mean[0]+mean[1]);
  *th_high = (2.0*mean[1]+mean[0])/3.0;
  std::cout<<"low threshold = "<<(*th_low)<<" , high threshold = "<<(*th_high)<<std::endl;
}

void
MRITissueClassifier::BoneDetection (float maskr)
{
  std::cout<<"BONE DETECTION\n";
  float th_low, th_high;
  BoneDetection_PDthreshold (&th_low, &th_high);
  m_BoneThreshold = th_low;
  BoneDetection_TopSection (maskr, th_low, th_high); // from top of head to 15
                                                     // mm above eye centers
  BoneDetection_SecondSection (th_low);              // a thin region above
                                                     // the eyes
  BoneDetection_ThirdSection (th_low);               // the rest
  AirDetection(2.0);
}

void
MRITissueClassifier::BoneDetection_TopSection (float maskr, float th_low, float th_high)
{
  int i, x, y, z, xp, yp, xn, yn, flag, flag2, flag_OFF;;
  float dist, dl, dh, th, val, ndx, ndy;
  int zt1 = m_TopOfEyeSlice + (int)ceil(35.0/m_SliceThickness);
  int zt2 = m_TopOfEyeSlice + (int)ceil(15.0/m_SliceThickness);

  NrrdDataHandle slice = create_nrrd_of_ints(m_width,m_height);
  int N[8][2];
  N[0][0] = 1;N[0][1] = 0;
  N[1][0] = -1;N[1][1] = 0;
  N[2][0] = 0;N[2][1] = 1;
  N[3][0] = 0;N[3][1] = -1;
  N[4][0] = 1;N[4][1] = 1;
  N[5][0] = 1;N[5][1] = -1;
  N[6][0] = -1;N[6][1] = 1;
  N[7][0] = -1;N[7][1] = -1;

  int rad = (int)ceil(maskr);
  int len = 2*rad+1;
  float xo, yo;
  NrrdDataHandle mask = create_nrrd_of_ints(len,len);;
  for (x=0;x<len;x++)
    for (y=0;y<len;y++)
    {
      xo = (float)( x - rad );
      yo = (float)( y - rad );
      if ( sqrt(xo*xo + yo*yo) <=maskr ) set_nrrd_int2(mask,1,x,y);
      else set_nrrd_int2(mask,0,x,y);
    }

  for (z=m_TopOfHead-1;z>zt2;z--)
  {
    if (z>zt1) th = th_high;
    else
    {
      val = ((float)(zt1 - z))/(float)(zt1-zt2);
      th = val*th_low + (1.0-val)*th_high;
    }
    for (y=0;y<m_height;y++) 
      for (x=0;x<m_width;x++)
        if (get_nrrd_int(m_Label,x,y,z)==T_FOREGROUND)
	{
          if ((get_nrrd_float(m_PD_Data,x,y,z) > th) ||
	      (get_nrrd_float(m_DistBG2D,x,y,z) <= 3.0)) 
	    set_nrrd_int2(slice,1,x,y); // scalp
          else 
	    set_nrrd_int2(slice,2,x,y); // bone
	}
        else 
	  set_nrrd_int2(slice,0,x,y);
    
    for (y=1;y<m_height-1;y++) 
      for (x=1;x<m_width-1;x++)
        if (get_nrrd_int2(slice,x,y)==1) // scalp
	{
          dist = get_nrrd_float(m_DistBG2D,x,y,z);
          ndx = get_nrrd_float(m_DistBG2D,x+1,y,z)-get_nrrd_float(m_DistBG2D,x-1,y,z);
          ndy = get_nrrd_float(m_DistBG2D,x,y+1,z)-get_nrrd_float(m_DistBG2D,x,y-1,z);
          val = sqrt(ndx*ndx+ndy*ndy);
          ndx /= val;
          ndy /= val;
          dl = dist - 1.5*m_PixelDim;
          dh = dist + 0.5*m_PixelDim;
          flag = flag2 = flag_OFF = 0;
          for (xp = -rad; xp<=rad; xp++)
	  {
            for (yp = -rad; yp<=rad; yp++)
              if (((xp!=0)||(yp!=0))&&get_nrrd_int2(mask,xp+rad,yp+rad))
	      {
                xn = x + xp;
                yn = y + yp;
                if ((xn>=0)&&(xn<m_width)&&(yn>=0)&&(yn<m_height)&&
                    (get_nrrd_float(m_DistBG2D,xn,yn,z)>=dl)&&
		    (get_nrrd_float(m_DistBG2D,xn,yn,z)<=dh))
		{
                  if (get_nrrd_int(m_Label,xn,yn,z)==T_INTRACRANIAL)
		  {
                    flag_OFF = 1;
                    break;
		  }
                  if (((!flag)||(!flag2))&&(get_nrrd_int2(slice,xn,yn)==2))
		  {
                    val = ndy*(float)xp - ndx*(float)yp;
                    if (val>0.0) flag = 1;
                    else if (val<0.0) flag2 = 1;
		  }
		}
	      }
            if (flag_OFF) break;
            }
          if (flag&&flag2&&(!flag_OFF)) set_nrrd_int2(slice,3,x,y);
	}

    // find scalp pixels touching the background
    for (y=1;y<m_height-1;y++)
      for (x=1;x<m_width-1;x++)
        if (get_nrrd_int2(slice,x,y)==1)
	{
          flag = 0;
          for (i=0;i<4;i++)
            if (get_nrrd_int(m_Label,x+N[i][0],y+N[i][1],z)==T_BACKGROUND)
	    {
              flag = 1;
              break;
	    }
          if (flag) 
	    floodFill (slice,1,0,x,y);
	}

    for (y=0;y<m_height;y++) 
      for (x=0;x<m_width;x++)
        if (get_nrrd_int2(slice,x,y)>0) set_nrrd_int2(slice,1,x,y);

    for (y=0;y<m_height;y++) 
      for (x=0;x<m_width;x++)
        if (get_nrrd_int2(slice,x,y)==1)
	{
          flag = 0;
          for (i=0;i<4;i++)
            if ((get_nrrd_int(m_Label,x+N[i][0],y+N[i][1],z)==T_INTRACRANIAL)||
                (get_nrrd_int(m_Label,x+N[i][0],y+N[i][1],z)==T_SINUS))
	    {
              flag = 1;
              break;
	    }
          if (flag) floodFill (slice,1,2,x,y);
	}

    for (y=0;y<m_height;y++) 
      for (x=0;x<m_width;x++)
        if (get_nrrd_int(m_Label,x,y,z)==T_FOREGROUND)
	{
          if (get_nrrd_int2(slice,x,y)==2)
	  {
            val = get_nrrd_float(m_PD_Data,x,y,z);	    
            if (m_FatSat)
	    {
              if (val<=th_low) 
		set_nrrd_int(m_Label,T_BONE,x,y,z);
              else if (get_nrrd_float(m_Fat_Energy,x,y,z)<=m_FatThreshold) 
		set_nrrd_int(m_Label,T_BONE,x,y,z);
              else 
		set_nrrd_int(m_Label,T_MARROW,x,y,z);
	    }
            else
	    {
              if (val>th_low) 
		set_nrrd_int(m_Label,T_MARROW,x,y,z);
              else 
		set_nrrd_int(m_Label,T_BONE,x,y,z);
	    }
	  }
          else 
	    set_nrrd_int(m_Label,T_FOREGROUND,x,y,z);
	}
  }
 
}

void
MRITissueClassifier::BoneDetection_SecondSection (float th)
{
  int i, x, y, z, flag;
  int zb = m_TopOfEyeSlice + (int)ceil(9.0/m_SliceThickness);
  int zt = m_TopOfEyeSlice + (int)ceil(15.0/m_SliceThickness);
  NrrdDataHandle boneim = create_nrrd_of_ints(m_width, m_height);
  int N[8][2];
  N[0][0] = 1;N[0][1] = 0;
  N[1][0] = -1;N[1][1] = 0;
  N[2][0] = 0;N[2][1] = 1;
  N[3][0] = 0;N[3][1] = -1;
  N[4][0] = 1;N[4][1] = 1;
  N[5][0] = 1;N[5][1] = -1;
  N[6][0] = -1;N[6][1] = 1;
  N[7][0] = -1;N[7][1] = -1;
  int ylow = (int)rint(Max(m_EyePosition[0][1],m_EyePosition[1][1])-12.5/m_PixelDim);
  int yhigh = (int)rint(Max(m_EyePosition[0][1],m_EyePosition[1][1])+12.5/m_PixelDim);
  
  for (z=zb;z<=zt;z++)
    {
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if (get_nrrd_int(m_Label,x,y,z)==T_FOREGROUND)
	{
	  if ((get_nrrd_float(m_PD_Data,x,y,z)<=th)&&
	      (get_nrrd_float(m_DistBG2D,x,y,z)>3.0))
	    set_nrrd_int2(boneim,1,x,y) ;
	  else 
	    set_nrrd_int2(boneim,2,x,y) ;
	}
        else 
	  set_nrrd_int2(boneim,0,x,y);
    
    for (y=1;y<m_height-1;y++)
      for (x=1;x<m_width-1;x++)
        if (get_nrrd_int2(boneim,x,y)==2)
	{
          flag = 0;
          for (i=0;i<4;i++)
            if (get_nrrd_int(m_Label,x+N[i][0],y+N[i][1],z)==T_BACKGROUND)
	    {
              flag = 1;
              break;
	    }
          if (flag) 
	    floodFill (boneim,2,0,x,y);
	}

    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if (get_nrrd_int2(boneim,x,y)==2) 
	  set_nrrd_int2(boneim,1,x,y);

    for (y=1;y<m_height-1;y++)
      for (x=1;x<m_width-1;x++)
        if (get_nrrd_int2(boneim,x,y)==1)
	{
          flag = 0;
          for (i=0;i<4;i++)
            if ((get_nrrd_int(m_Label,x+N[i][0],y+N[i][1],z)==T_INTRACRANIAL)||
                (get_nrrd_int(m_Label,x+N[i][0],y+N[i][1],z)==T_SINUS))               
	    {
              flag = 1;
              break;
	    }
          if (flag)
	    floodFill (boneim,1,2,x,y);
	}
    
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if ((get_nrrd_int2(boneim,x,y)==2)&&(get_nrrd_int(m_Label,x,y,z)==T_FOREGROUND))
	{
          if (get_nrrd_float(m_PD_Data,x,y,z)<=th) 
	    set_nrrd_int(m_Label,T_BONE,x,y,z);
          else
	  {
            if (m_FatSat)
	    {
              if (get_nrrd_float(m_Fat_Energy,x,y,z)>=m_FatThreshold)
	      {
                if ((y<ylow)||(y>yhigh)) 
		  set_nrrd_int(m_Label,T_MARROW,x,y,z);
                else 
		  set_nrrd_int(m_Label,T_FOREGROUND,x,y,z);
	      }
              else set_nrrd_int(m_Label,T_BONE,x,y,z);
	    }
            else
	    {
              if ((y<ylow)||(y>yhigh)) 
		set_nrrd_int(m_Label,T_MARROW,x,y,z);
              else 
		set_nrrd_int(m_Label,T_FOREGROUND,x,y,z);
	    }
	  }
	}
        else if ((get_nrrd_int2(boneim,x,y)==1)&&(get_nrrd_int(m_Label,x,y,z)==T_FOREGROUND)) 
	  set_nrrd_int(m_Label,T_FOREGROUND,x,y,z);
    }
}



void
MRITissueClassifier::BoneDetection_ThirdSection (float th)
{
  int x, y, z;
  int zt = m_TopOfEyeSlice + (int)ceil(9.0/m_SliceThickness);

  for (z=0;z<zt;z++)
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if ((get_nrrd_int(m_Label,x,y,z)==T_FOREGROUND)&&
            (get_nrrd_float(m_PD_Data,x,y,z)<=th)&&
            (get_nrrd_float(m_DistBG2D,x,y,z)>3.0))
          set_nrrd_int(m_Label,T_BONE,x,y,z);
        
}

float
MRITissueClassifier::BackgroundNoiseLevel ()
{
  // simple function to estimate background variance
  int x, y, z, cnt;
  float var;

  var = 0.0;
  cnt = 0;
  for (z=0;z<m_depth;z++)
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if (get_nrrd_int(m_Label,x,y,z)==T_BACKGROUND)
          {
          var += (get_nrrd_float(m_Energy,x,y,z)*get_nrrd_float(m_Energy,x,y,z));
          cnt++;
          }
  var/=(float)cnt;
  std::cout<<"Noise variance = "<<var<<"\n";

  return sqrt(var);
}

void
MRITissueClassifier::AirDetection (float maskr)
{
  int x, y, z;
  int zt = m_TopOfEyeSlice + (int)rint(20.0/m_SliceThickness);
  float th = BackgroundNoiseLevel ();
  float th2;
  //  int xl = (int)rint(m_EyePosition[0][0]-12.5/m_SliceThickness);
  //int xr = (int)rint(m_EyePosition[1][0]+12.5/m_SliceThickness);

  int rad = (int)floor(maskr);
  int len = 2*rad+1;
    
  float xo, yo;
  NrrdDataHandle mask = create_nrrd_of_ints(len,len);
  NrrdDataHandle tempI = create_nrrd_of_ints(m_width, m_height);
  NrrdDataHandle vol = create_nrrd_of_ints(m_width, m_height, zt);
  NrrdDataHandle seed = create_nrrd_of_ints(m_width, m_height, zt);
  
  
  for (y=0;y<len;y++)
    for (x=0;x<len;x++)
      {
      xo = (float)( x - rad );
      yo = (float)( y - rad );
      if ( sqrt(xo*xo + yo*yo) <=maskr ) set_nrrd_int2(mask,1,x,y);
      else set_nrrd_int2(mask,0,x,y);
      }

  for (z=0;z<=m_TopTemporalSlice;z++)
    {
    for (y=0;y<m_height;y++)
      {
      if (y>=m_brainlim[z]) th2 = 1.5*th; else th2 = th;
      for (x=0;x<m_width;x++)
        if (get_nrrd_int(m_Label,x,y,z)==T_BONE)
          {
          if ((get_nrrd_float(m_Energy,x,y,z)<=0.5*th)&&
	      (get_nrrd_float(m_PD_Data,x,y,z)<=0.5*m_BoneThreshold))
            {
            set_nrrd_int2(tempI,1,x,y);
            set_nrrd_int(seed,1,x,y,z);
            }
          else if (get_nrrd_float(m_Energy,x,y,z)<=th2)
            {
            set_nrrd_int2(tempI,1,x,y);
            set_nrrd_int(seed,0,x,y,z);
            }
          else
            {
            set_nrrd_int2(tempI,0,x,y);
            set_nrrd_int(seed,0,x,y,z);
            }
          }
        else
          {
          set_nrrd_int2(tempI,0,x,y);
          set_nrrd_int(seed,0,x,y,z);
          }
      }
    tempI = opening(tempI,mask);
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if (get_nrrd_int2(tempI,x,y)) set_nrrd_int(vol,1,x,y,z); else set_nrrd_int(vol,0,x,y,z);
    }

  for (z=m_TopTemporalSlice+1;z<zt;z++)
  {
    for (y=m_brainlim[z];y<m_height;y++)
      for (x=0;x<m_width;x++)
        if (get_nrrd_int(m_Label,x,y,z)==T_BONE)
	{
          if ((get_nrrd_float(m_Energy,x,y,z)<=0.5*th)&&
	      (get_nrrd_float(m_PD_Data,x,y,z)<=0.5*m_BoneThreshold))
	  {
	    set_nrrd_int2(tempI,1,x,y);
	    set_nrrd_int(seed,1,x,y,z);
	  }
          else if (get_nrrd_float(m_Energy,x,y,z)<=1.5*th)
	  {
            set_nrrd_int2(tempI,1,x,y);
            set_nrrd_int(seed,0,x,y,z);
	  }
          else
	  {
            set_nrrd_int2(tempI,0,x,y);
            set_nrrd_int(seed,0,x,y,z);
	  }
	}
        else
	{
          set_nrrd_int2(tempI,0,x,y);
          set_nrrd_int(seed,0,x,y,z);
	}
    for (y=0;y<m_brainlim[z];y++)
      for (x=0;x<m_width;x++)
      {
        set_nrrd_int2(tempI,0,x,y);
        set_nrrd_int(seed,0,x,y,z);
      }
    tempI = opening(tempI,mask);
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if (get_nrrd_int2(tempI,x,y)) set_nrrd_int(vol,1,x,y,z); else set_nrrd_int(vol,0,x,y,z);
  }
  
  for (z=0;z<zt;z++)
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if ((get_nrrd_int(vol,x,y,z)==1)&&(get_nrrd_int(seed,x,y,z)==1))
          floodFill(vol,1,2,x,y,z);

  for (z=0;z<zt;z++)
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if (get_nrrd_int(vol,x,y,z)==2)
          set_nrrd_int(m_Label,T_AIR,x,y,z);
}

void
MRITissueClassifier::BrainClassification ()
{
  std::cout<<"INTRACRANIAL CLASSIFICATION\n";
  int x, y, z, i;
  int cnt = 0;
  ColumnMatrix data;
  float min=0.0, max=0.0, temp, mean[3], prior[3], var[3];

  for (x=0;x<m_width;x++)
    for (y=0;y<m_height;y++)
      for (z=0;z<m_depth;z++)
        if (get_nrrd_int(m_Label,x,y,z)==T_INTRACRANIAL)
	{
	  const float val = get_nrrd_float(m_CSF_Energy,x,y,z);
          if (cnt==0) 
	    min=max=val;
          else
	  {
            if (val>max) 
	      max=val;
            else if (val<min) 
	      min=val;
	  }
          cnt++;
	}
  
  float stepsize = ceil((max-min)/(float)NBINMAX);
  int nbin = (int)rint((max-min)/stepsize);
  std::cout<<"Using "<<nbin<<" bins for histogram\n";
  int   *hist      = (int*)calloc(nbin, sizeof(int));
  float *cdf       = (float*)calloc(nbin, sizeof(int));
  float *pdf       = (float*)calloc(nbin, sizeof(int));
  float *centers   = (float*)calloc(nbin, sizeof(float));
  float *upperlims = (float*)calloc(nbin-1, sizeof(float));
  float *pdfest;
  
  upperlims[0]=min+stepsize;
  centers[0]=min;
  for (i=1;i<nbin-1;i++)
    {
    upperlims[i]=upperlims[i-1]+stepsize;
    centers[i]=centers[i-1]+stepsize;
    }
  centers[nbin-1]=max-stepsize;
  Clear1DHistogram(hist,nbin);
  Accumulate1DHistogramWithLabel (upperlims,hist,nbin,m_CSF_Energy,
                                  0,0,0,
				  m_width-1,m_height-1,m_depth-1,
                                  T_INTRACRANIAL);
  
  for (i=0;i<nbin;i++) pdf[i]=(float)hist[i]/(float)cnt;
  cdf[0] = pdf[0];
  for (i=1;i<nbin;i++) cdf[i] = cdf[i-1] + pdf[i];

  // initialize gaussian parameters
  for (i=0;i<nbin;i++) if (cdf[i]>=0.2) break;
  mean[0] = centers[i];                           // mean for WM
  for (;i<nbin;i++) if (cdf[i]>=0.65) break;
  mean[1] = centers[i];                           // mean for GM
  for (;i<nbin;i++) if (cdf[i]>=0.95) break;
  mean[2] = centers[Min(i,nbin-1)];            // mean for CSF
  prior[0]= 0.4;
  prior[1]= 0.5;
  prior[2]= 0.1;
  temp  = (mean[1]-mean[0])/4.0;
  var[0] = temp*temp;
  var[1] = 4.0*var[0];
  var[2] = 4.0*var[0];
  
  std::cout<<"Initialization for EM algorithm:\n";
  std::cout<<"WM  mean = "<<mean[0]<<" variance = "<<var[0]<<" prior = "<<prior[0]<<std::endl;
  std::cout<<"GM  mean = "<<mean[1]<<" variance = "<<var[1]<<" prior = "<<prior[1]<<std::endl;
  std::cout<<"CSF mean = "<<mean[2]<<" variance = "<<var[2]<<" prior = "<<prior[2]<<std::endl;

  // EM on histogram
  pdfest=EM1DWeighted (centers, pdf, nbin, 3, mean, prior, var);

  std::cout<<"After EM (on histogram):\n";
  std::cout<<"WM  mean = "<<mean[0]<<" variance = "<<var[0]<<" prior = "<<prior[0]<<std::endl;
  std::cout<<"GM  mean = "<<mean[1]<<" variance = "<<var[1]<<" prior = "<<prior[1]<<std::endl;
  std::cout<<"CSF mean = "<<mean[2]<<" variance = "<<var[2]<<" prior = "<<prior[2]<<std::endl;
  
  float diffl, diffh, pos, threshold1, threshold2;
  
  // find thresholf between CSF and GM
  for (i=nbin-1;i>=0;i--) if (pdfest[i*3+1]>=pdfest[i*3+2]) break;
  if (i>=(nbin-1)) threshold1 = centers[nbin-1];
  diffl=pdfest[i*3+2]-pdfest[i*3+1];
  diffh=pdfest[(i+1)*3+2]-pdfest[(i+1)*3+1];
  pos = diffl/(diffl-diffh);
  threshold1 = centers[i] + pos * stepsize;
  
  // find thresholf between WM and GM
  for (;i>=0;i--) if (pdfest[i*3]>=pdfest[i*3+1]) break;
  if (i>=(nbin-1)) threshold2= centers[nbin-1];
  diffl=pdfest[i*3+1]-pdfest[i*3];
  diffh=pdfest[(i+1)*3+1]-pdfest[(i+1)*3];
  pos = diffl/(diffl-diffh);
  threshold2 = centers[i] + pos * stepsize;
  std::cout<<"csf-gm threshold = "<<threshold1<<"\n";
  std::cout<<"gm-wm  threshold = "<<threshold2<<"\n";

  for (x=0;x<m_width;x++)
    for (y=0;y<m_height;y++)
      for (z=0;z<m_depth;z++)
        if (get_nrrd_int(m_Label,x,y,z)==T_INTRACRANIAL)
	{
          if (get_nrrd_float(m_CSF_Energy,x,y,z)>=threshold1) 
	    set_nrrd_int(m_Label,T_CSF,x,y,z);
          else if (get_nrrd_float(m_CSF_Energy,x,y,z)>=threshold2) 
	    set_nrrd_int(m_Label,T_GM,x,y,z);
          else 
	    set_nrrd_int(m_Label,T_WM,x,y,z);
	}

  int dim = 3;
  ColumnMatrix datavec(dim), diff(dim), feature(dim);
  ColumnMatrix CSF_mean(dim), GM_mean(dim), WM_mean(dim);
  DenseMatrix CSF_cov(dim,dim), GM_cov(dim,dim), WM_cov(dim,dim);
  float CSF_prior, GM_prior, WM_prior;

  CSF_mean.zero();GM_mean.zero();WM_mean.zero();
  CSF_cov.zero();GM_cov.zero();WM_cov.zero();
  CSF_prior = GM_prior = WM_prior = 0.0;

  for (x=0;x<m_width;x++)
    for (y=0;y<m_height;y++)
      for (z=0;z<m_depth;z++)
      {
	feature = get_m_Data(x,y,z);
	if (feature.nrows() != 3) feature.resize(3);
	if (get_nrrd_int(m_Label,x,y,z)==T_CSF)
	{
	  Add(CSF_mean, CSF_mean, feature);
	  CSF_prior += 1.0;
	}
	if (get_nrrd_int(m_Label,x,y,z)==T_GM)
	{
	  Add(GM_mean, GM_mean, feature);
	  GM_prior += 1.0;
	}
	if (get_nrrd_int(m_Label,x,y,z)==T_WM)
	{
	  Add(WM_mean, WM_mean, feature);
	  WM_prior += 1.0;
	}
      }
  CSF_mean.scalar_multiply(1.0 / CSF_prior);
  GM_mean.scalar_multiply(1.0 / GM_prior);
  WM_mean.scalar_multiply(1.0 / WM_prior);
  
  for (x=0;x<m_width;x++)
    for (y=0;y<m_height;y++)
      for (z=0;z<m_depth;z++)
      {	  
	feature = get_m_Data(x,y,z);
	if (feature.nrows() != 3) feature.resize(3);
	if (get_nrrd_int(m_Label,x,y,z)==T_CSF)
	{
	  Sub(diff, feature, CSF_mean);
	  Add(CSF_cov, CSF_cov, diff.exterior(diff));
	  set_nrrd_int(m_Label,T_INTRACRANIAL,x,y,z);
	}
        if (get_nrrd_int(m_Label,x,y,z)==T_GM)
	{
          Sub(diff, feature, GM_mean);
	  Add(GM_cov, GM_cov, diff.exterior(diff));
          set_nrrd_int(m_Label,T_INTRACRANIAL,x,y,z);
	}
        if (get_nrrd_int(m_Label,x,y,z)==T_WM)
	{
          Sub(diff, feature, WM_mean);
	  Add(WM_cov, WM_cov, diff.exterior(diff));
          set_nrrd_int(m_Label,T_INTRACRANIAL,x,y,z);
	}
      }
  
  CSF_cov.scalar_multiply(1.0 / CSF_prior);
  GM_cov.scalar_multiply(1.0 / GM_prior);
  WM_cov.scalar_multiply(1.0 / WM_prior);
  
  CSF_prior = prior[2];
  GM_prior = prior[1];
  WM_prior = prior[0];
  
  free (pdfest);
  free (hist);
  free (upperlims);
  free (pdf);
  free (centers);
  
  pdfest=EM_CSF_GM_WM(CSF_mean,CSF_cov,&CSF_prior,
                      GM_mean,GM_cov,&GM_prior,
                      WM_mean,WM_cov,&WM_prior,
                      T_INTRACRANIAL);
  
  int k = 0;
  for (z=0;z<m_depth;z++)
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if (get_nrrd_int(m_Label,x,y,z)==T_INTRACRANIAL)
	{
          if (pdfest[k]>pdfest[k+1])
	  {
            if (pdfest[k]>pdfest[k+2]) 
	      set_nrrd_int(m_Label,T_CSF,x,y,z);
            else 
	      set_nrrd_int(m_Label,T_WM,x,y,z);
	  }
          else 
	  {
            if (pdfest[k+1]>pdfest[k+2]) 
	      set_nrrd_int(m_Label,T_GM,x,y,z);
            else 
	      set_nrrd_int(m_Label,T_WM,x,y,z);
	  }
          k+=3;
	}
  
  for (z=0;z<m_depth;z++)
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if ((get_nrrd_int(m_Label,x,y,z)==T_CSF)||
            (get_nrrd_int(m_Label,x,y,z)==T_GM)||
            (get_nrrd_int(m_Label,x,y,z)==T_WM))
          set_nrrd_int(m_Label,T_INTRACRANIAL,x,y,z);
  
  float *spdf = SmoothProbabilities (pdfest,3,T_INTRACRANIAL,cnt,0.5);
  free (pdfest);
  pdfest = spdf;
  
  k = 0;
  for (z=0;z<m_depth;z++)
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if (get_nrrd_int(m_Label,x,y,z)==T_INTRACRANIAL)
	{
          if (pdfest[k]>pdfest[k+1])
	  {
            if (pdfest[k]>pdfest[k+2]) 
	      set_nrrd_int(m_Label,T_CSF,x,y,z);
            else 
	      set_nrrd_int(m_Label,T_WM,x,y,z);
	  }
          else 
	  {
            if (pdfest[k+1]>pdfest[k+2]) 
	      set_nrrd_int(m_Label,T_GM,x,y,z);
            else 
	      set_nrrd_int(m_Label,T_WM,x,y,z);
	  }
          k+=3;
	}  
  free(pdfest);
  //  write_flat_volume(m_Label, "/tmp/final.ppm");
}

NrrdDataHandle
MRITissueClassifier::gaussian(const NrrdDataHandle &data, double sigma)
{
  NrrdResampleInfo *info = nrrdResampleInfoNew();  
  NrrdKernel *kern;
  double p[NRRD_KERNEL_PARMS_NUM];
  memset(p, 0, NRRD_KERNEL_PARMS_NUM * sizeof(double));
  p[0] = 1.0;
  kern = nrrdKernelGaussian; 
  p[1] = sigma; 
  p[2] = 6.0; 
  for (int a = 0; a < data->nrrd->dim; a++) {
    info->kernel[a] = kern;
    info->samples[a]=data->nrrd->axis[a].size;
    if (a==0) {
      info->kernel[0]=0;
    } 
    memcpy(info->parm[a], p, NRRD_KERNEL_PARMS_NUM * sizeof(double));
    if (info->kernel[a] && 
	(!(airExists(data->nrrd->axis[a].min) && airExists(data->nrrd->axis[a].max)))) {
      nrrdAxisInfoMinMaxSet(data->nrrd, a, data->nrrd->axis[a].center ? 
			data->nrrd->axis[a].center : nrrdDefCenter);
    }
    info->min[a] = data->nrrd->axis[a].min;
    info->max[a] = data->nrrd->axis[a].max;
  }    
  info->boundary = nrrdBoundaryBleed;
  info->type = data->nrrd->type;
  info->renormalize = AIR_TRUE;
  NrrdData *nrrd = scinew NrrdData;
  if (nrrdSpatialResample(nrrd->nrrd=nrrdNew(), data->nrrd, info)) {
    char *err = biffGetDone(NRRD);
    error(string("Trouble resampling: ") +  err);
    msgStream_ << "  input Nrrd: data->nrrd->dim=" << data->nrrd->dim << "\n";
    free(err);
  }
  nrrdResampleInfoNix(info); 
  //nrrd->copy_sci_data(*data.get_rep());
  return nrrd;
}



float *MRITissueClassifier::SmoothProbabilities (float *prob, int c, int l, int n, float sigma)
{
  int x, y, z, k, j, m;
  float sum;
  float *sprob = (float*)calloc(n*c,sizeof(float));
  NrrdDataHandle p = create_nrrd_of_floats(m_width, m_height, m_depth);
  for (j=0;j<c;j++)
  {
    k = j;
    m = j;
    for (z=0;z<m_depth;z++)
      for (y=0;y<m_height;y++)
        for (x=0;x<m_width;x++)
          if (get_nrrd_int(m_Label,x,y,z)==l)
	  {
            set_nrrd_float(p,prob[k],x,y,z);
            k+=c;
	  }
          else 
	    set_nrrd_float(p,0.0f,x,y,z);
    
    p = gaussian(p,sigma);
    for (z=0;z<m_depth;z++)
      for (y=0;y<m_height;y++)
        for (x=0;x<m_width;x++)
          if (get_nrrd_int(m_Label,x,y,z)==l)
	  {
            sprob[m] = Max(get_nrrd_float(p,x,y,z),0.0f);
            m+=c;
	  }
  }
  
  for (k=0;k<(n*c);k+=c)
  {
    sum = 0.0f;
    for (j=0;j<c;j++) 
      sum+=sprob[k+j];
    sum = Max(sum, TINY);
    for (j=0;j<c;j++) 
      sprob[k+j]/=sum;
  }
  
  return sprob;
}


void
MRITissueClassifier::ScalpClassification ()
{
  std::cout<<"SCALP CLASSIFICATION\n";
  int x, y, z;
  int dim;
  if (m_FatSat) dim = 4; else dim = 3;
  ColumnMatrix datavec(dim), diff(dim);
  ColumnMatrix Muscle_mean(dim), Fat_mean(dim);
  DenseMatrix Muscle_cov(dim,dim), Fat_cov(dim,dim);
  float Muscle_prior, Fat_prior, fatenergy, *pdfest, sum;

  Muscle_mean.zero();Fat_mean.zero();
  Muscle_cov.zero();Fat_cov.zero();
  Muscle_prior = Fat_prior = 0.0;
  
  for (x=0;x<m_width;x++)
    for (y=0;y<m_height;y++)
      for (z=0;z<m_depth;z++)
        if (get_nrrd_int(m_Label,x,y,z)==T_FOREGROUND)
	{
	  datavec = get_m_Data(x,y,z);
          if (m_FatSat) fatenergy = get_nrrd_float(m_Fat_Energy,x,y,z);
          else fatenergy = datavec.get(0);
	  
          if (fatenergy>m_FatThreshold)
	  {	    
            Add(Fat_mean, Fat_mean, datavec);
            Fat_prior += 1.0;
            set_nrrd_int(m_Label,T_FAT,x,y,z);
	  }
          else 
	  {
            Add(Muscle_mean, Muscle_mean, datavec);
            Muscle_prior += 1.0;
            set_nrrd_int(m_Label,T_MUSCLE,x,y,z);
	  }
	}
  Muscle_mean.scalar_multiply(1.0 / Muscle_prior);
  Fat_mean.scalar_multiply(1.0 / Fat_prior);

  for (x=0;x<m_width;x++)
    for (y=0;y<m_height;y++)
      for (z=0;z<m_depth;z++)
      {
        datavec = get_m_Data(x,y,z);
        if (get_nrrd_int(m_Label,x,y,z)==T_MUSCLE)
	{
          Sub(diff,datavec,Muscle_mean);
	  Add(Muscle_cov, Muscle_cov, diff.exterior(diff));
          set_nrrd_int(m_Label,T_FOREGROUND,x,y,z);
	}
        if (get_nrrd_int(m_Label,x,y,z)==T_FAT)
	{
          Sub(diff,datavec, Fat_mean);
	  Add(Fat_cov, Fat_cov, diff.exterior(diff));
          set_nrrd_int(m_Label,T_FOREGROUND,x,y,z);
          }
        }

  Muscle_cov.scalar_multiply(1.0 / Muscle_prior);
  Fat_cov.scalar_multiply(1.0 / Fat_prior);

  sum = Muscle_prior + Fat_prior;
  Muscle_prior /= sum;
  Fat_prior /= sum;

  pdfest=EM_Muscle_Fat (Muscle_mean,Muscle_cov,&Muscle_prior,
			Fat_mean,Fat_cov,&Fat_prior,
                        T_FOREGROUND,dim);

  int k = 0;
  for (z=0;z<m_depth;z++)
    for (y=0;y<m_height;y++)
      for (x=0;x<m_width;x++)
        if (get_nrrd_int(m_Label,x,y,z)==T_FOREGROUND)
	{
          if (pdfest[k]>pdfest[k+1]) set_nrrd_int(m_Label,T_MUSCLE,x,y,z);
          else set_nrrd_int(m_Label,T_FAT,x,y,z);
          k+=2;
	}

  free (pdfest);
}



//  are the 2D morphology functions and floodFill from vispack that I
// use. The floodfill also uses some list classes, but I'm sure there are
// corresponding things in SCIRun


NrrdDataHandle
MRITissueClassifier::dilate(NrrdDataHandle data, NrrdDataHandle mask)
{
  ASSERT(data->nrrd->type == nrrdTypeInt);

  int val;
  int x, y, xm, ym, xo, yo;
  const int w = data->nrrd->axis[X_AXIS].size;
  const int h = data->nrrd->axis[Y_AXIS].size;
  const int wm = mask->nrrd->axis[X_AXIS].size;
  const int hm = mask->nrrd->axis[Y_AXIS].size;
  const int cx = (wm-1)/2;
  const int cy = (hm-1)/2;
  NrrdDataHandle r = create_nrrd_of_ints(w, h);
				 
  for (x = 0; x < w; x++)
    for (y = 0; y < h; y++)
    {
      val = 0;
      for (xm = 0; xm < wm; xm++)
      {
	for (ym = 0; ym < hm; ym++)
	  if (get_nrrd_int2(mask,xm,ym) > 0)
	  {
	    xo = x + xm - cx;
	    yo = y + ym - cy;
	    if (nrrd_check_bounds(data, xo, yo) &&
		(get_nrrd_int2(data,xo,yo) > 0))
	    {
	      val = 1;
	      break;
	    }
	  }
	if (val>0) break;
      }
      set_nrrd_int2(r, val, x, y);
    }
  return r;
}


// 2d nrrd of ints
NrrdDataHandle
MRITissueClassifier::erode(NrrdDataHandle data, NrrdDataHandle mask)
{
  ASSERT(data->nrrd->type == nrrdTypeInt);
  int val;
  int x, y, xm, ym, xo, yo;
  const int w = data->nrrd->axis[X_AXIS].size;
  const int h = data->nrrd->axis[Y_AXIS].size;
  const int wm = mask->nrrd->axis[X_AXIS].size;
  const int hm = mask->nrrd->axis[Y_AXIS].size;
  const int cx = (wm-1)/2;
  const int cy = (hm-1)/2;

  NrrdDataHandle r = create_nrrd_of_ints(w, h);
					 
  for (x = 0; x < w; x++)
    for (y = 0; y < h; y++)
    {
      val = 1;
      for (xm = 0; xm <wm; xm++)
      {
	for (ym = 0; ym <hm; ym++)
	  if (get_nrrd_int2(mask,xm,ym) > 0)
	  {
	    xo = x + xm - cx;
	    yo = y + ym - cy;
	    if (nrrd_check_bounds(data, xo, yo) &&
		(get_nrrd_int2(data, xo, yo) <= 0))
	    {
	      val = 0;
	      break;
	    }
	  }
	if (val < 1) break;
      }
      set_nrrd_int2(r,val,x,y);
    }
  return r;
}


// 2d nrrd of ints
NrrdDataHandle
MRITissueClassifier::opening(NrrdDataHandle data, NrrdDataHandle mask) 
{
  NrrdDataHandle temp = erode(data, mask);
  return dilate(temp,mask);
}

// 2d nrrd of ints
NrrdDataHandle
MRITissueClassifier::closing(NrrdDataHandle data, NrrdDataHandle mask)
{
  const int mw = mask->nrrd->axis[X_AXIS].size;
  const int mh = mask->nrrd->axis[Y_AXIS].size;
  const int px = (mw-1)/2;
  const int py = (mh-1)/2;
  const int w = data->nrrd->axis[X_AXIS].size;
  const int h = data->nrrd->axis[Y_AXIS].size;
  // we need to pad the image because dilation might spill outside the
  // boundaries of the original image making it impossible to recover the
  // original with erosion
  //  VISImage <T> padim = VISImage<T> (w+mask.width(), h+mask.height());
  NrrdDataHandle padim = create_nrrd_of_ints(w+mw, h+mh);
  memset(padim->nrrd->data, 0, (w+mw)*(h+mh)*sizeof(int));
  int x, y;
  for (x=0;x<w;x++)
    for (y=0;y<h;y++)     
      set_nrrd_int2(padim, get_nrrd_int2(data, x, y), x+px,y+py);




  //VISImage<T> temp  = padim.dilate (mask);
  //return (temp.erode (mask)).getROI(px,py,this->width(),this->height());
  NrrdDataHandle temp = dilate(padim, mask);
  padim = erode(temp, mask);

  int *min = scinew int[2];
  int *max = scinew int[2];
  min[X_AXIS] = px;
  max[X_AXIS] = px+w-1;
  min[Y_AXIS] = py;
  max[Y_AXIS] = py+h-1;
  
  if (nrrdCrop(temp->nrrd, padim->nrrd, min, max)) {
    char *err = biffGetDone(NRRD);
    error("Trouble cropping nrrd: "+string(err));
    free(err);
  }

  delete min;
  delete max;
  return temp;
}




void
MRITissueClassifier::floodFill(NrrdDataHandle data, int label_from, int label_to, unsigned int x, unsigned int y)
{
  // these are the neighbors
  int N[4][2];
  N[0][0] = 1;
  N[0][1] = 0;
  N[1][0] = -1;
  N[1][1] = 0;
  N[2][0] = 0;
  N[2][1] = 1;
  N[3][0] = 0;
  N[3][1] = -1;
  unsigned at_x, at_y, next_x, next_y;
  int i;
  
  if ((get_nrrd_int2(data,x, y) != label_from) || (label_from == label_to))
    return;
  //  if it passes the threshold put the starting pixel on the list
  list<int> listx, listy;
  listx.push_back(x);
  listy.push_back(y);
  //  mark that pixel as "in"
  set_nrrd_int2(data,label_to,x, y);
  while (!listx.empty())
  {
    at_x = listx.front();
    at_y = listy.front();
    // look at your neighbors and if they are: 1) in bounds, 2) more than the
    // threshod and 3) not yet visited, put them on the list and mark themas
    // visited.    
    for (i = 0; i < 4; i++)
    {
      next_x = at_x + N[i][0];
      next_y = at_y + N[i][1];
      if (nrrd_check_bounds(data, next_x, next_y))
      {
	if (get_nrrd_int2(data,next_x, next_y) == label_from)
	{
	  set_nrrd_int2(data, label_to, next_x, next_y);
	  listx.push_back(next_x);
	  listy.push_back(next_y);
	}
      }
    }
    //
    // remove the guy whose neighbors you have just visited
    // when the list is empty, you are done.
    //
    listx.pop_front();
    listy.pop_front();
  }
}





//3d floodfill
// assumes data is a volume of ints
void
MRITissueClassifier::floodFill(NrrdDataHandle data, int label_from, int label_to, unsigned int x, unsigned int y, unsigned int z)
{
  // these are the neighbors
  int N[6][3];
  N[0][0] = 1;
  N[0][1] = 0;
  N[0][2] = 0;
  N[1][0] = -1;
  N[1][1] = 0;
  N[1][2] = 0;
  N[2][0] = 0;
  N[2][1] = 1;
  N[2][2] = 0;
  N[3][0] = 0;
  N[3][1] = -1;
  N[3][2] = 0;
  N[4][0] = 0;
  N[4][1] = 0;
  N[4][2] = 1;
  N[5][0] = 0;
  N[5][1] = 0;
  N[5][2] = -1;
  unsigned at_x, at_y, at_z, next_x, next_y, next_z;
  int i;
  
  if ((get_nrrd_int(data, x, y, z) != label_from) ||(label_from == label_to))
    return;
  
  //  if it passes the threshold put the starting pixel on the list
  list<int> listx, listy, listz;
  listx.push_back(x);
  listy.push_back(y);
  listz.push_back(z);
  
  //  mark that pixel as "in"
  set_nrrd_int(data,label_to, x,y,z);
  
  while (!listx.empty())
  {
    at_x = listx.front();
    at_y = listy.front();
    at_z = listz.front();
    for (i = 0; i < 6; i++)
    {
      next_x = at_x + N[i][0];
      next_y = at_y + N[i][1];
      next_z = at_z + N[i][2];
      if (nrrd_check_bounds(data, next_x, next_y, next_z))
      {
	if (get_nrrd_int(data,next_x, next_y, next_z) == label_from)
	{
	  set_nrrd_int(data, label_to, next_x, next_y, next_z);
	  listx.push_back(next_x);
	  listy.push_back(next_y);
	  listz.push_back(next_z);
	}
      }
    }
    listx.pop_front();
    listy.pop_front();
    listz.pop_front();
  }
}



NrrdDataHandle
MRITissueClassifier::extract_nrrd_slice_int(NrrdDataHandle data, int z)
{
  const int dx = data->nrrd->axis[X_AXIS].size;
  const int dy = data->nrrd->axis[Y_AXIS].size;
  
  NrrdDataHandle ret = create_nrrd_of_ints(dx, dy);
  const int *source = (int *)data->nrrd->data;
  memcpy(ret->nrrd->data, source + z*dx*dy, dx*dy*sizeof(int));
  return ret;
}


NrrdDataHandle
MRITissueClassifier::create_nrrd_of_ints(int x, int y, int z)
{
  NrrdData *data = scinew NrrdData();
  data->nrrd = nrrdNew();  
  if (nrrdAlloc(data->nrrd, nrrdTypeInt, 3, x, y, z)) {
    char *err = biffGetDone(NRRD);
    error(string("Trouble create_nrrd_of_ints: ") +  err);
    free(err);
  }
  return data;
}

NrrdDataHandle
MRITissueClassifier::create_nrrd_of_floats(int x, int y, int z) 
{
  NrrdData *data = scinew NrrdData();
  data->nrrd = nrrdNew();
  nrrdAlloc(data->nrrd, nrrdTypeFloat, 3, x, y, z);
  return data;
}

bool
MRITissueClassifier::nrrd_check_bounds(NrrdDataHandle data, 
				       int x, int y, int z) 
{
  return (x >= 0 && x < data->nrrd->axis[X_AXIS].size &&
	  y >= 0 && y < data->nrrd->axis[Y_AXIS].size &&
	  z >= 0 && z < data->nrrd->axis[Z_AXIS].size);
}


NrrdDataHandle
MRITissueClassifier::create_nrrd_of_ints(int x, int y)
{
  NrrdData *data = scinew NrrdData();
  data->nrrd = nrrdNew();
  nrrdAlloc(data->nrrd, nrrdTypeInt, 2, x, y);
  return data;
}

NrrdDataHandle
MRITissueClassifier::create_nrrd_of_floats(int x, int y) 
{
  NrrdData *data = scinew NrrdData();
  data->nrrd = nrrdNew();
  nrrdAlloc(data->nrrd, nrrdTypeFloat, 2, x, y);
  return data;
}

bool
MRITissueClassifier::nrrd_check_bounds(NrrdDataHandle data, int x, int y) 
{
  return (x >= 0 && x < data->nrrd->axis[X_AXIS].size &&
	  y >= 0 && y < data->nrrd->axis[Y_AXIS].size);
}

ColumnMatrix
MRITissueClassifier::get_m_Data(int x, int y, int z)
{
  ColumnMatrix m_data(m_FatSat?4:3);
  ASSERT(m_T1_Data.get_rep());
  ASSERT(m_T2_Data.get_rep());
  ASSERT(m_PD_Data.get_rep());
  m_data[0] = get_nrrd_float(m_T1_Data, x, y, z);
  m_data[1] = get_nrrd_float(m_T2_Data, x, y, z);
  m_data[2] = get_nrrd_float(m_PD_Data, x, y, z);
  if (m_FatSat) 
    m_data[3] = get_nrrd_float(m_FATSAT_Data, x, y, z);
  
  return m_data;
}



void
MRITissueClassifier::write_flat_volume(NrrdDataHandle data, string filename)
{
  vector <char> r(23), g(23), b(23);
  r[0] =  0;   g[0] =  0;   b[0] = 0;
  r[1] =  0;   g[1] =  100; b[1] = 0;
  r[2] =  0;   g[2] =  0;   b[2] = 100;
  r[3] =  100; g[3] =  100; b[3] = 0;
  r[4] =  100; g[4] =  0;   b[4] = 100;
  r[5] =  000; g[5] =  100; b[5] = 100;
  r[6] =  100; g[6] =  100; b[6] = 100;
  r[7] =  100; g[7] =  200; b[7] = 0;
  r[8] =  100; g[8] =  0;   b[8] = 200;
  r[9] =  0;   g[9] =  200; b[9] = 200;
  r[10] = 100; g[10] = 0;   b[10] = 0;
  r[21] = 50;  g[21] = 100; b[21] = 200;
  r[22] = 200; g[22] = 100; b[22] = 50;

  const int *idata = (int *)data->nrrd->data;
  const float *fdata = (float *)data->nrrd->data;
  
  const int dx = data->nrrd->axis[X_AXIS].size;
  const int dy = data->nrrd->axis[Y_AXIS].size;
  const int dz = data->nrrd->axis[Z_AXIS].size;
  const int max = dx*dy*dz;

  set<int> done;
  const bool isfloat = (data->nrrd->type == nrrdTypeFloat);
  float fmax=0.0, fmin=0.0;
  int imax=0, imin=0, cnt=0;
  if (isfloat)
  {
    fmax = fmin = fdata[0];
    for (int k=0;k<max;k++)
    {
      if (fdata[k]>fmax) fmax=fdata[k];
      else if (fdata[k]<fmin) fmin=fdata[k];
    }
  }
  else
  {
    imax = imin = idata[0];
    for (int k=0;k<max;k++)
    {
      if (done.find(idata[k]) == done.end())
      {
	cnt++;
	done.insert(idata[k]);
      }

      if (idata[k]>imax) imax=idata[k];
      else if (idata[k]<imin) imin=idata[k];
    }
    std::cerr << "Count of colors in " << filename << ": " << cnt << std::endl;
  }


  const int num_col_tiles = (int)ceil(::sqrt((double)dz));
  const int num_row_tiles = (int)ceil((float)dz/
				      (float)num_col_tiles);
 
  const int new_width = num_col_tiles*dx;
  const int new_height = num_row_tiles*dy;
  
  const int zero = 0;
  int i, j, k, x, y;
  char c;

  FILE *out = fopen(filename.c_str(), "wb");
  if (!out) return;

  fprintf (out, "P6\n%d %d\n%d\n", new_width, new_height, 255);
  for (j = 0; j < num_row_tiles; j++)
    for (y = 0; y < dy; y++)
      for (i = 0; i < num_col_tiles; i++)
	for (x = 0; x < dx; x++) 
	{
	  k = x + dx*(y+dy*(i + j * num_col_tiles));
	  if (k > max) {
	    fwrite (&zero, 1, 3, out);
	    continue;
	  }

	  if (isfloat) 
	  {
	    c = (int)rint((fdata[k]-fmin)*255.0/(fmax-fmin));
	    fwrite (&c, 1, 1, out);
	    fwrite (&c, 1, 1, out);
	    fwrite (&c, 1, 1, out);
	  }
	  else
	  {
#if 1
	    c = (int)rint(255.0*(idata[k]-imin)/(imax-imin));
	    fwrite (&c, 1, 1, out);
	    fwrite (&c, 1, 1, out);
	    fwrite (&c, 1, 1, out);
#else
	    fwrite (&r[idata[k]], 1, 1, out);
	    fwrite (&g[idata[k]], 1, 1, out);
	    fwrite (&b[idata[k]], 1, 1, out);
#endif
	  }

	}

  fclose (out);
  std::cerr << "Wrote: " << filename << "." << std::endl;
} 



void
MRITissueClassifier::write_image(NrrdDataHandle data, string filename)
{
  vector <char> r(23), g(23), b(23);
  r[0] =  0;   g[0] =  0;   b[0] = 0;
  r[1] =  0;   g[1] =  100; b[1] = 0;
  r[2] =  0;   g[2] =  0;   b[2] = 100;
  r[3] =  100; g[3] =  100; b[3] = 0;
  r[4] =  100; g[4] =  0;   b[4] = 100;
  r[5] =  000; g[5] =  100; b[5] = 100;
  r[6] =  100; g[6] =  100; b[6] = 100;
  r[7] =  100; g[7] =  200; b[7] = 0;
  r[8] =  100; g[8] =  0;   b[8] = 200;
  r[9] =  0;   g[9] =  200; b[9] = 200;
  r[10] = 100; g[10] = 0;   b[10] = 0;
  r[21] = 50;  g[21] = 100; b[21] = 200;
  r[22] = 200; g[22] = 100; b[22] = 50;

  const int *idata = (int *)data->nrrd->data;
  const float *fdata = (float *)data->nrrd->data;

  const bool greyscale = (data->nrrd->type == nrrdTypeFloat);
  float fmax=0.0, fmin=0.0;
  if (greyscale)
  {
    fmax = fmin = fdata[0];
      for (int y=0;y<data->nrrd->axis[Y_AXIS].size;y++)
	for (int x=0;x<data->nrrd->axis[X_AXIS].size;x++)
	{
	  float fval = get_nrrd_float2(data, x, y);
	  if (fval>fmax) fmax=fval;
	  else if (fval<fmin) fmin=fval;
	}
  }
  
  const int dx = data->nrrd->axis[X_AXIS].size;
  const int dy = data->nrrd->axis[Y_AXIS].size;
  const int max = dx*dy;

  const int zero = 0;
  int  k, x, y;
  char c;

  FILE *out = fopen(filename.c_str(), "wb");
  if (!out) return;

  fprintf (out, "P6\n%d %d\n%d\n", dx, dy, 255);
  for (y = 0; y < dy; y++)
    for (x = 0; x < dx; x++) 
    {
      k = x + dx*y;
      if (k > max) {
	fwrite (&zero, 1, 3, out);
	continue;
      }
      
      if (greyscale) {
	c = (int)ceil((fdata[k]-fmin)*255.0/fmax);
	fwrite (&c, 1, 1, out);
	fwrite (&c, 1, 1, out);
	fwrite (&c, 1, 1, out);
      } else {
	fwrite (&r[idata[k]], 1, 1, out);
	fwrite (&g[idata[k]], 1, 1, out);
	fwrite (&b[idata[k]], 1, 1, out);
      }
    }
  
  fclose (out);
  std::cerr << "Wrote: " << filename << "." << std::endl;
} 



} //namespace SCIRun
