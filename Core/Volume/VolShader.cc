//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : VolShader.cc
//    Author : Milan Ikits
//    Date   : Tue Jul 13 02:28:09 2004

#include <sstream>
#include <iostream>
#include <Core/Volume/VolShader.h>
#include <Core/Geom/ShaderProgramARB.h>

using std::string;
using std::vector;
using std::ostringstream;

namespace SCIRun {

#define VOL_HEAD \
"!!ARBfp1.0 \n" \
"ATTRIB t = fragment.texcoord[0]; \n" \
"TEMP v; \n" \
"TEMP c; \n"

#define VOL_TAIL \
"END"

#define VOL_VLUP_1_1 \
"TEX v, t, texture[0], 3D; \n"
#define VOL_VLUP_1_4 \
"TEX v.w, t, texture[0], 3D; \n"
#define VOL_VLUP_2_1 VOL_VLUP_1_1
#define VOL_VLUP_2_4 VOL_VLUP_1_4
#define VOL_GLUP_2_1 \
"TEX v.y, t, texture[1], 3D; \n"
#define VOL_GLUP_2_4 \
"TEX v.x, t, texture[1], 3D; \n"


#define VOL_TFLUP_1_1 \
"TEX c, v.x, texture[2], 1D; \n"
#define VOL_TFLUP_1_4 \
"TEX c, v.w, texture[2], 1D; \n"
#define VOL_TFLUP_2_1 \
"TEX c, v, texture[2], 2D; \n"
#define VOL_TFLUP_2_4 \
"TEX c, v.wxyz, texture[2], 2D; \n"

#define VOL_TFLUP_MASK_HEAD \
"PARAM mask = program.local[3];\n" \
"TEMP f; \n" \
"TEMP m; \n" \
"TEMP b; \n" \
"MOV c, 0.0; \n" \
"MUL v.w, v.w, g.w; \n" \
"MOV m, mask; \n"

#define VOL_TFLUP_2_1_MASK \
"TEX b, v.wxyz, texture[2], 2D; \n" \
"MUL m, m, 0.5; \n" \
"FRC f, m; \n" \
"SGE f, f, 0.5; \n" \
"MAD_SAT c, b, f, c; \n" \
"ADD v.w, v.w, g.w; \n"

#define VOL_FOG_HEAD \
"PARAM fc = state.fog.color; \n" \
"PARAM fp = state.fog.params; \n" \
"ATTRIB tf = fragment.texcoord[1];\n" \
"TEMP fctmp; \n";
#define VOL_FOG_BODY \
"SUB v.x, fp.z, tf.x; \n" \
"MUL_SAT v.x, v.x, fp.w; \n" \
"MUL fctmp, c.w, fc; \n" \
"LRP c.xyz, v.x, c.xyzz, fctmp.xyzz; \n"

#define VOL_FRAG_HEAD \
"ATTRIB cf = fragment.color; \n"
#define VOL_FRAG_BODY \
"MUL c, c, cf; \n"

#define VOL_LIT_HEAD \
"PARAM l = program.local[0]; # {lx, ly, lz, alpha} \n" \
"PARAM k = program.local[1]; # {ka, kd, ks, ns} \n" \
"PARAM g = program.local[2]; # {1/gradrange, -gradmin/gradrange, 0, 0} \n" \
"TEMP n; \n" \
"TEMP w; \n"


#define VOL_LIT_BODY_NOGRAD \
"MAD n, v, 2.0, -1.0;		# rescale from [0,1] to [-1, 1]  \n" \
"DP3 n.w, n, n;			# n.w = x*x + y*y + z*z \n" \
"RSQ n.w, n.w;			# n.w = 1 / sqrt(x*x+y*y+z*z) \n" \
"MUL n, n, n.w;			# n = n / length(normal)\n" \
"DP3 n.w, l, n;			# calculate angle between light and normal. \n" \
"ABS_SAT n.w, n.w;		# two-sided lighting, n.w = abs(cos(angle))  \n" \
"MOV w, k;       # w.x = weight*ka, w.y = weight*kd, w.z = weight*ks \n" \
"SUB w.x, k.x, w.y; # w.x = ka - kd*weight \n" \
"ADD w.x, w.x, k.y; # w.x = ka + kd - kd*weight \n" \
"POW n.z, n.w, k.w;   # n.z = abs(cos(angle))^ns \n" \
"MAD n.w, n.w, w.y, w.x; # n.w = abs(cos(angle))*kd+ka\n" \
"MUL n.z, w.z, n.z; # n.z = weight*ks*abs(cos(angle))^ns \n"


#define VOL_LIT_BODY \
"MAD n, v, 2.0, -1.0;		# rescale from [0,1] to [-1, 1]  \n" \
"DP3 n.w, n, n;			# n.w = x*x + y*y + z*z \n" \
"RSQ n.w, n.w;			# n.w = 1 / sqrt(x*x+y*y+z*z) \n" \
"MUL n, n, n.w;			# n = n / length(normal)\n" \
"DP3 n.w, l, n;			# calculate angle between light and normal. \n" \
"ABS_SAT n.w, n.w;		# two-sided lighting, n.w = abs(cos(angle))  \n" \
"TEX w.x, t, texture[1], 3D;	# get the gradient magnitude \n" \
"MAD_SAT w.xyzw, w.x, g.x, g.y;	# compute saturated weight based on current gradient \n" \
"MUL w, w, k;       # w.x = weight*ka, w.y = weight*kd, w.z = weight*ks \n" \
"SUB w.x, k.x, w.y; # w.x = ka - kd*weight \n" \
"ADD w.x, w.x, k.y; # w.x = ka + kd - kd*weight \n" \
"POW n.z, n.w, k.w;   # n.z = abs(cos(angle))^ns \n" \
"MAD n.w, n.w, w.y, w.x; # n.w = abs(cos(angle))*kd+ka\n" \
"MUL n.z, w.z, n.z; # n.z = weight*ks*abs(cos(angle))^ns \n"




#define VOL_LIT_END \
"MUL n.z, n.z, c.w;\n" \
"MAD c.xyz, c.xyzz, n.w, n.z;\n"

/*
// "MAD n.w, n.z, k.z, n.w; \n"
// #define VOL_LIT_END \
// "MUL c.xyz, c.xyzz, n.w; \n"
*/

#define VOL_GRAD_COMPUTE_2_1 \
"PARAM dir = program.local[4]; \n" \
"TEMP r; \n" \
"TEMP p; \n" \
"PARAM tmat[]  = { state.matrix.texture[0].invtrans }; \n" \
"MOV v, v.xxxx; \n" \
"MOV n, 0; \n" \
"MOV w.x, dir.x; \n" \
"ADD_SAT p, fragment.texcoord[0], w; \n" \
"TEX r, p, texture[0], 3D; \n" \
"ADD n.x, r.x, n.x; \n" \
"SUB_SAT p, fragment.texcoord[0], w; \n" \
"TEX r, p, texture[0], 3D; \n" \
"SUB n.x, r.x, n.x; \n" \
"MOV w, 0; \n" \
"MOV w.y, dir.y; \n" \
"ADD_SAT p, fragment.texcoord[0], w; \n" \
"TEX r, p, texture[0], 3D; \n" \
"ADD n.y, r.x, n.y; \n" \
"SUB_SAT p, fragment.texcoord[0], w; \n" \
"TEX r, p, texture[0], 3D; \n" \
"SUB n.y, r.x, n.y; \n" \
"MOV w, 0; \n" \
"MOV w.z, dir.z; \n" \
"ADD_SAT p, fragment.texcoord[0], w; \n" \
"TEX r, p, texture[0], 3D; \n" \
"ADD n.z, r.x, n.z; \n" \
"SUB_SAT p, fragment.texcoord[0], w; \n" \
"TEX r, p, texture[0], 3D; \n" \
"SUB n.z, r.x, n.z; \n" \
"DP3 w.x, n.x, tmat[0]; \n" \
"DP3 w.y, n.y, tmat[1]; \n" \
"DP3 w.z, n.z, tmat[2]; \n" \
"DP3 r, w, w; \n" \
"RSQ p, r.x; \n" \
"DST p, r, p; \n" \
"MUL p.y, p, 1.75; \n" \
"MUL n.xyz, w, 1.0; \n#" // The # at the end of the line is a cheap way of disabling the next instruction


#define VOL_GRAD_COMPUTE_2_4 \
"PARAM dir = program.local[4]; \n" \
"TEMP r; \n" \
"TEMP p; \n" \
"PARAM tmat[]  = { state.matrix.texture[0].invtrans }; \n" \
"MOV v, v.wwww; \n" \
"MOV n, 0; \n" \
"MOV w.x, dir.x; \n" \
"ADD_SAT p, fragment.texcoord[0], w; \n" \
"TEX r, p, texture[0], 3D; \n" \
"ADD n.x, r.w, n.x; \n" \
"SUB_SAT p, fragment.texcoord[0], w; \n" \
"TEX r, p, texture[0], 3D; \n" \
"SUB n.x, r.w, n.x; \n" \
"MOV w, 0; \n" \
"MOV w.y, dir.y; \n" \
"ADD_SAT p, fragment.texcoord[0], w; \n" \
"TEX r, p, texture[0], 3D; \n" \
"ADD n.y, r.w, n.y; \n" \
"SUB_SAT p, fragment.texcoord[0], w; \n" \
"TEX r, p, texture[0], 3D; \n" \
"SUB n.y, r.w, n.y; \n" \
"MOV w, 0; \n" \
"MOV w.z, dir.z; \n" \
"ADD_SAT p, fragment.texcoord[0], w; \n" \
"TEX r, p, texture[0], 3D; \n" \
"ADD n.z, r.w, n.z; \n" \
"SUB_SAT p, fragment.texcoord[0], w; \n" \
"TEX r, p, texture[0], 3D; \n" \
"SUB n.z, r.w, n.z; \n" \
"DP3 w.x, n.x, tmat[0]; \n" \
"DP3 w.y, n.y, tmat[1]; \n" \
"DP3 w.z, n.z, tmat[2]; \n" \
"DP3 r, w, w; \n" \
"RSQ p, r.x; \n" \
"DST p, r, p; \n" \
"MUL p.y, p, 1.75; \n" \
"MUL n.xyz, w, 1.0; \n#" // The # at the end of the line is a cheap way of disabling the next instruction


#define VOL_COMPUTED_GRADIENT_LOOKUP \
"MOV v.y, p.y; \n"


#define VOL_FRAGMENT_BLEND_HEAD \
"TEMP n; \n"

#define VOL_GRAD_COMPUTE_NOLIGHT_HEAD \
"TEMP w; \n"

#define VOL_FRAGMENT_BLEND_OVER_NV \
"TEX v, fragment.position.xyyy, texture[3], RECT; \n" \
"SUB n.w, 1.0, c.w; \n" \
"MAD result.color, v, n.w, c; \n"
#define VOL_FRAGMENT_BLEND_MIP_NV \
"TEX v, fragment.position.xyyy, texture[3], RECT; \n" \
"MAX result.color, v, c; \n"

#define VOL_FRAGMENT_BLEND_OVER_ATI \
"MUL n.xy, fragment.position.xyyy, program.local[2].xyyy;\n" \
"TEX v, n.xyyy, texture[3], 2D; \n" \
"SUB n.w, 1.0, c.w; \n" \
"MAD result.color, v, n.w, c; \n"
#define VOL_FRAGMENT_BLEND_MIP_ATI \
"MUL n.xy, fragment.position.xyyy, program.local[2].xyyy;\n" \
"TEX v, n.xyyy, texture[3], 2D; \n" \
"MAX result.color, v, c; \n"

#define VOL_RASTER_BLEND \
"MOV result.color, c; \n"


VolShader::VolShader(int dim, int vsize, int channels, bool shading, 
		     bool frag, bool fog, int blend, int cmaps)
  : dim_(dim), 
    vsize_(vsize),
    channels_(channels),
    shading_(shading),
    fog_(fog),
    blend_(blend),
    frag_(frag),
    num_cmaps_(cmaps),
    program_(0)
{}

VolShader::~VolShader()
{
  delete program_;
}

bool
VolShader::create()
{
  string s;
  if (emit(s)) return true;
  program_ = new FragmentProgramARB(s);
  return false;
}

bool
VolShader::emit(string& s)
{
  if (dim_!=1 && dim_!=2) return true;
  if (vsize_!=1 && vsize_!=4) return true;
  if (blend_!=0 && blend_!=1 && blend_!=2) return true;
  ostringstream z;

  z << VOL_HEAD;

  // Set up light/blend variables and input parameters.
  if (shading_)
  {
    z << VOL_LIT_HEAD;
  }
  else if (blend_)
  {
    z << VOL_FRAGMENT_BLEND_HEAD;
  }

  if (frag_)
  {
    z << VOL_FRAG_HEAD;
  }
  
  // Set up fog variables and input parameters.
  if (fog_)
  {
    z << VOL_FOG_HEAD;
  }

  if (dim_ == 1)  // 1D colormap
  {
    // Get value
    z << VOL_VLUP_1_1;
    
    if (shading_)
    {
      if (vsize_ == 1)
      {
        // Compute the normal if needed and not there.
        z << VOL_GRAD_COMPUTE_2_1;
      }
      // Add the lighting.
      z << VOL_LIT_BODY_NOGRAD;
    }
    
    // Lookup the colormap entry for this value.
    if (vsize_ == 1)
    {
      z << VOL_TFLUP_1_1;
    }
    else
    {
      z << VOL_TFLUP_1_4;
    }

    // Apply the lighting.
    if (shading_)
    {
      z << VOL_LIT_END;
    }
  }
  else // dim_ == 2, 2D colormap
  {
    if (shading_)
    {
      z << VOL_VLUP_2_1;
      if (vsize_ == 1)
      {
	z << VOL_GRAD_COMPUTE_2_1;
      }

      z << VOL_LIT_BODY_NOGRAD;
      if (vsize_ == 1)
      {
	z << VOL_COMPUTED_GRADIENT_LOOKUP;
      }
      else
      {
	z << VOL_GLUP_2_4;
      }

      if (num_cmaps_ > 1)
      {
	z << VOL_TFLUP_MASK_HEAD;
	for (int n = 0; n < num_cmaps_; ++n)
        {
	  z << VOL_TFLUP_2_1_MASK;
        }
      }
      else
      {
	if (vsize_ == 1)
	  z << VOL_TFLUP_2_1;
	else
	  z << VOL_TFLUP_2_4;
      }
      z << VOL_LIT_END;
    }
    else // No shading, 2D colormap.
    {
      if (vsize_ == 1)
      {
        z << VOL_VLUP_2_1;
        if (channels_ == 1)
        {
          // Compute Gradient magnitude and use it.
          if (!blend_) z << VOL_FRAGMENT_BLEND_HEAD;
          z << VOL_GRAD_COMPUTE_NOLIGHT_HEAD;
          z << VOL_GRAD_COMPUTE_2_1;
          z << "\n";
          z << VOL_COMPUTED_GRADIENT_LOOKUP;
        }
        else
        {
          z << VOL_GLUP_2_1;
        }
        z << VOL_TFLUP_2_1;
      }
      else // vsize_ == 4
      {
        z << VOL_VLUP_2_4;
        if (channels_ == 1)
        {
          if (!blend_) z << VOL_FRAGMENT_BLEND_HEAD;
          z << VOL_GRAD_COMPUTE_NOLIGHT_HEAD;
          z << VOL_GRAD_COMPUTE_2_4;
          z << "\n";
          z << VOL_COMPUTED_GRADIENT_LOOKUP;
          z << VOL_TFLUP_2_1; // look it up as if 2_1
        }
        else
        {
          z << VOL_GLUP_2_4;
          z << VOL_TFLUP_2_4;
        }
      }
    }
  }

  // frag
  if (frag_)
  {
    z << VOL_FRAG_BODY;
  }

  // fog
  if (fog_)
  {
    z << VOL_FOG_BODY;
  }

  // blend
  if (blend_ == 0)
  {
    z << VOL_RASTER_BLEND;
  }
  else if (blend_ == 1)
  {
    z << VOL_FRAGMENT_BLEND_OVER_NV;
  }
  else if (blend_ == 2)
  {
    z << VOL_FRAGMENT_BLEND_MIP_NV;
  }
  else if (blend_ == 3)
  {
    z << VOL_FRAGMENT_BLEND_OVER_ATI;
  }
  else if (blend_ == 4)
  {
    z << VOL_FRAGMENT_BLEND_MIP_ATI;
  }

  z << VOL_TAIL;

  s = z.str();
  std::cerr << s << std::endl;
  return false;
}


VolShaderFactory::VolShaderFactory()
  : prev_shader_(-1)
{}

VolShaderFactory::~VolShaderFactory()
{
  for(unsigned int i=0; i<shader_.size(); i++) {
    delete shader_[i];
  }
}

FragmentProgramARB*
VolShaderFactory::shader(int dim, int vsize, int channels, bool shading, 
			 bool frag, bool fog, int blend, int cmaps)
{
  if(prev_shader_ >= 0) {
    if(shader_[prev_shader_]->match(dim, vsize, channels, shading, frag, fog, blend, cmaps)) {
      return shader_[prev_shader_]->program();
    }
  }
  for(unsigned int i=0; i<shader_.size(); i++) {
    if(shader_[i]->match(dim, vsize, channels, shading, frag, fog, blend, cmaps)) {
      prev_shader_ = i;
      return shader_[i]->program();
    }
  }
  std::cout << "dim = " << dim << ", vsize = " << vsize << ", channels = " << channels << ", shading = "  << shading << ", frag = " << frag << ", fog = " << fog << ", blend = " << blend << ", cmaps = " << cmaps << "\n";
  VolShader* s = new VolShader(dim, vsize, channels, shading, frag, fog, blend, cmaps);
  if(s->create()) {
    delete s;
    return 0;
  }
  shader_.push_back(s);
  prev_shader_ = shader_.size()-1;
  return s->program();
}

} // end namespace SCIRun

