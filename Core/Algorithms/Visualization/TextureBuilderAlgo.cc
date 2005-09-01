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
//    File   : TextureBuilderAlgo.cc
//    Author : Milan Ikits
//    Date   : Wed Jul 14 23:54:35 2004

#include <Core/Algorithms/Visualization/TextureBuilderAlgo.h>
#include <Core/Geom/ShaderProgramARB.h>

namespace SCIRun {


void
texture_build_bricks(vector<TextureBrickHandle>& bricks,
                     int nx, int ny, int nz,
                     int nc, int* nb,
                     const BBox& bbox, int card_mem,
                     bool use_nrrd_brick)
{
  int brick_mem = card_mem*1024*1024/2;
  const int data_size[3] = { nx, ny, nz };
  int brick_size[3];
  int brick_offset[3];
  int brick_pad[3];
  int num_brick[3];
  int brick_nb;
  // figure out largest possible brick size
  int size[3];
  size[0] = data_size[0]; size[1] = data_size[1]; size[2] = data_size[2];

  const unsigned int max_texture_size =
    (nb[0] == 1)?
    ShaderProgramARB::max_texture_size_1() :
    ShaderProgramARB::max_texture_size_4();
  // initial brick size
  brick_size[0] = Min(NextPowerOf2(data_size[0]), max_texture_size);
  brick_size[1] = Min(NextPowerOf2(data_size[1]), max_texture_size);
  brick_size[2] = Min(NextPowerOf2(data_size[2]), max_texture_size);
  // number of bytes per brick
  brick_nb = brick_size[0]*brick_size[1]*brick_size[2]*nb[0];
  
  // find brick size
  // subdivide until fits
  while (brick_nb > brick_mem)
  {
    // number of bricks along the axes
    // the divisions have to satisfy the equation:
    // data_size <= (brick_size-1)*num_brick + 1
    // we assume that brick_size > 1
    num_brick[0] = (int)ceil((double)(data_size[0]-1)/(brick_size[0]-1));
    num_brick[1] = (int)ceil((double)(data_size[1]-1)/(brick_size[1]-1));
    num_brick[2] = (int)ceil((double)(data_size[2]-1)/(brick_size[2]-1));
    // size of leftover volumes
    int sp[3];
    sp[0] = data_size[0] - (num_brick[0]-1)*(brick_size[0]-1);
    sp[1] = data_size[1] - (num_brick[1]-1)*(brick_size[1]-1);
    sp[2] = data_size[2] - (num_brick[2]-1)*(brick_size[2]-1);
    // size of padding
    brick_pad[0] = NextPowerOf2(sp[0]) - sp[0];
    brick_pad[1] = NextPowerOf2(sp[1]) - sp[1];
    brick_pad[2] = NextPowerOf2(sp[2]) - sp[2];
    // sort padding
    int idx[3];
    SortIndex(brick_pad, idx);
    // split largest one
    size[idx[2]] = (int)ceil(size[idx[2]]/2.0);
    brick_size[idx[2]] = NextPowerOf2(size[idx[2]]);
    brick_nb = brick_size[0]*brick_size[1]*brick_size[2]*nb[0];
  }

  // number of bricks along the axes
  // the divisions have to satisfy the equation:
  // data_size <= (brick_size-1)*num_brick + 1
  // we assume that brick_size > 1
  num_brick[0] = (int)ceil((double)(data_size[0]-1)/(brick_size[0]-1));
  num_brick[1] = (int)ceil((double)(data_size[1]-1)/(brick_size[1]-1));
  num_brick[2] = (int)ceil((double)(data_size[2]-1)/(brick_size[2]-1));
  // padded sizes of last bricks
  brick_pad[0] = NextPowerOf2(data_size[0] - (num_brick[0]-1)*(brick_size[0]-1));
  brick_pad[1] = NextPowerOf2(data_size[1] - (num_brick[1]-1)*(brick_size[1]-1));
  brick_pad[2] = NextPowerOf2(data_size[2] - (num_brick[2]-1)*(brick_size[2]-1));
  // delete previous bricks (if any)
  bricks.clear();
  bricks.reserve(num_brick[0]*num_brick[1]*num_brick[2]);

  // create bricks
  // data bbox
  double data_bbmin[3], data_bbmax[3];
  data_bbmin[0] = bbox.min().x();
  data_bbmin[1] = bbox.min().y();
  data_bbmin[2] = bbox.min().z();
  data_bbmax[0] = bbox.max().x();
  data_bbmax[1] = bbox.max().y();
  data_bbmax[2] = bbox.max().z();
  // bbox and tbox parameters
  double bmin[3], bmax[3], tmin[3], tmax[3];
  brick_offset[2] = 0;
  bmin[2] = data_bbmin[2];
  bmax[2] = num_brick[2] > 1 ?
    (double)(brick_size[2] - 0.5)/(double)data_size[2] *
    (data_bbmax[2] - data_bbmin[2]) + data_bbmin[2] : data_bbmax[2];
  tmin[2] = 0.0;
  tmax[2] = num_brick[2] > 1 ? (double)(brick_size[2] - 0.5)/(double)brick_size[2] :
    (double)data_size[2]/(double)brick_size[2];
  for (int k=0; k<num_brick[2]; k++)
  {
    brick_offset[1] = 0;
    bmin[1] = data_bbmin[1];
    bmax[1] = num_brick[1] > 1 ?
      (double)(brick_size[1] - 0.5)/(double)data_size[1] *
      (data_bbmax[1] - data_bbmin[1]) + data_bbmin[1] : data_bbmax[1];
    tmin[1] = 0.0;
    tmax[1] = num_brick[1] > 1 ?
      (double)(brick_size[1] - 0.5)/(double)brick_size[1] :
      (double)data_size[1]/(double)brick_size[1];
    for (int j=0; j<num_brick[1]; j++)
    {
      brick_offset[0] = 0;
      bmin[0] = data_bbmin[0];
      bmax[0] = num_brick[0] > 1 ?
        (double)(brick_size[0] - 0.5)/(double)data_size[0] *
        (data_bbmax[0] - data_bbmin[0]) + data_bbmin[0] : data_bbmax[0];
      tmin[0] = 0.0;
      tmax[0] = num_brick[0] > 1 ? (double)(brick_size[0] - 0.5)/(double)brick_size[0] :
        (double)data_size[0]/(double)brick_size[0];
      for (int i=0; i<num_brick[0]; i++)
      {
	// TODO: Not sure this logic is correct when a power of 2
	// brick is constucted.
        if (use_nrrd_brick &&
            num_brick[0] * num_brick[1] * num_brick[2] == 1 &&
	    brick_pad[0] == nx && brick_pad[1] == ny && brick_pad[2] == nz)
	{
	  NrrdTextureBrick *b = scinew NrrdTextureBrick(0, 0,
            i < num_brick[0]-1 ? brick_size[0] : brick_pad[0],
            j < num_brick[1]-1 ? brick_size[1] : brick_pad[1],
            k < num_brick[2]-1 ? brick_size[2] : brick_pad[2],
            nc, nb,
            brick_offset[0], brick_offset[1], brick_offset[2],
            i < num_brick[0]-1 ? brick_size[0] : data_size[0] - (num_brick[0]-1)*(brick_size[0]-1),
            j < num_brick[1]-1 ? brick_size[1] : data_size[1] - (num_brick[1]-1)*(brick_size[1]-1),
            k < num_brick[2]-1 ? brick_size[2] : data_size[2] - (num_brick[2]-1)*(brick_size[2]-1),
            BBox(Point(bmin[0], bmin[1], bmin[2]),
                 Point(bmax[0], bmax[1], bmax[2])),
            BBox(Point(tmin[0], tmin[1], tmin[2]),
                 Point(tmax[0], tmax[1], tmax[2])));
          bricks.push_back(b);
	}
	else
	{
	  TextureBrick* b = scinew TextureBrickT<unsigned char>(
            i < num_brick[0]-1 ? brick_size[0] : brick_pad[0],
            j < num_brick[1]-1 ? brick_size[1] : brick_pad[1],
            k < num_brick[2]-1 ? brick_size[2] : brick_pad[2],
            nc, nb,
            brick_offset[0], brick_offset[1], brick_offset[2],
            i < num_brick[0]-1 ? brick_size[0] : data_size[0] - (num_brick[0]-1)*(brick_size[0]-1),
            j < num_brick[1]-1 ? brick_size[1] : data_size[1] - (num_brick[1]-1)*(brick_size[1]-1),
            k < num_brick[2]-1 ? brick_size[2] : data_size[2] - (num_brick[2]-1)*(brick_size[2]-1),
            BBox(Point(bmin[0], bmin[1], bmin[2]),
                 Point(bmax[0], bmax[1], bmax[2])),
            BBox(Point(tmin[0], tmin[1], tmin[2]),
                 Point(tmax[0], tmax[1], tmax[2])));
          bricks.push_back(b);
	}

        // update x parameters                     
        brick_offset[0] += brick_size[0]-1;
        bmin[0] = bmax[0];
        bmax[0] += i < num_brick[0]-2 ?
          (double)(brick_size[0]-1)/(double)data_size[0]*(data_bbmax[0] - data_bbmin[0]) :
          (double)(data_size[0]-brick_offset[0]-0.5)/(double)data_size[0]*(data_bbmax[0]-data_bbmin[0]);
        tmin[0] = i < num_brick[0]-2 ? 0.5/(double)brick_size[0] : 0.5/(double)brick_pad[0];
        tmax[0] = i < num_brick[0]-2 ? (double)(brick_size[0]-0.5)/(double)brick_size[0] : (data_size[0] - brick_offset[0])/(double)brick_pad[0];
      }
      // update y parameters
      brick_offset[1] += brick_size[1]-1;
      bmin[1] = bmax[1];
      bmax[1] += j < num_brick[1]-2 ?
        (double)(brick_size[1]-1)/(double)data_size[1] * (data_bbmax[1]-data_bbmin[1]) :
        (double)(data_size[1]-brick_offset[1]-0.5)/(double)data_size[1] * (data_bbmax[1]-data_bbmin[1]);
      tmin[1] = j < num_brick[1]-2 ? 0.5/(double)brick_size[1] : 0.5/(double)brick_pad[1];
      tmax[1] = j < num_brick[1]-2 ? (double)(brick_size[1]-0.5)/(double)brick_size[1] : (data_size[1]-brick_offset[1])/(double)brick_pad[1];
    } // j
    // update z parameters
    brick_offset[2] += brick_size[2]-1;
    bmin[2] = bmax[2];
    bmax[2] += k < num_brick[2]-2 ?
      (double)(brick_size[2]-1)/(double)data_size[2] * (data_bbmax[2]-data_bbmin[2]) :
      (double)(data_size[2]-brick_offset[2]-0.5)/(double)data_size[2] * (data_bbmax[2]-data_bbmin[2]);
    tmin[2] = k < num_brick[2]-2 ? 0.5/(double)brick_size[2] : 0.5/(double)brick_pad[2];
    tmax[2] = k < num_brick[2]-2 ? (double)(brick_size[2]-0.5)/(double)brick_size[2] : (data_size[2] - brick_offset[2])/(double)brick_pad[2];
  } // k
}

CompileInfoHandle
TextureBuilderAlgo::get_compile_info(const TypeDescription *vftd,
				     const TypeDescription *gftd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("TextureBuilderAlgoT");
  static const string base_class_name("TextureBuilderAlgo");

  CompileInfo *rval = scinew CompileInfo(template_class_name + "." +
					 vftd->get_filename() + "." +
					 gftd->get_filename() + ".",
					 base_class_name, 
					 template_class_name, 
					 vftd->get_name() + ", " +
					 gftd->get_name() + "<Vector> " + ", " +
					 "TextureBrickT< unsigned char > " );
  
  // Add in the include path to compile this obj
  rval->add_include(include_path);  

  vftd->fill_compile_info(rval);
  gftd->fill_compile_info(rval);
  return rval;
}

} // namespace SCIRun
