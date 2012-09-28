/*

 The MIT License

 Copyright (c) 1997-2012 Center for the Simulation of Accidental Fires and
 Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI),
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


//Interpolate.cc Created Jan7. works for benchmark 99 absorb_coef
//Appears to work when T is constant.
#include <iostream>
#include <cmath>


double Interpolate(double value000, double value001, double value010, double value011,
    double value100, double value101, double value110, double value111,
    double cc_ray_location[3],
    int k_dist, int j_dist, int  i_dist, int k){
  //note, cc_ray_locations may be negative.  This is ok, because when I do k1-k0, this value will also be negative.
  double face_value[4];//These are the 4 values which represent the face which comes from interpolating the cube
  double line_value[2];//These are the 2 values which represent the line which comes from interpolating the face
  double interped_value;//This is the final value we end up with after interpolation.  

  //!! consider multiplying my i_dist,j_dist, etc., rather than dividing, since it will always be either 1 or -1...
  //the only down side to this is if, in doing composite mesh, that the distances between the cells is larger than 1.
  face_value[0] = (value001 - value000)*cc_ray_location[0]/i_dist + value000;
  face_value[1] = (value011 - value010)*cc_ray_location[0]/i_dist + value010;
  face_value[2] = (value101 - value100)*cc_ray_location[0]/i_dist + value100;
  face_value[3] = (value111 - value110)*cc_ray_location[0]/i_dist + value110;

  line_value[0] = (face_value[1] - face_value[0])*cc_ray_location[1]/j_dist + face_value[0];
  line_value[1] = (face_value[3] - face_value[2])*cc_ray_location[1]/j_dist + face_value[2];

  interped_value = (line_value[1] - line_value[0])*cc_ray_location[2]/k_dist + line_value[0];

/*  if (k==4){
  //  std::cout << "face_values " << face_value[0] << "  " << face_value[1] << "  " <<face_value[2] << " " << face_value[3] << std::endl;
//  }
  if (k==4){
  //  std::cout << "line_values " << line_value[0] << "  " << line_value[1] << endl;
    std::cout << "interped_value " << interped_value << std::endl;
  }
*/

  return (interped_value);
}//end Interpolate function

