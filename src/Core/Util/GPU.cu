/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
 
#include <Core/Util/GPU.h>
 
namespace Uintah {

//______________________________________________________________________
//  Returns true if threadID and blockID are 0.
//  Useful in conditional statements for limiting output.
//
__device__
bool
isThread0_Blk0(){
  int blockID  = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z; 
  int threadID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;
  
  bool test (blockID == 0 && threadID == 0);
  return test;
}

//______________________________________________________________________
//  Returns true if threadID = 0 for this block
//  Useful in conditional statements for limiting output.
//
__device__
bool
isThread0(){
  int threadID = threadIdx.x +  threadIdx.y +  threadIdx.z;
  bool test (threadID == 0 );
  return test;
}

//______________________________________________________________________
// Output the threadID
//
__device__
void 
printThread(){ 
  int threadID = threadIdx.x +  threadIdx.y +  threadIdx.z;
  printf( "Thread [%i,%i,%i], ID: %i\n", threadIdx.x,threadIdx.y,threadIdx.z, threadID);
}

//______________________________________________________________________
// Output the blockID
//
__device__
void 
printBlock(){ 
  int blockID  = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  printf( "Block  [%i,%i,%i], ID: %i\n", blockIdx.x,blockIdx.y,blockIdx.z, blockID);
}

}  // end namespace Uintah
