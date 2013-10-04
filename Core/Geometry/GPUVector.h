/*
 * The MIT License
 *
 * Copyright (c) 2013 The University of Utah
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


#include <sci_defs/cuda_defs.h>
typedef union Int3{
  int3 vector;
  int array[3];
  HOST_DEVICE Int3(const int3& copy):vector(copy){}
  HOST_DEVICE Int3& operator=(const int3& copy){
    vector=copy;
    return *this;
  }
  HOST_DEVICE int& operator[](const int& i){
    return array[i];
  }
  HOST_DEVICE operator int3() {
    return vector;
  }
} Int3;


typedef union uInt3{
  uint3 vector;
  unsigned int array[3];
  HOST_DEVICE uInt3(const uint3& copy):vector(copy){}
  HOST_DEVICE uInt3& operator=(const uint3& copy){
    vector=copy;
    return *this;
  }
  HOST_DEVICE unsigned int& operator[](const int& i){
    return array[i];
  }
  HOST_DEVICE operator uint3() {
    return vector;
  }
} uInt3;


typedef union Double3{
  double3 vector;
  double array[3];
  HOST_DEVICE Double3(const double3& copy):vector(copy){}
  HOST_DEVICE Double3& operator=(const double3& copy){
    vector=copy;
    return *this;
  }
  HOST_DEVICE double& operator[](const int& i){
    return array[i];
  }
  HOST_DEVICE operator double3() {
    return vector;
  }
} Double3;


typedef union Float3{
  float3 vector;
  float array[3];
  HOST_DEVICE Float3(const float3& copy):vector(copy){}
  HOST_DEVICE Float3& operator=(const float3& copy){
    vector=copy;
    return *this;
  }
  HOST_DEVICE float& operator[](const int& i){
    return array[i];
  }
  HOST_DEVICE operator float3() {
    return vector;
  }
} Float3;


