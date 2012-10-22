/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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


#include <CCA/Ports/SFC.h>
#include <climits>

namespace Uintah
{

int dir3[8][3]={
	{-1,1,1},
	{-1,1,-1},
	{-1,-1,1},
	{-1,-1,-1},
	{1,1,1},
	{1,1,-1},
	{1,-1,1},
	{1,-1,-1},
};

int dir2[4][3]={
	{1,1},
	{-1,1},
	{1,-1},
	{-1,-1},
};

int dir1[2][3]={
  {-1},
  {1}
};
  

int orient1[1][8]={{0,0}};
int order1[1][8]={{0,1}};
int inv1[1][8]={{0,1}};

  /********************Morton Curve***************************/
int morient3[1][8]={{0}};
int morient2[1][8]={{0}};
int morder3[1][8]={{0,1,2,3,4,5,6,7}};
int morder2[1][8]={{0,1,2,3}};
int minv3[1][8]={{0,1,2,3,4,5,6,7}};
int minv2[1][8]={{0,1,2,3}};


  /********************Hilbert Curve***************************/
#define hsize2 4
#define hsize3 24

int hinv3[hsize3][8]={
    {0,1,3,2,7,6,4,5},
    {0,7,3,4,1,6,2,5},
    {0,1,7,6,3,2,4,5},
    {2,1,5,6,3,0,4,7},
    {4,7,3,0,5,6,2,1},
    {4,5,3,2,7,6,0,1},
    {6,1,5,2,7,0,4,3},
    {0,3,7,4,1,2,6,5},
    {2,3,5,4,1,0,6,7},
    {6,7,5,4,1,0,2,3},
    {0,3,1,2,7,4,6,5},
    {2,5,3,4,1,6,0,7},
    {4,3,5,2,7,0,6,1},
    {4,3,7,0,5,2,6,1},
    {6,5,7,4,1,2,0,3},
    {0,7,1,6,3,4,2,5},
    {2,5,1,6,3,4,0,7},
    {6,5,1,2,7,4,0,3},
    {2,3,1,0,5,4,6,7},
    {4,5,7,6,3,2,0,1},
    {4,7,5,6,3,0,2,1},
    {6,7,1,0,5,4,2,3},
    {2,1,3,0,5,6,4,7},
    {6,1,7,0,5,2,4,3}
  };
int hinv2[hsize2][8]={
    {0,1,3,2},
    {0,3,1,2},
    {2,1,3,0},
    {2,3,1,0}
  };

int horder2[hsize2][8]={
    {0,1,3,2},
    {0,2,3,1},
    {3,1,0,2},
    {3,2,0,1}
  };
int horder3[hsize3][8]={
    {0,1,3,2,6,7,5,4},
    {0,4,6,2,3,7,5,1},
    {0,1,5,4,6,7,3,2},
    {5,1,0,4,6,2,3,7},
    {3,7,6,2,0,4,5,1},
    {6,7,3,2,0,1,5,4},
    {5,1,3,7,6,2,0,4},
    {0,4,5,1,3,7,6,2},
    {5,4,0,1,3,2,6,7},
    {5,4,6,7,3,2,0,1},
    {0,2,3,1,5,7,6,4},
    {6,4,0,2,3,1,5,7},
    {5,7,3,1,0,2,6,4},
    {3,7,5,1,0,4,6,2},
    {6,4,5,7,3,1,0,2},
    {0,2,6,4,5,7,3,1},
    {6,2,0,4,5,1,3,7},
    {6,2,3,7,5,1,0,4},
    {3,2,0,1,5,4,6,7},
    {6,7,5,4,0,1,3,2},
    {5,7,6,4,0,2,3,1},
    {3,2,6,7,5,4,0,1},
    {3,1,0,2,6,4,5,7},
    {3,1,5,7,6,4,0,2},
  };

int horient2[hsize2][8]={
    {1,0,0,2},
    {0,1,1,3},
    {3,2,2,0},
    {2,3,3,1}
  };
int horient3[hsize3][8]={
    {1,2,0,3,4,0,5,6},
    {0,7,1,8,5,1,4,9},
    {15,0,2,22,20,2,19,23},
    {20,6,3,23,15,3,16,22},
    {22,13,4,12,11,4,1,20},
    {11,19,5,20,22,5,0,12},
    {9,3,6,2,21,6,17,0},
    {10,1,7,11,12,7,13,14},
    {12,9,8,14,10,8,18,11},
    {6,8,9,7,17,9,21,1},
    {7,15,10,16,13,10,12,17},
    {5,14,11,9,0,11,22,8},
    {8,20,12,19,18,12,10,5},
    {18,4,13,5,8,13,7,19},
    {17,11,14,1,6,14,23,7},
    {2,10,15,18,19,15,20,21},
    {19,17,16,21,2,16,3,18},
    {14,16,17,15,23,17,6,10},
    {13,21,18,17,7,18,8,16},
    {16,5,19,4,3,19,2,13},
    {3,12,20,13,16,20,15,4},
    {23,18,21,10,14,21,9,15},
    {4,23,22,6,1,22,11,3},
    {21,22,23,0,9,23,14,2},
  };

  /*********************Grey Curve***********************/
#define gsize2 2
#define gsize3 4

int ginv3[gsize3][8]={
    {0,1,3,2,7,6,4,5},
    {6,7,5,4,1,0,2,3},
    {2,3,1,0,5,4,6,7},
    {4,5,7,6,3,2,0,1}
  };
int ginv2[gsize2][8]=
  {
    {0,1,3,2},
    {2,3,1,0}
  };
int gorder2[gsize2][8]={
    {0,1,3,2},
    {3,2,0,1}
  };
int gorder3[gsize3][8]={
    {0,1,3,2,6,7,5,4},
    {5,4,6,7,3,2,0,1},
    {3,2,0,1,5,4,6,7},
    {6,7,5,4,0,1,3,2}
  };
int gorient2[gsize2][8]={
    {0,1,1,0},
    {1,0,0,1}
  };
int gorient3[gsize3][8]={
    {0,1,2,3,3,2,1,0},
    {1,0,3,2,2,3,0,1},
    {2,3,0,1,1,0,3,2},
    {3,2,1,0,0,1,2,3}
  };




}  //end namespace
