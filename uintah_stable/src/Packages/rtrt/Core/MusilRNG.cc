/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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



#include <Packages/rtrt/Core/MusilRNG.h>

using namespace rtrt;

MusilRNG::MusilRNG(int seed)
{
    n[0]=4;
    n[1]=7;
    n[2]=10;
    n[3]=15;
    n[4]=6;
    n[5]=13;
    n[6]=1;
    n[7]=8;
    n[8]=12;
    n[9]=9;
    n[10]=0;
    n[11]=14;
    n[12]=5;
    n[13]=11;
    n[14]=2;
    n[15]=3;
    for(int i=0;i<32;i++){
	stab[0][i]=i;
	stab[1][i]=-i;
    }
    a1=0;
    a2=0;
    a3=0;
    a4=0;
    a5=(seed >> 28)&0x0f;
    a6=(seed >> 24)&0x0f;
    a7=(seed >> 20)&0x0f;
    a8=(seed >> 16)&0x0f;
    a9=(seed >> 12)&0x0f;
    a10=(seed >> 8)&0x0f;
    a11=(seed >> 4)&0x0f;
    a12=seed&0x0f;
    b1=b2=b3=b4=b5=b6=b7=b8=b9=b10=b11=b12=0;
    d1=d2=d3=d4=d5=d6=d7=d8=d9=d10=d11=d12=0;
    point=4;
    x2=seed;
}

