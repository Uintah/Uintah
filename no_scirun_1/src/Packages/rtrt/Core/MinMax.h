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



#ifndef MINMAX_H
#define MINMAX_H 1

namespace rtrt {

inline double Min(double x, double y)
{
    return x<y?x:y;
}

inline double Max(double x, double y)
{
    return x>y?x:y;
}

inline double Min(double x, double y, double z)
{
    return x<y?x<z?x:z:y<z?y:z;
}

inline double Max(double x, double y, double z)
{
    return x>y?x>z?x:z:y>z?y:z;
}

inline float Min(float x, float y)
{
    return x<y?x:y;
}

inline float Max(float x, float y)
{
    return x>y?x:y;
}

inline short Min(short x, short y)
{
    return x<y?x:y;
}

inline short Max(short x, short y)
{
    return x>y?x:y;
}

inline int Min(int x, int y)
{
    return x<y?x:y;
}

inline int Max(int x, int y)
{
    return x>y?x:y;
}

} // end namespace rtrt

#endif
