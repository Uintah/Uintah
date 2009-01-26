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



#include <cstdio>
#include <cstdlib>
#include <cmath>

int main(int argc, char* argv[])
{
    char out[200];
    sprintf(out, "xxmc.4");
    FILE* ofile=fopen(out, "w");
    float data[2][2][2];
    data[0][0][0]=100;
    data[0][0][1]=101;
    data[0][1][0]=101;
    data[0][1][1]=100;
    data[1][0][0]=101;
    data[1][0][1]=101;
    data[1][1][0]=101;
    data[1][1][1]=100;
    fwrite(data, sizeof(float), 8, ofile);
    fclose(ofile);
    char buf[100];
    sprintf(buf, "%s.hdr", out);
    FILE* hdr=fopen(buf, "w");
    fprintf(hdr, "2 2 2\n");
    fprintf(hdr, "-1 -1 -1\n");
    fprintf(hdr, "1 1 1\n");
    fprintf(hdr, "100 101\n");
    fclose(hdr);
}
