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

#ifndef Screen_h
#define Sceren_h

namespace SCIRun {

typedef unsigned long long Word;
typedef unsigned char Byte;
typedef Byte Block[8];      // 1 word, 64 bits

typedef Block Row[64];      // 64 blocks per 8 rows 


enum ScanEdgeType { LeftEdge, RightEdge};

struct Pt {
  double x,y;
};

struct ScanEdge {
  ScanEdgeType type;
  double from, to;
  double r;
  int si;
  int ymin, ymax;
  double inc, dec;
  int xi;
  ScanEdge *next, *prev;
  int depth;
};

class Screen {
public:
  Word pad;
  Row screen[64];
  Word pad1;
  Block map[8][8];
  Word pad2;
  Block root;
  Word pad3;
  int row_size, col_size, screen_size;
  int left, right;
  
  ScanEdge edges[30];
  int ET_size, ET[30];
  int AET_size, AET[30];
  Pt *pt;
  int YMAX;
  
public:
  Screen();
  ~Screen() {}
  
  int scan( Pt *_pt, int ne, int edge_list[], int edge_depth[] );
  void setup( int, int );
  int visible( int, int, int, int );
  int visible1( int, int, int, int );
  void display();
  void clear();
  int cover_pixel( int, int);
  void examine();
private:
  void init(int n, int *, int * );
  int scan();
  void add_edge( int from, int to, int depth, ScanEdge *&e, int &ne );
  
  void edge_update( ScanEdge *);
  void draw( int, int & );
  void fill( int, int, int);
  void fill( int, int, int, Byte & );
  //void check( int, int, int, int, int);
  void check( int, int, int, int, Row &);
  int visible_map( int, int, int, int, int, int);
  int visible_screen( int, int, int, int, int, int);
  int validate();
  
};

} // namespace SCIRun

#endif // Screen_h
