#ifndef SCREEN_H
#define SCEREN_H

namespace Yarden {

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
    
} // End namespace Yarden

#endif
