#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <GL/gl.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Containers/Stack.h>
#include <Packages/Yarden/Core/Datatypes/Screen.h>
using std::cerr;
using std::endl;

namespace Yarden {
using namespace SCIRun;
  
  inline
  int min(int a, int b) {return a<b ? a : b ;}
  
  int show = 0;
  
  int Byte2Bit[] =
  {0,
   7,
   6, 0,
   5, 0,0,0,	
   4, 0,0,0,0,0,0,0,
   3, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
   2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
   1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
   0 };


    
    Block LM[9] = {{0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff},
                  {0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff},
		  {0x7f,0x7f,0x7f,0x7f,0x7f,0x7f,0x7f,0x7f},
		  {0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f},
		  {0x1f,0x1f,0x1f,0x1f,0x1f,0x1f,0x1f,0x1f},
		  {0x0f,0x0f,0x0f,0x0f,0x0f,0x0f,0x0f,0x0f},
		  {0x07,0x07,0x07,0x07,0x07,0x07,0x07,0x07},
		  {0x03,0x03,0x03,0x03,0x03,0x03,0x03,0x03},
		  {0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01}};
    Word *LeftMask = (Word *) LM;

  
    Block RM[9] = {{0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff},
                  {0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
		  {0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0},
		  {0xe0,0xe0,0xe0,0xe0,0xe0,0xe0,0xe0,0xe0},
		  {0xf0,0xf0,0xf0,0xf0,0xf0,0xf0,0xf0,0xf0},
		  {0xf8,0xf8,0xf8,0xf8,0xf8,0xf8,0xf8,0xf8},
		  {0xfc,0xfc,0xfc,0xfc,0xfc,0xfc,0xfc,0xfc},
		  {0xfe,0xfe,0xfe,0xfe,0xfe,0xfe,0xfe,0xfe},
		  {0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff}};
    Word *RightMask =(Word *)  RM;

    Block BM[9] = {{0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff},
                   {0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff},
		   {0x00,0xff,0xff,0xff,0xff,0xff,0xff,0xff},
		   {0x00,0x00,0xff,0xff,0xff,0xff,0xff,0xff},
		   {0x00,0x00,0x00,0xff,0xff,0xff,0xff,0xff},
		   {0x00,0x00,0x00,0x00,0xff,0xff,0xff,0xff},
		   {0x00,0x00,0x00,0x00,0x00,0xff,0xff,0xff},
		   {0x00,0x00,0x00,0x00,0x00,0x00,0xff,0xff},
   	           {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xff}};
  
  Word *BottomMask = (Word *) BM;

    Block TM[18] = {{0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff},
                    {0xff,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
                    {0xff,0xff,0x00,0x00,0x00,0x00,0x00,0x00},
                    {0xff,0xff,0xff,0x00,0x00,0x00,0x00,0x00},
                    {0xff,0xff,0xff,0xff,0x00,0x00,0x00,0x00},
                    {0xff,0xff,0xff,0xff,0xff,0x00,0x00,0x00},
                    {0xff,0xff,0xff,0xff,0xff,0xff,0x00,0x00},
                    {0xff,0xff,0xff,0xff,0xff,0xff,0xff,0x00},
                    {0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff}};

  Word *TopMask = (Word *)TM;

  struct MapEntry {
    int r;
    int c;
    
    MapEntry(){}
    MapEntry(int i, int j) : r(i), c(j) {}
  };
  
  struct ScreenEntry {
    int r;
    int c;
    Word active;
    
    ScreenEntry() {}
    ScreenEntry( int i, int j, Word w ) :r(i), c(j), active(w) {}
  };
  
  Stack<MapEntry> map_stack;
  Stack<ScreenEntry> screen_stack;

Screen::Screen()
{
//   int i;
//   printf("LM = \n");
//   for (i=0;i<9; i++)
//     printf("\t0x%llx\n",LeftMask[i]);
//   printf("RM = \n");
//   for (i=0;i<9; i++)
//     printf("\t0x%llx\n",RightMask[i]);
//   printf("BM = \n");
//   for (i=0;i<9; i++)
//     printf("\t0x%llx\n",BottomMask[i]);
//   printf("TM = \n");
//   for (i=0;i<9; i++)
//     printf("\t0x%llx\n",TopMask[i]);
}  

int
Screen::scan( Pt *_pt, int ne, int edge_list[], int double_edges[] )
{
  static int counter = 0;
  counter++;
  pt = _pt;
  
  init( ne, edge_list, double_edges );
  
  int changed = scan();
  
  return changed;
  }
  
  
inline void
Screen::add_edge( int from, int to, int depth, ScanEdge *&e, int &ne )
{
  int ymin, ymax;
  double x, dx, dy, yfrac;
  
  double y = pt[from].y;
  double y1 = pt[to].y;
  if ( y < y1 ) {
    ymin = (int) floor(y);
    ymax = (int) floor(y1);
    yfrac = 1+ymin-y;
    x = pt[from].x;
    dx = pt[to].x - x;
    dy = y1 - y;
    if ( depth > 0 ) {
      e->type =  RightEdge;
      e->depth = -depth; 
    }
    else {
      e->type = LeftEdge;
      e->depth = -depth;
    }
  }
  else {
    ymin = (int) floor(y1);
    ymax = (int) floor(y);
    yfrac = 1+ymin-y1;
    x = pt[to].x;
    dx = pt[from].x - x;
    dy = y - y1;
    if ( depth > 0 ) {
      e->type = LeftEdge;
      e->depth = depth;
    }
    else {
      e->type = RightEdge;
      e->depth = depth;
    }
  }
  
  double xi;
  double si;
  double sf;
 
  if ( dy != 0 && ymin != ymax ) {
    double alpha = (x-floor(x))*dy + dx*yfrac;
    double beta = floor(alpha/dy);
    xi = floor(x)+beta;
    
    if ( dy >= 1 ) {
      si = floor(dx/dy);
      sf = dx - si*dy;
      
      e->r = alpha - beta*dy + sf -dy;
      e->si = int(si);
      e->inc = sf;
      e->dec = sf - dy;
    }
    e->xi = int(xi);
    
    e->ymin = ymin;
    e->ymax = ymax;
    if ( YMAX < ymax ) {
      YMAX = ymax;
      if (YMAX>=512 ) YMAX = 511;
    }
    e++;
    ne++;
  }
}

void
Screen::init( int n, int *edge_list, int *edge_depth )
{
//     glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
//     glColor3f(1,0,0);
//     glBegin(GL_TRIANGLES);
//     for ( int q=2; q<n; q++){
//       glVertex2i( pt[edge_list[0]].x, pt[edge_list[0]].y );
//       glVertex2i( pt[edge_list[q-1]].x, pt[edge_list[q-1]].y );
//       glVertex2i( pt[edge_list[q]].x, pt[edge_list[q]].y );
//     }
//     glEnd();
//     glFlush();
//     glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
  

  YMAX = 0;
  ScanEdge *e = edges;
  int ne = 0;
  for (int i=1; i<n; i++) {
    add_edge( edge_list[i-1], edge_list[i], edge_depth[i], e, ne);
    if ( edge_depth[i] != edge_depth[i+1]) 
      add_edge( edge_list[i], edge_list[0], 2*edge_depth[i], e, ne );
  }
  add_edge( edge_list[n-1], edge_list[0], edge_depth[0], e, ne);

  ET_size = ne;
  ET[0] = 0;
  for ( int i=1; i<ne; i++ ) {
    int j = i-1;
    int y = edges[i].ymin;
    for ( ; j>=0 && edges[ET[j]].ymin > y; j--)
      ET[j+1] = ET[j];
    ET[j+1] = i;
  }
}


inline void
Screen::edge_update( ScanEdge *e )
{
  if ( e->r >= 0 ) {
    e->xi += e->si +1;
    e->r += e-> dec;
  }
  else {
    e->xi += e->si;
    e->r += e->inc;
  }
}

inline void
Screen::fill( int y, int from, int to, Byte &updated)
{
  if ( y < 0 || (y>>3)>63)
    cerr << "BUG: y = " << y << " (" << (y>>3) <<")" << endl;

  if (from < 0 ) from = 0;
  if (to > 511 ) to = 511;
  int from_byte = from>>3;
  int to_byte = to>>3;

  Byte *byte = &screen[y>>3][from_byte][y&0x7];

  Byte left = FullByte>>(from & 0x7);
  Byte right = ~(FullByte >> (1+(to & 0x7)));

  if ( from_byte < to_byte ){
    updated |= (~*byte & left); 
    *byte |= left;
    for (from_byte++,byte+=8; from_byte<to_byte; from_byte++,byte+=8) {
      updated |= ~*byte;

      *byte = 0xff;
    }
    updated |= ~*byte & right;
    *byte |= right;
  }
  else if ( from_byte == to_byte ) {
    Byte mask = left & right;
    updated |= ~*byte & mask;
    *byte |= mask;
  }
//   else
//     cerr <<"BUG  from_byte "<<from_byte<<" ("<<from<<") > to_byte "
// 	 <<to_byte<<"  (" <<to << "}" << endl;

}

inline void
Screen::fill( int y, int from, int to )
{
  //  glFlush();
  glColor3f(1,1,1);
  glBegin(GL_LINES);
  glVertex2i( from, y);
  glVertex2i( to+1, y);
  glEnd();
  //  glFlush();
}

void
show_box( int left, int right, int top, int bottom )
{
  //  glFlush();
  glColor3f(1,1,1);
  glBegin(GL_LINE_STRIP);
  glVertex2i( left, bottom );
  glVertex2i( right, bottom );
  glVertex2i( right, top );
  glVertex2i( left, top);
  glVertex2i( left, bottom);
  glEnd();
  glFlush();
}

inline void
Screen::check( int y, int X, int left, int right, Row &row)
{
  int mask = 0;
  for ( int c=left; c<=right; c++) {
    if ( B2W(row[X*8+c]) == FullWord )
      mask |= 0x80 >> c;
  }
  Block &block = map[y>>6][X];
  Byte &change = block[(y>>3)&0x7];
  if ( (~change) & mask) {
    change |= mask;
    if ( B2W(block) == FullWord ) {
      root[y>>6] |=  0x80 >> X;
    }
  }
}

void
Screen::draw( int y, int &changed)
{
  Byte updated = 0;
  //int depth = 0;
  
  for ( int i = 0 ; i<AET_size; i++ ) {
    ScanEdge *edge = &edges[AET[i]];
    int xi = edge->xi+1;
    int depth = edge->depth;
    while ( depth && i < AET_size-1) {
      i++;
      depth += edges[AET[i]].depth;
    }
    int to = edges[AET[i]].xi;
    if ( xi <= to ) {
      //fill( y, xi, to );
      fill( y, xi, to, updated );
      if ( show ) display();
    }
  }

  if ( updated ) {
    int left_byte = (edges[AET[0]].xi) >> 3;;
    int right_byte = (edges[AET[AET_size-1]].xi) >> 3;
    if ( left_byte < left ) left = left_byte;
    if ( right_byte > right ) right = right_byte;
  }

  changed = changed || updated;
  
  if ( (y&0x7 == 7 || y == YMAX) && changed ) {
    int L = left>>3;
    int R = right>>3;
    //    int Y = y >> 6;
    Row &row = screen[y>>3];
    if (  L == R ) {
      check( y, L, left&0x7, right&0x7, row );
    }
    else {
      check( y, L, left&0x7, 7, row);
      for ( L++; L<R; L++)
	check( y, L, 0, 7,  row );
      check( y, L, 0, right&0x7, row );
    }
//     changed = 0;
//     left = 512;
//     right = -1;
  }
}



int
Screen::scan()
{
  int y = edges[ET[0]].ymin+1; // floor(edges[ET[0]].ymin)+1;
  int et = 0;
  AET_size = 0;
  int changed = 0;
  left = 512;
  right = -1;
  while ( y <= YMAX ) { //et < ET_size || AET_size) {
    
    // add new edges
    for ( ; et < ET_size && edges[ET[et]].ymin < y; et++ ) {
      int xi = edges[ET[et]].xi;
      int i=AET_size-1;
      for (; i>=0 && edges[AET[i]].xi > xi; i--)
	AET[i+1] = AET[i];
      AET[i+1] = ET[et];
      AET_size++;
    }

    if ( y >= 0 ) draw( y, changed );

    // update edges
    for (int i=0; i<AET_size; i++ )
      edge_update( &edges[AET[i]] );

    // reorder and remove
    y++;
    
    int size = 0;
    for ( int i=0; i<AET_size; i++) {
      int next = AET[i];
      if ( edges[next].ymax < y )
	continue;
      int xi = edges[next].xi;
      int j=size-1;
      for (; j>=0 && edges[AET[j]].xi > xi ; j--)
	AET[j+1] = AET[j];
      AET[j+1] = next;
      size++;
    }
    AET_size = size;
  }

  return changed;
}




void
Screen::setup( int xres, int yres )
{
  row_size = xres/8;
  col_size = yres/8;
  screen_size = row_size*col_size;


  clear();
}

void
Screen::clear()
{
  for (int i=0; i<64; i++) 
    for (int j=0; j<64; j++)
      for (int r=0; r<8; r++)
	screen[i][j][r] = 0;
  for (int i=0; i<8; i++)
    for (int j=0; j<8; j++)
      for (int c=0; c<8; c++)
	map[i][j][c] = 0;
  for (int i=0; i<8; i++ )
    root[i] = 0;
}


inline int
Screen::visible_map( int row, int col,
		      int left, int right, int top, int bottom)
{
  int lb = col == (left>>6);
  int rb = col == (right>>6);
  int bb = row == (bottom>>6);
  int tb = row == (top>>6);

  // add 1 since the *Mask tables starts at 1.
  // the case for entry 0 is an ignored.
  int ml = (((left>>3)&0x7)+1)  *lb;
  int mr = (((right>>3)&0x7)+1) *rb;
  int mt = (((top>>3)&0x7)+1)   *tb;
  int mb = (((bottom>>3)&0x7)+1)*bb;
  
  Word entry = B2W(map[row][col]);
  
  Word cover = LeftMask[ml] & RightMask[mr] & TopMask[mt] & BottomMask[mb];
  if ( !(~entry & cover) )     
    return 0; // not visible
  
  Word inner = LeftMask[ml+lb] & RightMask[mr-rb] &
    TopMask[mt-tb] & BottomMask[mb+bb];
  if ( ~entry & inner ) {
//     {if (show) printf("visible - map\n");}
    return 1; // visible
  }

  // unknown - check the boundary (cover - inner) at the screen level
  Word active =  (cover ^ inner) & ~entry;
  screen_stack.push( ScreenEntry(row, col, active) );
 
  return 0;
}

inline int
Screen::visible_screen( int row, int col, 
			 int left, int right, int top, int bottom)
{
  int sl = ((left&0x7)+1)  * (col == (left>>3));
  int sr = ((right&0x7)+1) * (col == (right>>3));
  int st = ((top&0x7)+1)   * (row == (top>>3));
  int sb = ((bottom&0x7)+1)* (row == (bottom>>3));
  
  Word entry = B2W(screen[row][col]);
  
  Word cover = LeftMask[sl] & RightMask[sr] & TopMask[st] & BottomMask[sb];
  return  (~entry & cover) != 0;
}

int visible_counter = 0;
int
Screen::visible1( int left, int bottom, int right , int top )
{
  int r;
  if ( visible_counter++ > 0 ) {
    if ( show) 
      fprintf( stderr, ">>Screen: visible %d at (%d %d %d %d)\n",
	       visible_counter, left,right,bottom,top);
    glLogicOp(GL_XOR);
    show_box( left, right, top, bottom );
    r = visible( left, bottom, right, top );
    //  validate();
    if ( !r )
      {if (show) printf("not visible\n");}
    else {
      {if (show) printf("    visible\n");}
    }
    show_box( left, right, top, bottom );
    glLogicOp(GL_COPY);
  }
  else
    r = visible( left, bottom, right, top );

//   fprintf( stderr, "<<\n");
  return r;
}

int
Screen::visible( int left, int bottom, int right, int top )
{
  if ( left > 511 || right < 0 || top < 0 || bottom > 511 )
    return 0;
  
  // check at top level
  
//   top += 1;
//   bottom -=1;
//   right+=1;
//   left -=1;

  if (top > 511) top = 511;
  if ( bottom < 0 ) bottom = 0;
  if ( right > 511 ) right = 511;
  if ( left < 0 ) left = 0;

  int loff = ( left >= 0);
  int roff = ( right <= 511 );
  int toff = ( top <= 511 );
  int boff = ( bottom >= 0 );
  
//   int tl = ((left >> 6)+1)   * loff;
//   int tr = ((right >> 6)+1)  * roff; 
//   int tt = ((top >> 6)+1)    * toff; 
//   int tb = ((bottom >> 6)+1) * boff;

  int tl,tr,tt,tb;

  if ( left < 0 ) tl = 0;
  else tl = min((left>>6)+1, 8 );

  if ( right > 511 ) tr = 8;
  else tr = min((right>>6)+1,8);

  if ( top > 511 ) tt = 8;
  else tt = min((top>>6)+1,8);

  if ( bottom < 0 ) tb = 0;
  else tb = min((bottom>>6)+1,8);

  if ( left < 0 ) left = 0;
  if ( right > 511) right = 511;
  if ( top > 511) top = 511;
  if( bottom < 0 ) bottom = 0;

  Word root_mask = B2W(root);
  Word cover = (LeftMask[tl] & RightMask[tr] &
		TopMask[tt] & BottomMask[tb]);

  if ( !(~root_mask & cover) ) {
    //    if ( show ) { printf("not visible - root\n"); scanf("%*c"); }
    return 0; // not visible
  }
  
  Word inner = (LeftMask[tl+loff] & RightMask[tr-roff] &
		TopMask[tt-toff] & BottomMask[tb+boff]);

  if ( ~root_mask & inner ) {
    //    { if (show)printf("visible - root\n");}
    return 1; // visible
  }

  Word active_mask =  (cover ^ inner) & ~root_mask;
  if ( active_mask ) {
    Block &active = W2B(active_mask);
    map_stack.remove_all();
    screen_stack.remove_all();
    
    for (int r=tb-boff; r<=tt-toff; r++) {
      Byte b = active[r];
      while (b) {
	Byte bit = b & (~b+1);
	int c = Byte2Bit[bit];
	map_stack.push( MapEntry(r, c) );
	b ^= bit;
      }
    }

    while ( !map_stack.empty() ) {
      MapEntry entry = map_stack.pop();
      if (visible_map( entry.r, entry.c, left, right, top, bottom)) {
	// 	{ if (show) printf(" visible - map\n");}
	return 1;
      }
    }

    while ( !screen_stack.empty() ) {
      ScreenEntry entry = screen_stack.pop();
      int r = entry.r << 3;
      int c = entry.c << 3;
      Block &active = W2B(entry.active);
      for ( int y=0; y<8; y++) {
	Byte b = active[y];
	while (b) {
	  Byte bit = b & (~b+1);
	  int x = Byte2Bit[bit];
	  if ( visible_screen( r+y, c+x, left, right, top, bottom )) {
	    // 	    if ( show ) printf(" visible - screen\n");
	    return 1;
	  }
	  b ^= bit;
	}
      }
    }
  }
  //   if ( show ) printf("not visible - root end\n");
  return 0;
}

int
Screen::validate()
{
  fprintf( stderr, ">> validate\n");
  int ok = 1;
  for (int i=0; i<8; i++) {
    for ( int j=0; j<8; j++ ) {
      if ( root[i] & (0x80>>j)) {
	if ( B2W(map[i][j]) != FullWord )
	  {ok=0,printf("not full: root[%d,%d]\n", i, j);}
      }
      else
	if ( B2W(map[i][j]) == FullWord )
	  {ok=0,printf(" is full: root[%d,%d]\n", i, j);}
      for ( int r=0; r<8; r++ ) {
	for ( int c=0; c<8; c++){
	  if ( map[i][j][r] & (0x80>>c) ) {
	    if ( B2W(screen[i*8+r][j*8+c]) != FullWord  )
	      { ok=1,printf(" not full: map[%d  %d %d,%d]\n", i, j, r,c);}
	  }
	  else
	    if ( B2W(screen[i*8+r][j*8+c]) == FullWord )
	      {ok=0,printf(" is full: map[%d  %d %d,%d]\n", i, j, r,c);}
	}
      }
    }
  }
  if ( !ok ) {
    fprintf(stderr,"bug \n");
    scanf("%*c");
  }
  fprintf( stderr, " << validate\n");
  return ok;
}

void 
Screen::examine()
{
  for (int i=1; i<65; i++) {
    int at = i*8;
    // line of points;
    glColor3f( 1, 1, 1);
    glBegin(GL_POINTS);
    for (int j=0; j<i; j++) 
      glVertex2i(10+j,  at);
    glEnd();
    // line 
    glColor3f( 0, 1, 1);
    glBegin(GL_LINES);
    glVertex2i(10, at+1  );
    glVertex2i(10+i, at+1 );
    glEnd();

    glColor3f( 0, 1, 0);
    glBegin(GL_LINES);
    glVertex2i(10, at+2  );
    glVertex2i(10+i, at+2 );
    glEnd();

    glColor3f( 1, 0, 1);
    glBegin(GL_LINES);
    glVertex2i(10, at+3  );
    glVertex2i(10+i, at+3 );
    glEnd();
    // polygon
    glColor3f(1,0,0); // red
    glBegin(GL_POLYGON);
    glVertex2i(10,   at+4);
    glVertex2i(10+i, at+4);
    glVertex2i(10+i, at+6);
    glVertex2i(10,   at+6);
    glEnd();
  }

  scanf("%*c");
}
    
  
	
void
Screen::display()
{
  for (int i=0; i<8; i++) {
    //     printf("root: 0x%x\n",root[i]);
    if ( root[i] == 0xff ) {
      glColor3f(1,0,0); // red
      glBegin(GL_POLYGON);
      glVertex2i(0,   i*64);
      glVertex2i(512, i*64);
      glVertex2i(512, i*64+64);
      glVertex2i(0,   i*64+64);
      glEnd();
    }
    else {
      for ( int j=0; j<8; j++ ) {
  	if ( root[i] & (0x80>>j)) {
  	  glColor3f(0.5, 0, 0);
  	  glBegin(GL_POLYGON);
  	  glVertex2i(j*64,    i*64);
  	  glVertex2i(j*64+64, i*64);
  	  glVertex2i(j*64+64, i*64+64);
  	  glVertex2i(j*64,    i*64+64);
  	  glEnd();
  	}
  	else {
	  for ( int r=0; r<8; r++ ) {
	    if ( map[i][j][r] == 0xff ) {
	      glColor3f(0,1,0);
	      glBegin(GL_POLYGON);
	      glVertex2i(j*64,    i*64+r*8);
	      glVertex2i(j*64+64, i*64+r*8);
	      glVertex2i(j*64+64, i*64+r*8+8);
	      glVertex2i(j*64,    i*64+r*8+8);
	      glEnd();
	    }
	    else {
	      for (int c=0; c<8; c++ ) {
		if ( map[i][j][r] & (0x80>>c)) {
		  glColor3f( 0, 0.5, 0);
		  glBegin(GL_POLYGON);
		  glVertex2i(j*64+c*8,   i*64+r*8);
		  glVertex2i(j*64+c*8+8, i*64+r*8);
		  glVertex2i(j*64+c*8+8, i*64+r*8+8);
		  glVertex2i(j*64+c*8,   i*64+r*8+8);
		  glEnd();
		}
		else {
		  for (int s=0; s<8; s++) {
		    if ( screen[i*8+r][j*8+c][s] == 0xff ) {
		      glColor3f( 0, 1, 1);
		      glBegin(GL_LINES);
		      glVertex2i(j*64+c*8,   i*64+r*8+s+1);
		      glVertex2i(j*64+c*8+8, i*64+r*8+s+1);
		      glEnd();
		    }
		    else {
		      glColor3f(1,1,0);
		      glBegin(GL_POINTS);
		      for (int t=0; t<8; t++)
			if ( (screen[i*8+r][j*8+c][s] & (0x80>>t))) 
			  glVertex2i(j*64+c*8+t,   i*64+r*8+s);
		      glEnd();
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
  glFlush();
}

	
int
Screen::cover_pixel( int x, int y )
{
  int Y = y>>3;
  int X = x>>3;
  if ( Y<0 || Y>63 ) {
    cerr << "CoverPixel err at " << x << " " << y << endl;
    return 0;
  }    
  if ( X<0 || X>63 ) {
    cerr << "CoverPixel err at " << x << " " << y << endl;
    return 0;
  }    
  Byte &byte = screen[y>>3][x>>3][y&0x7];
  Byte mask = Byte(0x80) >> (x&0x7);
  if ( byte & mask )
    return 0;

  if ( cover_flag )
    byte |= mask;
  Row &row = screen[y>>3];
  int bit = (x>>3)&0x7;
  check( y, x>>6, bit, bit, row );
  return 1;
  }
} // End namespace Yarden
  
