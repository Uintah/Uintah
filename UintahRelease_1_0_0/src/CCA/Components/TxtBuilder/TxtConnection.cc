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


/*
 *  TxtConnection.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   September 2005
 *
 */

#include <CCA/Components/TxtBuilder/TxtConnection.h>
#include <curses.h>
#include <panel.h>

using namespace SCIRun;

TxtConnection::TxtConnection(WINDOW *win_main){
  win=win_main;
}

void 
TxtConnection::resetPoints(Rect rUse, Rect rProvide)
{
  Point P = rUse.upoint();
  Point R = rProvide.ppoint();

    int t = 1;
    int h = 1;
    int mid;
  
    if ( (P.x + h) < (R.x - h) ) {
        mid = (P.y + R.y) / 2;
        int xm = (P.x + R.x) / 2;
        pa[0] = Point( P.x,     P.y);
        if (P.y <= mid) {
            pa[1] = Point( xm, pa[0].y );
        } else {
            pa[1] = Point( xm, pa[0].y );
        }
        pa[2] = Point( pa[1].x, R.y );
        pa[3] = Point( R.x, pa[2].y );
	n_pts=4;
    } else {
        if (rUse.top() > rProvide.bottom() + 2 * t) {
            mid = (rUse.top() + rProvide.bottom()) / 2;
        } else if (rProvide.top() > rUse.bottom() + 2 * t) {
            mid = (rUse.bottom() + rProvide.top()) / 2;     
        } else {
            mid = rUse.top() < rProvide.top() ? rUse.top() : rProvide.top();      
            mid -= 2 * t;
        }
        pa[0] = Point( P.x, P.y );
        if (P.y < mid) {
            pa[1] = Point( P.x + h , pa[0].y );
        } else {
            pa[1] = Point( P.x + h , pa[0].y );
        }
        if (P.x + h < R.x - h) {
            pa[2] = Point( pa[1].x, mid  );
        } else {
            pa[2] = Point( pa[1].x, mid  );
        }
        if (R.y > mid) {
            pa[3] = Point( R.x - h , pa[2].y );
        } else {
            pa[3] = Point( R.x - h , pa[2].y );
        }
        pa[4] = Point( pa[3].x, R.y );
        pa[5] = Point( R.x,     pa[4].y );
	n_pts=6;
    }
}

void 
TxtConnection::draw()
{
  TxtDirection d1=N, d2=N;
  for(int i=1; i<n_pts; i++){
    Point s=pa[i-1];
    Point e=pa[i];
    if(s.x==e.x && s.y==e.y) continue;
    int step;
    if(s.y==e.y){
      //horizontal
      step=e.x>s.x?1:-1;
      for(int x=s.x; x!=e.x; x+=step){
	//skip the corner
	if(i!=1 && x==s.x)continue;
	mvwaddch(win, s.y, x, cline(L,L));
      }
      d1=d2;
      d2=step>0? R: L;
    }else{
      //vertical
      step=e.y>s.y?1:-1;
      for(int y=s.y; y!=e.y; y+=step){
	//skip the corner
	if(i!=1 && y==s.y)continue;
	mvwaddch(win, y, s.x, cline(U,U));
      }
      d1=d2;
      d2=step>0? D: U;
    }
    if(i>1){
      //draw corner
      mvwaddch(win, s.y, s.x, cline(d1, d2));
    }
  }
  //do not call wrefresh(win);
  update_panels();
  doupdate();
}


void 
TxtConnection::erase()
{
  TxtDirection d1=N, d2=N;
  for(int i=1; i<n_pts; i++){
    Point s=pa[i-1];
    Point e=pa[i];
    if(s.x==e.x && s.y==e.y) continue;
    int step;
    if(s.y==e.y){
      //horizontal
      step=e.x>s.x?1:-1;
      for(int x=s.x; x!=e.x; x+=step){
	//skip the corner
	if(i!=1 && x==s.x)continue;
	mvwaddch(win, s.y, x, ' ');
      }
      d1=d2;
      d2=step>0? R: L;
    }else{
      //vertical
      step=e.y>s.y?1:-1;
      for(int y=s.y; y!=e.y; y+=step){
	//skip the corner
	if(i!=1 && y==s.y)continue;
	mvwaddch(win, y, s.x, ' ');
      }
      d1=d2;
      d2=step>0? D: U;
    }
    if(i>1){
      //draw corner
      mvwaddch(win, s.y, s.x, ' ');
    }
  }
  //do not call wrefresh(win);
  update_panels();
  doupdate();
}


int 
TxtConnection::cline(TxtDirection d1, TxtDirection d2){
  if((d1==U||d1==D) && (d2==U || d2==D)) return ACS_VLINE;
  if((d1==L||d1==R) && (d2==L || d2==R)) return ACS_HLINE;
  if(d1==U && d2==R) return ACS_ULCORNER; 
  if(d1==U && d2==L) return ACS_URCORNER;
  if(d1==D && d2==R) return ACS_LLCORNER; 
  if(d1==D && d2==L) return ACS_LRCORNER;
  if(d1==L && d2==U) return ACS_LLCORNER; 
  if(d1==L && d2==D) return ACS_ULCORNER;
  if(d1==R && d2==U) return ACS_LRCORNER; 
  if(d1==R && d2==D) return ACS_URCORNER;
  return '*'; //sth wrong if this happens
}

