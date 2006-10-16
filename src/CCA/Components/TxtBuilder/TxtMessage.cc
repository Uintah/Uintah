/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
 *  TxtMessage.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   September 2005
 *
 */

#include <CCA/Components/TxtBuilder/TxtMessage.h>
#include <sstream>

using namespace std;
using namespace SCIRun;


TxtMessage::TxtMessage(WINDOW *win){
  this->win=win;
  log.open("txtbuilder.log", fstream::in | fstream::out | fstream::binary | fstream::trunc);
  int prev=-(int)(sizeof(int));
  log.write((char*)&prev, sizeof(int));
  n_msg=0;
  cur_msg=0;
}

TxtMessage::~TxtMessage(){
  log.close();
}

void
TxtMessage::add(const string &msg){
  log.seekp(0, ios_base::end);
  int next=msg.size();
  log.write((char*)&next, sizeof(int));
  log.write(msg.c_str(),msg.size());
  int prev=-(int)(msg.size()+sizeof(int)*3); 
  log.write((char*)&prev, sizeof(int));
  log.seekg(-(int)sizeof(int), ios_base::end);
  cur_msg=++n_msg;
  print(-1,1);
}

void 
TxtMessage::printAll(){
  log.seekg(0, ios_base::beg);
  for(unsigned i=0; i<n_msg; i++){
    int prev, next;
    log.read((char*)&prev, sizeof(int));
    log.read((char*)&next, sizeof(int));
    char *s=new char[next+1];
    log.read(s, next);
    s[next]='\0';
    wprintw(win, s);
    delete []s;
  }
  wrefresh(win);
}

void 
TxtMessage::print(int offset, int num){
  if(offset<0){
    for(int i=0; i<-offset; i++){
      int prev;
      log.read((char*)&prev, sizeof(int));
      log.seekg(prev, ios_base::cur);
      if( --cur_msg==0) break;
    }
  }else{
    for(int i=0; i<offset; i++){
      int prev, next;
      if( cur_msg==n_msg-1) break;
      log.read((char*)&prev, sizeof(int));
      log.read((char*)&next, sizeof(int));
      log.seekg(next, ios_base::cur);
      cur_msg++;
    }
  }
  werase(win);
  for(int i=0; i<num; i++){
    int prev, next;
    if(cur_msg==n_msg) break;
    log.read((char*)&prev, sizeof(int));
    log.read((char*)&next, sizeof(int));
    char *s=new char[next+1];
    log.read(s, next);
    s[next]='\0';
    stringstream os;
    os<<"["<<cur_msg<<"]";
    wprintw(win, os.str().c_str());
    wprintw(win, s);
    wprintw(win, "\n\r");

    cur_msg++;

    delete []s;
  }
  wrefresh(win);
}



