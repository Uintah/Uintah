#include <Dataflow/Network/Module.h>
#include <Core/Containers/String.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/GeomCylinder.h>
#include <Core/Datatypes/ScalarFieldRG.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Runnable.h>
#include "Euler.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <stdio.h>

namespace McQ {
using namespace SCIRun;

  class SAMRAIListener: public Runnable {
    public:
      SAMRAIListener(Euler* myModule);
      void run(void);
    private:
      Euler* module;
      clString getLine(void);
      int bufLen,bufCur,geomSock;
      char buf[128];
  };

  SAMRAIListener::SAMRAIListener(Euler* myModule):
    module(myModule), geomSock(-1) { }

  void SAMRAIListener::run(void) {
    GeomGroup* group=new GeomGroup;
    ScalarFieldRGdouble* sf=NULL;

    for(;;) {
//fprintf(stderr,"SAMRAILISTENER LOOP\n");

      if(geomSock==-1) {
        printf("Waiting for Connection on port 4242\n");

        int in=socket(AF_INET,SOCK_STREAM,0);
        if(in<0) perror("socket");
        struct sockaddr_in me;
        bzero(&me,sizeof(me));
        me.sin_family=AF_INET;
        me.sin_addr.s_addr=htonl(INADDR_ANY);
        me.sin_port=htons(4242);
        if(bind(in,&me,sizeof(me))<0) perror("bind");
        if(listen(in,1)<0) perror("listen");
        struct sockaddr_in you;
        bzero(&you,sizeof(you));
        int youSize=sizeof(you);
        if((geomSock=accept(in,&you,&youSize))<0) perror("accept");
        close(in);

        printf("Connected to %lx\n",ntohl(you.sin_addr.s_addr));
      }

      clString line=getLine();
      if(line=="") continue;

      switch(line(0)) {
        case 'f': { //fprintf(stderr,"FRAME\n");
                  ScalarFieldHandle sfh(sf);
                  module->setState(group,sfh);
                  module->want_to_execute();
                  group=new GeomGroup;
                  break; }
        case 'b': { line=line.substr(2,-1);
                  double x1,y1,x2,y2,radius,depth;
                  line.substr(0,line.index(' ')).get_double(x1);
                  line=line.substr(line.index(' ')+1,-1);
                  line.substr(0,line.index(' ')).get_double(y1);
                  line=line.substr(line.index(' ')+1,-1);
                  line.substr(0,line.index(' ')).get_double(x2);
                  line=line.substr(line.index(' ')+1,-1);
                  line.substr(0,line.index(' ')).get_double(y2);
                  line=line.substr(line.index(' ')+1,-1);
                  line.substr(0,line.index(' ')).get_double(depth);
                  line=line.substr(line.index(' ')+1,-1);
                  line.substr(0,line.index(' ')).get_double(radius);
//                  fprintf(stderr,"BOX %f %f %f %f %f %f\n",x1,y1,x2,y2,
//                          depth,radius);
                  group->add(new GeomCylinder(Point(depth,x1,y1),
                                              Point(depth,x2,y1),radius));
                  group->add(new GeomCylinder(Point(depth,x2,y1),
                                              Point(depth,x2,y2),radius));
                  group->add(new GeomCylinder(Point(depth,x2,y2),
                                              Point(depth,x1,y2),radius));
                  group->add(new GeomCylinder(Point(depth,x1,y2),
                                              Point(depth,x1,y1),radius));
                  group->add(new GeomSphere(Point(depth,x1,y1),radius));
                  group->add(new GeomSphere(Point(depth,x2,y1),radius));
                  group->add(new GeomSphere(Point(depth,x1,y2),radius));
                  group->add(new GeomSphere(Point(depth,x2,y2),radius));
                  break;
                  }
        case 'u': { line=line.substr(2,-1);
                  double x1,y1,x2,y2;
                  int nx,ny;
                  line.substr(0,line.index(' ')).get_double(x1);
                  line=line.substr(line.index(' ')+1,-1);
                  line.substr(0,line.index(' ')).get_double(y1);
                  line=line.substr(line.index(' ')+1,-1);
                  line.substr(0,line.index(' ')).get_double(x2);
                  line=line.substr(line.index(' ')+1,-1);
                  line.substr(0,line.index(' ')).get_double(y2);
                  line=line.substr(line.index(' ')+1,-1);
                  line.substr(0,line.index(' ')).get_int(nx);
                  line=line.substr(line.index(' ')+1,-1);
                  line.substr(0,line.index(' ')).get_int(ny);
                  sf=new ScalarFieldRGdouble;
                  sf->set_bounds(Point(0,x1,y1),Point(0.1,x2,y2));
                  sf->set_minmax(0,5);
                  sf->resize(3,nx,ny);
                  for(int z=0;z<3;z++)
                    for(int x=0;x<nx;x++)
                      for(int y=0;y<ny;y++)
                        sf->grid(z,x,y)=-1;
                  break;
                  }
        case 's': { line=line.substr(2,-1);
                  int x1,y1,x2,y2,multx,multy;
                  line.substr(0,line.index(' ')).get_int(x1);
                  line=line.substr(line.index(' ')+1,-1);
                  line.substr(0,line.index(' ')).get_int(y1);
                  line=line.substr(line.index(' ')+1,-1);
                  line.substr(0,line.index(' ')).get_int(x2);
                  line=line.substr(line.index(' ')+1,-1);
                  line.substr(0,line.index(' ')).get_int(y2);
                  line=line.substr(line.index(' ')+1,-1);
                  line.substr(0,line.index(' ')).get_int(multx);
                  line=line.substr(line.index(' ')+1,-1);
                  line.substr(0,line.index(' ')).get_int(multy);
                  for(int y=y1;y<=y2;y++) {
                    line=getLine();
                    for(int x=x1;x<=x2;x++) {
                      double v;
                      line.substr(0,line.index(' ')).get_double(v);
                      line=line.substr(line.index(' ')+1,-1);
                      for(int ymult=0;ymult<multy;ymult++)
                        for(int xmult=0;xmult<multx;xmult++) {
                          sf->grid(0,x*multx+xmult,y*multy+ymult)=v;
                          sf->grid(1,x*multx+xmult,y*multy+ymult)=v;
                          sf->grid(2,x*multx+xmult,y*multy+ymult)=v;
                        }
                    }
                  }
//                  fprintf(stderr,"SCALARFIELD\n");
                  break;
                  }
      }

    }
  }

  clString SAMRAIListener::getLine(void) {
    clString line;
    do {
      if(bufLen==0) {
        bufLen=read(geomSock,&buf,128);
        if(bufLen<0) {
          perror("read");
          close(geomSock);
          geomSock=-1;
          return "";
        }
        bufCur=0;
      }
      if(bufLen==0) {
        fprintf(stderr,"THE DUDE HUNG UP!\n");
        close(geomSock);
        geomSock=-1;
        return "";
      }
      if(buf[bufCur]=='\r') {
        bufCur++;
        bufLen--;
        continue;
      }
      if(buf[bufCur]!='\n') {
        line+=buf[bufCur++];
        bufLen--;
      }
    } while(buf[bufCur]!='\n');
    if(geomSock==-1) return "";
    bufCur++;
    bufLen--;
//    printf("Got Line /%s/\n",line());
    return line;
  }

  //---

  Module* Euler::make(const clString& id) {
    return new Euler(id);
  }

  Euler::Euler(const clString& id): Module("Euler",id,Source),
      execMutex("Euler exec mutex"),curBoxes(NULL),curPressure(NULL) {
    add_oport(boxesOut=new GeometryOPort(this,"Patch Boundaries"));
    add_oport(pressureOut=new ScalarFieldOPort(this,"Pressure"));
    (new Thread(new SAMRAIListener(this),"SAMRAI Listener"))->detach();
  }

  // execMutex prevents clobberage resulting from the dataflow engine and
  // the SAMRAIListener calling execute() at the same time.

  void Euler::execute(void) {
    execMutex.lock();
    fprintf(stderr,"EULER EXEC\n");
    if(curBoxes) {
      boxesOut->delAll();
      boxesOut->addObj(curBoxes->clone(),"Patch Boundaries");
      boxesOut->flushViewsAndWait();
    }
    if(curPressure.get_rep()) {
      pressureOut->send(curPressure);
    }
    execMutex.unlock();
  }

  void Euler::setState(GeomGroup* boxes, ScalarFieldHandle& pressure) {
    fprintf(stderr,"EULER RX STATE\n");
    if(curBoxes) delete curBoxes;
    curBoxes=boxes;
// XXX: I can't ever delete this?
//    if(curPressure) delete curPressure;
    curPressure=pressure;
} // End namespace McQ
  }
