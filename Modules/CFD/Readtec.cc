/*
 *  Readtec.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/cfdlibParticleSet.h>
#include <Datatypes/ParticleSetPort.h>
#include <Geometry/Point.h>
#include <TCL/TCLvar.h>
#include <stdio.h>
#include <string.h>

class Readtec : public Module {
    TCLstring filebase;
    ParticleSetOPort* out;
public:
    Readtec(const clString& id);
    Readtec(const Readtec&, int deep);
    virtual ~Readtec();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_Readtec(const clString& id)
{
    return new Readtec(id);
}
};

Readtec::Readtec(const clString& id)
: Module("Readtec", id, Filter), filebase("filebase", id, this)
{
    // Create the output port
    out=new ParticleSetOPort(this, "ParticleSet", ParticleSetIPort::Atomic);
    add_oport(out);
}

Readtec::Readtec(const Readtec& copy, int deep)
: Module(copy, deep), filebase("filebase", id, this)
{
}

Readtec::~Readtec()
{
}

Module* Readtec::clone(int deep)
{
    return new Readtec(*this, deep);
}

void Readtec::execute()
{
    cfdlibParticleSet* ps=new cfdlibParticleSet();
    for(int i=1;i<100;i++){
	char filename[255];
	sprintf(filename, "%s%02d", filebase.get()(), i);
	FILE* in=fopen(filename, "r");
	if(!in){
	    if(i==1){
		cerr << "Couldn't read file: " << filename << '\n';
	    }
	    break;
	}
	char buf[1000];
	bool found=false;
	int npart;
	double time=-1;
	while(1){
	    if(fgets(buf, 1000, in) == 0){
		break;
	    }
	    char* p=strstr(buf, "T = \"T =");
	    if(p){
		int n=sscanf(p+8, "%lG", &time);
		if(n != 1){
		    cerr << "Error parsing time: " << buf << '\n';
		}
	    }
	    if(strstr(buf, "F=POINT")){
		cerr << "found it: " << buf << '\n';
		char* p=strstr(buf, "I=");
		if(!p){
		    cerr << "Cannot partse line!\n";
		} else {
		    p+=2;
		    cerr << "p=" << p << '\n';
		    int n=sscanf(p, "%d", &npart);
		    cerr << "n=" << n << ", npart=" << npart << '\n';
		    if(n == 1){
			found=true;
			break;
		    }
		}
	    }
	}
	if(found){
	    cfdlibTimeStep* ts=new cfdlibTimeStep;
	    ts->time=time;
	    ts->positions.resize(npart);
	    for(int i=0;i<npart;i++){
		double x,y,z;
		fscanf(in, "%lG", &x);
		fscanf(in, "%lG", &y);
		fscanf(in, "%lG", &z);
		ts->positions[i]=Vector(x,y,z);
		double tmp;
		for(int i=0;i<15;i++)
		    fscanf(in, "%lG", &tmp);
		if(feof(in)){
		    cerr << "Premature end of file!\n";
		}
	    }
	    ps->add(ts);
	} else {
	    cerr << "No points found!\n";
	}
	fclose(in);
    }
    out->send(ParticleSetHandle(ps));
}
