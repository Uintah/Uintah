/*
 * main.c - Does everything
 *
 *
 */

#include <Classlib/String.h>
#include <Classlib/Pstreams.h>
#include <Datatypes/ScalarFieldRGdouble.h>
#include <Datatypes/VectorFieldRG.h>
#include <Datatypes/ScalarFieldZone.h>
#include <Datatypes/VectorFieldZone.h>
#include <Classlib/Array3.h>
#include <iostream.h>
#include <fstream.h>
#include <stdio.h>
#include <stdlib.h>
#include <Classlib/Array1.h>
#include <ctype.h>

struct Var {
    Array1<Array3<double>*> data;
    char* name;
};

Array1<Var*> vars;

Var* findvar(char* str)
{
    for(int i=0;i<vars.size();i++){
	Var* var=vars[i];
	if(!strcmp(var->name, str)){
	    return var;
	}
    }
    return 0;
}

struct Zone {
    double dx, dy, dz;
    double x0, y0, z0;
    int ni, nj, nk;
};

int main(int argc, char *argv[])
{
    char* filename=0;
    for(int i=1;i<argc;i++){
	if(!strcmp(argv[i], "-vfield")){
	    i+=4;
	} else if(!strcmp(argv[i], "-sfield")){
	    i+=2;
	} else {
	    filename=argv[i];
	}
    }
    printf("Reading file: %s\n", filename);
    FILE* in=fopen(filename, "r");
    if(!in){
	perror("fopen");
	exit(1);
    }
    int zonec=1;
    while(1){
	char buf[1000];
	char what[1000];
	int done=0;
	while(1){
	    if(fgets(buf, 1000, in) == 0){
		done=1;
		break;
	    }
	    int n=sscanf(buf, "%s", what);
	    if(n == 1)
		break;
	}
	if(done)
	    break;
	if(!strcmp(what, "TITLE")){
	    //printf("title: %s\n", buf);
	} else if(!strcmp(what, "VARIABLES")){
	    //printf("variables: %s\n", buf);
	    char* p=buf;
	    while(*p != '=' && *p != 0)p++;
	    p++;
	    while(isspace(*p) && *p != 0)p++;
	    char* varnames=strdup(p);
	    p=varnames;
	    while(1){
		while(isspace(*p) && *p != 0)p++;
		if(!*p)
		    break;
		Var* var=new Var;
		var->name=p;
		vars.add(var);
		while(*p != ',' && !isspace(*p) && *p != 0)p++;
		*p=0;
		p++;
		//printf("var[%d]=%s\n", vars.size()-1, var->name);
	    }
	} else if(!strcmp(what, "TEXT")){
	    //printf("text: %s\n", buf);
	} else if(!strcmp(what, "T")){
	    //printf("t: %s\n", buf);
	} else if(!strcmp(what, "ZONE")){
	    if(strstr(buf, "F=BLOCK")){
		char* p=buf;
		while(*p != '"' && *p != 0)p++;
		while(*p != '"' && *p != 0)p++;
		while(*p != '=' && *p != 0)p++;
		int ni, nj, nk;
		p++;
		if(sscanf(p, "%d", &ni) != 1){
		    fprintf(stderr, "Error parsing ni\n");
		    exit(1);
		}
		while(*p != '=' && *p != 0)p++;
		p++;
		if(sscanf(p, "%d", &nj) != 1){
		    fprintf(stderr, "Error parsing nj\n");
		    exit(1);
		}
		while(*p != '=' && *p != 0)p++;
		p++;
		if(sscanf(p, "%d", &nk) != 1){
		    fprintf(stderr, "Error parsing nk: %s\n", p);
		    exit(1);
		}
		printf("Reading zone %d (%d %d %d)\n", zonec++, ni, nj, nk);
		for(int v=0;v<vars.size();v++){
		    Var* var=vars[v];
		    if(v>0)
			printf(", ");
		    printf("%s", var->name);
		    Array3<double>* dataptr=new Array3<double>(ni, nj, nk);
		    Array3<double>& data=*dataptr;
		    var->data.add(dataptr);
		    for(int k=0;k<nk;k++){
			for(int j=0;j<nj;j++){
			    for(int i=0;i<ni;i++){
				double d;
				int n=fscanf(in, "%lG", &d);
				if(n != 1){
				    fprintf(stderr, "Error parsing data\n");
				    exit(1);
				}
				data(i,j,k)=d;
			    }
			}
		    }
		}
		printf("\n");
	    } else {
		printf("not finished: %s\n", buf);
	    }
	} else {
		printf("unknown: %s\n", buf);
	}
    }
    int nzones=zonec-1;
    Zone* zones=new Zone[nzones];
    /* Find dx, dy and dz */
    Var* var=findvar("X");
    if(!var){
	printf("Can't find X\n");
	exit(1);
    }
    for(i=0;i<nzones;i++){
	Zone* zone=&zones[i];
	Array3<double>& data=*var->data[i];
	zone->dx=data(1,0,0)-data(0,0,0);
	zone->x0=data(0,0,0);
    }
    var=findvar("Y");
    if(!var){
	printf("Can't find Y\n");
	exit(1);
    }
    for(i=0;i<nzones;i++){
	Zone* zone=&zones[i];
	Array3<double>& data=*var->data[i];
	zone->dy=data(0,1,0)-data(0,0,0);
	zone->y0=data(0,0,0);
    }
    var=findvar("Z");
    if(!var){
	printf("Can't find Z\n");
	exit(1);
    }
    for(i=0;i<nzones;i++){
	Zone* zone=&zones[i];
	Array3<double>& data=*var->data[i];
	zone->dz=data(0,0,1)-data(0,0,0);
	zone->z0=data(0,0,0);
	zone->ni=data.dim1();
	zone->nj=data.dim2();
	zone->nk=data.dim3();
    }
    for(i=1;i<argc;i++){
	if(!strcmp(argv[i], "-vfield")){
	    printf("Writing variables %s,%s,%s to %s\n", argv[i+1], argv[i+2],
		   argv[i+3], argv[i+4]);
	    Var* var1=findvar(argv[i+1]);
	    Var* var2=findvar(argv[i+2]);
	    Var* var3=findvar(argv[i+3]);
	    VectorFieldZone* zfield=new VectorFieldZone(nzones);
	    for(int z=0;z<nzones;z++){
		Zone* zone=&zones[z];
		VectorFieldRG* field=new VectorFieldRG();
		Point min(zone->x0, zone->y0, zone->z0);
		Vector diag((zone->ni-1)*zone->dx, (zone->nj-1)*zone->dy, (zone->nk-1)*zone->dz);
		field->set_bounds(min, min+diag);
		field->resize(zone->ni, zone->nj, zone->nk);
		for(int k=0;k<zone->nk;k++){
		    for(int j=0;j<zone->nj;j++){
			for(int i=0;i<zone->ni;i++){
			    field->grid(i,j,k)=Vector((*var1->data[z])(i,j,k),
						      (*var2->data[z])(i,j,k),
						      (*var3->data[z])(i,j,k));
			}
		    }
		}
		zfield->zones[z]=VectorFieldHandle(field);
	    }
	    BinaryPiostream stream(argv[i+4], Piostream::Write);
	    VectorFieldHandle handle=zfield;
	    Pio(stream, handle);
	    i+=4;
	} else if(!strcmp(argv[i], "-sfield")){
	    printf("Writing variable %s to %s\n", argv[i+1], argv[i+2]);
	    Var* var=findvar(argv[i+1]);
	    ScalarFieldZone* zfield=new ScalarFieldZone(nzones);
	    for(int z=0;z<nzones;z++){
		Zone* zone=&zones[z];
		ScalarFieldRGdouble* field=new ScalarFieldRGdouble();
		Point min(zone->x0, zone->y0, zone->z0);
		Vector diag(zone->ni*zone->dx, zone->nj*zone->dy, zone->nk*zone->dz);
		field->set_bounds(min, min+diag);
		field->resize(zone->ni, zone->nj, zone->nk);
		for(int k=0;k<zone->nk;k++){
		    for(int j=0;j<zone->nj;j++){
			for(int i=0;i<zone->ni;i++){
			    field->grid(i,j,k)=(*var->data[z])(i,j,k);
			}
		    }
		}
		zfield->zones[z]=ScalarFieldHandle(field);
	    }
	    BinaryPiostream stream(argv[i+2], Piostream::Write);
	    ScalarFieldHandle handle=zfield;
	    Pio(stream, handle);
	    i+=2;
	} else {
	    filename=argv[i];
	}
    }
}
