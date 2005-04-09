
#include <Geom/Scene.h>
#include <Classlib/Persistent.h>
#include <Classlib/Pstreams.h>

#include <iostream.h>
#include <stdlib.h>
#include <string.h>

void usage(char* progname)
{
    cerr << "usage: " << progname << " [informat:]infile outformat:outfile\n";
    exit(-1);
}

int main(int argc, char* argv[])
{
    if(argc != 3)
	usage(argv[0]);
    char* p1=strchr(argv[1], ':');
    char* in_format;
    char* in_name;
    GeomScene scene;
    if(p1){
	in_name=p1+1;
	in_format=argv[1];
	*p1=0;
	cerr << "in_format: " << in_format << " not implemented\n";
	exit(-1);
    } else {
	// Try to figure out the file type...
	in_name=argv[1];
	Piostream* stream=auto_istream(in_name);
	if (!stream) {
	    cerr << "Couldn't open file " << in_name << endl;
	    exit(-1);
	}
	// Read the file...
	Pio(*stream, scene);
	if(stream->error()){
	    cerr << "Error reading file: " << in_name << endl;
	    exit(-1);
	}
	delete stream;
    }

    // Try to figure out how to write it out...
    char* p2=strchr(argv[2], ':');
    char* out_format;
    char* out_name;
    if(!p2){
	cerr << "output format must be specified\n";
	cerr << "Currently supported: sci_binary, sci_ascii/sci_text, vrml\n";
	usage(argv[0]);
    }
    out_name=p2+1;
    out_format=argv[2];
    *p2=0;

    clString format(out_format);
    if(format == "sci_binary" || format == "sci_text" || format == "sci_ascii"){
	Piostream* stream;
	if(format == "sci_binary")
	    stream=new BinaryPiostream(out_name, Piostream::Write);
	else
	    stream=new TextPiostream(out_name, Piostream::Write);
	if(stream->error()){
	    delete stream;
	    cerr << "Error opening: " << out_name << " for writing\n";
	    exit(-1);
	}
	Pio(*stream, scene);
	if(stream->error()){
	    cerr << "Error writing geom file: " << out_name << endl;
	} else {
	    cerr << "Done writing geom file: " << out_name << endl;
	}
	delete stream;
    } else {
	bool status=scene.save(out_name, format);
	if(!status){
	    cerr << "Error saving scene!\n";
	    exit(-1);
	}
    }
    exit(0);
}
