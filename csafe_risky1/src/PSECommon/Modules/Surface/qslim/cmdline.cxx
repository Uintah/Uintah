// $Id$

#include "Nautilus.h"

#include <string.h>
#include <fstream.h>
#ifdef WIN32
#  include <gfx/sys/getopt.h>
#else
#  include <unistd.h>
#endif
#include "avars.h"

extern ostream *outfile;

static char *options = "o:O:Q:t:s:e:bB:ma";

static char *usage_string =
"-o <file>	Output final model to given file.\n"
"-s <count>	Set the target number of faces.\n"
"-e <thresh>	Set the maximum error tolerance.\n"
"-t <t>		Set pair selection tolerance.\n"
"-Q[pv]		Select what constraint quadrics to use [default=p].\n"
"-On		Optimal placement policy.\n"
"			0=endpoints, 1=endormid, 2=line, 3=optimal [default]\n"
"-B <weight>	Use boundary preservation planes with given weight.\n"
"-m		Preserve mesh quality.\n"
"-a		Enable area weighting.\n"
"\n";

static void usage_error(char *msg = NULL)
{
    if( msg )
	cerr << msg << endl;

    cerr << endl << "usage: qslim <options> [filename|-]" << endl;
    cerr << endl
	 << "Standard Nautilus Options:" << endl
	 << "--------------------------" << endl
	 << nautilus_usage_string << endl
	 << "QSlim Options:" << endl
	 << "--------------" << endl
	 << usage_string << endl;

    exit(1);
}

void process_cmdline(int argc, char **argv)
{
    int opt;
    int ival;
    char *c;

    nautilus_add_options(options);
    while( (opt = getopt(argc, argv, global_cmdline_options)) != EOF )
    {
	int errflg = 0;

	switch( opt )
	{
	case 'o':
	    if( optarg[0]=='-' )
		outfile = &cout;
	    else
	    {
		outfile = new ofstream(optarg);
	    }
	    break;

	case 'O':
	    ival = atoi(optarg);
	    if( ival < 0 || ival > PLACE_OPTIMAL )
		errflg++;
	    else
		placement_policy = ival;
	    break;

	case 'Q':
	    will_use_plane_constraint = false;
	    will_use_vertex_constraint = false;
	    will_use_plane_constraint = false;

	    c = optarg;
	    while( *c )
	    {
		if( *c=='p' )
		    will_use_plane_constraint = true;
		else if( *c=='v' )
		    will_use_vertex_constraint = true;
		else
		    errflg++;
		c++;
	    }
	    break;

	case 's':
	    face_target = atoi(optarg);
	    break;

	case 'e':
	    error_tolerance = atof(optarg);
	    break;

	case 't':
	    pair_selection_tolerance = atof(optarg);
	    break;

	case 'b':
	    will_preserve_boundaries = true;
	    break;

	case 'B':
	    will_constrain_boundaries = true;
	    boundary_constraint_weight = atof(optarg);
	    break;

	case 'm':
	    will_preserve_mesh_quality = true;
	    break;

	case 'a':
	    will_weight_by_area = true;
	    break;

	default:
	    if( !nautilus_parseopt(opt, optarg) )
		errflg++;
	    break;
	}

	if( errflg )
	    usage_error();
    }

    ////////////////////////////////////////////////////////


    if( optind==argc )
	usage_error();

    SMF_Reader reader(argv[optind]);
    read_model(reader);
}
