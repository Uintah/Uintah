/************************************************************************

  $Id$

 ************************************************************************/

#include "Nautilus.h"

#include <string.h>
#include <fstream.h>

ostream *logfile = NULL;

unsigned int selected_output = OUTPUT_NONE;




static const char *nautilus_cmdline_opts = "L:l:";

static char _options_buf[512];
char *global_cmdline_options = _options_buf;


void nautilus_add_options(char *options)
{
    strcpy(global_cmdline_options, nautilus_cmdline_opts);

    strcpy(global_cmdline_options + strlen(global_cmdline_options),
	   options);
}

char *nautilus_usage_string = 
"-l <file>	Log all simplification operations to given file.\n"
"-L[xqcvfdA]	Set information to be output.\n"
"			x=contractions, q=quadrics, c=cost\n"
"			v=vert. notes, f=face notes\n"
"			d=model defn, A=All.\n";

bool nautilus_parseopt(int opt, char *optarg)
{
    char *c;
    int errflg = 0;

    switch( opt )
    {
    case 'l':
	if( optarg[0]=='-' )
	    logfile = &cout;
	else
	{
	    logfile = new ofstream(optarg);
	}
	break;

    case 'L':
	c = optarg;
	while( *c )
	{
	    if( *c=='x' )
		selected_output |= OUTPUT_CONTRACTIONS;
	    else if( *c=='q' )
		selected_output |= OUTPUT_QUADRICS;
	    else if( *c=='c' )
		selected_output |= OUTPUT_COST;
	    else if( *c=='v' )
		selected_output |= OUTPUT_VERT_NOTES;
	    else if( *c=='f' )
		selected_output |= OUTPUT_FACE_NOTES;
	    else if( *c=='d' )
		selected_output |= OUTPUT_MODEL_DEFN;
	    else if( *c=='A' )
		selected_output |= OUTPUT_ALL;
	    else
		errflg++;
	    c++;
	}
	break;

    default:
	errflg++;
	break;
    }


    return errflg==0;
}
