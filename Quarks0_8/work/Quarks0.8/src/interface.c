/*              Quarks distributed shared memory system.
 * 
 * Copyright (c) 1995 The University of Utah and
 * the Computer Systems Laboratory (CSL).  All rights reserved.
 *
 * Permission to use, copy, modify and distribute this software is hereby
 * granted provided that (1) source code retains these copyright, permission,
 * and disclaimer notices, and (2) redistributions including binaries
 * reproduce the notices in supporting documentation, and (3) all advertising
 * materials mentioning features or use of this software display the following
 * acknowledgement: ``This product includes software developed by the Computer 
 * Systems Laboratory at the University of Utah.''
 *
 * THE UNIVERSITY OF UTAH AND CSL ALLOW FREE USE OF THIS SOFTWARE IN ITS "AS
 * IS" CONDITION.  THE UNIVERSITY OF UTAH AND CSL DISCLAIM ANY LIABILITY OF
 * ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * CSL requests users of this software to return to csl-dist@cs.utah.edu any
 * improvements that they make and grant CSL redistribution rights.
 *
 * 	Utah $Hdr$
 *	Author: Dilip Khandekar, University of Utah CSL
 */
/**************************************************************************
 *
 * interface.c: start rsh and set up remote xterms
 *
 *************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

#include "thread.h"
#include "message.h"

extern int num_children_forked;

char *top_dir;
char *this_dir;

extern char *Qksrvhname;
int Qknumnodes=0;

#define YDISP 250

extern int setclockrate(int);


static int xt_width, xt_height, xt_xdisp, xt_ydisp, xt_numcol1;
static char xt_font[8];
static int use_gdb = 0;
static int use_xterm = 1;
static int sflag=0;

static int
getline(FILE *fp, char *line)
{
#ifdef IRIX
    signed char ch;
#else
    char ch;
#endif

    while ((ch = getc(fp)) != EOF)
    {
	if (ch == '\n')
	{
	    *line = 0;
	    return 0;
	}
	*line++ = ch;
    }
    *line = 0;
    return EOF;
}

static void
remove_comment(char *line)
{
    while (*line)
    {
	if (*line == '#') {*line = 0; return; }
	line++;
    }
}

static int 
blank_line(char *line)
{
    while (*line)
	if ((*line != ' ') || 
	    (*line != '\t'))
	    return 0;
    return 1;
}

static void 
get_resources()
{
    FILE *fp;
    char line[256];
    char recname[128], recval[128];

    fp = fopen("Qk.display", "r");
    if (!fp)
    {
	printf("No file Qk.display. Assuming default values\n");
	xt_width   = 68;
	xt_height  = 11;
	xt_xdisp   = 512;
	xt_ydisp   = 250;
	xt_numcol1 = 3;
	strcpy(xt_font, "10x20");
	return;
    }


    while (getline(fp, line) != EOF)
    {
	remove_comment(line);
	if (blank_line(line)) continue;
	sscanf(line,"%s%s", recname, recval);
	if (strcmp(recname, "width:") == 0)
	    xt_width = atoi(recval);
	if (strcmp(recname, "height:") == 0)
	    xt_height = atoi(recval);
	if (strcmp(recname, "xdisp:") == 0)
	    xt_xdisp = atoi(recval);
	if (strcmp(recname, "ydisp:") == 0)
	    xt_ydisp = atoi(recval);
	if (strcmp(recname, "numcol1:") == 0)
	    xt_numcol1 = atoi(recval);
	if (strcmp(recname, "font:") == 0)
	    strcpy(xt_font, recval);
    }

}    

static void
print_resources()
{
    printf("xt_width = %d\n", xt_width);
    printf("xt_height = %d\n", xt_height);
    printf("xt_xdisp = %d\n", xt_xdisp);
    printf("xt_ydisp = %d\n", xt_ydisp);
    printf("xt_numcol1 = %d\n", xt_numcol1);
    printf("xt_font = <%s>\n", xt_font);
}

void Qk_shutdown(int terminate)
{
    Id nodeid=2;
    int i;
    Message *msg, *reply;
    static int shutdown_done = 0;

    ASSERT(Qknodeid != 0);  /* server cannot shutdown! */

    if (!shutdown_done)
    {
	if (Qknodeid == 1)
	{
	    MSG_INIT(msg, MSG_OP_SHUTDOWN);
	    MSG_INSERT(msg, terminate);
	    for (i=0; i<num_children_forked; i++)
	    {
		reply = ssend_msg(construct_threadid(nodeid, DSM_THREAD), msg);
		free_buffer(reply);
		nodeid++;
	    }
	}
	quarks_dsm_shutdown();
	disconnect_all_clients();
	disconnect_server();
    }
    if (terminate) 
	exit(0);
    else
	shutdown_done = 1;
}

static void get_qargs(int argc, char *argv[])
{
    int  i, pos;
    char hname[256];
    static int qk_args_start = 0;

    for (i=1; i<argc; i++)
    {
	if (strcmp(argv[i], "--") == 0)
	    qk_args_start = i+1;
    }
    
    if (qk_args_start > 0)
    {
	pos = qk_args_start;
	while (pos < argc)
	{
	    if (strcmp(argv[pos], "-g") == 0)    /* use gdb */
		use_gdb = 1;
	    if (strcmp(argv[pos], "-nox") == 0)  /* use xterms */
		use_xterm = 0;
	    if (strcmp(argv[pos], "-n") == 0)    /* num procs */
	    {
		pos++;
		Qknumnodes = atoi(argv[pos]);
	    }
	    if (strcmp(argv[pos], "-s") == 0)    /* server hostname */
	    {
		pos++;
		Qksrvhname = (char *) malloc(strlen(argv[pos])+1);
		strcpy(Qksrvhname, argv[pos]);
		sflag = 1;
	    }
	    if (strcmp(argv[pos], "-m") == 0)    /* master node */
		Qkmaster = 1;
	    if (strcmp(argv[pos], "-c") == 0)    /* child node */
		Qkchild = 1;
	    
	    pos++;
	}
    }
    if (!Qkchild && (Qksrvhname == 0))
    {
	fprintf(stdout, "Give hostname of sh_server: ");
	fflush(stdout);
	scanf("%s", hname);
	Qksrvhname = (char *) malloc(strlen(hname)+1);
	strcpy(Qksrvhname, hname);
    }
}


void Qk_init(int argc, char **argv, char *script_file)
{
    FILE *fd;
    char hostname[32];
    char myhostname[32];
    int pid;
    char forrsh[1024];
    char geom_string[256];
    struct hostent *hp;
    int hindex,index,i,j,runi;
    int next_arg;
    struct in_addr *ptr;
    char **hptr;
    char *hosts[1024];  
    int childpid;
    char path[256],exe[100];
    char *display_name;
    char *chptr;
    int clockrate;
    int xdisp, ydisp;
    int NO_PROC;

    /* Cthread idiosyncracy. */
    clockrate = setclockrate(0);

    get_qargs(argc, argv);

    /* Qkchild should have been initialized by now, if this is 
     * a child process. If not, it better be a master.
     */
    if (!Qkchild)
    {
	top_dir  = (char *) malloc(sizeof(char)*256);
	this_dir = (char *) malloc(sizeof(char)*256);
	strcpy(top_dir, "..");
	strcpy(this_dir, ".");
	printf("top_dir = <%s>\n", top_dir);
	printf("this_dir = <%s>\n", this_dir);

	get_resources();

	if (use_xterm)
	{
	    if ((display_name = getenv("DISPLAY")) == NULL)
		PANIC("Cannot open display. Set the DISPLAY variable");
	    printf("Displaying on <%s>\n", display_name);
	}

	fd = fopen(script_file, "r");
	if (! fd)
	{
	    printf("script file <%s>\n", script_file);
	    PANIC("Could not open script file");
	}
	

	if (gethostname(myhostname, 32) < 0)
	    PANIC("Could not find current hostname");
	NO_PROC = 1;
	hp = gethostbyname(myhostname);
	hosts[1] = (char *) malloc((strlen(hp->h_name)+1)*sizeof(char));
	strcpy(hosts[1], hp->h_name);


	pid = 2;
	while(EOF != fscanf(fd,"%s\n",hostname))
	{
	    if (hostname[0] == '#') continue;
	    NO_PROC++;
	    /* check if the host is valid */
	    if((hp = gethostbyname(hostname)) == NULL)
		PANIC("ERROR: unknown host");
	    
	    hosts[pid] = (char *) malloc((strlen(hp->h_name)+1)*sizeof(char));
	    strcpy(hosts[pid], hp->h_name);
	    pid++;
	}
	fclose(fd);
	
	quarks_basic_init(0);

	if (Qknumnodes > 0) runi = Qknumnodes-1;
	else runi = NO_PROC-1;    

	if (! getwd(path))
	    PANIC("Could not get cwd");

	j=2;
	ydisp = 0;
	xdisp = 0;
	/* Build the xterm geometry specification string */
	while(runi)
	{
	    /* while(PEnt[j].hostptr==0) j++; */
	    /* pack the string for remote execution */

	    forrsh[0]=0;		    
	    sprintf(geom_string, "%s%d%s%d%s%d%s%d%s%s",
		    " -geometry ",
		    xt_width,
		    "x",
		    xt_height,
		    "+",
		    xdisp,
		    "+",
		    ydisp,
		    " -fn ",
		    xt_font);

	    ydisp += xt_ydisp;
	    if ((ydisp == xt_numcol1*xt_ydisp) /* && (xdisp == 0) */ )
	    {
		xdisp += xt_xdisp; 
		ydisp = 0;
	    }

	    /* Build the rsh command */
	    hindex = ((j-1) % NO_PROC) + 1;
	    if (use_xterm)
		sprintf(forrsh, "%s%s%s%s%s%d%s%s%s%s",
			"rsh ",
			hosts[hindex],
			" xterm -display ",
			display_name,
			" -title PID_",
			j,
			"_",
			hosts[hindex],
			geom_string,
			" -e ");
	    else
		sprintf(forrsh, "%s%s%s",
			"rsh ",
			hosts[hindex],
			" ");

	    if (use_gdb) 
	    {
		index = strlen(forrsh);
		sprintf(&forrsh[index], " gdb %s &", argv[0]);
	    }
	    else
	    {
		index = strlen(forrsh);
		sprintf(exe,"%s/%s",path,argv[0]); 
		sprintf(&forrsh[index]," %s", exe);
	    }

	    if (! use_gdb)
	    {   
		/* copy the command line params and append -c for child */
		for(i=1;i<argc;i++)
		{
		    if (strcmp(argv[i], "-m") != 0)
		    {
			index = strlen(forrsh);
			sprintf(&forrsh[index]," %s", argv[i]);
		    }
		}
		if (!sflag)
		{
		    index = strlen(forrsh);
		    sprintf(&forrsh[index]," -s %s ", Qksrvhname);
		}
		index = strlen(forrsh);
		sprintf(&forrsh[index]," -c &");
	    }

	    /* rsh here */
	    printf("Rsh command: <%s>\n", forrsh);
	    system(forrsh); 
	    
	    index = 0;
	    forrsh[0]=0;
	    
	    runi--;
	    j++; 
	}

	/* Cthread idiosyncracy. */
	setclockrate(clockrate);
	
	return;
    }
    else    /* This is a child */
    {
	top_dir  = (char *) malloc(strlen(argv[0])+1);
	this_dir = (char *) malloc(strlen(argv[0])+1);
	strcpy(top_dir, argv[0]);
	strcpy(this_dir, argv[0]);

	chptr = strrchr(top_dir, '/'); *chptr = 0;
	chptr = strrchr(top_dir, '/'); *chptr = 0;
	chptr = strrchr(this_dir, '/');  *chptr = 0;
	printf("top_dir = <%s>\n", top_dir);
	printf("this_dir = <%s>\n", this_dir);

	quarks_basic_init(0);

	/* Cthread idiosyncracy. */
	setclockrate(clockrate);
	return;
    }
}
