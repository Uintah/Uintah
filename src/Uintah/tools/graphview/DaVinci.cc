#include <errno.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <sys/wait.h>
#include "DaVinci.h"
#include "TaskGraph.h"

using namespace std;

static string readline(int fd);

DaVinci*
DaVinci::run()
{
    static char* const DAVINCI_ARGS[] = {"daVinci", "-pipe", 0};
    int pipes[2][2];
    
    // create the input and output communication channels
    pipe(pipes[0]);
    pipe(pipes[1]);

    pid_t pid = fork();
    if (pid == -1) {
    	// error
    	perror("fork() failed");
    	return 0;
    } else if (pid == 0) {
    	// child

    	// replace STDIN and STDOUT with the appropriate ends of the pipes
    	close(0);   	    // stdin
	dup(pipes[0][0]);   // new fd is 0==stdin
	close(1);   	    // stdout
	dup(pipes[1][1]);   // new fd is 1==stdout

    	// we won't need any of the original pipe FDs in the new process
    	close(pipes[0][0]);
    	close(pipes[0][1]);
	close(pipes[1][0]);
	close(pipes[1][1]);

    	// set STDIN and STDOUT to non-buffered mode
    	setbuf(stdin, 0);
	setbuf(stdout, 0);

    	if (execv("/usr/local/contrib/jehall/arch.IRIX64-mips/bin/davinci", DAVINCI_ARGS) == -1) {
    	    perror("execv() failed");
    	    exit(1);
	}
    }
    
    // parent
    close(pipes[0][0]);
    close(pipes[1][1]);

    DaVinci* davinci = new DaVinci(pid, pipes[0][1], pipes[1][0]);

    return davinci;
}

DaVinci::DaVinci(pid_t pid, int in_fd, int out_fd)
  : m_PID(pid), m_toDV(in_fd), m_fromDV(out_fd)
{
    string response = readline(m_fromDV);
    cout << "run: " << response << flush;
}

DaVinci::~DaVinci()
{
    ostringstream quit_cmd;
    quit_cmd << "menu(file(quit))" << endl;
    string cmdbuf = quit_cmd.str();
    
    ssize_t len = write(m_toDV, cmdbuf.c_str(), cmdbuf.size());
    if (len != cmdbuf.size())
    	cerr << "DaVinci::~DaVinci(): write() didn't finish ("
	     << strerror(errno) << ")" << endl;

    string response = readline(m_fromDV);
    cout << "quit: " << response << flush;

    while ((waitpid(m_PID, 0, 0) == -1) && (errno == EINTR))
    	;

    close(m_toDV);
    close(m_fromDV);
}

void
DaVinci::setGraph(const TaskGraph* graph)
{
    ostringstream graph_str;

    graph_str << "graph(new_placed([";

    bool first_node = true;
    const list<Task*> tasks = graph->getTasks();
    for (list<Task*>::const_iterator task_iter = tasks.begin();
    	 task_iter != tasks.end(); task_iter++) {
    	if (!first_node)
	    graph_str << ',';
	else
	    first_node = false;

    	graph_str << "l(\"" << (*task_iter)->getName()
    	    	  << "\",n(\"\",[a(\"OBJECT\",\"" << (*task_iter)->getName()
    	    	  << "\"),a(\"COLOR\",\"lightgrey\")],[";
	
	bool first_edge = true;
    	const list<Task*> dependencies = (*task_iter)->getDependencies();
	for (list<Task*>::const_iterator deps_iter = dependencies.begin();
	     deps_iter != dependencies.end(); deps_iter++) {
	    if (!first_edge)
	    	graph_str << ',';
	    else
	    	first_edge = false;
	    
	    graph_str << "l(\"" << (*task_iter)->getName() << "->"
    	    	      << (*deps_iter)->getName()
    	    	      << "\",e(\"\",[a(\"_DIR\",\"inverse\")],r(\""
    	    	      << (*deps_iter)->getName()
    	    	      << "\")))";
	}
	
	graph_str << "]))";
    }
    graph_str << "]))\n";

    string str = graph_str.str();
    ssize_t len = write(m_toDV, str.c_str(), str.size());
    if (len != str.size())
    	cerr << "DaVinci::setGraph(): write() didn't finish ("
	     << strerror(errno) << ")" << endl;

    string response = readline(m_fromDV);
    cout << "setGraph: " << response << flush;
}

void
DaVinci::setOrientation(Orientation orientation)
{
    ostringstream cmd;
    cmd << "menu(layout(orientation(";
    switch (orientation) {
    case TOP_DOWN:  	cmd << "top_down";	break;
    case BOTTOM_UP: 	cmd << "bottom_up"; break;
    case LEFT_RIGHT:	cmd << "left_right";	break;
    case RIGHT_LEFT:	cmd << "right_left";	break;
    }
    cmd << ")))\n";

    string cmd_str = cmd.str();
    ssize_t len = write(m_toDV, cmd_str.c_str(), cmd_str.size());
    if (len != cmd_str.size())
    	cerr << "DaVinci::setOrientation(): write() didn't finish ("
	     << strerror(errno) << ")" << endl;
    
    string response = readline(m_fromDV);
    cout << "setOrientation: " << response << flush;
}

static string readline(int fd)
{
    ostringstream line;
    ssize_t len;
    char buf[BUFSIZ + 1];
    bool need_more;
    do {
    	need_more = false;
    	cerr << "reading...";
    	while (((len = read(fd, buf, BUFSIZ)) == -1) && (errno == EINTR))
	    ;
    	if (len > 0) {
    	    buf[len] = '\0';
    	    cerr << "(" << len << "):" << buf;
	    line << buf;
	    if (strchr(buf, '\n') == 0)
    	    	need_more = true;
	} else if (len == -1)
    	    perror("Error reading from daVinci");
    } while (need_more);
    
    return line.str();
}
