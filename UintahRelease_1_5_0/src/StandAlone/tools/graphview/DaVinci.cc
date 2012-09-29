/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <cerrno>
#include <fstream>
#include <iostream>
#include <signal.h>
#include <sstream>
#include <string>
#include <list>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <cstdio>
#include <cstring>
#include <Core/Exceptions/ErrnoException.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include "DaVinci.h"
#include "graphview.h"
#include "GV_TaskGraph.h"

using namespace std;
using namespace SCIRun;

static ostream& operator<<(ostream& out, const GV_Task* task);
static string readline(int fd);
static void writeline(int fd, string str);

static void displayAttributes(std::ostream& out, const GV_Task* task);
static void displayAttributes(std::ostream& out, const Edge* edge);
static const char* getColor(float percent /* max incl path / critical path */,
			    float thresholdPercent);
static const char* getHidden(float percent /* max incl path / critical path */,
			     float thresholdPercent);

bool DaVinci::doExclusion = false;

DaVinci*
DaVinci::run()
{
  static const char* const DAVINCI_ARGS[] = {
    "/bin/sh", "-c", "davinci -pipe || daVinci -pipe", 0
  };
  int pipes[2][2];
    
  // block SIGPIPE; we'll notice when we get an EPIPE error from a syscall
  sigset_t sigset;
  sigaddset(&sigset, SIGPIPE);
  sigprocmask(SIG_BLOCK, &sigset, 0);

  // create the input and output communication channels
  pipe(pipes[0]);
  pipe(pipes[1]);

  pid_t pid = fork();
  if (pid == -1)
    throw ErrnoException("fork() failed", errno, __FILE__, __LINE__);
  else if (pid == 0) {
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

    if (execv("/bin/sh", const_cast<char* const*>(DAVINCI_ARGS)) == -1) {
      perror("execv() failed");
      if (errno == ENOENT) {
	cerr << "\n******************************************\n"
	     << "An executable named 'davinci' or 'daVinci'\n"
	     << "must be in your path to use graphview.\n"
	     << "******************************************\n\n";
      }
      exit(1);
    }
  }
    
  // parent
  close(pipes[0][0]);
  close(pipes[1][1]);

  DaVinci* davinci = scinew DaVinci(pid, pipes[0][1], pipes[1][0]);

  return davinci;
}

DaVinci::DaVinci(pid_t pid, int in_fd, int out_fd)
  : m_PID(pid), m_toDV(in_fd), m_fromDV(out_fd)
{
  string response = readline(m_fromDV);
  if ((response != "ok") && (response.size() > 0))
    cerr << "daVinci said: " << response << endl;
}

DaVinci::~DaVinci()
{
  try {
  
    ostringstream quit_cmd;
    quit_cmd << "menu(file(quit))" << endl;
    string cmdbuf = quit_cmd.str();

    writeline(m_toDV, cmdbuf);
    readline(m_fromDV);
	
    // just throw away the response; we need to do the read in case
    // davinci blocks forever trying to write
  } catch (Exception&) {}

  while ((waitpid(m_PID, 0, 0) == -1) && (errno == EINTR))
    ;

  close(m_toDV);
  close(m_fromDV);
}

void
DaVinci::setGraph(const GV_TaskGraph* graph)
{
  if (graph == NULL)
    return;
  
  ostringstream graph_str;    

  graph_str << "graph(new_placed([";

  bool first_node = true;
  const list<GV_Task*> tasks = graph->getTasks();
  for (list<GV_Task*>::const_iterator task_iter = tasks.begin();
       task_iter != tasks.end(); task_iter++) {

   if (doExclusion && ((*task_iter)->getMaxPathPercent() < (*task_iter)->getGraph()->getThresholdPercent()))
    continue;

   if (!first_node)
      graph_str << ',';
    else
      first_node = false;

    graph_str << *task_iter;
  }
  graph_str << "]))\n";

  //cout << graph_str.str() << endl;
  
  writeline(m_toDV, graph_str.str());
  string response = readline(m_fromDV);
  if (response != "ok")
    cerr << "daVinci said: " << response << endl;
}

void
DaVinci::setOrientation(Orientation orientation)
{
  ostringstream cmd;
  cmd << "menu(layout(orientation(";
  switch (orientation) {
  case TOP_DOWN:      cmd << "top_down";	    break;
  case BOTTOM_UP:     cmd << "bottom_up";     break;
  case LEFT_RIGHT:    cmd << "left_right";    break;
  case RIGHT_LEFT:    cmd << "right_left";    break;
  }
  cmd << ")))\n";

  writeline(m_toDV, cmd.str());
  string response = readline(m_fromDV);
  if (response != "ok")
    cout << "daVinci said: " << response << endl;
}

void
DaVinci::setFontSize(int font_size)
{
  ostringstream cmd;
  cmd << "set(font_size(" << font_size << "))\n";
  writeline(m_toDV, cmd.str());
  string response = readline(m_fromDV);
  if (response != "ok")
    cout << "daVinci said: " << response << endl;
}

void
DaVinci::handleInput()
{
  string input = readline(m_fromDV);

  // from the input, parse out the command and arguments list
  char* cmd = scinew char[input.size() + 1];
  strcpy(cmd, input.c_str());

  std::list<char*> args;
  parseAnswer(cmd, args);

  /* can uncomment when debugging
  cout << cmd << endl;
  int i = 0;
  for (std::list<char*>::iterator iter = args.begin(); iter != args.end();
       iter++) {
    cout << i++ << ": " << *iter << endl;
  }
  */
    
  // handle cmd
  if (strcmp(cmd, "quit") == 0)
    gEventQueue.push(Event(EVT_DV_QUIT));
  else if (strcmp(cmd, "communication_error") == 0) {
    cerr << "DaVinci error: " << args.front() << endl;
  }
  else if (strcmp(cmd, "node_selections_labels") == 0) {
    m_selectedNodes.clear();
    m_selectedEdge = "";
    for (std::list<char*>::iterator iter = args.begin();
	 iter != args.end(); iter++)
      m_selectedNodes.push_back(string(*iter));
    gEventQueue.push(Event(EVT_DV_SELECT_NODE));
  }
  else if (strcmp(cmd, "node_double_click") == 0) {
    gEventQueue.push(Event(EVT_DV_DOUBLE_CLICK_NODE));
  }
  else if (strcmp(cmd, "edge_selection_label") == 0) {
    m_selectedNodes.clear();
    if (args.size() > 0) {
      // there really should only be one argument
      m_selectedEdge = args.front(); 
      gEventQueue.push(Event(EVT_DV_SELECT_EDGE));
    }
  }
  else if (strcmp(cmd, "edge_double_click") == 0) {
    gEventQueue.push(Event(EVT_DV_DOUBLE_CLICK_EDGE));      
  }
  else if (strcmp(cmd, "menu_selection") == 0) {
      
  }

  delete[] cmd;
}

// Parses a DaVinci answer string simply by breaking it
// the cmd and arguments (by inserting '\0's in the cmd
// string and appending char*'s the the args list.
void DaVinci::parseAnswer(char* cmd, std::list<char*>& args)
{
  char* p = cmd;
  while (*p != '\0' && *p != '(')
    p++;

  if (*p == '\0') return;
  *p = '\0'; // NULL terminate cmd
  p++;

  // separate arguments
  // use the fact that all arguments are strings wrapped
  // in double quotes (with the technical exception of
  // the node_ids argument in node_selections_labels but
  // we want to extract its list of node_id's as arguments
  // instead of the list as a single argument which it
  // technically is).
  do {
    while (*p != '\0' && *p != '\"')
      p++;

    if (*p == '\0') break;

    p++; // pass first quote
    args.push_back(p);
	
    while (*p != '\0' && *p != '\"') {
      if (*p == '\\')
	p++; // skip passed excape sequence
      p++;
    }

    if (*p != '\0') {
      *p = '\0'; // null terminate argument at end quote
      p++;
    } 
  } while (*p != '\0');
}

static
ostream&
operator<<(ostream& out, const GV_Task* task)
{  
  out << "l(\"" << task->getName()
      << "\",n(\"\",[a(\"OBJECT\",\"" << task->getName()
      << "\"),";
  displayAttributes(out, task);
  out << "],[";
	
  bool first_edge = true;
  const list<Edge*> dependency_edges = task->getDependencyEdges();
  for (list<Edge*>::const_iterator dep_edge_iter = dependency_edges.begin();
       dep_edge_iter != dependency_edges.end(); dep_edge_iter++) {
    if (DaVinci::doExclusion && ((*dep_edge_iter)->getSource()->getMaxPathPercent() < (*dep_edge_iter)->getGraph()->getThresholdPercent() || (*dep_edge_iter)->isObsolete()))
      continue; // JUST TESTING -- NEED TO CHANGE BACK

   if (!first_edge)
      out << ',';
    else
      first_edge = false;
	    
    GV_Task* dep = (*dep_edge_iter)->getSource();
    out << "l(\"" << dep->getName() << " -> " << task->getName()
	<< "\",e(\"\",[a(\"_DIR\",\"inverse\"),";
    displayAttributes(out, (*dep_edge_iter));
    out << "],r(\"" << dep->getName() << "\")))";
  }
  out << "]))";

  return out;
}

static
string
readline(int fd)
{
  // this function will handle the case the line comes in several "packets",
  // but it won't handle the case where the \n is _not_ the last character
  // in a "packet". The extra data (probably part of the next response) will
  // be included with this line, and won't be seen on the next read.
  // So far it doesn't look like daVinci does this.

  ostringstream line;
  ssize_t len;
  char buf[BUFSIZ + 1];
  bool need_more;
  do {
    need_more = false;
    while (((len = read(fd, buf, BUFSIZ)) == -1) && (errno == EINTR))
      ;
    if (len > 0) {
      buf[len] = '\0';
      char* eol = strchr(buf, '\n');
      if (eol == 0)
	need_more = true;
      else
	*eol = '\0';
      line << buf;
    } else if (len == -1)
      throw ErrnoException("read() error", errno, __FILE__, __LINE__);
  } while (need_more);
    
  return line.str();
}

static
void
writeline(int fd, string str)
{
  ssize_t written = 0;
  ssize_t len;

  while (written < (int)str.size()) {
    len = write(fd, str.c_str() + written, str.size() - written);
    if (len > 0)
      written += len;
    else if ((len == -1) && (errno != EINTR))
      //	  cerr << "write() error: " << errno << endl;
      throw ErrnoException("write() error", errno, __FILE__, __LINE__);
    else {
      std::stringstream s;
      s << "Unexpected write() return code " << len << endl;
      std::string return_string = s.str();
      const char* buf = return_string.c_str();
      //char buf[64];
      //sprintf(buf, "Unexpected write() return code %ld", len);
      cerr << buf << endl;
      //throw InternalError(buf);
    }
  }
}


static
void displayAttributes(ostream& out, const GV_Task* task)
{
  float thresholdPercent = task->getGraph()->getThresholdPercent();
  float maxPathPercent = task->getMaxPathPercent();
  out << "a(\"COLOR\",\"#" << getColor(maxPathPercent, thresholdPercent)
      << "\"),";
  out << "a(\"HIDDEN\",\"" << getHidden(maxPathPercent, thresholdPercent)
      << "\")";
}

static
void displayAttributes(ostream& out, const Edge* edge)
{
  float thresholdPercent = edge->getGraph()->getThresholdPercent();
  float maxPathPercent = edge->getMaxPathPercent();
  out << "a(\"EDGECOLOR\",\"#" << getColor(maxPathPercent, thresholdPercent)
      << "\"),a(\"EDGEPATTERN\",";
  if (maxPathPercent < thresholdPercent || edge->isObsolete())
    out << "\"dashed\""; // below threshold
  else if (maxPathPercent == 1)
    out << "\"double\""; // critical path
  else
    out << "\"solid\""; // normal
  out << ")";
}

static
const char* getColor(float percent /* max incl path / critical path */,
		     float thresholdPercent)
{
  static char col_str[7];

  // percent range should be:
  // [thresholdPercent, 1]
  // adjust that to [0, 1]
  float adj_percent;
  if (thresholdPercent < 1)
    adj_percent = (percent - thresholdPercent) / (1 - thresholdPercent);
  else
    adj_percent = (percent == 1) ? 1 : -1;

  if (adj_percent >= 0 && adj_percent <= 1) {
    int red = static_cast<int>(adj_percent * 255);
    int green = static_cast<int>((1-adj_percent) * 255);
    int blue = 0;
    sprintf(col_str, "%02x%02x%02x", red, green, blue);
    col_str[6]='\0';
  }
  else {
    strcpy(col_str, "0000FF"); // blue if out of range
  }

  return col_str;
}

static
const char* getHidden(float percent /* max incl path / critical path */,
		      float thresholdPercent)
{
  return (percent < thresholdPercent) ? "true" : "false";
}


