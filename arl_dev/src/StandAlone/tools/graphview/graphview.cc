/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/DataArchive/DataArchive.h>

#include "DaVinci.h"
#include "graphview.h"
#include "GV_TaskGraph.h"

#include <cerrno>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstdlib>

using namespace Uintah;

static const char HELP_MSG[] = { "\nCommands\n"
                                 "--------\n"
                                 "Help\n"
                                 "    Display this command summary.\n"
                                 "Time <timestep>\n"
                                 "    Loads the taskgraph from the given timestep directory in the uda\n"
                                 "    directory.\n"
                                 "List\n"
                                 "    Lists the timestep directories in the uda directory.\n"
                                 "Prune <percent>\n"
                                 "    Hide nodes and edges with maximum path costs less than <percent>\n"
                                 "    of the critical path cost.\n"
                                 "Xclude\n"
                                 "    Turn exclusion on or off.  When it is on, it not only hides, but\n"
                                 "    exclude nodes with maximum path costs less than the set pruning\n"
                                 "    percent.  This is useful for very large graphs.\n"
                                 "FontSize <size>\n"
                                 "    Sets the font size in the daVinci graph.\n"
                                 "Quit\n"
                                 "    Exit the program.\n\n"
                                 "Each command can be given using its unique shortcut (indicated by the\n"
                                 "uppercase letter), e.g. \"p 0.5\" is the same as \"prune 0.5\"." };

// global variable instatiations
bool gQuit = false;
std::queue<Event> gEventQueue;

static GV_TaskGraph* gGraph = 0;
static DaVinci* gDavinci = 0;
static std::string udaDir;

static void handle_event(const Event& event);
static void handle_console_input();
       bool load_timestep(int timestep, float prune_percent);

void usage(char* prog_name)
{
  std::cerr << "usage: " << prog_name << " <uda directory> [-t <timestep>] [-p <prune percent>] [-x]" << std::endl;
  std::cerr << std::endl << "Options\n";
  std::cerr << "-t <timestep>\n" << "\tLoads the taskgraph from the given timestep directory in the uda\n" << "\tdirectory.\n";
  std::cerr << "-p <prune percent>\n" << "\tHide nodes and edges with maximum path costs less than <percent>\n"
            << "\tof the critical path cost.\n";
  std::cerr << "-x\n" << "\tNot just hide, but exclude nodes with maximum path costs less than\n"
       << "\tthe set pruning percent.  This is useful for very large graphs.\n";
}

int main(int argc, char* argv[])
{
  if (argc < 2) {
    usage(argv[0]);
    return 1;
  }

  udaDir = argv[1];

  int timestep = 1;
  float prune_percent = 0;
  bool do_exclusion = false;

  for (int i = 2; i < argc; i++) {
    if (argv[i][0] == '-') {
      if (argv[i][1] == 't') {
        if (++i >= argc) {
          usage(argv[0]);
          return 1;
        }
        timestep = atoi(argv[i]);
      } else if (argv[i][1] == 'p') {
        if (++i >= argc) {
          usage(argv[0]);
          return 1;
        }
        prune_percent = atof(argv[i]);
      } else if (argv[i][1] == 'x')
        do_exclusion = true;
      else {
        std::cerr << "Invalid option " << argv[i] << std::endl;
        usage(argv[0]);
        return 1;
      }
    } else {
      usage(argv[0]);
      return 1;
    }
  }

  gDavinci = DaVinci::run();
  gDavinci->setOrientation(DaVinci::BOTTOM_UP);
  DaVinci::doExclusion = do_exclusion;

  gGraph = NULL;

  bool loaded = load_timestep(timestep, prune_percent);
  if (!loaded) {
    std::cerr << "Failed reading task graph.  Quitting.\n";
    return 1;
  }

  std::cout << HELP_MSG << std::endl;

  while (!gQuit) {
    while (!gEventQueue.empty()) {
      Event event = gEventQueue.front();
      gEventQueue.pop();

      std::cout << "Handling event (type=" << event.type() << ")\n";
      std::cout << std::endl;
      handle_event(event);
    }

    /* Now that we've handled all pending events, we wait for input either
     * from the user at the console or from daVinci (presumably also
     * initiated by the user)
     */
    // select() changes the FD sets, so we have to reset them every time
    fd_set inputs, errors;
    int dv_fd = gDavinci->getOutput();
    FD_ZERO(&inputs);
    FD_ZERO(&errors);
    FD_SET(0 /* stdin */,&inputs);
    FD_SET(0 /* stdin */,&errors);
    FD_SET(dv_fd, &inputs);
    FD_SET(dv_fd, &errors);

    std::cout << std::endl << "? " << std::flush;

    while ((select(dv_fd + 1, &inputs, 0, &errors, 0) == -1) && (errno == EINTR)) {
      ;
    }
    if ((FD_ISSET(0, &errors)) || FD_ISSET(dv_fd, &errors)) {
      gQuit = true;
    }
    else if (FD_ISSET(0, &inputs)) {
      handle_console_input();
    } else if (FD_ISSET(dv_fd, &inputs)) {
      gDavinci->handleInput();
    }
  }

  delete gDavinci;
  delete gGraph;

  return 0;
}

static
void handle_event(const Event& event)
{
  switch (event.type()) {
    case DaVinci::EVT_DV_QUIT :
      gQuit = true;
      break;

    case DaVinci::EVT_DV_SELECT_NODE : {
      std::list<std::string> selected_nodes = gDavinci->getSelectedNodes();
      if (selected_nodes.size() == 1) {
        std::cout << selected_nodes.front() << std::endl;
        GV_Task* pTask = gGraph->findTask(selected_nodes.front());
        if (pTask != NULL) {
          std::cout << "\tCost (duration): " << pTask->getDuration() << std::endl;
          std::cout << "\tMax Path Cost: " << pTask->getMaxInclusivePathCost() << std::endl;
          std::cout << "\tMax Path Percent: " << pTask->getMaxPathPercent() << std::endl;
        } else {
          std::cout << "\tError, task not found\n";
        }
      } else if (selected_nodes.size() > 1) {
        double total_cost = 0;
        for (std::list<std::string>::iterator iter = selected_nodes.begin(); iter != selected_nodes.end(); iter++) {
          std::cout << *iter;
          GV_Task* pTask = gGraph->findTask(*iter);
          if (pTask == NULL) {
            std::cout << "\n\tError, task not found\n";
            return;
          }
          std::cout << "\t(" << pTask->getDuration() << ")\n";
          total_cost += pTask->getDuration();
        }
        std::cout << "\nTotal cost (duration): " << total_cost << std::endl;

      }
    }
      break;

    case DaVinci::EVT_DV_SELECT_EDGE :
      std::cout << gDavinci->getSelectedEdge() << std::endl;
      Edge* pEdge = gGraph->findEdge(gDavinci->getSelectedEdge());
      if (pEdge != NULL) {
        std::cout << "\tMax Path: " << pEdge->getMaxInclusivePathCost() << std::endl;
        std::cout << "\tMax Path Percent: " << pEdge->getMaxPathPercent() << std::endl;
      } else {
        std::cout << "\tError, edge not found\n";
      }
      break;
  }
}

bool load_timestep(int timestep,
                   float prune_percent)
{
  std::ostringstream timedir;
  timedir << "/t" << std::setw(5) << std::setfill('0') << timestep;
  std::cout << "Loading timestep " << timestep << "...\n";
  GV_TaskGraph* oldGraph = gGraph;
  gGraph = GV_TaskGraph::inflate(udaDir + timedir.str());

  if (gGraph != NULL) {
    gGraph->setThresholdPercent(prune_percent);
    std::cout << "Sending graph to daVinci...\n";
    gDavinci->setGraph(gGraph);
    std::cout << "Graph sent.\n";
    delete oldGraph;
    return true;
  } else {
    gGraph = oldGraph;
    return false;
  }
}

static void handle_console_input()
{
  std::string cmd;
  std::cin >> cmd;

  switch (tolower(cmd.c_str()[0])) {
    case 'h' :  // help
      std::cout << HELP_MSG << std::endl;
      break;

    case 'p' : { 	// prune
      float percent;
      std::cin >> percent;
      if (percent < 0) {
        percent = 0;
      }
      if (percent > 1) {
        percent = 1;
      }
      if (gGraph != NULL && gDavinci != NULL) {
        std::cout << "\nSetting threshold... " << percent << std::endl << std::endl;
        gGraph->setThresholdPercent(percent);
        gDavinci->setGraph(gGraph);  // refresh graph
      }
    }
      break;

    case 'x' :
      // turn exclusion on or off
      DaVinci::doExclusion = !DaVinci::doExclusion;
      std::cout << "Exclusion " << (DaVinci::doExclusion ? "on\n" : "off\n");
      gDavinci->setGraph(gGraph);

      break;

    case 'q' :
      gQuit = true;
      break;

    case 't' : {
      int timestep;
      if (std::cin.peek() == 't' || std::cin.peek() == 'T') {
        std::cin.get();
      }
      std::cin >> timestep;

      if (!load_timestep(timestep, gGraph->getThresholdPercent())) {
        std::cout << "Use the 'List' command to get a list of timestep directories\n";
      }
    }
      break;

    case 'f' : {
      // set the fontsize in daVinci
      int font_size;
      std::cin >> font_size;
      std::cout << "Setting font size to " << font_size << "...\n";
      gDavinci->setFontSize(font_size);

    }
      break;

    case 'l' : {
      DataArchive dataArchive(udaDir);
      std::vector<int> timeindices;
      std::vector<double> times;
      dataArchive.queryTimesteps(timeindices, times);

      std::cout << "\nTimesteps:\n";
      for (int i = 0; i < static_cast<int>(timeindices.size()); i++) {
        std::cout << timeindices[i] << std::endl;
      }

      //system((string("find ") +  udaDir + " -name 'taskgraph_00000.xml' | sed -e \"s/\\/taskgraph_00000\\.xml//g\" | sed -e \"s/.*\\///g\"").c_str());
    }
      break;

    default :
      std::cerr << "Unknown command: " << cmd << std::endl;
      break;
  }

  // clear is an easy way to prevent an error flag in cin from
  // causing the program to get caught in a loop saying "Unknown command"
  std::cin.clear();
}

