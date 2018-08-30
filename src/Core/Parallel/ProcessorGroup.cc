/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Util/DOUT.hpp>

#include <cstring>
#include <iostream>

using namespace Uintah;

ProcessorGroup::ProcessorGroup( const ProcessorGroup * parent
                              ,       MPI_Comm         comm
                              ,       int              rank
                              ,       int              nRanks
                              ,       int              threads
                              )
  : m_parent_group(parent)
  , m_comm(comm)
  , m_rank(rank)
  , m_nRanks(nRanks)
  , m_threads(threads)
{
  // Get the processor name for this rank.
  procName_t proc_name;
  int procNameLength;

  MPI::Get_processor_name( proc_name, &procNameLength );
  proc_name[procNameLength] = '\0';

  m_proc_name = std::string(proc_name);
  
  m_all_proc_names = new procName_t[m_nRanks];

  // Gather all of the processor names in rank order.
  MPI::Allgather(      proc_name,  MPI_MAX_PROCESSOR_NAME+1, MPI_CHAR,
                 m_all_proc_names, MPI_MAX_PROCESSOR_NAME+1, MPI_CHAR,
                 MPI_COMM_WORLD);

  // Loop through each rank and map its processor name to a node index.
  for( int i=0; i<nRanks; ++i )
  {
    // If the name is not found add it to the map.
    if( m_proc_name_map.find( std::string(m_all_proc_names[i]) ) ==
        m_proc_name_map.end() )
    {
      m_proc_name_map[ m_all_proc_names[i] ] = m_nNodes;
      m_nNodes = m_proc_name_map.size();
    }
  }

  // Get the index for this processor name.
  m_node = m_proc_name_map[ m_proc_name ];

  // More than one node so create a node based communicator.
  if( m_nNodes > 1 )
  {
    // The node index becomes the "color" while using the world rank
    // which is unique for the ordering.
    MPI::Comm_split(m_comm, m_node, m_rank, &m_node_comm);
    
    // Get the number of ranks for this node and the rank on this node.
    MPI::Comm_size(m_node_comm, &m_node_nRanks);
    MPI::Comm_rank(m_node_comm, &m_node_rank);
  }
  // Only one node so no need for a separtate communicator.
  else
  {
    m_node_nRanks = m_nRanks;
    m_node_rank   = m_rank;
    m_node_comm   = m_comm;
  }
}

ProcessorGroup::~ProcessorGroup() 
{
  // if( m_nNodes > 1 )
  // MPI::Comm_free( &m_node_comm );

  delete [] m_all_proc_names;  
};


void ProcessorGroup::setGlobalComm(int num_comms) const
{
  if (m_threads <= 1) {
    return;
  }

  int curr_size = m_global_comms.size();
  if (num_comms <= curr_size) {
    return;
  }

  m_global_comms.resize(num_comms);
  for (int i = curr_size; i < num_comms; i++) {
    if (Uintah::MPI::Comm_dup(m_comm, &m_global_comms[i]) != MPI_SUCCESS) {
      std::cerr << "Rank: " << m_rank << " - MPI Error in Uintah::MPI::Comm_dup\n";
      Parallel::exitAll(1);
    }
  }
}

// From any rank get the node index.
int ProcessorGroup::getNodeFromRank( int rank ) const
{
  // Make sure the rank is valid.
  if( 0 <= rank && rank < m_nRanks )
  {
    // First get the process name for this rank
    std::string proc_name = m_all_proc_names[rank];

    // Make sure the name exists in the map.
    if( m_proc_name_map.find(proc_name) != m_proc_name_map.end() )
    {
      // Return the index.
      return m_proc_name_map.at(proc_name);
    }
    else
      return -1;
  }
  else
    return -1;
}
