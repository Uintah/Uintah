/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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
  procName_t my_proc_name;
  int procNameLength;

  MPI::Get_processor_name( my_proc_name, &procNameLength );
  my_proc_name[procNameLength] = '\0';

  procName_t *all_proc_names = new procName_t[m_nRanks];

  m_all_proc_indexs.resize(m_nRanks);

  // Gather all of the processor names in rank order.
  MPI::Allgather(  my_proc_name,  MPI_MAX_PROCESSOR_NAME+1, MPI_CHAR,
                  all_proc_names, MPI_MAX_PROCESSOR_NAME+1, MPI_CHAR,
                  m_comm );

  std::map< std::string, unsigned int > proc_name_map;

  // Loop through each rank and map its processor name to a node index.
  for( int i=0; i<nRanks; ++i )
  {
    std::string proc_name = std::string(all_proc_names[i]);

    // If the name is not found add it to the map.
    if( proc_name_map.find( proc_name ) == proc_name_map.end() )
    {
      int nNodes = proc_name_map.size();

      // Add the name to map so to track unique processor names.
      proc_name_map[ proc_name ] = nNodes;

      // Store the processor name for later look up by node index.
      m_all_proc_names.push_back( proc_name );
    }

    // For each rank save it's node index. This index can be
    // used to get the node name.
    m_all_proc_indexs[i] = proc_name_map[ proc_name ];
  }

  delete [] all_proc_names;

  // More than one node so create a node based communicator.
  if( proc_name_map.size() > 1 )
  {
    // The node index becomes the "color" while using the world rank
    // which is unique for the ordering.
    MPI::Comm_split(m_comm, m_all_proc_indexs[m_rank], m_rank, &m_node_comm);

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
  // This free call causes a hard crash.
  // if( m_all_proc_names.size() > 1 )
  // MPI::Comm_free( &m_node_comm );
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

// For this rank get the node name.
std::string ProcessorGroup::myNodeName() const
{
  return m_all_proc_names[ m_all_proc_indexs[ m_rank ] ];
}

// For any rank get the node index.
int ProcessorGroup::getNodeIndexFromRank( unsigned int rank ) const
{
  // Make sure the rank is valid.
  if( 0 <= rank && rank < m_all_proc_indexs.size() )
    return m_all_proc_indexs[ rank ];
  else
    return -1;
}

// For any rank get the node name.
std::string ProcessorGroup::getNodeNameFromRank( unsigned int rank ) const
{
  // Make sure the rank is valid.
  if( 0 <= rank && rank < m_all_proc_indexs.size() )
    return m_all_proc_names[ m_all_proc_indexs[ rank ] ];
  else
    return std::string( "" );
}

// For any node get the node name.
std::string ProcessorGroup::getNodeName( unsigned int node ) const
{
  // Make sure the node is valid.
  if( 0 <= node && node < m_all_proc_names.size() )
    return m_all_proc_names[ node ];
  else
    return std::string( "" );
}
