/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 *   Copyright (c) 2020 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License version 2 as
 *   published by the Free Software Foundation;
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

#include <ns3/nstime.h>
#include <string>
#include <ostream>
#include "ns3/dash-module.h"
#include "ns3/core-module.h"
#include "ns3/config-store.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/applications-module.h"
#include "ns3/mobility-module.h"
#include "ns3/antenna-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/lte-module.h"
#include <ns3/radio-environment-map-helper.h>
#include "ns3/config-store-module.h"
#include "lena-v1-utils.h"
#include "lena-v2-utils.h"
#include <iomanip>
#include "ns3/log.h"

#include "lena-lte-comparison.h"

#ifndef TRACE_CALLBACKS_H
#define TRACE_CALLBACKS_H

namespace ns3 {

void LogPosition (Ptr<OutputStreamWrapper> stream);
void udpServerTrace(std::pair<uint16_t, uint16_t> DelayPortNums,
               std::pair<uint16_t, uint16_t> FlowPortNums,
                const Ptr<Node> &remoteHost,
                std::string context,
                Ptr<const Packet> packet, 
               const Address &from, const Address &localAddress);
void delayTrace (Ptr<OutputStreamWrapper> stream,
                const Ptr<Node> &remoteHost,
                std::string context,
                Ptr<const Packet> packet, const Address &from, const Address &localAddress);
void dashClientTrace (
                Ptr<OutputStreamWrapper> stream,
                std::string context, 
		ClientDashSegmentInfo & info);
void mpegPlayerTrace (
                Ptr<OutputStreamWrapper> stream,
                std::string context,
                MpegPlayerFrameInfo & mpegInfo);
void httpClientTraceRxDelay ( Ptr<OutputStreamWrapper> stream,
                std::string context, 
		const Time & delay, const Address & from, const uint32_t & size);
void httpClientTraceRxRtt ( Ptr<OutputStreamWrapper> stream,
                std::string context,
		const Time & rtt, const Address & from, const uint32_t & size);
void httpServerTraceRxDelay ( Ptr<OutputStreamWrapper> stream,
                std::string context,
		const Time & delay, const Address & from);
void flowTrace (Ptr<OutputStreamWrapper> stream,
                const Ptr<Node> &remoteHost,
                std::string context,
                Ptr<const Packet> packet, const Address &from, const Address &localAddress);
void rttTrace (Ptr<OutputStreamWrapper> stream,
                std::string context, 
                Ptr<const Packet> packet, const Address &from, const Address &localAddress);


/***************************
 * Trace callbacks 
 ***************************/

void LogPosition (Ptr<OutputStreamWrapper> stream)
{
  for (uint32_t ueId = 0; ueId < ueNodes.GetN (); ++ueId){
          Ptr<Node> ue_node = ueNodes.Get (ueId);
          Ptr<MobilityModel> mob_model = ue_node->GetObject<MobilityModel>();
          Vector pos = mob_model->GetPosition ();
          Vector vel = mob_model->GetVelocity ();
	  
          *stream->GetStream() 
	    << Simulator::Now ().GetMicroSeconds ()
            << "\t" << ueId
	    << "\t" << GetImsi_from_ueId(ueId)
	    << "\t" << GetCellId_from_ueId(ueId)
            << "\t" << pos.x << "\t" << pos.y << "\t" << pos.z
            << "\t" << vel.x << "\t" << vel.y << "\t" << vel.z
            << std::endl;
          }
  Simulator::Schedule (MilliSeconds(500), &LogPosition, stream);
}

// Trace Callback for UdpServer. This included both delayTrace and flowTrace 
// since they both use UdpServers
void
udpServerTrace(std::pair<uint16_t, uint16_t> DelayPortNums,
               std::pair<uint16_t, uint16_t> FlowPortNums,
                const Ptr<Node> &remoteHost,
                std::string context,
                Ptr<const Packet> packet, 
               const Address &from, const Address &localAddress){
    
    if ( (InetSocketAddress::ConvertFrom (localAddress).GetPort () == DelayPortNums.first) || (InetSocketAddress::ConvertFrom (localAddress).GetPort () == DelayPortNums.second)) {
        delayTrace (delayStream, remoteHost, 
                    context, packet, from, localAddress);     
    }
    
    if ( (InetSocketAddress::ConvertFrom (localAddress).GetPort () == FlowPortNums.first) || (InetSocketAddress::ConvertFrom (localAddress).GetPort () == FlowPortNums.second)) {
        flowTrace (flowStream, remoteHost, 
                    context, packet, from, localAddress);
        }
        
}
    
    
// This includes both the UL and DL delay trace callbacks 
void
delayTrace (Ptr<OutputStreamWrapper> stream, 
                const Ptr<Node> &remoteHost,
                std::string context,
                Ptr<const Packet> packet, const Address &from, const Address &localAddress)
{

  uint16_t receiver_nodeId = GetNodeIdFromContext(context);
  uint16_t remoteHostId = remoteHost->GetId ();
  //Create a copy of the packet so we can modify it
  Ptr<Packet> packet_copy = packet->Copy();
  SeqTsHeader seqTs;
  packet_copy->RemoveHeader (seqTs);

  std::string dir;
  uint16_t ueId;
  // UL 
  if(receiver_nodeId == remoteHostId){
    dir = "UL";
    ueId = GetUeNodeIdFromIpAddr (from, &ueNodes, &ueIpIfaces);
  }
  // DL
  else {
    dir = "DL";
    ueId = GetUeIdFromNodeId( GetNodeIdFromContext( context ) );
  } 
    
  Ptr<Node> ue_node = ueNodes.Get (ueId);
  if (InetSocketAddress::IsMatchingType (from)){
      *stream->GetStream()
          << Simulator::Now ().GetMicroSeconds ()
              << "\t" << dir
              << "\t" << ueId // ue global id
	 << "\t" << GetImsi_from_ueId(ueId)
	 << "\t" << GetCellId_from_ueId(ueId)
         << "\t" << packet_copy->GetSize () // received size
         << "\t" << seqTs.GetSeq () //current sequence number
         << "\t" << packet_copy->GetUid ()
         << "\t" << (seqTs.GetTs ()).GetMicroSeconds ()
         << "\t" << (Simulator::Now () - seqTs.GetTs ()).GetMicroSeconds ()
         << std::endl;
    }
}

void
flowTrace (Ptr<OutputStreamWrapper> stream,
                const Ptr<Node> &remoteHost,
                std::string context,
                Ptr<const Packet> packet, const Address &from, const Address &localAddress)
{

  uint16_t receiver_nodeId = GetNodeIdFromContext(context);
  uint16_t remoteHostId = remoteHost->GetId ();
  //Create a copy of the packet so we can modify it
  Ptr<Packet> packet_copy = packet->Copy();
  SeqTsHeader seqTs;
  packet_copy->RemoveHeader (seqTs);

  std::string dir;
  uint16_t ueId;
  // UL 
  if(receiver_nodeId == remoteHostId){
    dir = "UL";
    ueId = GetUeNodeIdFromIpAddr (from, &ueNodes, &ueIpIfaces);
  }
  // DL
  else {
    dir = "DL";
    ueId = GetUeIdFromNodeId( GetNodeIdFromContext( context ) );
  }
  Ptr<Node> ue_node = ueNodes.Get (ueId);
  if (InetSocketAddress::IsMatchingType (from)){
    *stream->GetStream()
         << Simulator::Now ().GetMicroSeconds ()
         << "\t" << dir
         << "\t" << ueId // ue global id
	 << "\t" << GetImsi_from_ueId(ueId)
	 << "\t" << GetCellId_from_ueId(ueId)
         << "\t" << packet_copy->GetSize () // received size
         << "\t" << seqTs.GetSeq () //current sequence number
         << "\t" << packet_copy->GetUid ()
         << "\t" << (seqTs.GetTs ()).GetMicroSeconds () // tx TimeStamp 
         << "\t" << (Simulator::Now () - seqTs.GetTs ()).GetMicroSeconds () // delay
         << std::endl;
    }
}

void rttTrace (Ptr<OutputStreamWrapper> stream,
		std::string context, 
		Ptr<const Packet> packet, const Address &from, const Address &localAddress)
{
  //Create a copy of the packet so we can modify it
  Ptr<Packet> packet_copy = packet->Copy();
  SeqTsHeader seqTs;
  packet_copy->RemoveHeader (seqTs);
  uint16_t ueId = GetUeIdFromNodeId( GetNodeIdFromContext( context ) );

  if (InetSocketAddress::IsMatchingType (from))
  {
    *stream->GetStream() 
	 << Simulator::Now ().GetMicroSeconds ()
	 << "\t" << ueId // ue global id
	 << "\t" << GetImsi_from_ueId(ueId)
	 << "\t" << GetCellId_from_ueId(ueId)
         << "\t" << packet_copy->GetSize () // received size
         << "\t" << seqTs.GetSeq () //current sequence number
         << "\t" << packet_copy->GetUid ()
         << "\t" << (seqTs.GetTs ()).GetMicroSeconds () // tx TimeStamp
         << "\t" << (Simulator::Now () - seqTs.GetTs ()).GetMicroSeconds () // rtt
         << std::endl;
  }
}

void
dashClientTrace (
                Ptr<OutputStreamWrapper> stream,
                std::string context,
                ClientDashSegmentInfo & info)
{
	
  uint16_t ueId = GetUeIdFromNodeId (GetNodeIdFromContext(context));

  *stream->GetStream()
          << Simulator::Now ().GetMicroSeconds () //tstamp_us
          //<< "\t" << (info.node_id+1) //IMSI the + 1 is because 
	  //this is indexed from 0 and IMSI is indexed from 1
          << "\t" << ueId
	  << "\t" << GetImsi_from_ueId(ueId)
	  << "\t" << GetCellId_from_ueId(ueId)
          << "\t" << info.videoId
          << "\t" << info.segId
          << "\t" << info.bitRate //newBitRate
          << "\t" << info.oldBitRate //oldBitRate
          << "\t" << info.thputOverLastSeg_bps
          << "\t" << info.estBitRate //estBitRate, average segment bitrate over last window seconds.
          << "\t" << info.frameQueueSize //frameQueueSize in number of frames
          << "\t" << info.interTime_s //interTime_s, the time for which it was paused due to empty queue
          << "\t" << info.playbackTime
          << "\t" << info.realplayTime // realplayTime or bufferTime, 
	  //this is GetRealPlaytime (mpeg_header.GetPlaybackTime())
          // currDt does not seem to indicate the current size of the mpeg player buffer,
          // but is still refered to as buffering time
          << "\t" << info.Dt
          << "\t" << info.timeToNextReq //time to wait before next segment is requested.
          << std::endl;

}

void
mpegPlayerTrace (
                Ptr<OutputStreamWrapper> stream,
                std::string context,
                MpegPlayerFrameInfo & mpegInfo)
{

  uint16_t ueId = GetUeIdFromNodeId (GetNodeIdFromContext(context));
  *stream->GetStream()
          << Simulator::Now ().GetMicroSeconds ()
          << "\t" << ueId
	  << "\t" << GetImsi_from_ueId(ueId)
	  << "\t" << GetCellId_from_ueId(ueId)
          //<< "\t" << mpegInfo.node_id //The NS_LOG below is using m_dashClient->m_id which is different from the older version from which I copied m_dashClient->GetId(), so watch out this ID could be wrong. 
          << "\t" << mpegInfo.videoId
          << "\t" << mpegInfo.segId
          << "\t" << mpegInfo.resolution
          << "\t" << mpegInfo.frame_id
          << "\t" << mpegInfo.playback_time
          << "\t" << mpegInfo.type
          << "\t" << mpegInfo.size
          << "\t" << mpegInfo.interruption_time
          << "\t" << mpegInfo.queue_size
          << std::endl;
}

void httpClientTraceRxDelay ( Ptr<OutputStreamWrapper> stream,
		std::string context,
		const Time & delay, const Address & from, const uint32_t & size)
{
  uint16_t ueId = GetUeIdFromNodeId (GetNodeIdFromContext(context));

  *stream->GetStream()
          << Simulator::Now ().GetMicroSeconds () //tstamp_us
          << "\t" << ueId
	  << "\t" << GetImsi_from_ueId(ueId)
	  << "\t" << GetCellId_from_ueId(ueId)
	  << "\t" << size
	  << "\t" << delay.GetMicroSeconds () // dl delay 
          << std::endl;
}

void httpClientTraceRxRtt ( Ptr<OutputStreamWrapper> stream,
                std::string context,
		const Time & rtt, const Address & from, const uint32_t & size)
{
  uint16_t ueId = GetUeIdFromNodeId (GetNodeIdFromContext(context));

  *stream->GetStream()
          << Simulator::Now ().GetMicroSeconds () //tstamp_us
          << "\t" << ueId
	  << "\t" << GetImsi_from_ueId(ueId)
	  << "\t" << GetCellId_from_ueId(ueId)
	  << "\t" << size
          << "\t" << rtt.GetMicroSeconds () // dl delay 
          << std::endl;
}


void httpServerTraceRxDelay ( Ptr<OutputStreamWrapper> stream,
                std::string context,
		const Time & delay, const Address & from)
{
	
  uint16_t ueId = GetUeNodeIdFromIpAddr (from, &ueNodes, &ueIpIfaces);

  *stream->GetStream()
          << Simulator::Now ().GetMicroSeconds () //tstamp_us
          << "\t" << ueId
	  << "\t" << GetImsi_from_ueId(ueId)
	  << "\t" << GetCellId_from_ueId(ueId)
	  << "\t" << delay.GetMicroSeconds() 
          << std::endl;
	  
}

} // namespace ns3

#endif // TRACE_CALLBACKS_H
