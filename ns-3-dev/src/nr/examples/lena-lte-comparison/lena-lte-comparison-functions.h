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
//#include <ns3/sqlite-output.h>
//#include "flow-monitor-output-stats.h"
#include "lena-v1-utils.h"
#include "lena-v2-utils.h"
#include <iomanip>
#include "ns3/log.h"


#include "lena-lte-comparison.h"


#ifndef LENA_LTE_COMPARISON__FUNCTION_H
#define LENA_LTE_COMPARISON_FUNCTION_H

namespace ns3 {

/***************************
 * Global declarations
 ***************************/

const Time appStartWindow = MilliSeconds (50);
uint16_t NUM_GNBS = 0; // needs to be set in code 

NodeContainer gnbNodes;
NodeContainer ueNodes;
Ipv4InterfaceContainer ueIpIfaces;

/***************************
 * Structure Definitions
 ***************************/

struct CallbackStruct
{
    Ptr<OutputStreamWrapper> stream;
    NodeDistributionScenarioInterface* scenario;

    // equality comparison. doesn't modify object. therefore const.
    bool operator==(const CallbackStruct& a) const
    {
        return (stream == a.stream && scenario == a.scenario);
    }
    // NOT equality comparison. doesn't modify object. therefore const.
    bool operator!=(const CallbackStruct& a) const
    {
        return (!(stream == a.stream && scenario == a.scenario));
    }
};

/***************************
 * Function Declarations
 ***************************/

uint16_t GetUeIdFromNodeId (uint16_t nodeId);
uint16_t GetImsiFromUeId (uint16_t ueId);
uint16_t GetNodeIdFromContext (std::string context);
uint16_t GetUeNodeIdFromIpAddr (Address ip_addr, const NodeContainer* ueNodes, const Ipv4InterfaceContainer* ueIpIfaces);
void LogPosition (Ptr<OutputStreamWrapper> stream, NodeDistributionScenarioInterface* scenario);
void delayTrace (Ptr<OutputStreamWrapper> stream, NodeDistributionScenarioInterface* scenario,
                const Ptr<Node> &remoteHost,
                std::string context,
                Ptr<const Packet> packet, const Address &from, const Address &localAddress);
void dashClientTrace (
                Ptr<OutputStreamWrapper> stream, NodeDistributionScenarioInterface* scenario,
                std::string context, 
		ClientDashSegmentInfo & info);
void mpegPlayerTrace (
                Ptr<OutputStreamWrapper> stream, NodeDistributionScenarioInterface* scenario,
                std::string context,
                MpegPlayerFrameInfo & mpegInfo);
void httpClientTraceRxDelay ( Ptr<OutputStreamWrapper> stream, 
		NodeDistributionScenarioInterface* scenario,
                std::string context, 
		const Time & delay, const Address & from, const uint32_t & size);
void httpClientTraceRxRtt ( Ptr<OutputStreamWrapper> stream, 
                NodeDistributionScenarioInterface* scenario,
                std::string context,
		const Time & rtt, const Address & from, const uint32_t & size);
void httpServerTraceRxDelay ( Ptr<OutputStreamWrapper> stream,  
                NodeDistributionScenarioInterface* scenario,
                std::string context,
		const Time & delay, const Address & from);
void flowTrace (Ptr<OutputStreamWrapper> stream, NodeDistributionScenarioInterface* scenario,
                const Ptr<Node> &remoteHost,
                std::string context,
                Ptr<const Packet> packet, const Address &from, const Address &localAddress);
void rttTrace (Ptr<OutputStreamWrapper> stream, NodeDistributionScenarioInterface* scenario,
                std::string context, 
                Ptr<const Packet> packet, const Address &from, const Address &localAddress);
std::pair<ApplicationContainer, Time> 
InstallDashApps (const Ptr<Node> &ue, const Ptr<NetDevice> &ueDevice,
             const Address &ueAddress,
             DashClientHelper *dashClient, const Ptr<Node> &remoteHost,
             const Ipv4Address &remoteHostAddr, Time appStartTime,
             uint16_t portNum, const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime,
             const Ptr<LteHelper> &lteHelper, const Ptr<NrHelper> &nrHelper);
std::pair<ApplicationContainer, Time> 
InstallHttpApps (const Ptr<Node> &ue, const Ptr<NetDevice> &ueDevice,
             const Address &ueAddress,
             ThreeGppHttpClientHelper *httpClient, const Ptr<Node> &remoteHost,
             const Ipv4Address &remoteHostAddr, Time appStartTime,
             uint16_t portNum, const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime,
             const Ptr<LteHelper> &lteHelper, const Ptr<NrHelper> &nrHelper);
std::pair<ApplicationContainer, Time> 
InstallUdpEchoApps (const Ptr<Node> &ue, const Ptr<NetDevice> &ueDevice,
             const Address &ueAddress,
             UdpEchoClientHelper *echoClient, const Ptr<Node> &remoteHost,
             const Ipv4Address &remoteHostAddr, Time appStartTime,
             uint16_t portNum, const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime,
             const Ptr<LteHelper> &lteHelper, const Ptr<NrHelper> &nrHelper);
std::pair<ApplicationContainer, Time> 
InstallUlFlowTrafficApps (const Ptr<Node> &ue, const Ptr<NetDevice> &ueDevice,
             const Address &ueAddress,
             UdpClientHelper *ulFlowClient,
             const Ptr<Node> &remoteHost,
             const Ipv4Address &remoteHostAddr, Time appStartTime,
             uint16_t ulFlowPortNum, const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime,
             const Ptr<LteHelper> &lteHelper, const Ptr<NrHelper> &nrHelper);
std::pair<ApplicationContainer, Time> 
InstallDlFlowTrafficApps (const Ptr<Node> &ue, const Ptr<NetDevice> &ueDevice,
             const Address &ueAddress,
             UdpClientHelper *dlFlowClient,
             const Ptr<Node> &remoteHost,
             const Ipv4Address &remoteHostAddr, Time appStartTime,
             uint16_t dlFlowPortNum, const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime,
             const Ptr<LteHelper> &lteHelper, const Ptr<NrHelper> &nrHelper);
std::pair<ApplicationContainer, Time> 
InstallUlDelayTrafficApps (const Ptr<Node> &ue, const Ptr<NetDevice> &ueDevice,
             const Address &ueAddress,
             UdpClientHelper *ulDelayClient,
             const Ptr<Node> &remoteHost,
             const Ipv4Address &remoteHostAddr, Time appStartTime,
             uint16_t ulDelayPortNum,
             const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime,
             const Ptr<LteHelper> &lteHelper, const Ptr<NrHelper> &nrHelper);
std::pair<ApplicationContainer, Time> 
InstallDlDelayTrafficApps (const Ptr<Node> &ue, const Ptr<NetDevice> &ueDevice,
             const Address &ueAddress,
             UdpClientHelper *dlDelayClient,
             const Ptr<Node> &remoteHost,
             const Ipv4Address &remoteHostAddr, Time appStartTime,
             uint16_t dlDelayPortNum,
             const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime,
             const Ptr<LteHelper> &lteHelper, const Ptr<NrHelper> &nrHelper);

/***************************
 * Function Definitions
 ***************************/

uint16_t
GetUeIdFromNodeId (uint16_t nodeId){
  return(nodeId - NUM_GNBS);
}

uint16_t
GetImsiFromUeId (uint16_t ueId){
  return(ueId + 1);
}

uint16_t
GetNodeIdFromContext (std::string context){
  std::string path = context.substr (10, context.length());
  std::string nodeIdStr = path.substr (0, path.find ("/"));
  uint16_t nodeId = stoi(nodeIdStr);
  return (nodeId);
}

uint16_t
GetUeNodeIdFromIpAddr (Address ip_addr, const NodeContainer* ueNodes,
                const Ipv4InterfaceContainer* ueIpIfaces
                ) {
  bool knownSender = false;
  for (uint32_t ueId = 0; ueId < ueNodes->GetN (); ++ueId){
    Ptr<Node> ue_node = ueNodes->Get (ueId);
    Ipv4Address ue_addr = ueIpIfaces->GetAddress (ueId);

    if (ue_addr == InetSocketAddress::ConvertFrom (ip_addr).GetIpv4 ()) {
      knownSender = true;
      return (ueId);
    }
  }
  NS_ASSERT (knownSender);
  return(666); // the number of the beast
}


/***************************
 * Trace callbacks 
 ***************************/
void LogPosition (Ptr<OutputStreamWrapper> stream, NodeDistributionScenarioInterface* scenario)
{
  for (uint32_t ueId = 0; ueId < ueNodes.GetN (); ++ueId){
          Ptr<Node> ue_node = ueNodes.Get (ueId);
          Ptr<MobilityModel> mob_model = ue_node->GetObject<MobilityModel>();
          Vector pos = mob_model->GetPosition ();
          Vector vel = mob_model->GetVelocity ();
	  
          *stream->GetStream() 
	    << Simulator::Now ().GetMicroSeconds ()
            << "\t" << ueId
            << "\t" << scenario->GetCellIndex (ueId)
            << "\t" << pos.x << "\t" << pos.y << "\t" << pos.z
            << "\t" << vel.x << "\t" << vel.y << "\t" << vel.z
            << std::endl;
          }
  Simulator::Schedule (MilliSeconds(500), &LogPosition, stream, scenario);
}

// This includes both the UL and DL delay trace callbacks 
void
delayTrace (Ptr<OutputStreamWrapper> stream, NodeDistributionScenarioInterface* scenario,
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
         << "\t" << scenario->GetCellIndex (ueId) // cell global id
         << "\t" << packet_copy->GetSize () // received size
         << "\t" << seqTs.GetSeq () //current sequence number
         << "\t" << packet_copy->GetUid ()
         << "\t" << (seqTs.GetTs ()).GetMicroSeconds ()
         << "\t" << (Simulator::Now () - seqTs.GetTs ()).GetMicroSeconds ()
         << std::endl;
    }
}

void
flowTrace (Ptr<OutputStreamWrapper> stream, NodeDistributionScenarioInterface* scenario,
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
         << "\t" << scenario->GetCellIndex (ueId) // cell global id
         << "\t" << packet_copy->GetSize () // received size
         << "\t" << seqTs.GetSeq () //current sequence number
         << "\t" << packet_copy->GetUid ()
         << "\t" << (seqTs.GetTs ()).GetMicroSeconds () // tx TimeStamp 
         << "\t" << (Simulator::Now () - seqTs.GetTs ()).GetMicroSeconds () // delay
         << std::endl;
    }
}

void rttTrace (Ptr<OutputStreamWrapper> stream, NodeDistributionScenarioInterface* scenario,
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
	 << "\t" << scenario->GetCellIndex (ueId) // cell global id
         << "\t" << packet_copy->GetSize () // received size
         << "\t" << seqTs.GetSeq () //current sequence number
         << "\t" << packet_copy->GetUid ()
         << "\t" << (seqTs.GetTs ()).GetMicroSeconds () // rtt
         << "\t" << (seqTs.GetTs ()).GetMicroSeconds () // tx TimeStamp
         << "\t" << (Simulator::Now () - seqTs.GetTs ()).GetMicroSeconds () // rtt
         << std::endl;
  }
}

void
dashClientTrace (
                Ptr<OutputStreamWrapper> stream, NodeDistributionScenarioInterface* scenario,
                std::string context,
                ClientDashSegmentInfo & info)
{
	
  uint16_t ueId = GetUeIdFromNodeId (GetNodeIdFromContext(context));
  uint16_t cellId = scenario->GetCellIndex (ueId);

  *stream->GetStream()
          << Simulator::Now ().GetMicroSeconds () //tstamp_us
          //<< "\t" << (info.node_id+1) //IMSI the + 1 is because 
	  //this is indexed from 0 and IMSI is indexed from 1
          << "\t" << ueId
          << "\t" << cellId
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
                Ptr<OutputStreamWrapper> stream, NodeDistributionScenarioInterface* scenario,
                std::string context,
		//uint16_t & temp)
                MpegPlayerFrameInfo & mpegInfo)
{

  uint16_t ueId = GetUeIdFromNodeId (GetNodeIdFromContext(context));
  uint16_t cellId = scenario->GetCellIndex (ueId);
  *stream->GetStream()
          << Simulator::Now ().GetMicroSeconds ()
          << "\t" << ueId
          << "\t" << cellId
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
		NodeDistributionScenarioInterface* scenario,
		std::string context,
		const Time & delay, const Address & from, const uint32_t & size)
{
  uint16_t ueId = GetUeIdFromNodeId (GetNodeIdFromContext(context));
  uint16_t cellId = scenario->GetCellIndex (ueId);

  *stream->GetStream()
          << Simulator::Now ().GetMicroSeconds () //tstamp_us
          << "\t" << ueId
          << "\t" << cellId
	  << "\t" << size
	  << "\t" << delay.GetMicroSeconds () // dl delay 
          << std::endl;
}

void httpClientTraceRxRtt ( Ptr<OutputStreamWrapper> stream,
                NodeDistributionScenarioInterface* scenario,
                std::string context,
		const Time & rtt, const Address & from, const uint32_t & size)
{
  uint16_t ueId = GetUeIdFromNodeId (GetNodeIdFromContext(context));
  uint16_t cellId = scenario->GetCellIndex (ueId);

  *stream->GetStream()
          << Simulator::Now ().GetMicroSeconds () //tstamp_us
          << "\t" << ueId
          << "\t" << cellId
	  << "\t" << size
          << "\t" << rtt.GetMicroSeconds () // dl delay 
          << std::endl;
}


void httpServerTraceRxDelay ( Ptr<OutputStreamWrapper> stream,
                NodeDistributionScenarioInterface* scenario,
                std::string context,
		const Time & delay, const Address & from)
{
	
  uint16_t ueId = GetUeNodeIdFromIpAddr (from, &ueNodes, &ueIpIfaces);
  uint16_t cellId = scenario->GetCellIndex (ueId);

  *stream->GetStream()
          << Simulator::Now ().GetMicroSeconds () //tstamp_us
          << "\t" << ueId
          << "\t" << cellId
	  << "\t" << delay.GetMicroSeconds() 
          << std::endl;
	  
}

/***********************************************
 * Install client applications
 **********************************************/

std::pair<ApplicationContainer, Time>
InstallDashApps (const Ptr<Node> &ue, const Ptr<NetDevice> &ueDevice,
             const Address &ueAddress,
             DashClientHelper *dashClient, const Ptr<Node> &remoteHost,
             const Ipv4Address &remoteHostAddr, Time appStartTime,
             uint16_t portNum, const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime,
             const Ptr<LteHelper> &lteHelper, const Ptr<NrHelper> &nrHelper)
{
  ApplicationContainer app;
  app = dashClient->Install (ue);
  double start = x->GetValue (appStartTime.GetMilliSeconds (),
                              (appStartTime + appStartWindow).GetMilliSeconds ());
  Time startTime = MilliSeconds (start);
  app.Start (startTime);
  app.Stop (startTime + appGenerationTime);

  return std::make_pair (app, startTime);
}

std::pair<ApplicationContainer, Time>
InstallHttpApps (const Ptr<Node> &ue, const Ptr<NetDevice> &ueDevice,
             const Address &ueAddress,
             ThreeGppHttpClientHelper *httpClient, const Ptr<Node> &remoteHost,
             const Ipv4Address &remoteHostAddr, Time appStartTime,
             uint16_t portNum, const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime,
             const Ptr<LteHelper> &lteHelper, const Ptr<NrHelper> &nrHelper)
{
  ApplicationContainer app;
  app = httpClient->Install (ue);
  double start = x->GetValue (appStartTime.GetMilliSeconds (),
                              (appStartTime + appStartWindow).GetMilliSeconds ());
  Time startTime = MilliSeconds (start);
  app.Start (startTime);
  app.Stop (startTime + appGenerationTime);
  return std::make_pair (app, startTime);
}


std::pair<ApplicationContainer, Time>
InstallUdpEchoApps (const Ptr<Node> &ue, const Ptr<NetDevice> &ueDevice,
             const Address &ueAddress,
             UdpEchoClientHelper *echoClient, const Ptr<Node> &remoteHost,
             const Ipv4Address &remoteHostAddr, Time appStartTime,
             uint16_t portNum, const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime,
             const Ptr<LteHelper> &lteHelper, const Ptr<NrHelper> &nrHelper)
{
  ApplicationContainer app;
  app = echoClient->Install (ue);
  double start = x->GetValue (appStartTime.GetMilliSeconds (),
                              (appStartTime + appStartWindow).GetMilliSeconds ());
  Time startTime = MilliSeconds (start);
  app.Start (startTime);
  app.Stop (startTime + appGenerationTime);
  return std::make_pair (app, startTime);
}

std::pair<ApplicationContainer, Time>
InstallUlFlowTrafficApps (const Ptr<Node> &ue, const Ptr<NetDevice> &ueDevice,
             const Address &ueAddress,
             UdpClientHelper *ulFlowClient,
             const Ptr<Node> &remoteHost,
             const Ipv4Address &remoteHostAddr, Time appStartTime,
             uint16_t ulFlowPortNum, const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime,
             const Ptr<LteHelper> &lteHelper, const Ptr<NrHelper> &nrHelper)
{
  ApplicationContainer app;
  ulFlowClient->SetAttribute ("RemoteAddress", AddressValue (remoteHostAddr));
  app = ulFlowClient->Install (ue);

  double start = x->GetValue (appStartTime.GetMilliSeconds (),
                              (appStartTime + appStartWindow).GetMilliSeconds ());
  Time startTime = MilliSeconds (start);
  app.Start (startTime);
  app.Stop (startTime + appGenerationTime);
  return std::make_pair (app, startTime);
}

std::pair<ApplicationContainer, Time>
InstallDlFlowTrafficApps (const Ptr<Node> &ue, const Ptr<NetDevice> &ueDevice,
             const Address &ueAddress,
             UdpClientHelper *dlFlowClient,
             const Ptr<Node> &remoteHost,
             const Ipv4Address &remoteHostAddr, Time appStartTime,
             uint16_t dlFlowPortNum, const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime,
             const Ptr<LteHelper> &lteHelper, const Ptr<NrHelper> &nrHelper)
{
  ApplicationContainer app;
  dlFlowClient->SetAttribute ("RemoteAddress", AddressValue (ueAddress));
  app = dlFlowClient->Install (remoteHost);

  double start = x->GetValue (appStartTime.GetMilliSeconds (),
                              (appStartTime + appStartWindow).GetMilliSeconds ());
  Time startTime = MilliSeconds (start);
  app.Start (startTime);
  app.Stop (startTime + appGenerationTime);
  return std::make_pair (app, startTime);
}

std::pair<ApplicationContainer, Time>
InstallUlDelayTrafficApps (const Ptr<Node> &ue, const Ptr<NetDevice> &ueDevice,
             const Address &ueAddress,
             UdpClientHelper *ulDelayClient,
             const Ptr<Node> &remoteHost,
             const Ipv4Address &remoteHostAddr, Time appStartTime,
             uint16_t ulDelayPortNum,
             const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime,
             const Ptr<LteHelper> &lteHelper, const Ptr<NrHelper> &nrHelper)
{
  ApplicationContainer app;
  ulDelayClient->SetAttribute ("RemoteAddress", AddressValue (remoteHostAddr));
  app.Add(ulDelayClient->Install (ue));

  double start = x->GetValue (appStartTime.GetMilliSeconds (),
                              (appStartTime + appStartWindow).GetMilliSeconds ());
  Time startTime = MilliSeconds (start);
  app.Start (startTime);
  app.Stop (startTime + appGenerationTime);
  return std::make_pair (app, startTime);
}

std::pair<ApplicationContainer, Time>
InstallDlDelayTrafficApps (const Ptr<Node> &ue, const Ptr<NetDevice> &ueDevice,
             const Address &ueAddress,
             UdpClientHelper *dlDelayClient,
             const Ptr<Node> &remoteHost,
             const Ipv4Address &remoteHostAddr, Time appStartTime,
             uint16_t dlDelayPortNum,
             const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime,
             const Ptr<LteHelper> &lteHelper, const Ptr<NrHelper> &nrHelper)
{
  ApplicationContainer app;
  dlDelayClient->SetAttribute ("RemoteAddress", AddressValue (ueAddress));
  app = dlDelayClient->Install (remoteHost);

  double start = x->GetValue (appStartTime.GetMilliSeconds (),
                              (appStartTime + appStartWindow).GetMilliSeconds ());
  Time startTime = MilliSeconds (start);
  app.Start (startTime);
  app.Stop (startTime + appGenerationTime);
  return std::make_pair (app, startTime);
}


// Print the scenario parameters

std::ostream &
operator << (std::ostream & os, const Parameters & parameters)
{
  // Use p as shorthand for arg parametersx
  auto p {parameters};

#define MSG(m) \
  os << "\n" << m << std::left \
     << std::setw (40 - strlen (m)) << (strlen (m) > 0 ? ":" : "")


  MSG ("LENA LTE Scenario Parameters");
  MSG ("");
  MSG ("Model version")
    << p.simulator << (p.simulator == "LENA" ? " (v1)" : " (v2)");
  if (p.simulator == "5GLENA")
    {
      //MSG ("LTE Standard")
      //  << p.radioNetwork << (p.radioNetwork == "LTE" ? " (4G)" : " (5G NR)");
      MSG ("4G-NR calibration mode") << (p.calibration ? "ON" : "off");
      MSG ("4G-NR ULPC mode") << (p.enableUlPc ? "Enabled" : "Disabled");
      MSG ("Operation mode") << p.operationMode;
      if (p.operationMode == "TDD")
        {
          MSG ("Numerology") << p.numerologyBwp;
          MSG ("TDD pattern") << p.pattern;
        }
      if (p.errorModel != "")
        {
          MSG ("Error model") << p.errorModel;
        }
      //else if (p.radioNetwork == "LTE")
      //  {
      //    MSG ("Error model") << "ns3::LenaErrorModel";
      //  }
     // else if (p.radioNetwork == "NR")
     //   {
          MSG ("Error model") << "ns3::NrEesmCcT2";
     //   }

    }
  else
    {
      // LENA v1
      p.operationMode = "FDD";
      MSG ("LTE Standard") << "4G";
      MSG ("Calibration mode") << (p.calibration ? "ON" : "off");
      MSG ("LTE ULPC mode") << (p.enableUlPc ? "Enabled" : "Disabled");
      MSG ("Operation mode") << p.operationMode;
    }

  if (p.baseStationFile != "" and p.useSiteFile)
    {
      MSG ("Base station positions") << "read from file " << p.baseStationFile;
    }
  else
    {
      MSG ("Base station positions") << "regular hexaonal lay down";
      MSG ("Number of rings") << p.numOuterRings;
    }
  MSG ("");
  MSG ("Channel bandwidth") << p.bandwidthMHz << " MHz";
  MSG ("Spectrum configuration")
    <<    (p.freqScenario == 0 ? "non-" : "") << "overlapping";
  MSG ("LTE Scheduler") << p.scheduler;

  MSG ("");
  MSG ("Basic scenario") << p.scenario;
  if (p.scenario == "UMa")
    {
      os << "\n  (ISD: 1.7 km, BS: 30 m, UE: 1.5 m, UE-BS min: 30.2 m)";
    }
  else if (p.scenario == "UMi")
    {
      os << "\n  (ISD: 0.5 km, BS: 10 m, UE: 1.5 m, UE-BS min: 10 m)";
    }
  else if (p.scenario == "RMa")
    {
      os << "\n  (ISD: 7.0 km, BS: 45 m, UE: 1.5 m, UE-BS min: 44.6 m)";
    }
  else
    {
      os << "\n  (unknown configuration)";
    }
  if (p.baseStationFile == "" and p.useSiteFile)
    {
      MSG ("Number of outer rings") << p.numOuterRings;
    }
  MSG ("Number of UEs per sector") << p.ueNumPergNb;
  MSG ("Antenna down tilt angle (deg)") << p.downtiltAngle;

  MSG ("");
  MSG ("Network loading") << p.trafficScenario;
  switch (p.trafficScenario)
    {
      case 0:
        MSG ("  Max loading (80 Mbps/20 MHz)");
        MSG ("  Number of packets") << "infinite";
        MSG ("  Packet size");
        switch (p.bandwidthMHz)
          {
            case 20:
              os << "1000 bytes";
              break;
            case 10:
              os << "500 bytes";
              break;
            case 5:
              os << "250 bytes";
              break;
            default:
              os << "1000 bytes";
          }
        // 1 s / (10000 / nUes)
        MSG ("  Inter-packet interval (per UE)") << p.ueNumPergNb / 10.0 << " ms";
        break;

      case 1:
        MSG ("  Latency");
        MSG ("  Number of packets") << 1;
        MSG ("  Packet size") << "12 bytes";
        MSG ("  Inter-packet interval (per UE)") << "1 s";
        break;

      case 2:
        MSG ("  Moderate loading");
        MSG ("  Number of packets") << "infinite";
        MSG ("  Packet size");
        switch (p.bandwidthMHz)
          {
            case 20:
              os << "125 bytes";
              break;
            case 10:
              os << "63 bytes";
              break;
            case 5:
              os << "32 bytes";
              break;
            default:
              os << "125 bytes";
          }
        // 1 s / (1000 / nUes)
        MSG ("  Inter-packet interval (per UE)") << 1 / (1000 / p.ueNumPergNb) << " s";

        break;

      case 3:
        MSG ("  Moderate-high loading");
        MSG ("  Number of packets") << "infinite";
        MSG ("  Packet size");
        switch (p.bandwidthMHz)
          {
            case 20:
              os << "250 bytes";
              break;
            case 10:
              os << "125 bytes";
              break;
            case 5:
              os << "75 bytes";
              break;
            default:
              os << "250 bytes";
          }
        // 1 s / (10000 / nUes)
        MSG ("  Inter-packet interval (per UE)") << 1 / (10000.0 / p.ueNumPergNb) << " s";

        break;
      default:
        os << "\n  (Unknown configuration)";
    }

  MSG ("Application start window") << p.udpAppStartTime.As (Time::MS) << " + " << appStartWindow.As (Time::MS);
  MSG ("Application on duration") << p.appGenerationTime.As (Time::MS);
  MSG ("Traffic direction") << p.direction;

  MSG ("");
  MSG ("Output file name") << p.simTag;
  MSG ("Output directory") << p.outputDir;
  MSG ("Logging") << (p.logging ? "ON" : "off");
  MSG ("Trace file generation") << (p.traces ? "ON" : "off");
  MSG ("");
  os << std::endl;
  return os;
}


} // namespace ns3

#endif // LENA_LTE_COMPARISON_FUNCTION_H
