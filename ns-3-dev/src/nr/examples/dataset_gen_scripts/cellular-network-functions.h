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
#include "lte-utils.h"
#include "nr-utils.h"
#include <iomanip>
#include "ns3/log.h"

#include "cellular-network.h"

#ifndef CELLULAR_NETWORK_FUNCTION_H
#define CELLULAR_NETWORK_FUNCTION_H

namespace ns3 {

/***************************
 * Global declarations
 ***************************/

const Time appStartWindow = MilliSeconds (500); // 50
//uint16_t NUM_GNBS = 0; // needs to be set in code 

NodeContainer macroLayerGnbNodes;
NodeContainer microLayerGnbNodes;
NodeContainer allGnbNodes;
NodeContainer ueNodes;
Ipv4InterfaceContainer ueIpIfaces;

Parameters global_params;
    
AsciiTraceHelper traceHelper;
Ptr<OutputStreamWrapper> mobStream;
Ptr<OutputStreamWrapper> delayStream;
Ptr<OutputStreamWrapper> dashClientStream;
Ptr<OutputStreamWrapper> mpegPlayerStream; 
Ptr<OutputStreamWrapper> httpClientDelayStream;
Ptr<OutputStreamWrapper> httpClientRttStream;
Ptr<OutputStreamWrapper> httpServerDelayStream;
Ptr<OutputStreamWrapper> flowStream;
Ptr<OutputStreamWrapper> rttStream;
Ptr<OutputStreamWrapper> topologyStream;         
Ptr<OutputStreamWrapper> fragmentRxStream;
Ptr<OutputStreamWrapper> burstRxStream;   
Ptr<OutputStreamWrapper> handoverStream;
Ptr<OutputStreamWrapper> thputStream;

// Save the ueIds as part of the different groups they are part of 
// fast moving ueIds // no need to do slow since we have alist of all Ids
std::vector<uint32_t> fastUes;
std::vector<uint32_t> dashAppUes;
std::vector<uint32_t> vrAppUes;
std::vector<uint32_t> httpAppUes;
std::vector<uint32_t> onlyDelayUes;

    
    
/***************************
 * Structure Definitions
 ***************************/

/*struct CallbackStruct
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
};*/

/***************************
 * Function Declarations
 ***************************/

uint16_t GetUeIdFromNodeId (uint16_t nodeId);
uint16_t GetNodeIdFromContext (std::string context);
uint16_t GetUeNodeIdFromIpAddr (Address ip_addr, const NodeContainer* ueNodes, const Ipv4InterfaceContainer* ueIpIfaces);
//uint16_t GetCellId_from_cellIndex (uint16_t index, uint16_t num_sites, uint16_t num_sectors);
uint16_t GetCellId_from_cellIndex (uint16_t index);
Ptr<LteEnbNetDevice> GetServingEnbDev_from_ueId (uint16_t ueId);
bool isFromMicroLayer (uint16_t cellId);
bool isFromMacroLayer (uint16_t cellId);    
uint16_t GetImsi_from_ueId(uint16_t ueId);
uint16_t GetCellId_from_ueId(uint16_t ueId);
Ipv4Address GetIpAddrFromUeId(uint16_t ueId);
void ScenarioInfo (NodeDistributionScenarioInterface* scenario);
void NotifyConnectionEstablishedUe (std::string context, uint64_t imsi,
                               uint16_t cellid, uint16_t rnti);
void NotifyConnectionEstablishedEnb (std::string context, uint64_t imsi,
                                uint16_t cellid, uint16_t rnti);
void NotifyHandoverStartUe (std::string context, uint64_t imsi,
                       uint16_t cellid, uint16_t rnti, uint16_t targetCellId);
void NotifyHandoverEndOkUe (std::string context, uint64_t imsi,
                       uint16_t cellid, uint16_t rnti);   
void NotifyHandoverStartEnb (Ptr<OutputStreamWrapper> stream, std::string context, uint64_t imsi,
                        uint16_t cellid, uint16_t rnti, uint16_t targetCellId);
void NotifyHandoverEndOkEnb (std::string context, uint64_t imsi,
                        uint16_t cellid, uint16_t rnti);
void CheckForManualHandovers (Ptr<LteHelper> &lteHelper);
void LogPosition (Ptr<OutputStreamWrapper> stream);
void ThputMeasurement (Ptr<OutputStreamWrapper> stream, std::string context, 
                       Ptr< const Packet > p, const Address &from, const Address &to, 
                       const SeqTsSizeHeader &header);    
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
		const uint16_t & webpageId, const std::string & object_type, const Time & rtt, const Address & from, const uint32_t & size);
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
void BurstRx (Ptr<OutputStreamWrapper> stream,
                std::string context, Ptr<const Packet> burst, const Address &from, const Address &to,
         const SeqTsSizeFragHeader &header);
void FragmentRx (Ptr<OutputStreamWrapper> stream,
                std::string context, Ptr<const Packet> fragment, const Address &from, const Address &to,
         const SeqTsSizeFragHeader &header);    
    
    
std::pair<ApplicationContainer, Time> 
InstallDashApps (const Ptr<Node> &ue,
             DashClientHelper *dashClient, Time appStartTime, 
             const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime);
std::pair<ApplicationContainer, Time> 
InstallHttpApps (const Ptr<Node> &ue,
             ThreeGppHttpClientHelper *httpClient, Time appStartTime,
             const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime);
std::pair<ApplicationContainer, Time> 
InstallUdpEchoApps (const Ptr<Node> &ue,
             UdpEchoClientHelper *echoClient, Time appStartTime,
             const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime);
std::pair<ApplicationContainer, Time> 
InstallUlFlowTrafficApps (const Ptr<Node> &ue,
             const Address &ueAddress,
             UdpClientHelper *ulFlowClient,
             const Ptr<Node> &remoteHost,
             const Ipv4Address &remoteHostAddr, Time appStartTime,
             const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime);
std::pair<ApplicationContainer, Time> 
InstallDlFlowTrafficApps (const Ptr<Node> &ue,
             const Address &ueAddress,
             UdpClientHelper *dlFlowClient,
             const Ptr<Node> &remoteHost,
             const Ipv4Address &remoteHostAddr, Time appStartTime,
             const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime);
std::pair<ApplicationContainer, Time> 
InstallUlDelayTrafficApps (const Ptr<Node> &ue,
             UdpClientHelper *ulDelayClient,
             const Ptr<Node> &remoteHost,
             const Ipv4Address &remoteHostAddr, Time appStartTime,
             const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime);
std::pair<ApplicationContainer, Time> 
InstallDlDelayTrafficApps (const Ptr<Node> &ue,
             const Address &ueAddress,
             UdpClientHelper *dlDelayClient,
             const Ptr<Node> &remoteHost,
             const Ipv4Address &remoteHostAddr, Time appStartTime,
             const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime);
void CreateTraceFiles (void);
    
/***************************
 * Function Definitions
 ***************************/

uint16_t
GetUeIdFromNodeId (uint16_t nodeId){
  return(nodeId - macroLayerGnbNodes.GetN());
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
  // From the Ip addr get NetDevice and then Node
  // There is probably a way to do this through indexing 
  // but I am looping through the entire UE list to get the right node 
  for (uint32_t ueId = 0; ueId < ueNodes->GetN (); ++ueId){
    Ptr<Node> ue_node = ueNodes->Get (ueId);
    Ptr<Ipv4> ipv4 = ue_node->GetObject<Ipv4> ();
    Ipv4InterfaceAddress iaddr = ipv4->GetAddress (1,0); 
    Ipv4Address ue_addr = iaddr.GetLocal ();
    
    if (ue_addr == InetSocketAddress::ConvertFrom (ip_addr).GetIpv4 ()) {
      knownSender = true;
      return (ueId);
    }
  }
  NS_ASSERT (knownSender);
  return(666); // the number of the beast
}

Ipv4Address 
GetIpAddrFromUeId (uint16_t ueId)
{
    Ptr<Node> ue_node = ueNodes.Get (ueId);
    Ptr<Ipv4> ipv4 = ue_node->GetObject<Ipv4> ();
    Ipv4InterfaceAddress iaddr = ipv4->GetAddress (1,0);
    Ipv4Address addr = iaddr.GetLocal ();
    return(addr);
}

bool 
isFromMicroLayer (uint16_t cellId)
{
    for(uint16_t index = 0; index < microLayerGnbNodes.GetN() ; ++index)
    {
        if ( cellId == ( microLayerGnbNodes.Get (index)->GetDevice (0)->GetObject<LteEnbNetDevice> ()->GetCellId() ) )
            return (true);
    }
    return (false);  
}
    
    
bool 
isFromMacroLayer (uint16_t cellId)
{
    for(uint16_t index = 0; index < macroLayerGnbNodes.GetN() ; ++index)
    {
        if ( cellId == ( macroLayerGnbNodes.Get (index)->GetDevice (0)->GetObject<LteEnbNetDevice> ()->GetCellId() ) )
            return (true);
    }
    return (false);
}

/*uint16_t 
GetCellId_from_cellIndex (uint16_t index, uint16_t num_sites, uint16_t num_sectors) {
    return ( (num_sites * (index % num_sectors)) + (index / num_sectors) + 1) ; 
}*/

uint16_t
GetCellId_from_cellIndex (uint16_t index)
{
    return( allGnbNodes.Get (index)->GetDevice (0)->GetObject<LteEnbNetDevice> ()->GetCellId() );
}

uint16_t 
GetImsi_from_ueId(uint16_t ueId){
  uint16_t imsi=0;
  Ptr<Node> ue_node = ueNodes.Get (ueId);
  if (global_params.rat == "NR"){
    imsi = ue_node->GetDevice (0)->GetObject<NrUeNetDevice>()->GetImsi ();
  }
  else if (global_params.rat == "LTE"){
    imsi = ue_node->GetDevice (0)->GetObject<LteUeNetDevice> ()->GetImsi ();
  }
  return(imsi);
}

uint16_t
GetCellId_from_ueId(uint16_t ueId){
  uint16_t cellId=0;
  Ptr<Node> ue_node = ueNodes.Get (ueId);
  if (global_params.rat == "NR"){
    cellId = ue_node->GetDevice (0)->GetObject<NrUeNetDevice>()->GetRrc ()->GetCellId ();
  }
  else if (global_params.rat == "LTE"){
    cellId = ue_node->GetDevice (0)->GetObject<LteUeNetDevice> ()->GetRrc ()->GetCellId ();
  }
  return(cellId);
}


Ptr<LteEnbNetDevice> 
GetServingEnbDev_from_ueId (uint16_t ueId)
{
    uint16_t cellId = GetCellId_from_ueId (ueId);
    // Iterate through the cells and find the right NetDev
    for (uint16_t cellIndex = 0; cellIndex < allGnbNodes.GetN (); ++cellIndex)
    {
        Ptr<LteEnbNetDevice> enbDev = allGnbNodes.Get(cellIndex)->GetDevice (0)->GetObject<LteEnbNetDevice> ();
        if ( enbDev->GetCellId () ==  cellId ) 
        {
            return (enbDev);
        }
    }
    NS_ABORT_MSG ("Could not find serving cell for UE with IMSI: "<< GetImsi_from_ueId (ueId));
}
    
void
ScenarioInfo (NodeDistributionScenarioInterface* scenario)
{
  // Iterate through the ue container and print info 
  std::cout << "\n================ BS Info =================" << std::endl;
  std::cout << "All basestation locations: " << std::endl;
  *topologyStream->GetStream()
          << "cellId," << "gnbpos_x," << "gnbpos_y," << "gnbpos_z" << std::endl;   
  
    for (uint32_t cellIndex = 0; cellIndex < allGnbNodes.GetN (); ++cellIndex)
    {
      Ptr<Node> gnb = allGnbNodes.Get (cellIndex);
      uint32_t cellId = 0; 
      //double bsTxPower = 0;
      Vector gnbpos = gnb->GetObject<MobilityModel> ()->GetPosition ();
      if (global_params.rat == "NR"){
          cellId = gnb->GetDevice (0)->GetObject<NrGnbNetDevice>()->GetCellId ();
          //bsTxPower = gnb->GetDevice (0)->GetObject<NrGnbNetDevice>()->GetObject<Nr>()
      }
      else if (global_params.rat == "LTE"){
          cellId = gnb->GetDevice (0)->GetObject<LteEnbNetDevice>()->GetCellId ();
          //bsTxPower = gnb->GetDevice (0)->GetObject<LteEnbNetDevice>()->GetObject<LteEnbPhy>()->GetTxPower();
      }
      std::cout << "gnbNodes Index:" << cellIndex << " CellId: " << cellId << " pos: (" << gnbpos.x << ", " << gnbpos.y <<  ", " << gnbpos.z <<  ")  "
              //<< "   BSTxPower " << bsTxPower
              << std::endl;
     *topologyStream->GetStream()
          << cellId << "," << gnbpos.x << "," << gnbpos.y <<  "," << gnbpos.z << std::endl; 
    }
    std::cout << "\n";   
     
  // Iterate through the ue container and print info 
  std::cout << "\n================ UE Info =================" << std::endl;
  for (uint32_t ueId = 0; ueId < ueNodes.GetN (); ++ueId){
     
      if (global_params.rat == "NR"){
          Ptr<Node> ue_node = ueNodes.Get (ueId);
          Ptr<NrUeNetDevice> uedev = ue_node->GetDevice (0)->GetObject<NrUeNetDevice> ();
          Vector uepos = ue_node->GetObject<MobilityModel> ()->GetPosition ();
          Ptr<Ipv4> ipv4 = ue_node->GetObject<Ipv4> ();
          Ipv4InterfaceAddress iaddr = ipv4->GetAddress (1,0); 
          Ipv4Address addr = iaddr.GetLocal ();
          
          // couldn't get mobility model object from netDev, 
          // I need to get it form the node object
          // Need to write code to get node from netdev ?? How ?  
          //Vector gnbpos = uedev->GetTargetEnb ()->GetNode() ->GetObject<MobilityModel> ()->GetPosition ();
          //double distance = CalculateDistance (gnbpos, uepos);
        
          //const double ueTxpower = uedev->GetObject<NrUePhy> ()->GetTxPower ();
          std::cout << "NodeId "<< ue_node->GetId () 
              << "   UeIndex(ueId) " << ueId
              << "   IMSI " << uedev->GetImsi ()
              << "   cellId " << GetCellId_from_ueId (ueId)
              << "   UeIpAddr " << addr 
              << "   UePos " << uepos
              //<< "   ueTxPower " << ueTxpower
              //<< "   distance to gnb " << distance << " meters"
              << std::endl;
      }
      else if (global_params.rat == "LTE"){
          Ptr<Node> ue_node = ueNodes.Get (ueId);
          Ptr<LteUeNetDevice> uedev = ue_node->GetDevice (0)->GetObject<LteUeNetDevice> ();
          Vector uepos = ue_node->GetObject<MobilityModel> ()->GetPosition ();
          Ptr<Ipv4> ipv4 = ue_node->GetObject<Ipv4> ();
          Ipv4InterfaceAddress iaddr = ipv4->GetAddress (1,0); 
          Ipv4Address addr = iaddr.GetLocal ();
        
          
          //double ueTxpower = ue_node->GetDevice (0)->GetObject<LteUePhy> ()->GetTxPower ();
          //Vector gnbpos = uedev->GetTargetEnb ()->GetNode()  >GetObject<MobilityModel> ()->GetPosition ();
          //double distance = CalculateDistance (gnbpos, uepos);
          std::cout << "NodeId "<< ue_node->GetId () 
              << "   UeIndex(ueId) " << ueId
              << "   IMSI " << uedev->GetImsi () 
              << "   cellId " << GetCellId_from_ueId (ueId)
              << "   UeIpAddr " << addr 
              << "   UePos " << uepos
             // << "   ueTxPower " << ueTxpower
              //<< "   distance to gnb " << distance << " meters"
              << std::endl;
      }
  }  
  std::cout << "\n";  
  //list all the fast UEs
  std::cout << "Fast moving UeIds" << std::endl;
  for(std::vector<uint32_t>::iterator iter = fastUes.begin(); iter < fastUes.end(); iter++)
  {
    if (global_params.rat == "NR"){
      Ptr<Node> ue_node = ueNodes.Get (*iter);
      Ptr<NrUeNetDevice> uedev = ue_node->GetDevice (0)->GetObject<NrUeNetDevice> ();
      std::cout <<"UE ID: " << *iter << "  UE IMSI: " << uedev->GetImsi () << "\n";
    }
    else if (global_params.rat == "LTE"){
      Ptr<Node> ue_node = ueNodes.Get (*iter);
      Ptr<LteUeNetDevice> uedev = ue_node->GetDevice (0)->GetObject<LteUeNetDevice> ();
      std::cout <<"UE ID: " << *iter << "  UE IMSI: " << uedev->GetImsi () << "\n";
    }
  }
  std::cout << "\n";
}
    
    
/*********************************
 * Distance based manual handover 
 *********************************/
void CheckForManualHandovers (Ptr<LteHelper> &lteHelper)
{
    NS_ASSERT (lteHelper != nullptr);
    // Iterate over UEs and check if there is a nearer cell to connect to 
    for (uint32_t ueId = 0; ueId < ueNodes.GetN (); ++ueId) 
    {
        Ptr<Node> ue_node = ueNodes.Get (ueId);
        Ptr<MobilityModel> ue_mob_model = ue_node->GetObject<MobilityModel>();
        Ptr<Node> attachedGnb_node = GetServingEnbDev_from_ueId (ueId)->GetNode();
        Ptr<MobilityModel> attachedGnb_mob_model = attachedGnb_node->GetObject<MobilityModel>();
        // distance to current cell
        double current_dist = attachedGnb_mob_model->GetDistanceFrom (ue_mob_model);
        
        uint32_t attached_cellId = GetCellId_from_ueId(ueId);
        uint16_t target_cellId = attached_cellId;
        uint16_t target_cellIndex=666;
        
        for (uint16_t cellIndex = 0; cellIndex < allGnbNodes.GetN (); ++cellIndex)
        {
            Ptr<Node> gnb_node = allGnbNodes.Get (cellIndex);
            Ptr<MobilityModel> gnb_mob_model = gnb_node->GetObject<MobilityModel>();
            double dist = gnb_mob_model->GetDistanceFrom (ue_mob_model);
            
            if (dist < current_dist)
            {
                current_dist = dist;
                target_cellId = GetCellId_from_cellIndex (cellIndex);
                target_cellIndex = cellIndex;
            }
        }     
        
        // If the target or current cell are from the micro layer, and target is closer to UE than attached         
        if ( ( target_cellId != attached_cellId) && (isFromMicroLayer(target_cellId) || isFromMicroLayer(attached_cellId) ) )
        {
            lteHelper->HandoverRequest (MilliSeconds (0), 
                                        ue_node->GetDevice (0), 
                                        attachedGnb_node->GetDevice(0),
                                        allGnbNodes.Get (target_cellIndex)->GetDevice (0));
                    
            std::cout << " Manual handover done for UE IMSI: " << GetImsi_from_ueId(ueId) 
                      << " from CellId: " << attached_cellId
                      << " to CellId: " << target_cellId << " which is at dist: " << current_dist
                      << std::endl;
        }     
    }
    Simulator::Schedule (MilliSeconds(global_params.manualHoTriggerTime), &CheckForManualHandovers, lteHelper);
}
    
    
/**********************************
 * Connection and Handover Events 
 **********************************/
    
void
NotifyConnectionEstablishedUe (std::string context, uint64_t imsi,
                               uint16_t cellid, uint16_t rnti)
{
  std::cout << "ConnectionEstablished at "
            << " UE IMSI " << imsi
            << " to CellId " << cellid
            << " with RNTI " << rnti
            << std::endl;
}

void
NotifyConnectionEstablishedEnb (std::string context, uint64_t imsi,
                                uint16_t cellid, uint16_t rnti)
{
  std::cout << "ConnectionEstablished at "
            << " gNB CellId " << cellid
            << " with UE IMSI " << imsi
            << " RNTI " << rnti
            << std::endl;
}
 
void
NotifyHandoverStartUe (std::string context, uint64_t imsi,
                       uint16_t cellid, uint16_t rnti, uint16_t targetCellId)
{
  std::cout << "HandoverStart "
            << " UE IMSI " << imsi
            << ": previously connected to CellId " << cellid
            << " with RNTI " << rnti
            << ", doing handover to CellId " << targetCellId
            << std::endl;
}

void
NotifyHandoverEndOkUe (std::string context, uint64_t imsi,
                       uint16_t cellid, uint16_t rnti)
{
  std::cout << "HandoverEnd "
            << " UE IMSI " << imsi
            << ": successful handover to CellId " << cellid
            << " with RNTI " << rnti
            << std::endl;
}    
        
void
NotifyHandoverStartEnb (Ptr<OutputStreamWrapper> stream, std::string context, 
                        uint64_t imsi, uint16_t cellid, 
                        uint16_t rnti, uint16_t targetCellId)
{
  /*std::cout << "HandoverStart " << context 
            << " eNB CellId " << cellid
            << ": start handover of UE with IMSI " << imsi
            << " RNTI " << rnti
            << " to CellId " << targetCellId
            << std::endl;*/

    *stream->GetStream() 
            << Simulator::Now ().GetMicroSeconds ()
            << "\t" << imsi // check that this is same as previous 
            << "\t" << cellid // current cellID. Make sure these are same 
            << "\t" << targetCellId
            << std::endl;   
}

void
NotifyHandoverEndOkEnb (std::string context, uint64_t imsi,
                        uint16_t cellid, uint16_t rnti)
{
  std::cout << "HandoverEnd "
            << " eNB CellId " << cellid
            << ": completed handover of UE with IMSI " << imsi
            << " RNTI " << rnti
            << std::endl;
}
    
    
    
    
/***************************
 * Trace callbacks 
 ***************************/

void 
LogPosition (Ptr<OutputStreamWrapper> stream)
{
    for (uint32_t ueId = 0; ueId < ueNodes.GetN (); ++ueId)
    {
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

void 
ThputMeasurement (Ptr<OutputStreamWrapper> stream, std::string context, 
                       Ptr< const Packet > p, const Address &from, const Address &to, 
                       const SeqTsSizeHeader &header)
{   
    // TO DO 
    // Collect the bytes in windows instead of printing every packet to file 
    // This is to avoid generating very large files 
    // expected file length for 10 MHz BW, 1400 pktSize for 1000s if all pkts printed is around 3 Million lines.  
    // expected file length if I use 100 ms window is 10 Thousand lines. 

    if (InetSocketAddress::IsMatchingType (from)){
      *stream->GetStream() 
          << Simulator::Now ().GetMicroSeconds ()
         << "\t" << InetSocketAddress::ConvertFrom (from).GetIpv4 ()
         << "\t" << InetSocketAddress::ConvertFrom (to).GetIpv4 ()     
         //<< "\t" << p->GetSize () // received packet size 
         << "\t" << header.GetSize () // received packet size with header    
         << "\t" << header.GetSeq () //current sequence number
         << "\t" << header.GetTs ().GetMicroSeconds ()
         << "\t" << (Simulator::Now () - header.GetTs ()).GetMicroSeconds ()
         << std::endl;
    }
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
		const uint16_t & webpageId, const std::string & object_type, const Time & rtt, const Address & from, const uint32_t & size)
{
  uint16_t ueId = GetUeIdFromNodeId (GetNodeIdFromContext(context));

  *stream->GetStream()
          << Simulator::Now ().GetMicroSeconds () //tstamp_us
          << "\t" << ueId
          << "\t" << GetImsi_from_ueId(ueId)
          << "\t" << GetCellId_from_ueId(ueId)
          << "\t" << webpageId
          << "\t" << object_type
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

void
BurstRx (Ptr<OutputStreamWrapper> stream, std::string context,
         Ptr<const Packet> burst, const Address &from, const Address &to,
         const SeqTsSizeFragHeader &header)
{
    uint16_t ueId = GetUeIdFromNodeId (GetNodeIdFromContext(context));
    Time now = Simulator::Now ();
    *stream->GetStream()
        << now.GetMicroSeconds () //tstamp_us
        << "\t" << ueId 
        << "\t" << GetImsi_from_ueId(ueId)
        << "\t" << GetCellId_from_ueId(ueId)   
        << "\t" << header.GetSeq () // burst seqnum
        << "\t" << header.GetSize () //burst size 
        << "\t" << header.GetFrags () // total num of fragments in this burst
        << std::endl;
}    

void
FragmentRx (Ptr<OutputStreamWrapper> stream, std::string context,
            Ptr<const Packet> fragment, const Address &from, const Address &to,
         const SeqTsSizeFragHeader &header)
{
    uint16_t ueId = GetUeIdFromNodeId (GetNodeIdFromContext(context));
    Time now = Simulator::Now ();
    *stream->GetStream()
        << now.GetMicroSeconds () //tstamp_us
        << "\t" << ueId 
        << "\t" << GetImsi_from_ueId(ueId)
        << "\t" << GetCellId_from_ueId(ueId)   
        << "\t" << header.GetSeq () // burst seq num
        << "\t" << header.GetSize () // burst size
        << "\t" << header.GetFrags () // total num of fragments in this burst 
        << "\t" << header.GetFragSeq () // fragment seq num
        << "\t" << header.GetTs().GetMicroSeconds () // Tx time of the fragment     
        << "\t" << (now - header.GetTs ()).GetMicroSeconds () // delay to receive this fragment
        // NOTE: You cannnot sum fragment delays to get burst delay since 
        // many fragments are not sent one after the other and instead in a burst, 
        // so many fragments could be scheduled together  
        << std::endl;
}    
    
/***********************************************
 * Install client applications
 **********************************************/

std::pair<ApplicationContainer, Time>
InstallDashApps (const Ptr<Node> &ue,
             DashClientHelper *dashClient, Time appStartTime,
             const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime)
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
InstallHttpApps (const Ptr<Node> &ue,
             ThreeGppHttpClientHelper *httpClient, Time appStartTime,
             const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime)
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
InstallUdpEchoApps (const Ptr<Node> &ue,
             UdpEchoClientHelper *echoClient, Time appStartTime,
             const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime)
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
InstallUlFlowTrafficApps (const Ptr<Node> &ue,
             UdpClientHelper *ulFlowClient,
             const Ptr<Node> &remoteHost,
             const Ipv4Address &remoteHostAddr, Time appStartTime,
             const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime)
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
InstallDlFlowTrafficApps (const Ptr<Node> &ue,
             const Address &ueAddress,
             UdpClientHelper *dlFlowClient,
             const Ptr<Node> &remoteHost,
             const Ipv4Address &remoteHostAddr, Time appStartTime,
             const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime)
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
InstallUlDelayTrafficApps (const Ptr<Node> &ue,
             UdpClientHelper *ulDelayClient,
             const Ptr<Node> &remoteHost,
             const Ipv4Address &remoteHostAddr, Time appStartTime,
             const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime)
{
  ApplicationContainer app;
  ulDelayClient->SetAttribute ("RemoteAddress", AddressValue (remoteHostAddr));
  app = ulDelayClient->Install (ue);

  double start = x->GetValue (appStartTime.GetMilliSeconds (),
                              (appStartTime + appStartWindow).GetMilliSeconds ());
  Time startTime = MilliSeconds (start);
  app.Start (startTime);
  app.Stop (startTime + appGenerationTime);
  return std::make_pair (app, startTime);
}

std::pair<ApplicationContainer, Time>
InstallDlDelayTrafficApps (const Ptr<Node> &ue,
             const Address &ueAddress,
             UdpClientHelper *dlDelayClient,
             const Ptr<Node> &remoteHost,
             const Ipv4Address &remoteHostAddr, Time appStartTime,
             const Ptr<UniformRandomVariable> &x,
             Time appGenerationTime)
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
  
 /***********************************************
 * Create Log files and write column names
 **********************************************/
void CreateTraceFiles (void)
{
    mobStream = traceHelper.CreateFileStream ("mobility_trace.txt");
    *mobStream->GetStream() 
          << "tstamp_us\t" << "ueId\t" << "IMSI\t" << "cellId\t"
          << "pos_x\t" << "pos_y\t" << "pos_z\t"
          << "vel_x\t" << "vel_y\t" << "vel_z" <<std::endl;
    
    handoverStream = traceHelper.CreateFileStream ("handover_trace.txt");
    *handoverStream->GetStream()
          << "tstamp_us\t" << "IMSI\t" 
          << "cellId\t" << "targetCellId" << std::endl;
  
    topologyStream = traceHelper.CreateFileStream ("gnb_locations.txt"); 
    if(global_params.traceDelay)
    {
        delayStream = traceHelper.CreateFileStream ("delay_trace.txt");
        *delayStream->GetStream()
              << "tstamp_us\t" << "dir\t" << "ueId\t" << "IMSI\t" << "cellId\t"
              << "pktSize\t" << "seqNum\t" << "pktUid\t" << "txTstamp_us\t" << "delay" << std::endl;
    }
    if(global_params.traceDash)
    {
        dashClientStream = traceHelper.CreateFileStream ("dashClient_trace.txt");
        *dashClientStream->GetStream()
              << "tstamp_us\t" << "ueId\t" << "IMSI\t" << "cellId\t" << "videoId\t" << "segmentId\t"
              << "newBitRate_bps\t" << "oldBitRate_bps\t"
              << "thputOverLastSeg_bps\t" << "avgThputOverWindow_bps(estBitRate)\t" << "frameQueueSize\t"
              << "interTime_s\t" << "playBackTime_s\t" << "BufferTime_s\t"
              << "deltaBufferTime_s\t" << "delayToNxtReq_s"
              << std::endl;
        mpegPlayerStream = traceHelper.CreateFileStream ("mpegPlayer_trace.txt");
        *mpegPlayerStream->GetStream()
              << "tstamp_us\t" << "ueId\t"  << "IMSI\t" << "cellId\t" << "videoId\t" << "segmentId\t"
              << "resolution\t" << "frameId\t"
              << "playbackTime\t" << "type\t" << "size\t"
              << "interTime\t" << "queueSize"
              << std::endl;
    }
    if(global_params.traceHttp)
    {
        httpClientDelayStream = traceHelper.CreateFileStream ("httpClientDelay_trace.txt");
        *httpClientDelayStream->GetStream()
              << "tstamp_us\t" << "ueId\t" << "IMSI\t" << "cellId\t"
              << "pktSize\t" << "delay" << std::endl;
        httpClientRttStream = traceHelper.CreateFileStream ("httpClientRtt_trace.txt");
        *httpClientRttStream->GetStream()
              << "tstamp_us\t" << "ueId\t" << "IMSI\t" << "cellId\t" 
              << "webpageId\t" << "objectType\t"    
              << "objectSize\t" << "delay" << std::endl;
        httpServerDelayStream = traceHelper.CreateFileStream ("httpServerDelay_trace.txt");
        *httpServerDelayStream->GetStream()
              << "tstamp_us\t" << "ueId\t" << "IMSI\t" << "cellId\t"
              << "delay" << std::endl;
    }
    if(global_params.traceFlow)
    {
        flowStream = traceHelper.CreateFileStream ("flow_trace.txt");
        *flowStream->GetStream()
              << "tstamp_us\t" << "dir\t" << "ueId\t" << "IMSI\t" << "cellId\t"
              << "pktSize\t" << "seqNum\t" << "pktUid\t" << "txTstamp_us\t" << "delay" << std::endl;
    }
    if(global_params.traceRtt)
    {
        rttStream = traceHelper.CreateFileStream ("rtt_trace.txt");
        *rttStream->GetStream()
              << "tstamp_us\t" << "ueId\t" << "IMSI\t" << "cellId\t"
              << "pktSize\t" << "seqNum\t" << "pktUid\t" << "txTstamp_us\t" << "delay" << std::endl;
    }
    if(global_params.traceVr)
    {
        fragmentRxStream = traceHelper.CreateFileStream ("vrFragment_trace.txt");
        *fragmentRxStream->GetStream()
              << "tstamp_us\t" << "ueId\t" << "IMSI\t" << "cellId\t"
              << "burstSeqNum\t" << "burstSize\t" << "numFragsInBurst\t" << "fragSeqNum\t" << "txTstamp_us\t" << "delay" << std::endl;  
        burstRxStream = traceHelper.CreateFileStream ("vrBurst_trace.txt");
        *burstRxStream->GetStream()
              << "tstamp_us\t" << "ueId\t" << "IMSI\t" << "cellId\t"
              << "burstSeqNum\t" << "burstSize\t" << "numFragsInBurst" << std::endl;
    }
    if(global_params.traceUlThput || global_params.traceDlThput)
    {
        if(global_params.traceUlThput)
            thputStream = traceHelper.CreateFileStream ("ulThroughput_trace.txt");
        else
            thputStream = traceHelper.CreateFileStream ("dlThroughput_trace.txt");
        
        *thputStream->GetStream()
            << "tstamp_us\t" << "fromAddr\t" << "toAddr\t" 
            << "pktSize\t" 
            << "seqNum\t" << "txTstamp_us\t" << "delay" << std::endl;
    }
}    
    
    
    
// Print the scenario parameters

std::ostream & operator << (std::ostream & os, const Parameters & parameters)
{
    // Use p as shorthand for arg parameters
    auto p {parameters};

    #define MSG(m) \
    os << "\n" << m << std::left \
     << std::setw (40 - strlen (m)) << (strlen (m) > 0 ? ":" : "")

    MSG ("Scenario Topology Parameters");
    MSG ("");
    
    // Topology related
    if (p.baseStationFile != "" and p.useSiteFile)
    {
        MSG ("Base station positions") << "Read from file " << p.baseStationFile;
    }
    else
    {
        MSG ("Macro layer BS positions") << "Regular hexaonal lay down";
        MSG ("Macro layer num of rings") << p.numOuterRings;
        MSG ("Macro layer num of BSs") << macroLayerGnbNodes.GetN();
        MSG ("Micro layer BS antenna pattern") << "Cosine 120 degrees";
        MSG ("Micro layer of BSs") << (p.useMicroLayer ? "Enabled" : "Disabled");
        if (p.useMicroLayer)
        {
            MSG ("Micro layer BS positions") << "Random drop";
            MSG ("Micro layer num of BSs") << p.numMicroCells;
            MSG ("Micro layer BS antenna pattern") << "Isotropic";
            MSG ("Micro cell Tx power") << p.microCellTxPower;
        }
    }  
    MSG ("Num of UEs") << ueNodes.GetN();
    MSG ("Number of UEs per BS (per sector)") << p.ueNumPergNb;
    MSG ("Antenna down tilt angle") << p.downtiltAngle << " deg";
    MSG ("3GPP Scenario") << p.scenario;

    if (p.scenario == "UMa")
    {
        os << "\n  (ISD: 1.7 km, BS height: 30 m, UE height: 1.5 m, UE-BS min dist: 30.2 m)";
    }
    else if (p.scenario == "UMi")
    {
        os << "\n  (ISD: 0.5 km, BS height: 10 m, UE height: 1.5 m, UE-BS min dist: 10 m)";
    }
    else if (p.scenario == "RMa")
    {
        os << "\n  (ISD: 7.0 km, BS height: 45 m, UE height: 1.5 m, UE-BS min dist: 44.6 m)";
    }
    else
    {
        os << "\n  (unknown configuration)";
    }
    MSG ("");
    
    MSG ("Radio Parameters");
    MSG ("");
    
    // Radio related 
    MSG ("Radio Access Technology") << p.rat;
    MSG ("UL/DL Duplexing mode") << p.operationMode;
    MSG ("Uplink Power Control mode") << (p.enableUlPc ? "Enabled" : "Disabled");
    MSG ("Macro layer Channel bandwidth") << p.bandwidthMHz << " MHz";
    if(p.useMicroLayer)
    {
        MSG ("Micro layer Channel bandwidth") << p.microBandwidthMHz << " MHz";
        MSG ("Macro and Micro on the same spectrum?") << (p.macroMicroSharedSpectrum ? "Yes" : "No");
    }
    MSG ("Spectrum configuration") << (p.freqScenario == 0 ? "non-" : "") << "overlapping";
    MSG ("Scheduler") << p.scheduler;    
    MSG ("RLC buffer size") << p.rlcUmTxBuffSize << " Bytes"; 
    
    if (p.rat == "NR") // NR
    {
        MSG ("Numerology") << p.numerologyBwp;
        if (p.operationMode == "TDD")
            MSG ("TDD pattern") << p.pattern;
    }
    else // LTE
    {
        MSG ("Handover Algorithm") << p.handoverAlgo;
    }
    MSG ("");

    MSG ("TCP Send/REcv Buffer Size") << p.tcpSndRcvBuf << " Bytes";
    MSG ("");
    
    
    MSG ("Mobility Parameters");
    MSG ("");
    MSG ("Mobility Model") << "Random Way point";
    MSG ("Fraction of fast moving UEs") << p.fracFastUes;
    MSG ("");
    
    MSG ("Traffic App Parameters");
    MSG ("");
    MSG ("Apps included");
    os << (p.traceDelay? "Delay probes  " : "")
       << (p.traceRtt? "RTT probes  " : "")
       << (p.traceHttp? "Web Browsing  " : "")
       << (p.traceDash? "Video Streaming  " : "")
       << (p.traceVr? "VR  " : "") << std::endl;
    
    MSG ("Delay/RTT Pkt size") << p.delayPacketSize;
    MSG ("Delay/RTT Probe interval") << p.delayInterval.As (Time::S);
    MSG ("Video streaming Client Buffer") << p.bufferSpace << " Bytes";

    MSG ("");
    MSG ("Simulations Parameters");
    MSG ("Simulations Duration") << p.appGenerationTime.As (Time::S);
    MSG ("Trace file generation") << (p.traces ? "ON" : "off");
    MSG ("Rand Seed") << p.randSeed;
    
    os << std::endl;
    return os;
}

} // namespace ns3

#endif // CELLULAR_NETWORK_FUNCTION_H
