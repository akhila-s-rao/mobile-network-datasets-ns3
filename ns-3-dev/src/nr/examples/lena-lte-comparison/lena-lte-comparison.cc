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
#include <ns3/sqlite-output.h>
#include "flow-monitor-output-stats.h"
#include "lena-v1-utils.h"
#include "lena-v2-utils.h"
#include <iomanip>
#include "ns3/log.h"

#include "lena-lte-comparison.h"
#include "lena-lte-comparison-functions.h"

NS_LOG_COMPONENT_DEFINE ("LenaLteComparison");

namespace ns3 {

/***********************************************
 * Validate scenario parameter settings
 **********************************************/


bool Parameters::Validate (void) const{
  NS_ABORT_MSG_IF (bandwidthMHz != 20 && bandwidthMHz != 10 && bandwidthMHz != 5,
                   "Valid bandwidth values are 20, 10, 5, you set " << bandwidthMHz);
  NS_ABORT_MSG_IF (trafficScenario > 3,
                   "Traffic scenario " << trafficScenario << " not valid. Valid values are 0 1 2 3");
  NS_ABORT_MSG_IF (numerologyBwp > 4,
                   "At most 4 bandwidth parts supported.");
  NS_ABORT_MSG_IF (direction != "DL" && direction != "UL",
                   "Flow direction can only be DL or UL: " << direction);
  NS_ABORT_MSG_IF (operationMode != "TDD" && operationMode != "FDD",
                   "Operation mode can only be TDD or FDD: " << operationMode);
  //NS_ABORT_MSG_IF (radioNetwork != "LTE" && radioNetwork != "NR",
  //                 "Unrecognized radio network technology: " << radioNetwork);
  //NS_ABORT_MSG_IF (radioNetwork == "LTE" && operationMode != "FDD",
  //                 "Operation mode must be FDD in a 4G LTE network: " << operationMode);
  NS_ABORT_MSG_IF (simulator != "LENA" && simulator != "5GLENA",
                   "Unrecognized simulator: " << simulator);
  NS_ABORT_MSG_IF (scheduler != "PF" && scheduler != "RR",
                   "Unrecognized scheduler: " << scheduler);
  return true;
}


void LenaLteComparison (const Parameters &params){
  params.Validate ();

  RngSeedManager::SetSeed (3);  // Changes seed from default of 1 to 3
  RngSeedManager::SetRun (params.randSeed);   // Changes run number from default of 1 to 7
  
  // Enables NS logging 
  if (params.logging)
    {
      //LogComponentEnable ("UdpClient", LOG_LEVEL_INFO);
      //LogComponentEnable ("UdpServer", LOG_LEVEL_INFO);
      //LogComponentEnable ("UdpEchoServerApplication", LOG_LEVEL_ALL);
      //LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_ALL);
      //LogComponentEnable ("NrAmc", LOG_LEVEL_LOGIC);
      //LogComponentEnable ("DashServer", LOG_LEVEL_ALL);
      //LogComponentEnable ("DashClient", LOG_LEVEL_ALL);
      //LogComponentEnable ("ThreeGppHttpServer", LOG_LEVEL_INFO);
      //LogComponentEnable ("ThreeGppHttpClient", LOG_LEVEL_INFO);

    }

 /***********************************************
 * Create Log files and write column names
 **********************************************/

  AsciiTraceHelper traceHelper;
  Ptr<OutputStreamWrapper> mobStream = traceHelper.CreateFileStream ("mobility_trace.txt");
  *mobStream->GetStream() 
	  << "tstamp_us\t" << "ueId\t" << "cellId\t"
          << "pos_x\t" << "pos_y\t" << "pos_z\t"
          << "vel_x\t" << "vel_y\t" << "vel_z" <<std::endl;
  Ptr<OutputStreamWrapper> delayStream = traceHelper.CreateFileStream ("delay_trace.txt");
  *delayStream->GetStream()
          << "tstamp_us\t" << "dir\t" << "ueId\t" << "cellId\t"
          << "pktSize\t" << "seqNum\t" << "pktUid\t" << "txTstamp\t" << "delay" << std::endl;
  Ptr<OutputStreamWrapper> dashClientStream = traceHelper.CreateFileStream ("dashClient_trace.txt");
  *dashClientStream->GetStream()
          << "tstamp_us\t" << "ueId\t" << "cellId\t" << "videoId\t" << "segmentId\t"
          << "newBitRate_bps\t" << "oldBitRate_bps\t"
          << "thputOverLastSeg_bps\t" << "avgThputOverWindow_bps(estBitRate)\t" << "frameQueueSize\t"
          << "interTime_s\t" << "playBackTime_s\t" << "BufferTime_s\t"
          << "deltaBufferTime_s\t" << "delayToNxtReq_s"
          << std::endl;
  Ptr<OutputStreamWrapper> mpegPlayerStream = traceHelper.CreateFileStream ("mpegPlayer_trace.txt");
  *mpegPlayerStream->GetStream()
          << "tstamp_us\t" << "ueId\t" << "cellId\t" << "videoId\t" << "segmentId\t"
          << "resolution\t" << "frameId\t"
          << "playbackTime\t" << "type\t" << "size\t"
          << "interTime\t" << "queueSize"
          << std::endl;
  Ptr<OutputStreamWrapper> httpClientDelayStream = traceHelper.CreateFileStream ("httpClientDelay_trace.txt");
  *httpClientDelayStream->GetStream()
          << "tstamp_us\t" << "ueId\t" << "cellId\t"
          << "pktSize\t" << "delay" << std::endl;
  Ptr<OutputStreamWrapper> httpClientRttStream = traceHelper.CreateFileStream ("httpClientRtt_trace.txt");
  *httpClientRttStream->GetStream()
          << "tstamp_us\t" << "ueId\t" << "cellId\t"
          << "pktSize\t" << "delay" << std::endl;
  Ptr<OutputStreamWrapper> httpServerDelayStream = traceHelper.CreateFileStream ("httpServerDelay_trace.txt");
  *httpServerDelayStream->GetStream()
          << "tstamp_us\t" << "ueId\t" << "cellId\t"
          << "pktSize\t" << "delay" << std::endl;
  Ptr<OutputStreamWrapper> flowStream = traceHelper.CreateFileStream ("flow_trace.txt");
  *flowStream->GetStream()
          << "tstamp_us\t" << "dir\t" << "ueId\t" << "cellId\t"
          << "pktSize\t" << "seqNum\t" << "pktUid\t" << "txTstamp\t" << "delay" << std::endl;
  Ptr<OutputStreamWrapper> rttStream = traceHelper.CreateFileStream ("rtt_trace.txt");
  *rttStream->GetStream()
          << "tstamp_us\t" << "ueId\t" << "cellId\t"
          << "pktSize\t" << "seqNum\t" << "pktUid\t" << "txTstamp\t" << "delay" << std::endl;



  /*
   * Default values for the simulation. We are progressively removing all
   * the instances of SetDefault, but we need it for legacy code (LTE)
   */
  std::cout << "  max tx buffer size\n"; 
  Config::SetDefault ("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue (999999999));
  Config::SetDefault ("ns3::LteEnbRrc::SrsPeriodicity", UintegerValue (160));
  // Set time granularity for observations that are periodic
  Config::SetDefault ("ns3::LteUePhy::RsrpSinrSamplePeriod", UintegerValue (100));//millisecond
  Config::SetDefault ("ns3::LteEnbPhy::UeSinrSamplePeriod", UintegerValue (10));//millisecond
  Config::SetDefault ("ns3::LteEnbPhy::InterferenceSamplePeriod", UintegerValue (100));//millisecond
  

  //Config::SetDefault ("ns3::TcpSocket::SndBufSize", UintegerValue (75000000));

  /***********************************************
  * Create scenario
  **********************************************/

  ScenarioParameters scenarioParams;
  scenarioParams.SetScenarioParameters (params.scenario);

  // The essentials describing a laydown
  uint32_t gnbSites = 0;
  //NodeContainer gnbNodes;
  //NodeContainer ueNodes;

  double sector0AngleRad = 0;
  const uint32_t sectors = 3;

  NodeDistributionScenarioInterface * scenario {NULL};
  FileScenarioHelper fileScenario;
  HexagonalGridScenarioHelper gridScenario;

  std::cout << "  hexagonal grid: ";
  gridScenario.SetScenarioParameters (scenarioParams);
  gridScenario.SetNumRings (params.numOuterRings);
  gnbSites = gridScenario.GetNumSites ();
  uint32_t ueNum = params.ueNumPergNb * gnbSites * sectors;
  gridScenario.SetUtNumber (ueNum);
  sector0AngleRad = gridScenario.GetAntennaOrientationRadians (0);
  std::cout << sector0AngleRad << std::endl;

  // Creates and plots the network deployment
  gridScenario.CreateScenario ();
  gnbNodes = gridScenario.GetBaseStations ();
  ueNodes = gridScenario.GetUserTerminals ();
  scenario = &gridScenario;

  // Log the topology configuration 
  std::cout << "\n    Topology configuration: " << gnbSites << " sites, "
            << sectors << " sectors/site, "
            << gnbNodes.GetN ()   << " cells, "
            << ueNodes.GetN ()    << " UEs\n";


  /*
   * Create different gNB NodeContainer for the different sectors.
   *
   * Relationships between ueId, cellId, sectorId and siteId:
   * ~~~{.cc}
   *   cellId = scenario->GetCellIndex (ueId);
   *   sector = scenario->GetSectorIndex (cellId);
   *   siteId = scenario->GetSiteIndex (cellId);
   * ~~~{.cc}
   *
   * Iterate/index gnbNodes, gnbNetDevs by `cellId`.
   * Iterate/index gnbSector<N>Container, gnbNodesBySector[sector],
   *   gnbSector<N>NetDev, gnbNdBySector[sector] by `siteId`
   */
  NodeContainer gnbSector1Container, gnbSector2Container, gnbSector3Container;
  std::vector<NodeContainer*> gnbNodesBySector{&gnbSector1Container, 
	  &gnbSector2Container, &gnbSector3Container};
  for (uint32_t cellId = 0; cellId < gnbNodes.GetN (); ++cellId)
    {
      Ptr<Node> gnb = gnbNodes.Get (cellId);
      auto sector = scenario->GetSectorIndex (cellId);
      gnbNodesBySector[sector]->Add (gnb);
    }
  std::cout << "    gNb containers: "
            << gnbSector1Container.GetN () << ", "
            << gnbSector2Container.GetN () << ", "
  	    << gnbSector3Container.GetN ()
            << std::endl;

  NUM_GNBS = gnbNodes.GetN();

  /*
   * Create different UE NodeContainer for the different sectors.
   *
   * Multiple UEs per sector!
   * Iterate/index ueNodes, ueNetDevs, ueIpIfaces by `ueId`.
   * Iterate/Index ueSector<N>Container, ueNodesBySector[sector],
   *   ueSector<N>NetDev, ueNdBySector[sector] with i % gnbSites
   */
  NodeContainer ueSector1Container, ueSector2Container, ueSector3Container;
  std::vector<NodeContainer*> ueNodesBySector {&ueSector1Container, &ueSector2Container, &ueSector3Container};
  for (uint32_t ueId = 0; ueId < ueNodes.GetN (); ++ueId)
    {
      Ptr<Node> ue = ueNodes.Get (ueId);
      auto cellId = scenario->GetCellIndex (ueId);
      auto sector = scenario->GetSectorIndex (cellId);
      ueNodesBySector[sector]->Add (ue);
    }
  std::cout << "    UE containers: "
            << ueSector1Container.GetN () << ", "
            << ueSector2Container.GetN () << ", "
            << ueSector3Container.GetN ()
            << std::endl;
  

 /***********************************************
 * Mobility model for UEs
 **********************************************/
  // Set the bounding box for the scenario within which the UEs move
  double boundingBoxMinX = 0;
  double boundingBoxMinY = 0;
  double boundingBoxMaxX = 0;
  double boundingBoxMaxY = 0;
  std::cout << " Base station locations: " << std::endl;
  for (uint32_t cellId = 0; cellId < gnbNodes.GetN (); ++cellId)
    {
      Ptr<Node> gnb = gnbNodes.Get (cellId);
      Vector gnbpos = gnb->GetObject<MobilityModel> ()->GetPosition ();
      boundingBoxMinX = std::min(gnbpos.x, boundingBoxMinX);
      boundingBoxMinY = std::min(gnbpos.y, boundingBoxMinY);
      boundingBoxMaxX = std::max(gnbpos.x, boundingBoxMaxX);
      boundingBoxMaxY = std::max(gnbpos.y, boundingBoxMaxY);
      std::cout << "(" << gnbpos.x << ", " << gnbpos.y <<  ", " << gnbpos.z <<  ")  "
              << std::endl;
    }
  // Add the cell width to the min and max boundaries to create the bounding box
  double hexCellRadius = gridScenario.GetHexagonalCellRadius();
  boundingBoxMinX = boundingBoxMinX - hexCellRadius;
  boundingBoxMinY = boundingBoxMinY - hexCellRadius;
  boundingBoxMaxX = boundingBoxMaxX + hexCellRadius;
  boundingBoxMaxY = boundingBoxMaxY + hexCellRadius;
  std::cout << "   Bounding box for the scenario is "
            << "(" << boundingBoxMinX << ", " << boundingBoxMinY << ")  "
            << "(" << boundingBoxMinX << ", " << boundingBoxMaxY << ")  "
            << "(" << boundingBoxMaxX << ", " << boundingBoxMinY << ")  "
            << "(" << boundingBoxMaxX << ", " << boundingBoxMaxY << ")  "
            //<< " UE height: " << ueZ
            << "\n Xwidth: " << (boundingBoxMaxX - boundingBoxMinX)
            << "  Ywidth: " << (boundingBoxMaxY - boundingBoxMinY)
            << std::endl;

  double ueMinSpeed = 1.4; // m/s // This should be in params
  double ueMaxSpeed = 10; // m/s // This should be in params
  double ueZ =1.5; // THIS SHOULD NOT BE HARD CODED. CHANGE

  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::SteadyStateRandomWaypointMobilityModel");
  Config::SetDefault ("ns3::SteadyStateRandomWaypointMobilityModel::MinX",
                          DoubleValue (boundingBoxMinX));
  Config::SetDefault ("ns3::SteadyStateRandomWaypointMobilityModel::MinY",
                          DoubleValue (boundingBoxMinY));
  Config::SetDefault ("ns3::SteadyStateRandomWaypointMobilityModel::MaxX",
                          DoubleValue (boundingBoxMaxX));
  Config::SetDefault ("ns3::SteadyStateRandomWaypointMobilityModel::MaxY",
                          DoubleValue (boundingBoxMaxY));
  Config::SetDefault ("ns3::SteadyStateRandomWaypointMobilityModel::Z", DoubleValue (ueZ));
  Config::SetDefault ("ns3::SteadyStateRandomWaypointMobilityModel::MaxSpeed",
                          DoubleValue (ueMaxSpeed));
  Config::SetDefault ("ns3::SteadyStateRandomWaypointMobilityModel::MinSpeed",
                          DoubleValue (ueMinSpeed));

  Ptr<PositionAllocator> positionAlloc = CreateObject<RandomBoxPositionAllocator> ();
  Ptr<UniformRandomVariable> xPos = CreateObject<UniformRandomVariable> ();
  xPos->SetAttribute ("Min", DoubleValue (boundingBoxMinX));
  xPos->SetAttribute ("Max", DoubleValue (boundingBoxMaxX));
  positionAlloc->SetAttribute ("X", PointerValue (xPos));
  Ptr<UniformRandomVariable> yPos = CreateObject<UniformRandomVariable> ();
  yPos->SetAttribute ("Min", DoubleValue (boundingBoxMinY));
  yPos->SetAttribute ("Max", DoubleValue (boundingBoxMaxY));
  positionAlloc->SetAttribute ("Y", PointerValue (yPos));
  Ptr<ConstantRandomVariable> zPos = CreateObject<ConstantRandomVariable> ();
  zPos->SetAttribute ("Constant", DoubleValue (ueZ));
  positionAlloc->SetAttribute ("Z", PointerValue (zPos));

  mobility.SetPositionAllocator (positionAlloc);
  mobility.Install(ueNodes);

  /*
   * Setup the LTE or NR module. We create the various helpers needed inside
   * their respective configuration functions
   */
  std::cout << "  helpers\n";
  Ptr<PointToPointEpcHelper> epcHelper;

  NetDeviceContainer gnbSector1NetDev, gnbSector2NetDev, gnbSector3NetDev;
  std::vector<NetDeviceContainer *> gnbNdBySector {&gnbSector1NetDev, &gnbSector2NetDev, &gnbSector3NetDev};
  NetDeviceContainer ueSector1NetDev, ueSector2NetDev,ueSector3NetDev;
  std::vector<NetDeviceContainer *> ueNdBySector {&ueSector1NetDev, &ueSector2NetDev, &ueSector3NetDev};

  Ptr <LteHelper> lteHelper = nullptr;
  Ptr <NrHelper> nrHelper = nullptr;

  if (params.simulator == "LENA")
    {
      epcHelper = CreateObject<PointToPointEpcHelper> ();
      LenaV1Utils::SetLenaV1SimulatorParameters (sector0AngleRad,
                                                 params.scenario,
                                                 gnbSector1Container,
                                                 gnbSector2Container,
                                                 gnbSector3Container,
                                                 ueSector1Container,
                                                 ueSector2Container,
                                                 ueSector3Container,
                                                 epcHelper,
                                                 lteHelper,
                                                 gnbSector1NetDev,
                                                 gnbSector2NetDev,
                                                 gnbSector3NetDev,
                                                 ueSector1NetDev,
                                                 ueSector2NetDev,
                                                 ueSector3NetDev,
                                                 params.enableUlPc,
                                                 params.scheduler,
                                                 params.bandwidthMHz,
                                                 params.freqScenario,
                                                 params.downtiltAngle);
    }
  else if (params.simulator == "5GLENA")
    {
      epcHelper = CreateObject<NrPointToPointEpcHelper> ();
      LenaV2Utils::SetLenaV2SimulatorParameters (sector0AngleRad,
                                                 params.scenario,
                                                 params.errorModel,
                                                 params.operationMode,
                                                 params.direction,
                                                 params.numerologyBwp,
                                                 params.pattern,
                                                 gnbSector1Container,
                                                 gnbSector2Container,
                                                 gnbSector3Container,
                                                 ueSector1Container,
                                                 ueSector2Container,
                                                 ueSector3Container,
                                                 epcHelper,
                                                 nrHelper,
                                                 gnbSector1NetDev,
                                                 gnbSector2NetDev,
                                                 gnbSector3NetDev,
                                                 ueSector1NetDev,
                                                 ueSector2NetDev,
                                                 ueSector3NetDev,
                                                 params.enableUlPc,
                                                 params.powerAllocation,
                                                 params.scheduler,
                                                 params.bandwidthMHz,
                                                 params.freqScenario,
                                                 params.downtiltAngle);
    }

  // Check we got one valid helper
  if ( (lteHelper == nullptr) && (nrHelper == nullptr) )
    {
      NS_ABORT_MSG ("Programming error: no valid helper");
    }

  // create the internet and install the IP stack on the UEs
  // get SGW/PGW and create a single RemoteHost
  std::cout << "  pgw and internet\n";
  Ptr<Node> pgw = epcHelper->GetPgwNode ();
  NodeContainer remoteHostContainer;
  remoteHostContainer.Create (1);
  Ptr<Node> remoteHost = remoteHostContainer.Get (0);
  InternetStackHelper internet;
  internet.Install (remoteHostContainer);

  // connect a remoteHost to pgw. Setup routing too
  PointToPointHelper p2ph;
  p2ph.SetDeviceAttribute ("DataRate", DataRateValue (DataRate ("100Gb/s")));
  p2ph.SetDeviceAttribute ("Mtu", UintegerValue (2500));
  p2ph.SetChannelAttribute ("Delay", TimeValue (Seconds (0.000)));
  NetDeviceContainer internetDevices = p2ph.Install (pgw, remoteHost);
  Ipv4AddressHelper ipv4h;
  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  ipv4h.SetBase ("1.0.0.0", "255.0.0.0");
  Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign (internetDevices);
  Ptr<Ipv4StaticRouting> remoteHostStaticRouting = ipv4RoutingHelper.GetStaticRouting (remoteHost->GetObject<Ipv4> ());
  remoteHostStaticRouting->AddNetworkRouteTo (Ipv4Address ("7.0.0.0"), Ipv4Mask ("255.0.0.0"), 1);
  internet.Install (ueNodes);

  NetDeviceContainer gnbNetDevs (gnbSector1NetDev, gnbSector2NetDev);
  gnbNetDevs.Add (gnbSector3NetDev);
  NetDeviceContainer ueNetDevs (ueSector1NetDev, ueSector2NetDev);
  ueNetDevs.Add (ueSector3NetDev);
  //ueIpIfaces {epcHelper->AssignUeIpv4Address (ueNetDevs)};
  ueIpIfaces = epcHelper->AssignUeIpv4Address (ueNetDevs);
  Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress (1);

  // Set the default gateway for the UEs
  std::cout << "  default gateway\n";
  for (auto ue = ueNodes.Begin (); ue != ueNodes.End (); ++ue)
    {
      Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting ((*ue)->GetObject<Ipv4> ());
      ueStaticRouting->SetDefaultRoute (epcHelper->GetUeDefaultGatewayAddress (), 1);
    }

  // attach UEs to their gNB. Try to attach them per cellId order
  std::cout << " attach UEs to gNBs\n";
  std::cout << " Number of gNBs: " << gnbNodes.GetN() << std::endl;
  for (uint32_t ueId = 0; ueId < ueNodes.GetN (); ++ueId)
    {
      auto cellId = scenario->GetCellIndex (ueId);
      Ptr<NetDevice> gnbNetDev = gnbNetDevs.Get (cellId);
      Ptr<NetDevice> ueNetDev = ueNetDevs.Get (ueId);
      if (lteHelper != nullptr)
        {
          lteHelper->Attach (ueNetDev, gnbNetDev);
        }
      else if (nrHelper != nullptr)
        {
          nrHelper->AttachToEnb (ueNetDev, gnbNetDev);
          // UL phy
          uint32_t bwp = (params.operationMode == "FDD" ? 1 : 0);
          auto ueUlPhy {nrHelper->GetUePhy (ueNetDev, bwp)};
          auto rnti = ueUlPhy->GetRnti ();
          Vector gnbpos = gnbNetDev->GetNode ()->GetObject<MobilityModel> ()->GetPosition ();
          Vector uepos = ueNetDev->GetNode ()->GetObject<MobilityModel> ()->GetPosition ();
          double distance = CalculateDistance (gnbpos, uepos);
          std::cout << "ueId: " << ueId
                    << ", rnti: " << rnti
                    << ", at " << uepos
                    << ", attached to eNB " << cellId
                    << " at " << gnbpos
                    << ", range: " << distance << " meters"
                    << std::endl;
        }
    }


 /***********************************************
 * Traffic generation applications
 **********************************************/
  uint16_t ulFlowPortNum = 1234;
  uint16_t dlFlowPortNum = 1235;
  uint16_t echoPortNum = 9; // well known echo port
  uint16_t ulDelayPortNum = 17000;
  uint16_t dlDelayPortNum = 18000;
  uint16_t dashPortNum = 15000;
  uint16_t httpPortNum = 16000;

  // Configuration parameters for echo application
  uint32_t echoPacketSize = 1024;
  uint32_t echoPacketCount = 100000; // how to send packets until the end of the simulation ? 
  Time echoInterPacketInterval = Seconds (0.1);

  // Configuration parameters for UL and DL delay application 
  uint32_t delayPacketSize = 1024;
  uint32_t delayPacketCount = 100000; // how to send packets until the end of the simulation ? 
  Time delayInterval = Seconds (0.1);

  // Configuration parameters for DASH application
  double targetDt = 35.0;  // The target time difference between receiving and playing a frame. [s].
  double window = 10.0; // The window for measuring the average throughput. [s].
  std::istringstream iss ("ns3::FdashClient");
  uint32_t bufferSpace = 30000000; // The space in bytes that is used for buffering the video

  // Configuration parameters for UL and DL Traffic parameters for the Flows 
  uint32_t flowPacketSize = 1000;
  uint32_t lambda;
  uint32_t flowPacketCount;

  std::cout << "  traffic parameters\n";
  switch (params.trafficScenario)
    {
      case 0: // let's put 80 Mbps with 20 MHz of bandwidth. Everything else is scaled
        flowPacketCount = 0xFFFFFFFF;
        switch (params.bandwidthMHz)
          {
            case 20:
              flowPacketSize = 1000;
              break;
            case 10:
              flowPacketSize = 500;
              break;
            case 5:
              flowPacketSize = 250;
              break;
            default:
              flowPacketSize = 1000;
          }
        lambda = 10000 / params.ueNumPergNb;
        break;
      case 1:
        flowPacketCount = 1;
        flowPacketSize = 12;
        lambda = 1;
        break;
      case 2: // 1 Mbps == 0.125 MB/s in case of 20 MHz, everything else is scaled
        flowPacketCount = 0xFFFFFFFF;
        switch (params.bandwidthMHz)
          {
            case 20:
              flowPacketSize = 125;
              break;
            case 10:
              flowPacketSize = 63;
              break;
            case 5:
              flowPacketSize = 32;
              break;
            default:
              flowPacketSize = 125;
          }
	lambda = 1000 / params.ueNumPergNb;
        break;
      case 3: // 20 Mbps == 2.5 MB/s in case of 20 MHz, everything else is scaled
        flowPacketCount = 0xFFFFFFFF;
        switch (params.bandwidthMHz)
          {
            case 20:
              flowPacketSize = 250;
              break;
            case 10:
              flowPacketSize = 125;
              break;
            case 5:
              flowPacketSize = 75;
              break;
            default:
              flowPacketSize = 250;
          }
        lambda = 10000 / params.ueNumPergNb;
        break;
      default:
        NS_FATAL_ERROR ("Traffic scenario " << params.trafficScenario << " not valid. Valid values are 0 1 2 3");
    }

  // Create the server applications
  ApplicationContainer serverApps;

  UdpServerHelper ulFlowPacketSink (ulFlowPortNum);
  UdpServerHelper dlFlowPacketSink (dlFlowPortNum);
  UdpServerHelper ulDelayPacketSink (ulDelayPortNum);
  UdpServerHelper dlDelayPacketSink (dlDelayPortNum);
  UdpEchoServerHelper echoServer (echoPortNum);
  DashServerHelper dashServer ("ns3::TcpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), dashPortNum));
  ThreeGppHttpServerHelper httpServer (remoteHostAddr);

  // Install the server applications on the desired devices
  //  App Id as seen in the context string is based on order on App installation
  serverApps.Add (ulDelayPacketSink.Install (remoteHost)); // appId is 0  
  serverApps.Add (ulFlowPacketSink.Install (remoteHost)); // appId is 1
  serverApps.Add (dashServer.Install (remoteHost)); // appId is 2
  serverApps.Add (httpServer.Install (remoteHost)); // appId is 3
  serverApps.Add (echoServer.Install (remoteHost)); // appId is 4

  serverApps.Add (dlDelayPacketSink.Install (ueNodes)); // appId is 0
  serverApps.Add (dlFlowPacketSink.Install (ueNodes)); // appId is 1
  Ptr<ThreeGppHttpServer> httpSer = serverApps.Get (3)->GetObject<ThreeGppHttpServer> ();
  PointerValue varPtr;
  httpSer->GetAttribute ("Variables", varPtr);
  Ptr<ThreeGppHttpVariables> httpVariables = varPtr.Get<ThreeGppHttpVariables> ();
  httpVariables->SetMainObjectSizeMean (102400); // 100kB
  httpVariables->SetMainObjectSizeStdDev (40960); // 40kB

  // start all servers
  serverApps.Start (params.udpAppStartTime);
  
  // Configure UL and DL flow client applications 
  Time flowInterval = Seconds (1.0 / lambda);
  UdpClientHelper ulFlowClient;
  ulFlowClient.SetAttribute ("RemotePort", UintegerValue (ulFlowPortNum));
  ulFlowClient.SetAttribute ("MaxPackets", UintegerValue (flowPacketCount));
  ulFlowClient.SetAttribute ("PacketSize", UintegerValue (flowPacketSize));
  ulFlowClient.SetAttribute ("Interval", TimeValue (flowInterval));
  UdpClientHelper dlFlowClient;
  dlFlowClient.SetAttribute ("RemotePort", UintegerValue (dlFlowPortNum));
  dlFlowClient.SetAttribute ("MaxPackets", UintegerValue (flowPacketCount));
  dlFlowClient.SetAttribute ("PacketSize", UintegerValue (flowPacketSize));
  dlFlowClient.SetAttribute ("Interval", TimeValue (flowInterval));

  // Configure UL and DL delay client applications 
  UdpClientHelper ulDelayClient;
  ulDelayClient.SetAttribute ("RemotePort", UintegerValue (ulDelayPortNum));
  ulDelayClient.SetAttribute ("MaxPackets", UintegerValue (delayPacketCount));
  ulDelayClient.SetAttribute ("PacketSize", UintegerValue (delayPacketSize));
  ulDelayClient.SetAttribute ("Interval", TimeValue (delayInterval));
  UdpClientHelper dlDelayClient;
  dlDelayClient.SetAttribute ("RemotePort", UintegerValue (dlDelayPortNum));
  dlDelayClient.SetAttribute ("MaxPackets", UintegerValue (delayPacketCount));
  dlDelayClient.SetAttribute ("PacketSize", UintegerValue (delayPacketSize));
  dlDelayClient.SetAttribute ("Interval", TimeValue (delayInterval));

  // Configure echo client application
  UdpEchoClientHelper echoClient (remoteHostAddr, echoPortNum);
  echoClient.SetAttribute ("MaxPackets", UintegerValue (echoPacketCount));
  echoClient.SetAttribute ("Interval", TimeValue (echoInterPacketInterval));
  echoClient.SetAttribute ("PacketSize", UintegerValue (echoPacketSize));

  // Configure DASH client application 
  DashClientHelper dashClient ("ns3::TcpSocketFactory", InetSocketAddress (remoteHostAddr, dashPortNum), "ns3::FdashClient");
  dashClient.SetAttribute ("VideoId", UintegerValue (1)); // VideoId should be positive
  dashClient.SetAttribute ("TargetDt", TimeValue (Seconds (targetDt)));
  dashClient.SetAttribute ("window", TimeValue (Seconds (window)));
  dashClient.SetAttribute ("bufferSpace", UintegerValue (bufferSpace));

  // Configure http client application
  ThreeGppHttpClientHelper httpClient (remoteHostAddr);

  // Install the client applications on the desired devices
  ApplicationContainer clientApps;  
  Ptr<UniformRandomVariable> startRng = CreateObject<UniformRandomVariable> ();
  startRng->SetStream (RngSeedManager::GetRun ());
  Time maxStartTime;

  for (uint32_t ueId = 0; ueId < ueNodes.GetN (); ++ueId)
    {
      Ptr<Node> node = ueNodes.Get (ueId);
      Ptr<NetDevice> dev = ueNetDevs.Get (ueId);
      Address addr = ueIpIfaces.GetAddress (ueId);

      if (params.simulator == "5GLENA"){
        Ptr<NrUeNetDevice> uedev = node->GetDevice (0)->GetObject<NrUeNetDevice> ();
	std::cout << "   UeId "<< node->GetId ()
              << "   UeIndex " << ueId
              << "   RNTI " << uedev->GetRrc ()->GetRnti ()
              << "   IMSI " << uedev->GetImsi ()
              << "   cellId " << scenario->GetCellIndex (ueId)
              << "   UeIpAddr " << ueIpIfaces.GetAddress (ueId)
              << std::endl;
      }
      else if (params.simulator == "LENA"){
        Ptr<LteUeNetDevice> uedev = node->GetDevice (0)->GetObject<LteUeNetDevice> ();
        std::cout << "   UeId "<< node->GetId () 
	      << "   UeIndex " << ueId
	      << "   RNTI " << uedev->GetRrc ()->GetRnti () 
	      << "   IMSI " << uedev->GetImsi () 
	      << "   cellId " << scenario->GetCellIndex (ueId)
	      << "   UeIpAddr " << ueIpIfaces.GetAddress (ueId)
	      << std::endl;
      }

      // These are the apps that are on all devices 
      auto appsClass1 = InstallUdpEchoApps (node, dev, addr,
                              &echoClient,
                              remoteHost, remoteHostAddr,
                              params.udpAppStartTime, echoPortNum,
                              startRng, params.appGenerationTime,
                              lteHelper, nrHelper);
      clientApps.Add (appsClass1.first);
      auto appsClass2 = InstallUlDelayTrafficApps (node, dev, addr,
                              &ulDelayClient,
                              remoteHost, remoteHostAddr, params.udpAppStartTime,
                              ulDelayPortNum,
                              startRng, params.appGenerationTime,
                              lteHelper, nrHelper);
      clientApps.Add (appsClass2.first);
      auto appsClass3 = InstallDlDelayTrafficApps (node, dev, addr,
                              &dlDelayClient,
                              remoteHost, remoteHostAddr, params.udpAppStartTime,
                              dlDelayPortNum,
                              startRng, params.appGenerationTime,
                              lteHelper, nrHelper);
      clientApps.Add (appsClass3.first);

      // These are the apps tht are on some devices 
      if (ueId % 5 == 0){
	      // These UEs do just delay measurement. So nothing extra to add
      } 
      else if (ueId % 2 == 0) {// These UEs do video streaming + background CBR traffic + delay measurement 
	      auto appsClass4 = InstallUlFlowTrafficApps (node, dev, addr,
                              &ulFlowClient,
                              remoteHost, remoteHostAddr, params.udpAppStartTime,
                              ulFlowPortNum,
                              startRng, params.appGenerationTime,
                              lteHelper, nrHelper);
              clientApps.Add (appsClass4.first);
              auto appsClass5 = InstallDlFlowTrafficApps (node, dev, addr,
                              &dlFlowClient,
                              remoteHost, remoteHostAddr, params.udpAppStartTime,
                              dlFlowPortNum,
                              startRng, params.appGenerationTime,
                              lteHelper, nrHelper);
              clientApps.Add (appsClass5.first);
 	      auto appsClass6 = InstallDashApps (node, dev, addr,
                              &dashClient,
                              remoteHost, remoteHostAddr,
                              params.udpAppStartTime, dashPortNum,
                              startRng, params.appGenerationTime,
                              lteHelper, nrHelper);
	      clientApps.Add (appsClass6.first);
      }
      else {// These UEs do web browsing + background CBR traffic + delay measurement 
              auto appsClass4 = InstallUlFlowTrafficApps (node, dev, addr,
                              &ulFlowClient,
                              remoteHost, remoteHostAddr, params.udpAppStartTime,
                              ulFlowPortNum,
                              startRng, params.appGenerationTime,
                              lteHelper, nrHelper);
              clientApps.Add (appsClass4.first);
              auto appsClass5 = InstallDlFlowTrafficApps (node, dev, addr,
                              &dlFlowClient,
                              remoteHost, remoteHostAddr, params.udpAppStartTime,
                              dlFlowPortNum,
                              startRng, params.appGenerationTime,
                              lteHelper, nrHelper);
              clientApps.Add (appsClass5.first);
              auto appsClass6 = InstallHttpApps (node, dev, addr,
                              &httpClient,
                              remoteHost, remoteHostAddr,
                              params.udpAppStartTime, httpPortNum,
                              startRng, params.appGenerationTime,
                              lteHelper, nrHelper);
	      clientApps.Add (appsClass6.first);
      
      }
      maxStartTime = std::max (MilliSeconds(500), maxStartTime);

    } // end of for over UEs
  std::cout << clientApps.GetN () << " apps\n";

  // enable the RAN traces provided by the lte/nr module
  std::cout << "  tracing\n";
  if (params.traces == true)
    {
      if (lteHelper != nullptr)
        {
          lteHelper->EnableTraces ();
        }
      else if (nrHelper != nullptr)
        {
          nrHelper->EnableTraces ();
        }
    }

  Config::Connect ("/NodeList/*/ApplicationList/0/$ns3::UdpServer/RxWithAddresses", MakeBoundCallback (&delayTrace, delayStream, scenario, remoteHost));
  Config::Connect ("/NodeList/*/ApplicationList/*/$ns3::DashClient/RxSegment", MakeBoundCallback (&dashClientTrace, dashClientStream, scenario));
  Config::Connect ("/NodeList/*/ApplicationList/*/$ns3::DashClient/PlayedFrame", MakeBoundCallback (&mpegPlayerTrace, mpegPlayerStream, scenario));
  Config::Connect ("/NodeList/*/ApplicationList/*/$ns3::ThreeGppHttpClient/RxDelay", MakeBoundCallback (&httpClientTraceRxDelay, httpClientDelayStream, scenario));
  Config::Connect ("/NodeList/*/ApplicationList/*/$ns3::ThreeGppHttpClient/RxRtt", MakeBoundCallback (&httpClientTraceRxRtt, httpClientRttStream, scenario));
  Config::Connect ("/NodeList/*/ApplicationList/*/$ns3::ThreeGppHttpServer/RxDelay", MakeBoundCallback (&httpServerTraceRxDelay, httpServerDelayStream, scenario));
  Config::Connect ("/NodeList/*/ApplicationList/1/$ns3::UdpServer/RxWithAddresses", MakeBoundCallback (&flowTrace, flowStream, scenario, remoteHost));
  Config::Connect ("/NodeList/*/ApplicationList/*/$ns3::UdpEchoClient/RxWithAddresses", MakeBoundCallback (&rttTrace, rttStream, scenario));

  // FlowMonitor for aggregated logs
  std::cout << "  flowmon\n";
  FlowMonitorHelper flowmonHelper;
  NodeContainer endpointNodes;
  endpointNodes.Add (remoteHost);
  endpointNodes.Add (ueNodes);
  Ptr<FlowMonitor> monitor = flowmonHelper.Install (endpointNodes);
  monitor->SetAttribute ("DelayBinWidth", DoubleValue (0.001));
  monitor->SetAttribute ("JitterBinWidth", DoubleValue (0.001));
  monitor->SetAttribute ("PacketSizeBinWidth", DoubleValue (20));
  std::string tableName = "e2e";

  std::cout << "\n----------------------------------------\n"
            << "Start simulation"
            << std::endl;

  // Add some extra time for the last generated packets to be received
  const Time appStopWindow = MilliSeconds (50);
  Time stopTime = maxStartTime + params.appGenerationTime + appStopWindow;
  Simulator::Stop (stopTime);
  // schedule the periodic logging of UE positions
  Simulator::Schedule (MilliSeconds(0), &LogPosition, mobStream, scenario);
  Simulator::Run ();

  FlowMonitorOutputStats flowMonStats;
  //flowMonStats.SetDb (&db, tableName);
  flowMonStats.Save (monitor, flowmonHelper, params.outputDir + "/" + params.simTag);

  std::cout << "\n----------------------------------------\n"
            << "End simulation"
            << std::endl;

  Simulator::Destroy ();
}

} // namespace ns3

