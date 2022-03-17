/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 *   Copyright (c) 2019 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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

/**
 * \ingroup examples
 * \file cttc-nr-demo.cc
 * \brief A cozy, simple, NR demo (in a tutorial style)
 *
 * This example describes how to setup a simulation using the 3GPP channel model
 * from TR 38.900. This example consists of a simple grid topology, in which you
 * can choose the number of gNbs and UEs. Have a look at the possible parameters
 * to know what you can configure through the command line.
 *
 * With the default configuration, the example will create two flows that will
 * go through two different subband numerologies (or bandwidth parts). For that,
 * specifically, two bands are created, each with a single CC, and each CC containing
 * one bandwidth part.
 *
 * The example will print on-screen the end-to-end result of one (or two) flows,
 * as well as writing them on a file.
 *
 * \code{.unparsed}
$ ./waf --run "cttc-nr-demo --Help"
    \endcode
 *
 */

/*
 * Include part. Often, you will have to include the headers for an entire module;
 * do that by including the name of the module you need with the suffix "-module.h".
 */

#include "ns3/core-module.h"
#include "ns3/config-store-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/applications-module.h"
#include "ns3/mobility-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/buildings-module.h"
#include "ns3/nr-module.h"
#include "ns3/antenna-module.h"
#include <ns3/dash-module.h>

/*
 * Use, always, the namespace ns3. All the NR classes are inside such namespace.
 */
using namespace ns3;

/*
 * With this line, we will be able to see the logs of the file by enabling the
 * component "Scenario1"
 */
NS_LOG_COMPONENT_DEFINE ("Scenario1");

int
main (int argc, char *argv[])
{
  /*
   * Variables that represent the parameters we will accept as input by the
   * command line. Each of them is initialized with a default value, and
   * possibly overridden below when command-line arguments are parsed.
   */
  // Deployment parameters:
  uint16_t gNbNum = 4; // for now choose a number whose square root is a whole number 
  uint16_t ueNumPergNb = 3;
  bool doubleOperationalBand = false;
  ScenarioParameters scenarioParams;
  scenarioParams.SetScenarioParameters ("UMi");

  // Simulation parameters: Please don't use double to indicate seconds; use
  // ns-3 Time values which use integers to avoid portability issues.
  Time simTime = MilliSeconds (100000);
  Time appStartTime = MilliSeconds (400);

  // NR parameters:
  uint16_t numerologyBwp1 = 4;
  double centralFrequencyBand1 = 28e9;
  double bandwidthBand1 = 20e6;
  uint16_t numerologyBwp2 = 2;
  double centralFrequencyBand2 = 28.2e9;
  double bandwidthBand2 = 100e6;
  double totalTxPower = 30; // dont know if this is dB or dBm.  

  // Traffic parameters: for DASH application
  double targetDt = 35.0;  // The target time difference between receiving and playing a frame. [s].
  double window = 10.0; // The window for measuring the average throughput. [s].
  std::istringstream iss ("ns3::FdashClient");
  uint32_t bufferSpace = 30000000; // The space in bytes that is used for buffering the video
  // The adaptation algorithms used. It can be a comma seperated list of
  // protocolos, such as 'ns3::FdashClient,ns3::OsmpClient'.
  // You may find the list of available algorithms in src/dash/model/algorithms
  std::vector<std::string> algorithms{std::istream_iterator<std::string>{iss},
                                      std::istream_iterator<std::string>{}};

  // Logging parameters: Where we will store the output files.
  bool logging = true;
  std::string simTag = "default";
  std::string outputDir = "./logs/";

  /*
   * From here, we instruct the ns3::CommandLine class of all the input parameters
   * that we may accept as input, as well as their description, and the storage
   * variable.
   */
  CommandLine cmd;

  cmd.AddValue ("gNbNum",
                "The number of gNbs in multiple-ue topology",
                gNbNum);
  cmd.AddValue ("ueNumPergNb",
                "The number of UE per gNb in multiple-ue topology",
                ueNumPergNb);
  cmd.AddValue ("logging",
                "Enable logging",
                logging);
  cmd.AddValue ("doubleOperationalBand",
                "If true, simulate two operational bands with one CC for each band,"
                "and each CC will have 1 BWP that spans the entire CC.",
                doubleOperationalBand);
  cmd.AddValue ("simTime",
                "Simulation time",
                simTime);
  cmd.AddValue ("numerologyBwp1",
                "The numerology to be used in bandwidth part 1",
                numerologyBwp1);
  cmd.AddValue ("centralFrequencyBand1",
                "The system frequency to be used in band 1",
                centralFrequencyBand1);
  cmd.AddValue ("bandwidthBand1",
                "The system bandwidth to be used in band 1",
                bandwidthBand1);
  cmd.AddValue ("numerologyBwp2",
                "The numerology to be used in bandwidth part 2",
                numerologyBwp2);
  cmd.AddValue ("centralFrequencyBand2",
                "The system frequency to be used in band 2",
                centralFrequencyBand2);
  cmd.AddValue ("bandwidthBand2",
                "The system bandwidth to be used in band 2",
                bandwidthBand2);
  cmd.AddValue ("totalTxPower",
                "total tx power in dBm that will be proportionally assigned to"
                " bands, CCs and bandwidth parts depending on each BWP bandwidth ",
                totalTxPower);
  cmd.AddValue ("simTag",
                "tag to be appended to output filenames to distinguish simulation campaigns",
                simTag);
  cmd.AddValue ("outputDir",
                "directory where to store simulation results",
                outputDir);


  // Parse the command line
  cmd.Parse (argc, argv);

  /*
   * Check if the frequency is in the allowed range.
   * If you need to add other checks, here is the best position to put them.
   */
  NS_ABORT_IF (centralFrequencyBand1 > 100e9);
  NS_ABORT_IF (centralFrequencyBand2 > 100e9);

  /*
   * If the logging variable is set to true, enable the log of some components
   * through the code. The same effect can be obtained through the use
   * of the NS_LOG environment variable:
   *
   * export NS_LOG="UdpClient=level_info|prefix_time|prefix_func|prefix_node:UdpServer=..."
   *
   * Usually, the environment variable way is preferred, as it is more customizable,
   * and more expressive.
   */
  if (logging)
    {
      // To see BS and UE initial positions 
      LogComponentEnable ("GridScenarioHelper", LOG_LEVEL_ALL);
      // These are all traffic sources and their sinks 		    
      //LogComponentEnable ("UdpEchoClient", LOG_LEVEL_INFO);
      //LogComponentEnable ("UdpEchoServer", LOG_LEVEL_INFO);
      //LogComponentEnable ("ThreeGppHttpClient", LOG_INFO);
      //LogComponentEnable ("ThreeGppHttpServer", LOG_INFO);
      LogComponentEnable ("DashServer", LOG_LEVEL_ALL);
      LogComponentEnable ("DashClient", LOG_LEVEL_ALL);
    }

  /*
   * Default values for the simulation. We are progressively removing all
   * the instances of SetDefault, but we need it for legacy code (LTE)
   */
  Config::SetDefault ("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue (999999999));

  /*
   * Create the scenario. In our examples, we heavily use helpers that setup
   * the gnbs and ue following a pre-defined pattern. Please have a look at the
   * HexagonalGridScenarioHelper documentation to see how the nodes will be distributed.
   */
/*
  NodeDistributionScenarioInterface * scenario {NULL};
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

  // Log the configuration
  std::cout << "\n    Topology configuration: " << gnbSites << " sites, "
            << sectors << " sectors/site, "
            << gnbNodes.GetN ()   << " cells, "
            << ueNodes.GetN ()    << " UEs\n";

*/


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
  /*
  NodeContainer gnbSector1Container, gnbSector2Container, gnbSector3Container;
  std::vector<NodeContainer*> gnbNodesBySector{&gnbSector1Container, &gnbSector2Container, &gnbSector3Container};
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
*/
  /*
   * Create different UE NodeContainer for the different sectors.
   *
   * Multiple UEs per sector!
   * Iterate/index ueNodes, ueNetDevs, ueIpIfaces by `ueId`.
   * Iterate/Index ueSector<N>Container, ueNodesBySector[sector],
   *   ueSector<N>NetDev, ueNdBySector[sector] with i % gnbSites
   */
/*
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

*/


  /*
   * Create the scenario. In our examples, we heavily use helpers that setup
   * the gnbs and ue following a pre-defined pattern. Please have a look at the
   * GridScenarioHelper documentation to see how the nodes will be distributed.
   */
  
  int64_t randomStream = 1;
  GridScenarioHelper gridScenario;
  gridScenario.SetRows (uint16_t(sqrt(gNbNum)));
  gridScenario.SetColumns (uint16_t(sqrt(gNbNum)));
  gridScenario.SetHorizontalBsDistance (300.0);
  gridScenario.SetVerticalBsDistance (300.0);
  gridScenario.SetBsHeight (10);
  gridScenario.SetUtHeight (1.5);
  // must be set before BS number
  gridScenario.SetSectorization (GridScenarioHelper::SINGLE);
  gridScenario.SetBsNumber (gNbNum);
  gridScenario.SetUtNumber (ueNumPergNb * gNbNum);
  gridScenario.SetScenarioHeight (400); // Create a 3x3 scenario where the UE will
  gridScenario.SetScenarioLength (400); // be distribuited.
  randomStream += gridScenario.AssignStreams (randomStream);
  gridScenario.CreateScenario ();


  /*
   * Create two different NodeContainer for the different traffic type.
   * In ueLowLat we will put the UEs that will receive low-latency traffic,
   * while in ueVideo we will put the UEs that will receive the video traffic.
   */
  NodeContainer ueEchoContainer, ueVideoContainer, ueHttpContainer;

  for (uint32_t j = 0; j < gridScenario.GetUserTerminals ().GetN (); ++j)
    {
      Ptr<Node> ue = gridScenario.GetUserTerminals ().Get (j);
      if (j % 3 == 0)
        {
          ueEchoContainer.Add (ue);
        }
      else if (j % 3 == 1)
        {
          ueVideoContainer.Add (ue);
        }
      else
        {
          ueHttpContainer.Add (ue);
        }
    }

  /*
   * TODO: Add a print, or a plot, that shows the scenario.
   */

  /*
   * Setup the NR module. We create the various helpers needed for the
   * NR simulation:
   * - EpcHelper, which will setup the core network
   * - IdealBeamformingHelper, which takes care of the beamforming part
   * - NrHelper, which takes care of creating and connecting the various
   * part of the NR stack
   */
  Ptr<NrPointToPointEpcHelper> epcHelper = CreateObject<NrPointToPointEpcHelper> ();
  Ptr<IdealBeamformingHelper> idealBeamformingHelper = CreateObject<IdealBeamformingHelper>();
  Ptr<NrHelper> nrHelper = CreateObject<NrHelper> ();

  // Put the pointers inside nrHelper
  nrHelper->SetBeamformingHelper (idealBeamformingHelper);
  nrHelper->SetEpcHelper (epcHelper);

  /*
   * Spectrum division. We create two operational bands, each of them containing
   * one component carrier, and each CC containing a single bandwidth part
   * centered at the frequency specified by the input parameters.
   * Each spectrum part length is, as well, specified by the input parameters.
   * Both operational bands will use the StreetCanyon channel modeling.
   */
  BandwidthPartInfoPtrVector allBwps;
  CcBwpCreator ccBwpCreator;
  const uint8_t numCcPerBand = 1;  // in this example, both bands have a single CC

  // Create the configuration for the CcBwpHelper. SimpleOperationBandConf creates
  // a single BWP per CC
  CcBwpCreator::SimpleOperationBandConf bandConf1 (centralFrequencyBand1, bandwidthBand1, numCcPerBand, BandwidthPartInfo::UMi_StreetCanyon);
  CcBwpCreator::SimpleOperationBandConf bandConf2 (centralFrequencyBand2, bandwidthBand2, numCcPerBand, BandwidthPartInfo::UMi_StreetCanyon);

  // By using the configuration created, it is time to make the operation bands
  OperationBandInfo band1 = ccBwpCreator.CreateOperationBandContiguousCc (bandConf1);
  OperationBandInfo band2 = ccBwpCreator.CreateOperationBandContiguousCc (bandConf2);

  /*
   * The configured spectrum division is:
   * ------------Band1--------------|--------------Band2-----------------
   * ------------CC1----------------|--------------CC2-------------------
   * ------------BWP1---------------|--------------BWP2------------------
   */

  /*
   * Attributes of ThreeGppChannelModel still cannot be set in our way.
   * TODO: Coordinate with Tommaso
   */
  Config::SetDefault ("ns3::ThreeGppChannelModel::UpdatePeriod",TimeValue (MilliSeconds (0)));
  nrHelper->SetChannelConditionModelAttribute ("UpdatePeriod", TimeValue (MilliSeconds (0)));
  nrHelper->SetPathlossAttribute ("ShadowingEnabled", BooleanValue (false));

  /*
   * Initialize channel and pathloss, plus other things inside band1. If needed,
   * the band configuration can be done manually, but we leave it for more
   * sophisticated examples. For the moment, this method will take care
   * of all the spectrum initialization needs.
   */
  nrHelper->InitializeOperationBand (&band1);

  /*
   * Start to account for the bandwidth used by the example, as well as
   * the total power that has to be divided among the BWPs.
   */
  double x = pow (10, totalTxPower / 10);
  double totalBandwidth = bandwidthBand1;

  /*
   * if not single band simulation, initialize and setup power in the second band
   */
  if (doubleOperationalBand)
    {
      // Initialize channel and pathloss, plus other things inside band2
      nrHelper->InitializeOperationBand (&band2);
      totalBandwidth += bandwidthBand2;
      allBwps = CcBwpCreator::GetAllBwps ({band1, band2});
    }
  else
    {
      allBwps = CcBwpCreator::GetAllBwps ({band1});
    }

  /*
   * allBwps contains all the spectrum configuration needed for the nrHelper.
   *
   * Now, we can setup the attributes. We can have three kind of attributes:
   * (i) parameters that are valid for all the bandwidth parts and applies to
   * all nodes, (ii) parameters that are valid for all the bandwidth parts
   * and applies to some node only, and (iii) parameters that are different for
   * every bandwidth parts. The approach is:
   *
   * - for (i): Configure the attribute through the helper, and then install;
   * - for (ii): Configure the attribute through the helper, and then install
   * for the first set of nodes. Then, change the attribute through the helper,
   * and install again;
   * - for (iii): Install, and then configure the attributes by retrieving
   * the pointer needed, and calling "SetAttribute" on top of such pointer.
   *
   */

  Packet::EnableChecking ();
  Packet::EnablePrinting ();

  /*
   *  Case (i): Attributes valid for all the nodes
   */
  // Beamforming method
  idealBeamformingHelper->SetAttribute ("BeamformingMethod", TypeIdValue (DirectPathBeamforming::GetTypeId ()));

  // Core latency
  epcHelper->SetAttribute ("S1uLinkDelay", TimeValue (MilliSeconds (0)));

  // Antennas for all the UEs
  nrHelper->SetUeAntennaAttribute ("NumRows", UintegerValue (2));
  nrHelper->SetUeAntennaAttribute ("NumColumns", UintegerValue (4));
  nrHelper->SetUeAntennaAttribute ("AntennaElement", PointerValue (CreateObject<IsotropicAntennaModel> ()));

  // Antennas for all the gNbs
  nrHelper->SetGnbAntennaAttribute ("NumRows", UintegerValue (4));
  nrHelper->SetGnbAntennaAttribute ("NumColumns", UintegerValue (8));
  nrHelper->SetGnbAntennaAttribute ("AntennaElement", PointerValue (CreateObject<IsotropicAntennaModel> ()));

  uint32_t bwpIdForEcho = 0;
  uint32_t bwpIdForVideo = 0;
  if (doubleOperationalBand)
    {
      bwpIdForVideo = 1;
      bwpIdForEcho = 0;
    }

  // gNb routing between Bearer and bandwidh part
  nrHelper->SetGnbBwpManagerAlgorithmAttribute ("NGBR_LOW_LAT_EMBB", UintegerValue (bwpIdForEcho));
  nrHelper->SetGnbBwpManagerAlgorithmAttribute ("NGBR_VIDEO_TCP_DEFAULT", UintegerValue (bwpIdForVideo));

  // Ue routing between Bearer and bandwidth part
  nrHelper->SetUeBwpManagerAlgorithmAttribute ("NGBR_LOW_LAT_EMBB", UintegerValue (bwpIdForEcho));
  nrHelper->SetUeBwpManagerAlgorithmAttribute ("NGBR_VIDEO_TCP_DEFAULT", UintegerValue (bwpIdForVideo));

  /*
   * We miss many other parameters. By default, not configuring them is equivalent
   * to use the default values. Please, have a look at the documentation to see
   * what are the default values for all the attributes you are not seeing here.
   */

  /*
   * Case (ii): Attributes valid for a subset of the nodes
   */

  // NOT PRESENT IN THIS SIMPLE EXAMPLE

  /*
   * We have configured the attributes we needed. Now, install and get the pointers
   * to the NetDevices, which contains all the NR stack:
   */

  NetDeviceContainer enbNetDev = nrHelper->InstallGnbDevice (gridScenario.GetBaseStations (), allBwps);
  NetDeviceContainer ueEchoNetDev = nrHelper->InstallUeDevice (ueEchoContainer, allBwps);
  NetDeviceContainer ueVideoNetDev = nrHelper->InstallUeDevice (ueVideoContainer, allBwps);
  NetDeviceContainer ueHttpNetDev = nrHelper->InstallUeDevice (ueHttpContainer, allBwps);

  randomStream += nrHelper->AssignStreams (enbNetDev, randomStream);
  randomStream += nrHelper->AssignStreams (ueEchoNetDev, randomStream);
  randomStream += nrHelper->AssignStreams (ueVideoNetDev, randomStream);
  randomStream += nrHelper->AssignStreams (ueHttpNetDev, randomStream);
  /*
   * Case (iii): Go node for node and change the attributes we have to setup
   * per-node.
   */

  // Get the first netdevice (enbNetDev.Get (0)) and the first bandwidth part (0)
  // and set the attribute.
  nrHelper->GetGnbPhy (enbNetDev.Get (0), 0)->SetAttribute ("Numerology", UintegerValue (numerologyBwp1));
  nrHelper->GetGnbPhy (enbNetDev.Get (0), 0)->SetAttribute ("TxPower", DoubleValue (10 * log10 ((bandwidthBand1 / totalBandwidth) * x)));

  if (doubleOperationalBand)
    {
      // Get the first netdevice (enbNetDev.Get (0)) and the second bandwidth part (1)
      // and set the attribute.
      nrHelper->GetGnbPhy (enbNetDev.Get (0), 1)->SetAttribute ("Numerology", UintegerValue (numerologyBwp2));
      nrHelper->GetGnbPhy (enbNetDev.Get (0), 1)->SetTxPower (10 * log10 ((bandwidthBand2 / totalBandwidth) * x));
    }

  // When all the configuration is done, explicitly call UpdateConfig ()

  for (auto it = enbNetDev.Begin (); it != enbNetDev.End (); ++it)
    {
      DynamicCast<NrGnbNetDevice> (*it)->UpdateConfig ();
    }

  for (auto it = ueEchoNetDev.Begin (); it != ueEchoNetDev.End (); ++it)
    {
      DynamicCast<NrUeNetDevice> (*it)->UpdateConfig ();
    }

  for (auto it = ueVideoNetDev.Begin (); it != ueVideoNetDev.End (); ++it)
    {
      DynamicCast<NrUeNetDevice> (*it)->UpdateConfig ();
    }
  for (auto it = ueHttpNetDev.Begin (); it != ueHttpNetDev.End (); ++it)
    {
      DynamicCast<NrUeNetDevice> (*it)->UpdateConfig ();
    }

  // From here, it is standard NS3. In the future, we will create helpers
  // for this part as well.

  // create the internet and install the IP stack on the UEs
  // get SGW/PGW and create a single RemoteHost
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
  //in this container, interface 0 is the pgw, 1 is the remoteHost
  Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress (1);
  Ptr<Ipv4StaticRouting> remoteHostStaticRouting = ipv4RoutingHelper.GetStaticRouting (remoteHost->GetObject<Ipv4> ());
  remoteHostStaticRouting->AddNetworkRouteTo (Ipv4Address ("7.0.0.0"), Ipv4Mask ("255.0.0.0"), 1);
  internet.Install (gridScenario.GetUserTerminals ());

  Ipv4InterfaceContainer ueEchoIpIface = epcHelper->AssignUeIpv4Address (NetDeviceContainer (ueEchoNetDev));
  Ipv4InterfaceContainer ueVideoIpIface = epcHelper->AssignUeIpv4Address (NetDeviceContainer (ueVideoNetDev));
  Ipv4InterfaceContainer ueHttpIpIface = epcHelper->AssignUeIpv4Address (NetDeviceContainer (ueHttpNetDev));

  // Set the default gateway for the UEs
  for (uint32_t j = 0; j < gridScenario.GetUserTerminals ().GetN (); ++j)
    {
      Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting (gridScenario.GetUserTerminals ().Get (j)->GetObject<Ipv4> ());
      ueStaticRouting->SetDefaultRoute (epcHelper->GetUeDefaultGatewayAddress (), 1);
    }

  // attach UEs to the closest eNB
  nrHelper->AttachToClosestEnb (ueEchoNetDev, enbNetDev);
  nrHelper->AttachToClosestEnb (ueVideoNetDev, enbNetDev);
  nrHelper->AttachToClosestEnb (ueHttpNetDev, enbNetDev);

  /*
   * Traffic part. Install two kind of traffic: low-latency and voice, each
   * identified by a particular source port.
   */
  //uint16_t dlPortLowLat = 1234;
  //uint16_t dlPortVoice = 1235;
  uint16_t BSdashPort = 15000;
  uint16_t echoPort = 9; // well known echo port 
  uint16_t httpPort = 16000;

  // Application servers 
  ApplicationContainer httpServerApps;
  ApplicationContainer dashServerApps;
  ApplicationContainer echoServerApps;

  // Create a UdpEchoServer application on remote host 
  UdpEchoServerHelper echoServer (echoPort);
  //echoServerApps = echoServer.Install (remoteHost);
  
  // Create a DASH server on the remote host 
  DashServerHelper dashServerHelper ("ns3::TcpSocketFactory",
                                             InetSocketAddress (Ipv4Address::GetAny (), BSdashPort));


  // Create HTTP server helper
  ThreeGppHttpServerHelper httpServerHelper (remoteHostAddr);

  // Install the servers of all applications on the remote host 
  echoServerApps.Add (echoServer.Install (remoteHost));
  dashServerApps.Add (dashServerHelper.Install (remoteHost));
  httpServerApps.Add(httpServerHelper.Install (remoteHost));
  Ptr<ThreeGppHttpServer> httpServer = httpServerApps.Get (0)->GetObject<ThreeGppHttpServer> ();
  // Setup HTTP variables for the server
  PointerValue varPtr;
  httpServer->GetAttribute ("Variables", varPtr);
  Ptr<ThreeGppHttpVariables> httpVariables = varPtr.Get<ThreeGppHttpVariables> ();
  httpVariables->SetMainObjectSizeMean (102400); // 100kB
  httpVariables->SetMainObjectSizeStdDev (40960); // 40kB

  // The bearer and epc filter for echo traffic 
  EpsBearer echoBearer (EpsBearer::NGBR_LOW_LAT_EMBB);
  Ptr<EpcTft> echoTft = Create<EpcTft> ();
  EpcTft::PacketFilter pfEcho;
  pfEcho.localPortStart = echoPort;
  pfEcho.localPortEnd = echoPort;
  echoTft->Add (pfEcho);

  // The bearer and epc filter for video traffic
  EpsBearer videoBearer (EpsBearer::NGBR_VIDEO_TCP_DEFAULT); // remember to change this 
  Ptr<EpcTft> videoTft = Create<EpcTft> (); 
  EpcTft::PacketFilter dlpfVideo;
  dlpfVideo.localPortStart = BSdashPort;
  dlpfVideo.localPortEnd = BSdashPort;
  videoTft->Add (dlpfVideo);

  // The bearer and epc filter for web surfing http traffic  
  EpsBearer httpBearer (EpsBearer::NGBR_VIDEO_TCP_DEFAULT); // remember to change this 
  Ptr<EpcTft> httpTft = Create<EpcTft> ();
  EpcTft::PacketFilter pfHttp;
  pfHttp.localPortStart = httpPort;
  pfHttp.localPortEnd = httpPort;
  httpTft->Add (pfHttp);

  /*
   * Let's install the applications!
   */
  ApplicationContainer echoClientApps;
  ApplicationContainer dashClientApps;
  ApplicationContainer httpClientApps;
  
  //
  // Create a UdpEchoClient application to send UDP datagrams from node zero to
  // node one.
  //
  uint32_t packetSize = 1024;
  uint32_t maxPacketCount = 36000; // how to send packets until the end of the simulation ? 
  Time interPacketInterval = Seconds (0.1);

  for (uint32_t i = 0; i < ueEchoContainer.GetN (); ++i)
    {
      Ptr<Node> ue = ueEchoContainer.Get (i);
      Ptr<NetDevice> ueDevice = ueEchoNetDev.Get (i);
      UdpEchoClientHelper echoClient (remoteHostAddr, echoPort);
      echoClient.SetAttribute ("MaxPackets", UintegerValue (maxPacketCount));
      echoClient.SetAttribute ("Interval", TimeValue (interPacketInterval));
      echoClient.SetAttribute ("PacketSize", UintegerValue (packetSize));
      echoClientApps.Add(echoClient.Install (ue));
      // Activate a dedicated bearer for the traffic type
      nrHelper->ActivateDedicatedEpsBearer (ueDevice, echoBearer, echoTft);
    }

  for (uint32_t i = 0; i < ueVideoContainer.GetN (); ++i)
    {
      Ptr<Node> ue = ueVideoContainer.Get (i);
      Ptr<NetDevice> ueDevice = ueVideoNetDev.Get (i);
      DashClientHelper dashClientHelper ("ns3::TcpSocketFactory",
                                                     InetSocketAddress (remoteHostAddr, BSdashPort),
                                                     algorithms[i % algorithms.size ()]);
      dashClientHelper.SetAttribute (
                  "VideoId", UintegerValue (1)); // VideoId should be positive
		  //"VideoId", UintegerValue ((u % numVideos) + 1));
                  //"VideoId", UintegerValue (u + 1)); // VideoId should be positive
                  dashClientHelper.SetAttribute ("TargetDt", TimeValue (Seconds (targetDt)));
                  dashClientHelper.SetAttribute ("window", TimeValue (Seconds (window)));
                  dashClientHelper.SetAttribute ("bufferSpace", UintegerValue (bufferSpace));
                  dashClientApps.Add (dashClientHelper.Install (ue));      

      // Activate a dedicated bearer for the traffic type
      nrHelper->ActivateDedicatedEpsBearer (ueDevice, videoBearer, videoTft);
    }

    for (uint32_t i = 0; i < ueHttpContainer.GetN (); ++i)
    {
      Ptr<Node> ue = ueHttpContainer.Get (i);
      Ptr<NetDevice> ueDevice = ueHttpNetDev.Get (i);
      // Create HTTP client helper
      ThreeGppHttpClientHelper httpClientHelper (remoteHostAddr);
      // Install HTTP client
      httpClientApps.Add(httpClientHelper.Install (ue));
      Ptr<ThreeGppHttpClient> httpClient = httpClientApps.Get (0)->GetObject<ThreeGppHttpClient> ();

      // Activate a dedicated bearer for the traffic type
      nrHelper->ActivateDedicatedEpsBearer (ueDevice, httpBearer, httpTft);
    }

  // start UDP server and client apps
  echoServerApps.Start (appStartTime);
  dashServerApps.Start (appStartTime);
  httpServerApps.Start (appStartTime);
  echoClientApps.Start (appStartTime);
  dashClientApps.Start (appStartTime);
  httpClientApps.Start (appStartTime);

  echoServerApps.Stop (simTime);
  dashServerApps.Stop (simTime);
  httpServerApps.Stop (simTime);
  echoClientApps.Stop (simTime);
  dashClientApps.Stop (simTime);
  httpClientApps.Stop (simTime);

  // enable the traces provided by the nr module
  nrHelper->EnableTraces();

  FlowMonitorHelper flowmonHelper;
  NodeContainer endpointNodes;
  endpointNodes.Add (remoteHost);
  endpointNodes.Add (gridScenario.GetUserTerminals ());

  Ptr<ns3::FlowMonitor> monitor = flowmonHelper.Install (endpointNodes);
  monitor->SetAttribute ("DelayBinWidth", DoubleValue (0.001));
  monitor->SetAttribute ("JitterBinWidth", DoubleValue (0.001));
  monitor->SetAttribute ("PacketSizeBinWidth", DoubleValue (20));

  Simulator::Stop (simTime);
  Simulator::Run ();

  /*
   * To check what was installed in the memory, i.e., BWPs of eNb Device, and its configuration.
   * Example is: Node 1 -> Device 0 -> BandwidthPartMap -> {0,1} BWPs -> NrGnbPhy -> Numerology,
  GtkConfigStore config;
  config.ConfigureAttributes ();
  */

  // Print per-flow statistics
  monitor->CheckForLostPackets ();
  Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier> (flowmonHelper.GetClassifier ());
  FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats ();

  double averageFlowThroughput = 0.0;
  double averageFlowDelay = 0.0;

  std::ofstream outFile;
  std::string filename = outputDir + "/" + simTag;
  outFile.open (filename.c_str (), std::ofstream::out | std::ofstream::trunc);
  if (!outFile.is_open ())
    {
      std::cerr << "Can't open file " << filename << std::endl;
      return 1;
    }

  outFile.setf (std::ios_base::fixed);

  double flowDuration = (simTime - appStartTime).GetSeconds ();
  for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin (); i != stats.end (); ++i)
    {
      Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow (i->first);
      std::stringstream protoStream;
      protoStream << (uint16_t) t.protocol;
      if (t.protocol == 6)
        {
          protoStream.str ("TCP");
        }
      if (t.protocol == 17)
        {
          protoStream.str ("UDP");
        }
      outFile << "Flow " << i->first << " (" << t.sourceAddress << ":" << t.sourcePort << " -> " << t.destinationAddress << ":" << t.destinationPort << ") proto " << protoStream.str () << "\n";
      outFile << "  Tx Packets: " << i->second.txPackets << "\n";
      outFile << "  Tx Bytes:   " << i->second.txBytes << "\n";
      outFile << "  TxOffered:  " << i->second.txBytes * 8.0 / flowDuration / 1000.0 / 1000.0  << " Mbps\n";
      outFile << "  Rx Bytes:   " << i->second.rxBytes << "\n";
      if (i->second.rxPackets > 0)
        {
          // Measure the duration of the flow from receiver's perspective
          averageFlowThroughput += i->second.rxBytes * 8.0 / flowDuration / 1000 / 1000;
          averageFlowDelay += 1000 * i->second.delaySum.GetSeconds () / i->second.rxPackets;

          outFile << "  Throughput: " << i->second.rxBytes * 8.0 / flowDuration / 1000 / 1000  << " Mbps\n";
          outFile << "  Mean delay:  " << 1000 * i->second.delaySum.GetSeconds () / i->second.rxPackets << " ms\n";
          //outFile << "  Mean upt:  " << i->second.uptSum / i->second.rxPackets / 1000/1000 << " Mbps \n";
          outFile << "  Mean jitter:  " << 1000 * i->second.jitterSum.GetSeconds () / i->second.rxPackets  << " ms\n";
        }
      else
        {
          outFile << "  Throughput:  0 Mbps\n";
          outFile << "  Mean delay:  0 ms\n";
          outFile << "  Mean jitter: 0 ms\n";
        }
      outFile << "  Rx Packets: " << i->second.rxPackets << "\n";
    }

  outFile << "\n\n  Mean flow throughput: " << averageFlowThroughput / stats.size () << "\n";
  outFile << "  Mean flow delay: " << averageFlowDelay / stats.size () << "\n";

  outFile.close ();

  std::ifstream f (filename.c_str ());

  if (f.is_open ())
    {
      std::cout << f.rdbuf ();
    }

  Simulator::Destroy ();
  return 0;
}


