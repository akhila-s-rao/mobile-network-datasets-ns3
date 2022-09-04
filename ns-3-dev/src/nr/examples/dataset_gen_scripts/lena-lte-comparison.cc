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
#include "lena-lte-comparison-functions.h"

NS_LOG_COMPONENT_DEFINE ("LenaLteComparison");

namespace ns3 {
    
void LenaLteComparison (const Parameters &params){

    // Validate the parameter settings  
    params.Validate ();
    // Set as global for easy access
    global_params = params;

    RngSeedManager::SetSeed (3); 
    RngSeedManager::SetRun (params.randSeed);

    // NS logging 
    // NOTE: these logs will show only when the code is NOT build as optimized   
    /*LogComponentEnableAll (LOG_PREFIX_TIME);
    LogComponentEnable ("TcpSocketBase", LOG_LEVEL_ALL);
    LogComponentEnable ("UdpClient", LOG_LEVEL_INFO);
    LogComponentEnable ("UdpServer", LOG_LEVEL_INFO);
    LogComponentEnable ("UdpEchoServerApplication", LOG_LEVEL_ALL);
    LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_ALL);
    LogComponentEnable ("DashServer", LOG_LEVEL_ALL);
    LogComponentEnable ("DashClient", LOG_LEVEL_ALL);
    LogComponentEnable ("ThreeGppHttpServer", LOG_LEVEL_INFO);
    LogComponentEnable ("ThreeGppHttpClient", LOG_LEVEL_INFO);*/

    // Create Trace files and write the column names 
    // To prevent the should trigger HO Preparation Failure, but it is not implemented ASSERT
    Config::SetDefault ("ns3::LteEnbMac::NumberOfRaPreambles", UintegerValue (40));
    // Default values for the simulation. 
    Config::SetDefault ("ns3::LteEnbRrc::SrsPeriodicity", UintegerValue (320));
    //Config::SetDefault ("ns3::LteHelper::UseIdealRrc", BooleanValue (true));
    Config::SetDefault ("ns3::LteSpectrumPhy::CtrlErrorModelEnabled", BooleanValue (false));

    // Set time granularity for RAN traces that are periodic
    Config::SetDefault ("ns3::LteUePhy::RsrpSinrSamplePeriod", UintegerValue (params.ranSamplePeriodMilli));
    Config::SetDefault ("ns3::LteEnbPhy::UeSinrSamplePeriod", UintegerValue (params.ranSamplePeriodMilli));
    Config::SetDefault ("ns3::LteEnbPhy::InterferenceSamplePeriod", UintegerValue (params.ranSamplePeriodMilli));

    // This is set in the lena-v1-utils (not v2) 
    //Config::SetDefault ("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue (10 * 1024)); //default is 10240 

    Config::SetDefault ("ns3::TcpSocket::SndBufSize", UintegerValue (params.tcpSndRcvBuf));// default is 131072
    Config::SetDefault ("ns3::TcpSocket::RcvBufSize", UintegerValue (params.tcpSndRcvBuf));// default is 131072

   
    // Need to increase size from default for larger webpages.  
    Config::SetDefault ("ns3::ThreeGppHttpVariables::MainObjectSizeMean", UintegerValue (params.httpMainObjMean));
    Config::SetDefault ("ns3::ThreeGppHttpVariables::MainObjectSizeStdDev", UintegerValue (params.httpMainObjStd));
    
    // Create user created trace files with corresponding column names
    CreateTraceFiles ();
    
    /**********************************************
    * Create scenario
    **********************************************/

    /***  ScenarioParameters ***/
    // Sets the right values for ISD etc. for scenario type (UMa, UMi ..)
    // ScenarioParameters sets a 3 sector per site deployment
    // Can be set to NONE if I want omnidirectional 
    ScenarioParameters scenarioParams;
    scenarioParams.SetScenarioParameters (params.scenario);

    // The essentials describing a laydown
    uint32_t gnbSites = 0;
    double sector0AngleRad = 0;
    const uint32_t sectors = 3;

    /*** NodeDistributionScenarioInterface ***/
    // Creates framework for sites, sectors and antenna orientation  
    NodeDistributionScenarioInterface * scenario {NULL};
    
    /*** HexagonalGridScenarioHelper ***/
    // Sets the locations of base stations for a hex topology 
    // Ingerits from NodeDistributionScenarioInterface
    HexagonalGridScenarioHelper gridScenario;

    gridScenario.SetScenarioParameters (scenarioParams);
    gridScenario.SetNumRings (params.numOuterRings);
    gnbSites = gridScenario.GetNumSites ();
    // I changed this to reflect the number of macro and micro layers
    // Also I am not using allGnbNodes.GetN() here because it hasn't been initialized yet 
    uint32_t ueNum;
    if (params.useMicroLayer)
        ueNum = params.ueNumPergNb * ( gnbSites * sectors + params.numMicroCells);
    else
        ueNum = params.ueNumPergNb * gnbSites * sectors;
    
    gridScenario.SetUtNumber (ueNum);
    sector0AngleRad = gridScenario.GetAntennaOrientationRadians (0);

    // Creates and plots the network deployment and assigns gnb and ue containers
    gridScenario.CreateScenario ();
    macroLayerGnbNodes = gridScenario.GetBaseStations ();
    ueNodes = gridScenario.GetUserTerminals ();
    scenario = &gridScenario;

    // Log the topology configuration 
    std::cout << "\nTopology Configuration: " << gnbSites << " sites, "
            << sectors << " sectors/site, "
            << macroLayerGnbNodes.GetN ()   << " macro cells, "
            << ueNodes.GetN ()    << " UEs\n";

    // Create gNodeB containers by sector 
    NodeContainer gnbSector1Container, gnbSector2Container, gnbSector3Container;
    std::vector<NodeContainer*> gnbNodesBySector{&gnbSector1Container, 
      &gnbSector2Container, &gnbSector3Container};
    for (uint32_t cellId = 0; cellId < macroLayerGnbNodes.GetN (); ++cellId)
    {
        Ptr<Node> gnb = macroLayerGnbNodes.Get (cellId);
        auto sector = scenario->GetSectorIndex (cellId);
        gnbNodesBySector[sector]->Add (gnb);
    }
    
    /*
    * Create different UE NodeContainer for the different sectors.
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

    /***********************************************
    * Bounding box for the topology
    **********************************************/ 

    // Compute the bounding box for the scenario within which the UEs move
    double boundingBoxMinX = 0;
    double boundingBoxMinY = 0;
    double boundingBoxMaxX = 0;
    double boundingBoxMaxY = 0;  
    for (uint32_t cellIndex = 0; cellIndex < macroLayerGnbNodes.GetN (); ++cellIndex)
    {
        // NOTE: gnb mobility and position already set in hexagonalGridScenarioHelper 
        Ptr<Node> gnb = macroLayerGnbNodes.Get (cellIndex);
        Vector gnbpos = gnb->GetObject<MobilityModel> ()->GetPosition ();
        boundingBoxMinX = std::min(gnbpos.x, boundingBoxMinX);
        boundingBoxMinY = std::min(gnbpos.y, boundingBoxMinY);
        boundingBoxMaxX = std::max(gnbpos.x, boundingBoxMaxX);
        boundingBoxMaxY = std::max(gnbpos.y, boundingBoxMaxY);
    }

    // Add the cell width to the min and max boundaries to create the bounding box
    double hexCellRadius = gridScenario.GetHexagonalCellRadius();
    boundingBoxMinX = boundingBoxMinX - hexCellRadius;
    boundingBoxMinY = boundingBoxMinY - hexCellRadius;
    boundingBoxMaxX = boundingBoxMaxX + hexCellRadius;
    boundingBoxMaxY = boundingBoxMaxY + hexCellRadius;
    std::cout << "Topology Bounding box: "
            << "(" << boundingBoxMinX << ", " << boundingBoxMinY << ")  "
            << "(" << boundingBoxMinX << ", " << boundingBoxMaxY << ")  "
            << "(" << boundingBoxMaxX << ", " << boundingBoxMinY << ")  "
            << "(" << boundingBoxMaxX << ", " << boundingBoxMaxY << ")  "
            //<< " UE height: " << ueZ
            << "\nX width: " << (boundingBoxMaxX - boundingBoxMinX)
            << "  Y width: " << (boundingBoxMaxY - boundingBoxMinY)
            << std::endl;
    
    
    /***********************************************
    * Mobility model for UEs
    **********************************************/ 
    
    double ueZ =params.ueHeight;

    // Set these bounds for the mobility model of choice 
    Config::SetDefault ("ns3::SteadyStateRandomWaypointMobilityModel::MinX",
                          DoubleValue (boundingBoxMinX));
    Config::SetDefault ("ns3::SteadyStateRandomWaypointMobilityModel::MinY",
                          DoubleValue (boundingBoxMinY));
    Config::SetDefault ("ns3::SteadyStateRandomWaypointMobilityModel::MaxX",
                          DoubleValue (boundingBoxMaxX));
    Config::SetDefault ("ns3::SteadyStateRandomWaypointMobilityModel::MaxY",
                          DoubleValue (boundingBoxMaxY));
    Config::SetDefault ("ns3::SteadyStateRandomWaypointMobilityModel::Z", DoubleValue (ueZ));
    
    // Create random initial position allocator for UEs within this box
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

    // Create mobility model object 
    MobilityHelper mobility;
    mobility.SetPositionAllocator (positionAlloc);
    
    // We create 2 mobility categories. One slow moving and the other fast
    // Iterate over UEs to put them in slow or fast categories
    for (uint32_t ueId = 0; ueId < ueNodes.GetN (); ++ueId)
    {
        Ptr<Node> node = ueNodes.Get (ueId);
        if ( ueId < params.fracFastUes*ueNodes.GetN () ) 
        {
            // fast moving 
            mobility.SetMobilityModel ("ns3::SteadyStateRandomWaypointMobilityModel", 
                                       "MaxSpeed", DoubleValue (params.fastUeMaxSpeed), 
                                       "MinSpeed", DoubleValue (params.fastUeMinSpeed) );
            mobility.Install(node);
        }
        else
        {
            // slow moving
            mobility.SetMobilityModel ("ns3::SteadyStateRandomWaypointMobilityModel", 
                                       "MaxSpeed", DoubleValue (params.slowUeMaxSpeed), 
                                       "MinSpeed", DoubleValue (params.slowUeMinSpeed) );
            mobility.Install(node);            
        }
    }
    
    /***********************************************
    * Setup the LTE or NR RAN module
    **********************************************/

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
        LenaV1Utils::SetLenaV1SimulatorParameters (params, 
                                                 sector0AngleRad,
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
                                                 ueSector3NetDev);
    }
    else if (params.simulator == "5GLENA")
    {
        epcHelper = CreateObject<NrPointToPointEpcHelper> ();
        LenaV2Utils::SetLenaV2SimulatorParameters (sector0AngleRad,
                                                 params.scenario,
                                                 params.operationMode,
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
                                                 params.scheduler,
                                                 params.bandwidthMHz,
                                                 params.freqScenario,
                                                 params.downtiltAngle);
    }
    
    // Check that we got a valid helper
    if ( (lteHelper == nullptr) && (nrHelper == nullptr) )
    {
        NS_ABORT_MSG ("Programming error: no valid helper");
    }

    allGnbNodes.Add (macroLayerGnbNodes);
    
    if (params.useMicroLayer) 
    {
        /*************************************************************************
        ************************* Micro layer********************************
        **************************************************************************/    

        microLayerGnbNodes.Create(params.numMicroCells); 
        MobilityHelper microGnbMobility;
        microGnbMobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
        // reusing the same position allocator as the one used for UEs 
        // since the bounding box is the same 
        microGnbMobility.SetPositionAllocator (positionAlloc);
        microGnbMobility.Install (microLayerGnbNodes);

        // I could also just use the same lte helper we had before since 
        // those gnbs have already been installed 
        // Configure a new lteHelper and install all the RAN stuff
        // Is it okay to have 2 lteHelpers ?

        Config::SetDefault ("ns3::LteEnbPhy::TxPower", DoubleValue (params.microCellTxPower)); // Don't know what this shoudl be set to 

        /*
        Ptr<LteHelper> microLteHelper = CreateObject<LteHelper> ();

        // Does this work ? Will it get attached to the same EPC as the macro layer ? 

        microLteHelper->SetEpcHelper (epcHelper);

        // Do these without setting default 

        // same as macro layer can apply
        /////////////////////////Config::SetDefault ("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue (10*1024)); //default is 10*1024 = 10KB

        lteHelmicroLteHelperper->SetAttribute ("PathlossModel", StringValue ("ns3::ThreeGppUmiStreetCanyonPropagationLossModel"));

        if (params.handoverAlgo == "A3Rsrp")
        {
            microLteHelper->SetHandoverAlgorithmType ("ns3::A3RsrpHandoverAlgorithm");
            microLteHelper->SetHandoverAlgorithmAttribute ("Hysteresis", 
                                                    DoubleValue (6.0)); // used to be 3
            microLteHelper->SetHandoverAlgorithmAttribute ("TimeToTrigger",
                                                  TimeValue (MilliSeconds (500))); // used to be 256 
        }
        else if (params.handoverAlgo == "A2A4Rsrq")
        {
            microLteHelper->SetHandoverAlgorithmType ("ns3::A2A4RsrqHandoverAlgorithm");
            microLteHelper->SetHandoverAlgorithmAttribute ("ServingCellThreshold",
                                                UintegerValue (30));
            microLteHelper->SetHandoverAlgorithmAttribute ("NeighbourCellOffset",
                                                UintegerValue (1));
        }
        microLteHelper->SetPathlossModelAttribute ("ShadowingEnabled", BooleanValue (true));
        if (params.scheduler == "PF")
        {
            microLteHelper->SetSchedulerType ("ns3::PfFfMacScheduler");
        }
        else if (params.scheduler == "RR")
        {
            microLteHelper->SetSchedulerType ("ns3::RrFfMacScheduler");
        }*/



        // same as macro layer can apply
        //Config::SetDefault ("ns3::LteUePhy::EnableUplinkPowerControl", BooleanValue (params.enableUlPc));
        // antenna stuff
        // Not doing anything since default is omnidirectional (is it ? or is it isotropic)
        uint32_t microDlRB;
        uint32_t microUlRB;
      if (params.microBandwidthMHz == 20)
        {
          microDlRB = 100;
          microUlRB = 100;
        }
      else if (params.microBandwidthMHz == 15)
        {
          microDlRB = 75;
          microUlRB = 75;
        }
      else if (params.microBandwidthMHz == 10)
        {
          microDlRB = 50;
          microUlRB = 50;
        }
      else if (params.microBandwidthMHz == 5)
        {
          microDlRB = 25;
          microUlRB = 25;
        }
      else
        {
          NS_ABORT_MSG ("The configured micro layer bandwidth in MHz not supported:" << params.microBandwidthMHz);
        }
        lteHelper->SetEnbDeviceAttribute ("DlBandwidth", UintegerValue (microDlRB));
        lteHelper->SetEnbDeviceAttribute ("UlBandwidth", UintegerValue (microUlRB));
        
        if (!params.macroMicroSharedSpectrum) 
        {
            lteHelper->SetEnbDeviceAttribute ("DlEarfcn", UintegerValue (2850)); // 2620 MHz, band 7
            lteHelper->SetEnbDeviceAttribute ("UlEarfcn", UintegerValue (20850)); // 2620 MHz, band 7
        }
        
        // If using lteHelper I need to reset the antenna stuff as well... 
        lteHelper->SetEnbAntennaModelType ("ns3::IsotropicAntennaModel");
        lteHelper->SetEnbAntennaModelAttribute ("Gain", DoubleValue (0));

        // install netDev
        NetDeviceContainer microLayerGnbLteDevs = lteHelper->InstallEnbDevice (microLayerGnbNodes);
        //Ptr<LteEnbPhy> enbPhy = microLayerGnbLteDevs->GetPhy();
        //enbPhy->SetAttribute("TxPower", DoubleValue (23.0));
        // Install internet 
        // figure out the epc connection etc. 
        // connect to remote host 


        allGnbNodes.Add (microLayerGnbNodes);   
        // global declaration to make things easy   
        //NUM_GNBS = allGnbNodes.GetN();

        /*************************************************************************
        ************************* Micro layer END ********************************
    **************************************************************************/
    }
    
    
    // create the internet and install the IP stack on the UEs
    // get SGW/PGW and create a single RemoteHost
    Ptr<Node> pgw = epcHelper->GetPgwNode ();
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create (1);
    Ptr<Node> remoteHost = remoteHostContainer.Get (0);
    InternetStackHelper internet;
    internet.Install (remoteHostContainer);

    // Connect a remoteHost to pgw. Setup routing too
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
    // CHECK IF THIS IS THE REASON FOR MISMATCH 
    //ueIpIfaces {epcHelper->AssignUeIpv4Address (ueNetDevs)}
    ueIpIfaces = epcHelper->AssignUeIpv4Address (ueNetDevs);
    Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress (1);

    // Set the default gateway for the UEs
    for (auto ue = ueNodes.Begin (); ue != ueNodes.End (); ++ue)
    {
        Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting ((*ue)->GetObject<Ipv4> ());
        ueStaticRouting->SetDefaultRoute (epcHelper->GetUeDefaultGatewayAddress (), 1);
    }

    
    if (lteHelper != nullptr)
    {
        // This does not work. It connects all UEs to CellId 1
        //lteHelper->AttachToClosestEnb (ueNetDevs, gnbNetDevs); 
        lteHelper->Attach (ueNetDevs);
    }
    else if (nrHelper != nullptr)
    {
        nrHelper->AttachToClosestEnb (ueNetDevs, gnbNetDevs);
    }
    
    
    // Attach UEs to their gNB. Try to attach them per cellId order
    /*for (uint32_t ueId = 0; ueId < ueNodes.GetN (); ++ueId)
    {
        auto cellId = scenario->GetCellIndex (ueId);
        Ptr<NetDevice> gnbNetDev = gnbNetDevs.Get (cellId);
        Ptr<NetDevice> ueNetDev = ueNetDevs.Get (ueId);
        if (lteHelper != nullptr)
        {
            // This call attaches the ues in the ue container to the 
            // gnbs in the gnb container
            lteHelper->Attach (ueNetDev, gnbNetDev);
            
            // This call attaches the ues automatically to the best gnb available to it 
            //lteHelper->Attach (ueNetDev);
        }
        else if (nrHelper != nullptr)
        {
            nrHelper->AttachToEnb (ueNetDev, gnbNetDev);
            //nrHelper->Attach (ueNetDev);
            // UL phy
            uint32_t bwp = (params.operationMode == "FDD" ? 1 : 0);
            auto ueUlPhy {nrHelper->GetUePhy (ueNetDev, bwp)};
        }
    }*/



    /***********************************************
    * Traffic generation applications
    **********************************************/
    uint16_t ulFlowPortNum = 1234;
    uint16_t dlFlowPortNum = 1235;
    uint16_t echoPortNum = 9; // well known echo port
    uint16_t ulDelayPortNum = 17000;
    uint16_t dlDelayPortNum = 18000;
    uint16_t dashPortNum = 15000;

    uint32_t echoPacketCount = 0xFFFFFFFF; 
    uint32_t delayPacketCount = 0xFFFFFFFF; 
    uint32_t flowPacketCount = 0xFFFFFFFF;

    // Configuration parameters for UL and DL Traffic parameters for the Flow app 
    //uint32_t flowPacketSize = 1000; // this gets set based on trafficScenario and BW 
    //uint32_t lambda;
    double dataRate;
    switch (params.bandwidthMHz)
    {
        case 5: // 5MHz = 25 PRBS
            dataRate = 18; // X 3 for the 3 beams Mbps 
        case 10: // 10 MHz = 50 PRBs
            dataRate = 36; // Mbps 
        case 20: // 20 MHz = 100 PRBs
            dataRate = 75; // Mbps 
    } 
    uint16_t flowUeCount = 0;
    for (uint32_t ueId = 0; ueId < ueNodes.GetN (); ++ueId)
    {
        if (ueId % 5 != 0)
        {
            flowUeCount++;
        }
    }  
    double load = params.trafficLoadFrac * dataRate * 1000000; // bps
    double flowPerUe = load / (flowUeCount / allGnbNodes.GetN()); // bps
    Time flowInterval = Seconds ( (params.flowPacketSize*8)/load );  
    std::cout << "\nUDP flow has a per flow rate of " << (flowPerUe/1000000) << " Mbps\n" << std::endl; 

    // Server Config 
    ApplicationContainer serverApps;

    // Declaration of Helpers for Servers 
    UdpServerHelper ulDelayPacketSink (ulDelayPortNum);
    UdpServerHelper dlDelayPacketSink (dlDelayPortNum);
    UdpServerHelper ulFlowPacketSink (ulFlowPortNum);
    UdpServerHelper dlFlowPacketSink (dlFlowPortNum);
    ThreeGppHttpServerHelper httpServer (remoteHostAddr);
    DashServerHelper dashServer ("ns3::TcpSocketFactory",
                    InetSocketAddress (Ipv4Address::GetAny (), dashPortNum));
    UdpEchoServerHelper echoServer (echoPortNum);

    // Server Creation 
    if(params.traceDelay)
    {
        serverApps.Add (ulDelayPacketSink.Install (remoteHost)); // appId updated on remoteHost
    }
    if(params.traceFlow)
    {
        serverApps.Add (ulFlowPacketSink.Install (remoteHost)); // appId updated on remoteHost
    }
    if(params.traceHttp)
    {   
        serverApps.Add (httpServer.Install (remoteHost)); // appId updated on remoteHost        
        //Ptr<ThreeGppHttpServer> httpSer = serverApps.Get (2)->GetObject<ThreeGppHttpServer> (); // appId on remoteHost used
        //PointerValue varPtr;
        //httpSer->GetAttribute ("Variables", varPtr);
        //Ptr<ThreeGppHttpVariables> httpVariables = varPtr.Get<ThreeGppHttpVariables> ();
        //httpVariables->SetMainObjectSizeMean (params.httpMainObjMean); // 100kB
        //httpVariables->SetMainObjectSizeStdDev (params.httpMainObjStd); // 40kB
        
    }
    if(params.traceDash)
    {
        serverApps.Add (dashServer.Install (remoteHost)); // appId updated on remoteHost
    }
    if(params.traceRtt)
    {
        serverApps.Add (echoServer.Install (remoteHost)); // appId updated on remoteHost
    }


    //========================================================
    // Client Config 
    ApplicationContainer clientApps;

    // Declarations of Helpers for Clients
    UdpClientHelper ulFlowClient;
    UdpClientHelper dlFlowClient;
    UdpClientHelper ulDelayClient;
    UdpClientHelper dlDelayClient;
    UdpEchoClientHelper echoClient (remoteHostAddr, echoPortNum);
    DashClientHelper dashClient ("ns3::TcpSocketFactory", InetSocketAddress (remoteHostAddr, dashPortNum), params.abr);
    ThreeGppHttpClientHelper httpClient (remoteHostAddr);

    // Client Config
    if(params.traceFlow)
    {  
        // Configure UL and DL flow client applications 
        //Time flowInterval = Seconds (1.0 / lambda);
        ulFlowClient.SetAttribute ("RemotePort", UintegerValue (ulFlowPortNum));
        ulFlowClient.SetAttribute ("MaxPackets", UintegerValue (flowPacketCount));
        ulFlowClient.SetAttribute ("PacketSize", UintegerValue (params.flowPacketSize));
        ulFlowClient.SetAttribute ("Interval", TimeValue (flowInterval));

        dlFlowClient.SetAttribute ("RemotePort", UintegerValue (dlFlowPortNum));
        dlFlowClient.SetAttribute ("MaxPackets", UintegerValue (flowPacketCount));
        dlFlowClient.SetAttribute ("PacketSize", UintegerValue (params.flowPacketSize));
        dlFlowClient.SetAttribute ("Interval", TimeValue (flowInterval));
    }
    if(params.traceDelay)
    {
        // Configure UL and DL delay client applications 
        ulDelayClient.SetAttribute ("RemotePort", UintegerValue (ulDelayPortNum));
        ulDelayClient.SetAttribute ("MaxPackets", UintegerValue (delayPacketCount));
        ulDelayClient.SetAttribute ("PacketSize", UintegerValue (params.delayPacketSize));
        ulDelayClient.SetAttribute ("Interval", TimeValue (params.delayInterval));

        dlDelayClient.SetAttribute ("RemotePort", UintegerValue (dlDelayPortNum));
        dlDelayClient.SetAttribute ("MaxPackets", UintegerValue (delayPacketCount));
        dlDelayClient.SetAttribute ("PacketSize", UintegerValue (params.delayPacketSize));
        dlDelayClient.SetAttribute ("Interval", TimeValue (params.delayInterval));
    }
    if(params.traceRtt)
    {
        // Configure echo client application
        echoClient.SetAttribute ("MaxPackets", UintegerValue (echoPacketCount));
        echoClient.SetAttribute ("Interval", TimeValue (params.echoInterPacketInterval));
        echoClient.SetAttribute ("PacketSize", UintegerValue (params.echoPacketSize));
    }
    if(params.traceDash)
    {
        // Configure DASH client application 
        dashClient.SetAttribute ("VideoId", UintegerValue (1)); // VideoId should be positive
        dashClient.SetAttribute ("TargetDt", TimeValue (Seconds (params.targetDt)));
        dashClient.SetAttribute ("window", TimeValue (Seconds (params.window)));
        dashClient.SetAttribute ("bufferSpace", UintegerValue (params.bufferSpace));
    }
    if(params.traceHttp)
    {
        // Configure http client application
    }

    // Client Creation on the desired devices
    Ptr<UniformRandomVariable> startRng = CreateObject<UniformRandomVariable> ();
    startRng->SetStream (RngSeedManager::GetRun ());


    for (uint32_t ueId = 0; ueId < ueNodes.GetN (); ++ueId)
    {
        Ptr<Node> node = ueNodes.Get (ueId);
        Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();
        Ipv4InterfaceAddress iaddr = ipv4->GetAddress (1,0); 
        Ipv4Address addr = iaddr.GetLocal ();

        // Client apps
        // These are the apps that are on all devices 
        if (params.traceDelay)
        {
            serverApps.Add (dlDelayPacketSink.Install (node));  
            auto appsClass2 = InstallUlDelayTrafficApps (node,
                                  &ulDelayClient,
                                  remoteHost, remoteHostAddr, params.appStartTime,
                                  startRng, params.appGenerationTime);
            clientApps.Add (appsClass2.first);
            auto appsClass3 = InstallDlDelayTrafficApps (node, addr,
                              &dlDelayClient,
                              remoteHost, remoteHostAddr, params.appStartTime,
                              startRng, params.appGenerationTime);
            clientApps.Add (appsClass3.first);
            // Print the IMSI of the ues that are doing this	
            std::cout << "ueId: " << ueId << " IMSI: " << GetImsi_from_ueId(ueId) 
                << " Ip_addr: " << addr 
                << " has UL and DL Delay app installed " << std::endl;
        }
        if(params.traceRtt)
        {
            auto appsClass1 = InstallUdpEchoApps (node,
                              &echoClient,
                              params.appStartTime,
                              startRng, params.appGenerationTime);
            clientApps.Add (appsClass1.first);
            std::cout << "ueId: " << ueId << " IMSI: " << GetImsi_from_ueId(ueId)  
              << " IP_addr: " << addr
              <<" has RTT app installed" << std::endl;
        } 


        // These are the apps that are on a subset of devices 
        if (ueId % 5 == 0)
        {
            // These UEs do just delay measurement. So nothing extra to add
            // Print the IMSI of the ues that are doing this
        } 


        else if (ueId % 2 == 0)
        {
            // These UEs do video + background CBR traffic + delay measurement
            if(params.traceDash)
            {
                auto appsClass6 = InstallDashApps (node,
                                  &dashClient,
                                  params.appStartTime,
                                  startRng, params.appGenerationTime);
                clientApps.Add (appsClass6.first);
                // Print the IMSI of the ues that are doing this
                std::cout << "ueId: " << ueId << " IMSI: " << GetImsi_from_ueId(ueId)  
                << " IP_addr: " << addr
                <<" has DASH app installed" << std::endl;
            }
            if(params.traceFlow)
            {
                serverApps.Add (dlFlowPacketSink.Install (node)); 
                auto appsClass4 = InstallUlFlowTrafficApps (node,
                                  &ulFlowClient,
                                  remoteHost, remoteHostAddr, params.appStartTime,
                                  startRng, params.appGenerationTime);
                clientApps.Add (appsClass4.first);
                auto appsClass5 = InstallDlFlowTrafficApps (node, addr,
                                  &dlFlowClient,
                                  remoteHost, remoteHostAddr, params.appStartTime,
                                  startRng, params.appGenerationTime);
                clientApps.Add (appsClass5.first);
                // Print the IMSI of the ues that are doing this	
                std::cout << "ueId: " << ueId << " IMSI: " << GetImsi_from_ueId(ueId) 
                << " Ip_addr: " << addr 
                << " has UL and DL FLOW app installed " << std::endl;
            }
        }


        else 
        {
                // These UEs do web browsing + background CBR traffic + delay measurement
                if(params.traceHttp)
                {
                    auto appsClass6 = InstallHttpApps (node,
                                  &httpClient,
                                  params.appStartTime,
                                  startRng, params.appGenerationTime);
                    clientApps.Add (appsClass6.first);
                    // Print the IMSI of the ues that are doing this
                    std::cout << "ueId: " << ueId << " IMSI: " << GetImsi_from_ueId(ueId) 
                      << " IP_addr: " << addr 
                      << " has HTTP app installed" << std::endl;
                }
                if(params.traceFlow)
                {
                    serverApps.Add (dlFlowPacketSink.Install (node));
                    auto appsClass4 = InstallUlFlowTrafficApps (node,
                                      &ulFlowClient,
                                      remoteHost, remoteHostAddr, params.appStartTime,
                                      startRng, params.appGenerationTime);
                    clientApps.Add (appsClass4.first);
                    auto appsClass5 = InstallDlFlowTrafficApps (node, addr,
                                      &dlFlowClient,
                                      remoteHost, remoteHostAddr, params.appStartTime,
                                      startRng, params.appGenerationTime);
                    clientApps.Add (appsClass5.first);
                    // Print the IMSI of the ues that are doing this	
                    std::cout << "ueId: " << ueId << " IMSI: " << GetImsi_from_ueId(ueId) 
                    << " Ip_addr: " << addr 
                    << " has UL and DL FLOW app installed " << std::endl;
                }
        }
    } // end of for over UEs

    // Server Start  
    serverApps.Start (params.appStartTime);
    // client apps are started individually using the Install function 

    // Setup X2 interface to enable handover   
    if (lteHelper != nullptr)
    {
        //lteHelper->AddX2Interface (macroLayerGnbNodes);
        lteHelper->AddX2Interface (allGnbNodes);
        /*if (microLteHelper != nullptr)
        {         
            microLteHelper->AddX2Interface (microLayerGnbNodes);
        }   */
    }
    else if (nrHelper != nullptr)
    {
        // BooHoo Handover is not yet supported for NR 
    }


    // enable the RAN traces provided by the LTE or NR module
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

    // enable packet tracing from the application layer 
    if (params.traces == true)
    {
        // appId is being used here BE CAREFUL about changing the order 
        // in which the apps get added to the server container
        if (params.traceDelay || params.traceFlow)
        {
            Config::Connect ("/NodeList/*/ApplicationList/*/$ns3::UdpServer/RxWithAddresses", 
            MakeBoundCallback (&udpServerTrace, 
                               std::make_pair(ulDelayPortNum, dlDelayPortNum),
                               std::make_pair(ulFlowPortNum, dlFlowPortNum), 
                               remoteHost));
        }

        // connect custom trace sinks for RRC connection establishment and handover notification
        
        //Config::Connect ("/NodeList/*/DeviceList/*/LteUeRrc/ConnectionEstablished",
        //               MakeCallback (&NotifyConnectionEstablishedUe));
        Config::Connect ("/NodeList/*/DeviceList/*/LteEnbRrc/ConnectionEstablished",
                       MakeCallback (&NotifyConnectionEstablishedEnb));
        //Config::Connect ("/NodeList/*/DeviceList/*/LteUeRrc/HandoverStart",
        //               MakeCallback (&NotifyHandoverStartUe));
        Config::Connect ("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverStart",
                       MakeCallback (&NotifyHandoverStartEnb));
        

        if(params.traceRtt)
        {
            Config::Connect ("/NodeList/*/ApplicationList/*/$ns3::UdpEchoClient/RxWithAddresses", 
                             MakeBoundCallback (&rttTrace, rttStream));
        }
        if(params.traceDash)
        {
            Config::Connect ("/NodeList/*/ApplicationList/*/$ns3::DashClient/RxSegment", 
                             MakeBoundCallback (&dashClientTrace, dashClientStream));
            Config::Connect ("/NodeList/*/ApplicationList/*/$ns3::DashClient/PlayedFrame", 
                             MakeBoundCallback (&mpegPlayerTrace, mpegPlayerStream));
        }
        if(params.traceHttp)
        {
            Config::Connect ("/NodeList/*/ApplicationList/*/$ns3::ThreeGppHttpClient/RxDelay", 
                             MakeBoundCallback (&httpClientTraceRxDelay, httpClientDelayStream));
            Config::Connect ("/NodeList/*/ApplicationList/*/$ns3::ThreeGppHttpClient/RxRtt", 
                             MakeBoundCallback (&httpClientTraceRxRtt, httpClientRttStream));
            Config::Connect ("/NodeList/*/ApplicationList/*/$ns3::ThreeGppHttpServer/RxDelay", 
                             MakeBoundCallback (&httpServerTraceRxDelay, httpServerDelayStream));
        }

    }

    // Add some extra time for the last generated packets to be received
    const Time appStopWindow = MilliSeconds (50);
    Time stopTime = params.appStartTime + appStartWindow + params.appGenerationTime + appStopWindow;
    std::cout << "\n------------------------------------------------------\n";
    std::cout << "Start Simulation ! Runtime: " << stopTime.GetSeconds() << " seconds\n";
    Simulator::Stop (stopTime);
    // schedule the periodic logging of UE positions
    Simulator::Schedule (MilliSeconds(500), &LogPosition, mobStream);
    // To print info, some of which is only available a few milliseconds 
    // after the simulation is setup and UEs attached
    Simulator::Schedule (MilliSeconds(100), &ScenarioInfo, scenario);
    
    // To schedule on/off of apps that I would like to pause and resume randomly 
    Ptr<UniformRandomVariable> appOnTime = CreateObject<UniformRandomVariable> ();
    appOnTime->SetAttribute ("Min", DoubleValue (1));
    appOnTime->SetAttribute ("Max", DoubleValue (300));
    Ptr<UniformRandomVariable> appOffTime = CreateObject<UniformRandomVariable> ();
    appOffTime->SetAttribute ("Min", DoubleValue (1));
    appOffTime->SetAttribute ("Max", DoubleValue (30));
    // sample ON time and pass both on and off random variables  
    // 
    
    
    //Simulator::Schedule (Seconds(appOnTime->GetValue()), &PauseApp, appOnTime, appOffTime);
    
    
    
    if ( (params.simulator == "LENA") && (params.useMicroLayer) && (!params.macroMicroSharedSpectrum))
    {
        // This is a nearest cell manual HO that works even between frequencies 
        Simulator::Schedule (MilliSeconds(params.manualHoTriggerTime), &CheckForManualHandovers, lteHelper); 
    }
    Simulator::Run ();
    std::cout << "\n------------------------------------------------------\n"
            << "End simulation"
            << std::endl;
    Simulator::Destroy ();
    }

} // namespace ns3

