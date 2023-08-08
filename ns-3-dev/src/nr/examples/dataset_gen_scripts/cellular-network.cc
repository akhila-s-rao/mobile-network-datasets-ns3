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
#include<cmath>

#include <filesystem>

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

//For VR app
#include "ns3/seq-ts-size-frag-header.h"
#include "ns3/bursty-helper.h"
#include "ns3/burst-sink-helper.h"
#include "ns3/trace-file-burst-generator.h"

#include "cellular-network.h"
#include "cellular-network-functions.h"

NS_LOG_COMPONENT_DEFINE ("CellularNetwork");

namespace ns3 {
    
void CellularNetwork (const Parameters &params){
    LogComponentEnable ("DashServer", LOG_LEVEL_INFO);
    LogComponentEnable ("DashClient", LOG_LEVEL_INFO);
    // Validate the parameter settings  
    params.Validate ();
    // Set as global for easy access
    global_params = params;

    RngSeedManager::SetSeed (params.randSeed+1); 
    RngSeedManager::SetRun (params.randSeed);
    
    // Default values for the simulation. 
    Config::SetDefault ("ns3::LteEnbMac::NumberOfRaPreambles", UintegerValue (40));
    // SrsPeriodicity must be larger than the max number of UEs that could be connected to a BS
    // The UeSinrSamplePeriod and InterferenceSamplePeriod from LteEnbPhy are multiplied by this SrsPeriodicity
    // because I guess they are measured on every srsReport. So make to account for this when you set the logging period   
    Config::SetDefault ("ns3::LteEnbRrc::SrsPeriodicity", UintegerValue (80));// 320 is max {0, 2, 5, 10, 20, 40,  80, 160, 320}
    Config::SetDefault ("ns3::LteHelper::UseIdealRrc", BooleanValue (true));
    Config::SetDefault ("ns3::LteSpectrumPhy::CtrlErrorModelEnabled", BooleanValue (false));
    // Set time granularity for RAN traces that are periodic
    Config::SetDefault ("ns3::LteUePhy::RsrpSinrSamplePeriod", UintegerValue (params.ranSamplePeriodMilli));
    Config::SetDefault ("ns3::LteEnbPhy::UeSinrSamplePeriod", UintegerValue (1)); // The real sample period is multiplied by ns3::LteEnbRrc::SrsPeriodicity
    Config::SetDefault ("ns3::LteEnbPhy::InterferenceSamplePeriod", UintegerValue (1)); // The real sample period is multiplied by ns3::LteEnbRrc::SrsPeriodicity
    // This is set in the lte-utils (not nr-utils however) 
    //Config::SetDefault ("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue (10 * 1024)); //default is 10240 
    Config::SetDefault ("ns3::TcpSocket::SndBufSize", UintegerValue (params.tcpSndRcvBuf));// default is 131072
    Config::SetDefault ("ns3::TcpSocket::RcvBufSize", UintegerValue (params.tcpSndRcvBuf));// default is 131072
    // Need to increase size from default for larger webpages.  
    Config::SetDefault ("ns3::ThreeGppHttpVariables::MainObjectSizeMean", UintegerValue (params.httpMainObjMean));
    Config::SetDefault ("ns3::ThreeGppHttpVariables::MainObjectSizeStdDev", UintegerValue (params.httpMainObjStd));
    
    Config::SetDefault ("ns3::ThreeGppHttpVariables::EmbeddedObjectSizeMean", UintegerValue (params.httpEmbeddedObjectSizeMean));
    Config::SetDefault ("ns3::ThreeGppHttpVariables::NumOfEmbeddedObjectsScale", UintegerValue (params.httpNumOfEmbeddedObjectsScale));
    Config::SetDefault ("ns3::ThreeGppHttpVariables::NumOfEmbeddedObjectsMax", UintegerValue (params.httpNumOfEmbeddedObjectsMax));
    
    
    // Create user created trace files with corresponding column names
    CreateTraceFiles ();
    std::string traceFolder = params.ns3Dir + "src/vr-app/model/BurstGeneratorTraces/"; // example traces can be found here	

    // Trace files for VR     
    std::string vrTraceFiles[8]
        = {"mc_10mbps_30fps.csv", "ge_cities_10mbps_30fps.csv", "ge_tour_10mbps_30fps.csv", "vp_10mbps_30fps.csv", 
        "mc_10mbps_60fps.csv", "ge_cities_10mbps_60fps.csv", "ge_tour_10mbps_60fps.csv", "vp_10mbps_60fps.csv"};
    uint16_t vrTraceFileIndex = 0;
    
    int64_t randomStream = 1;
    
    /****************************************************
    * Macro gNBs and all UEs: topology scenario screation
    *****************************************************/

    // Sets the right values for ISD etc. for scenario type (UMa, UMi ..)
    // ScenarioParameters sets a 3 sector per site deployment
    // Can be set to NONE if I want omnidirectional 
    ScenarioParameters scenarioParams;
    scenarioParams.SetScenarioParameters (params.scenario);
    // The essentials describing a laydown
    uint32_t gnbSites = 0;
    double sector0AngleRad = 0;
    const uint32_t sectors = 3;
    // Creates framework for sites, sectors and antenna orientation  
    NodeDistributionScenarioInterface * scenario {NULL};
    // Sets the locations of base stations for a hex topology 
    // Inherits from NodeDistributionScenarioInterface
    HexagonalGridScenarioHelper gridScenario;
    gridScenario.SetScenarioParameters (scenarioParams);
    gridScenario.SetNumRings (params.numOuterRings);
    gnbSites = gridScenario.GetNumSites ();
    // I changed this to reflect the number of macro and micro layers
    uint32_t ueNum;
    if (params.useMicroLayer)
        ueNum = (params.ueNumPerMacroGnb * gnbSites * sectors) + (params.ueNumPerMicroGnb * params.numMicroCells) ;
    else
        ueNum = params.ueNumPerMacroGnb * gnbSites * sectors;

    sector0AngleRad = gridScenario.GetAntennaOrientationRadians (0);
    std::cout << "sector0AngleRad: " << sector0AngleRad << std::endl;

    // Creates and plots the network deployment and assigns gnb and ue containers
    gridScenario.CreateScenario ();
    macroLayerGnbNodes = gridScenario.GetBaseStations ();
    allGnbNodes.Add (macroLayerGnbNodes);
    ueNodes.Create(ueNum);
    scenario = &gridScenario;

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
  
    // Separate out the macro and micro ueNodes so that we can handle their mobility models and app 
    // installations separately. Since we are separating them inside here, this section of code needs 
    // to be before the installation of macro UE mobility and applications  
    // The first numMacro UEs are macro and the rest are micro     
    if (params.useMicroLayer) 
    {
        for (uint32_t ueId = 0; ueId < ueNodes.GetN (); ++ueId)
        {
            // First set of nodes are macro UE nodes moving within the larger bounding box
            if (ueId < (params.ueNumPerMacroGnb * gnbSites * sectors))
            {
                ueNodesMacro.Add(ueNodes.Get (ueId));
            }
            // The rest are going to be initialised into smaller micro UE boundign boxes
            else
            {
                ueNodesMicro.Add(ueNodes.Get (ueId));
            }
        }
    }
    else
    {
        ueNodesMacro.Add(ueNodes);
    }

    /*********************************************************
    * Macro UEs: Position Bounding box and Mobility model
    **********************************************************/ 

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
    boundingBoxMinX = boundingBoxMinX - sqrt(2)*hexCellRadius;
    boundingBoxMinY = boundingBoxMinY - sqrt(2)*hexCellRadius;
    boundingBoxMaxX = boundingBoxMaxX + sqrt(2)*hexCellRadius;
    boundingBoxMaxY = boundingBoxMaxY + sqrt(2)*hexCellRadius;
    std::cout << "Topology Bounding box (x,y): "
            << "(" << boundingBoxMinX << ", " << boundingBoxMinY << ")  "
            << "(" << boundingBoxMinX << ", " << boundingBoxMaxY << ")  "
            << "(" << boundingBoxMaxX << ", " << boundingBoxMinY << ")  "
            << "(" << boundingBoxMaxX << ", " << boundingBoxMaxY << ")  "
            << "\nArea X width: " << (boundingBoxMaxX - boundingBoxMinX) << " meters"
            << "  Area Y width: " << (boundingBoxMaxY - boundingBoxMinY) << " meters\n"
            << std::endl;
    
    double ueZ =params.ueHeight;    
    
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
    
    // Create mobility model object 
    MobilityHelper mobility;
    mobility.SetPositionAllocator (positionAlloc);
    
    // We create 2 mobility categories. One slow moving and the other fast
    // Iterate over UEs to put them in slow or fast categories
    for (uint32_t ueId = 0; ueId < ueNodesMacro.GetN (); ++ueId)
    {
        Ptr<Node> node = ueNodesMacro.Get (ueId);
        //The first few UeIds are the fast moving ones 
        if ( ueId < (params.fracFastUes*ueNodesMacro.GetN ()) ) // use floor(params.fracFastUes*ueNodesMacro.GetN ()) to allow only slow in 3 UE case 
        {
            // fast moving 
            mobility.SetMobilityModel ("ns3::SteadyStateRandomWaypointMobilityModel", 
                                       "MaxSpeed", DoubleValue (params.fastUeMaxSpeed), 
                                       "MinSpeed", DoubleValue (params.fastUeMinSpeed) );
            mobility.Install(node);
            // save list of UE IDs that are fast moving for later use 
            fastUes.push_back (ueId);
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
    * Macro gNBs and all UEs: RAN settings
    **********************************************/
  
    Ptr<PointToPointEpcHelper> epcHelper;

    NetDeviceContainer gnbSector1NetDev, gnbSector2NetDev, gnbSector3NetDev;
    std::vector<NetDeviceContainer *> gnbNdBySector {&gnbSector1NetDev, &gnbSector2NetDev, &gnbSector3NetDev};
    NetDeviceContainer ueNetDevs;
    //NetDeviceContainer ueSector1NetDev, ueSector2NetDev,ueSector3NetDev;
    //std::vector<NetDeviceContainer *> ueNdBySector {&ueSector1NetDev, &ueSector2NetDev, &ueSector3NetDev};

    Ptr <LteHelper> lteHelper = nullptr;
    Ptr <NrHelper> nrHelper = nullptr;

    if (params.rat == "LTE")
    {
        epcHelper = CreateObject<PointToPointEpcHelper> ();
        LteUtils::SetLteSimulatorParameters (params, 
                                                 sector0AngleRad,
                                                 gnbSector1Container,
                                                 gnbSector2Container,
                                                 gnbSector3Container,
                                                 ueNodes,
                                                 epcHelper,
                                                 lteHelper,
                                                 gnbSector1NetDev,
                                                 gnbSector2NetDev,
                                                 gnbSector3NetDev);
    }
    else if (params.rat == "NR")
    {
        epcHelper = CreateObject<NrPointToPointEpcHelper> ();
        NrUtils::SetNrSimulatorParameters (sector0AngleRad,
                                                 params.scenario,
                                                 params.operationMode,
                                                 params.numerologyBwp,
                                                 params.pattern,
                                                 gnbSector1Container,
                                                 gnbSector2Container,
                                                 gnbSector3Container,
                                                 ueNodes,
                                                 epcHelper,
                                                 nrHelper,
                                                 gnbSector1NetDev,
                                                 gnbSector2NetDev,
                                                 gnbSector3NetDev,
                                                 ueNetDevs,
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
    
    NetDeviceContainer gnbNetDevs (gnbSector1NetDev, gnbSector2NetDev);
    gnbNetDevs.Add (gnbSector3NetDev);
    
    
    if (params.useMicroLayer) 
    {
        /****************************************************
        * Micro gNBs: creation
        *****************************************************/   
        microLayerGnbNodes.Create(params.numMicroCells);    
        
        /*********************************************************
        * Micro gNBs: Position allocation 
        **********************************************************/         
        
        Ptr<ListPositionAllocator> microPositionAlloc = CreateObject<ListPositionAllocator> ();  
        microPositionAlloc->Add (Vector(-100,-175,7));
        microPositionAlloc->Add (Vector(175,175,7));
        microPositionAlloc->Add (Vector(190,-180,7));
       
        
        // Random position allocation for micro basestations
        /*
        Ptr<PositionAllocator> microPositionAlloc = CreateObject<RandomBoxPositionAllocator> ();
        Ptr<UniformRandomVariable> xPos = CreateObject<UniformRandomVariable> ();
        xPos->SetAttribute ("Min", DoubleValue (boundingBoxMinX));
        xPos->SetAttribute ("Max", DoubleValue (boundingBoxMaxX));
        microPositionAlloc->SetAttribute ("X", PointerValue (xPos));
        Ptr<UniformRandomVariable> yPos = CreateObject<UniformRandomVariable> ();
        yPos->SetAttribute ("Min", DoubleValue (boundingBoxMinY));
        yPos->SetAttribute ("Max", DoubleValue (boundingBoxMaxY));
        microPositionAlloc->SetAttribute ("Y", PointerValue (yPos));
        Ptr<ConstantRandomVariable> zPos = CreateObject<ConstantRandomVariable> ();
        // make this a parameter and set it in the .h file
        zPos->SetAttribute ("Constant", DoubleValue (7.0));
        microPositionAlloc->SetAttribute ("Z", PointerValue (zPos));
        */
      
        MobilityHelper microGnbMobility;
        microGnbMobility.SetPositionAllocator (microPositionAlloc);
        microGnbMobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
        microGnbMobility.Install (microLayerGnbNodes);
    
        /****************************************************
        * Micro UEs: Position Bounding box and Mobility model
        *****************************************************/  
        
        // Create a bounding box around each micro gNB for the "micro" UEs
        for (uint32_t microCellIndex = 0; microCellIndex < microLayerGnbNodes.GetN (); ++microCellIndex)
        {  
            Ptr<Node> microGnb = microLayerGnbNodes.Get (microCellIndex); 
            Vector microGnbpos = microGnb->GetObject<MobilityModel> ()->GetPosition ();
            std::cout << "microGnb Position " << microGnbpos.x << "," << microGnbpos.y <<  "," << microGnbpos.z << std::endl; 

            // Compute the bounding box for the scenario within which the UEs move
            double microCellRadius = 80;
            double boundingBoxMinX = microGnbpos.x - microCellRadius;
            double boundingBoxMinY = microGnbpos.y - microCellRadius;
            double boundingBoxMaxX = microGnbpos.x + microCellRadius;
            double boundingBoxMaxY = microGnbpos.y + microCellRadius;
            std::cout << "Micro Topology Bounding box (x,y) for index : " << microCellIndex
                << " (" << boundingBoxMinX << ", " << boundingBoxMinY << ")  "
                << "(" << boundingBoxMinX << ", " << boundingBoxMaxY << ")  "
                << "(" << boundingBoxMaxX << ", " << boundingBoxMinY << ")  "
                << "(" << boundingBoxMaxX << ", " << boundingBoxMaxY << ")  "
                << std::endl;
 
            double ueZ =params.ueHeight;
            
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

            // Create mobility model object 
            MobilityHelper mobility;
            mobility.SetPositionAllocator (positionAlloc);
            mobility.SetMobilityModel ("ns3::SteadyStateRandomWaypointMobilityModel", 
                                       "MaxSpeed", DoubleValue (params.microUeMaxSpeed), 
                                       "MinSpeed", DoubleValue (params.microUeMinSpeed) );
            
            // Install this for the UEs that are to be in the bounding box of this micro BS 
            for (uint32_t ueId = microCellIndex*params.ueNumPerMicroGnb; ueId < microCellIndex*params.ueNumPerMicroGnb +  params.ueNumPerMicroGnb; ++ueId)
            {
                Ptr<Node> node = ueNodesMicro.Get (ueId);  
                mobility.Install(node);
            }
        } // end of for over micro cells 
        
        allGnbNodes.Add (microLayerGnbNodes);  
        /***********************************************
        * Micro gNBs: RAN settings
        **********************************************/        
        // Some settings are borrowed from the macro case so it is not being set again here
        // e.g. handover algo, scheduler type etc.
        Config::SetDefault ("ns3::LteEnbPhy::TxPower", DoubleValue (params.microCellTxPower)); 

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
         
        lteHelper->SetEnbAntennaModelType ("ns3::IsotropicAntennaModel");

        NetDeviceContainer microLayerGnbLteDevs = lteHelper->InstallEnbDevice (microLayerGnbNodes);
        randomStream += lteHelper->AssignStreams (microLayerGnbLteDevs, randomStream); 
    } // end of if add micro layer 
    
    ueNetDevs = lteHelper->InstallUeDevice (ueNodes);  
    
    randomStream += lteHelper->AssignStreams (ueNetDevs, randomStream);   
    randomStream += lteHelper->AssignStreams (gnbNetDevs, randomStream); 
    
    // This part is for asserts or sanity checks   
    for (auto nd = ueNetDevs.Begin (); nd != ueNetDevs.End (); ++nd)
    {
        auto ueNetDevice = DynamicCast<LteUeNetDevice> (*nd);
        NS_ASSERT (ueNetDevice->GetCcMap ().size () == 1);
        auto uePhy = ueNetDevice->GetPhy ();
    }
    
    std::cout << "All gNB nodes created" << std::endl;
    for (uint32_t gnb_idx = 0; gnb_idx < allGnbNodes.GetN (); ++gnb_idx)
    {
        Ptr<Node> gnb = allGnbNodes.Get (gnb_idx);  

        std::cout << "gnb_idx: " << gnb_idx 
            << " cellId: " << gnb->GetDevice (0)->GetObject<LteEnbNetDevice> ()->GetCellId()
            << std::endl;
    }

    /****************************************************
    * Install Internet for all Nodes
    *****************************************************/
    
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
    
    ueIpIfaces = epcHelper->AssignUeIpv4Address (ueNetDevs);
    Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress (1);

    // Set the default gateway for the UEs
    for (auto ue = ueNodes.Begin (); ue != ueNodes.End (); ++ue)
    {
        Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting ((*ue)->GetObject<Ipv4> ());
        ueStaticRouting->SetDefaultRoute (epcHelper->GetUeDefaultGatewayAddress (), 1);
    }

    /****************************************************
    * Attach UEs to gNBs
    *****************************************************/      
    
    if (lteHelper != nullptr)
    { 
        lteHelper->Attach (ueNetDevs);
    }
    else if (nrHelper != nullptr)
    {
        nrHelper->AttachToClosestEnb (ueNetDevs, gnbNetDevs);
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
    uint16_t vrPortNum = 16000;
    //uint16_t httpPortNum = 31000;

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
    if(params.traceFlow)
    {
        std::cout << "\nFlow App (UDP) has a per flow rate of " << (flowPerUe/1000000) << " Mbps\n" << std::endl;
    }

    // Server Config 
    ApplicationContainer serverApps;

    // Declaration of Helpers for Sinks and Servers 
    UdpServerHelper ulDelayPacketSink (ulDelayPortNum);
    UdpServerHelper dlDelayPacketSink (dlDelayPortNum);
    UdpServerHelper ulFlowPacketSink (ulFlowPortNum);
    UdpServerHelper dlFlowPacketSink (dlFlowPortNum);
    ThreeGppHttpServerHelper httpServer (remoteHostAddr);
    //DashServerHelper dashServer ("ns3::TcpSocketFactory", 
    //                             InetSocketAddress (Ipv4Address::GetAny (), dashPortNum));
    UdpEchoServerHelper echoServer (echoPortNum);
    //vr  
    Ptr<UniformRandomVariable> vrStart = CreateObject<UniformRandomVariable> ();
    vrStart->SetAttribute ("Min", DoubleValue (params.vrStartTimeMin));
    vrStart->SetAttribute ("Max", DoubleValue (params.vrStartTimeMax));
    
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
    /*if(params.traceDash)
    {
        serverApps.Add (dashServer.Install (remoteHost)); // appId updated on remoteHost
    }*/
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
    //DashClientHelper dashClient ("ns3::TcpSocketFactory", InetSocketAddress (remoteHostAddr, dashPortNum), params.abr);
    ThreeGppHttpClientHelper httpClient (remoteHostAddr);
    //vr
    BurstSinkHelper burstSinkHelper ("ns3::UdpSocketFactory",
                                   InetSocketAddress (Ipv4Address::GetAny (), vrPortNum));
    
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
    /*if(params.traceDash)
    {
        // Configure DASH client application 
        dashClient.SetAttribute ("VideoId", UintegerValue (1)); // VideoId should be positive
        dashClient.SetAttribute ("TargetDt", TimeValue (Seconds (params.targetDt)));
        dashClient.SetAttribute ("window", TimeValue (Seconds (params.window)));
        dashClient.SetAttribute ("bufferSpace", UintegerValue (params.videoBufferSize));
    }*/
    if(params.traceHttp)
    {
        // Configure http client application
    }
    if(params.traceVr)
    {
        // Nothing to configure
    }

    // Client Creation on the desired devices
    Ptr<UniformRandomVariable> startRng = CreateObject<UniformRandomVariable> ();
    startRng->SetStream (RngSeedManager::GetRun ());

    
    /***********************************************
    * Iterate through macro UEs and install apps 
    **********************************************/    
    
    for (uint32_t ueId = 0; ueId < ueNodesMacro.GetN (); ++ueId)
    {
        Ptr<Node> node = ueNodesMacro.Get (ueId);
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
        }
        if(params.traceRtt)
        {
            auto appsClass1 = InstallUdpEchoApps (node,
                              &echoClient,
                              params.appStartTime,
                              startRng, params.appGenerationTime);
            clientApps.Add (appsClass1.first);
        } 
        // Install full buffer BulkSend traffic on only one UE to test the 
        // TCP throughput achieved as the UE moves within the topology
        // This should be installed on one of the fast speed UEs so that it can 
        // cover more distance and visit more cells 
        if(params.traceUlThput)
        {
            if(ueId == 0)
            {
                uint32_t port = 20000;
                BulkSendHelper ulBulkSendHelper ("ns3::TcpSocketFactory",
                                             InetSocketAddress (remoteHostAddr, port));
                ulBulkSendHelper.SetAttribute ("MaxBytes", UintegerValue (0)); // send forever
                ulBulkSendHelper.SetAttribute ("EnableSeqTsSizeHeader", BooleanValue (true));
                clientApps.Add (ulBulkSendHelper.Install (node));
                PacketSinkHelper ulPacketSinkHelper ("ns3::TcpSocketFactory",
                                                   InetSocketAddress (Ipv4Address::GetAny (), port));
                ulPacketSinkHelper.SetAttribute ("EnableSeqTsSizeHeader", BooleanValue (true));   
                serverApps.Add (ulPacketSinkHelper.Install (remoteHost));
                std::cout << " IMSI: " << GetImsi_from_node(node) 
                    << " IP_addr: " << addr 
                    <<" has UL Throughput measurement (BulkSend) app installed" << std::endl;
            }
        }
        if(params.traceDlThput)
        {
            if(ueId == 0)
            {
                uint32_t port = 19000;
                BulkSendHelper dlBulkSendHelper ("ns3::TcpSocketFactory",
                                             InetSocketAddress (addr, port));
                dlBulkSendHelper.SetAttribute ("MaxBytes", UintegerValue (0)); // send forever
                dlBulkSendHelper.SetAttribute ("EnableSeqTsSizeHeader", BooleanValue (true));
                clientApps.Add (dlBulkSendHelper.Install (remoteHost));
                PacketSinkHelper dlPacketSinkHelper ("ns3::TcpSocketFactory",
                                                   InetSocketAddress (Ipv4Address::GetAny (), port));
                dlPacketSinkHelper.SetAttribute ("EnableSeqTsSizeHeader", BooleanValue (true)); 
                serverApps.Add (dlPacketSinkHelper.Install (node));
                std::cout << " IMSI: " << GetImsi_from_node(node)  
                    << " IP_addr: " << addr 
                    <<" has DL Throughput measurement (BulkSend) app installed" << std::endl;
                
            }
        }
        
        
        if (params.traceVr) 
        {
          // Install VR app on the last n UEs which are slow moving UEs 
            if (ueId >= (ueNodesMacro.GetN ()-params.numMacroVrUes) )
            {
                //Install VR and continue so that the regular logic is not encountered
                // Random sample for the start time fo the VR session for each UE  
                double vrStartTime = vrStart->GetValue();
                // The sender of VR traffic to be installed on remoteHost
                BurstyHelper burstyHelper ("ns3::UdpSocketFactory", 
                                           InetSocketAddress (addr, vrPortNum)); 
                burstyHelper.SetAttribute ("FragmentSize", UintegerValue (1200));
                burstyHelper.SetBurstGenerator ("ns3::TraceFileBurstGenerator", 
                                                "TraceFile", StringValue (traceFolder + vrTraceFiles[vrTraceFileIndex]), 
                                                "StartTime", DoubleValue (vrStartTime));
                vrTraceFileIndex = (vrTraceFileIndex + 1)%8;
                serverApps.Add (burstyHelper.Install (remoteHost));
                // The receiver of the VR traffic to be installed on UEs
                clientApps.Add (burstSinkHelper.Install (node));
                // Print the IMSI of the ues that are doing this	
                std::cout << " IMSI: " << GetImsi_from_node(node) 
                    << " Ip_addr: " << addr 
                    << " has VR app installed " << std::endl;
                //}
                continue;
            }
        }
        
        // These are the apps that are on a subset of devices 
        if (ueId % 5 == 0)
        {
            // This is on 20% of the macro devices
            // These UEs do video + delay probes
            if(params.traceDash)
            {
                DashServerHelper dashServer ("ns3::TcpSocketFactory", 
                                 InetSocketAddress (Ipv4Address::GetAny (), dashPortNum+ueId));
                serverApps.Add (dashServer.Install (remoteHost)); // appId updated on remoteHost
                DashClientHelper dashClient ("ns3::TcpSocketFactory", InetSocketAddress (remoteHostAddr, dashPortNum+ueId), params.abr);
                // Configure DASH client application 
                dashClient.SetAttribute ("VideoId", UintegerValue (1)); // VideoId should be positive
                dashClient.SetAttribute ("TargetDt", TimeValue (Seconds (params.targetDt)));
                dashClient.SetAttribute ("window", TimeValue (Seconds (params.window)));
                dashClient.SetAttribute ("bufferSpace", UintegerValue (params.videoBufferSize));

                
                auto appsClass6 = InstallDashApps (node,
                                  &dashClient,
                                  params.appStartTime,
                                  startRng, params.appGenerationTime);
                clientApps.Add (appsClass6.first);
                // Print the IMSI of the ues that are doing this
                std::cout << " IMSI: " << GetImsi_from_node(node) 
                << " IP_addr: " << addr
                <<" has DASH app installed" << std::endl;
            }
        } 
        else if (ueId % 2 == 0)
        {
            // This is on 40% of the devices
            // only delay probes
            onlyDelayUes.push_back (GetImsi_from_node(node));
        }
        else 
        {
            // This is on 40% of the devices
            // These UEs do web browsing + delay measurement
            if(params.traceHttp)
            {   
                // if dedicated server for each client 
                /*ThreeGppHttpClientHelper httpClient (remoteHostAddr);
                httpClient.SetAttribute("RemoteServerPort", UintegerValue (httpPortNum+ueId));
                ThreeGppHttpServerHelper httpServer (remoteHostAddr);
                httpServer.SetAttribute("LocalPort", UintegerValue (httpPortNum+ueId));
                serverApps.Add (httpServer.Install (remoteHost));*/
                
                //option 2
                /*Ptr<ThreeGppHttpServer> httpCli = s ->GetObject<ThreeGppHttpServer> ();
                PointerValue varPtr;
                httpCli->GetAttribute ("Variables", varPtr);
                Ptr<ThreeGppHttpVariables> httpVariables = varPtr.Get<ThreeGppHttpVariables> ();
                httpVariables->SetMainObjectSizeMean (params.httpMainObjMean); // 100kB*/
                
                //option 1
                Ptr<ThreeGppHttpVariables> httpVariables = CreateObject<ThreeGppHttpVariables> ();
                randomStream += httpVariables->AssignStreams(randomStream);
                httpClient.SetAttribute("Variables",  PointerValue (httpVariables));
                
                auto appsClass6 = InstallHttpApps (node,
                              &httpClient,
                              params.appStartTime,
                              startRng, params.appGenerationTime);
                clientApps.Add (appsClass6.first);
                // Print the IMSI of the ues that are doing this
                std::cout << " IMSI: " << GetImsi_from_node(node) 
                  << " IP_addr: " << addr 
                  << " has HTTP app installed" << std::endl;
            }
        }
    } // end of for over macro layer UEs
   
    /***********************************************
    * Iterate through micro UEs and install apps 
    **********************************************/ 
    
    for (uint32_t ueId = 0; ueId < ueNodesMicro.GetN (); ++ueId)
    {
        Ptr<Node> node = ueNodesMicro.Get (ueId);
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
        }
        if(params.traceRtt)
        {
            auto appsClass1 = InstallUdpEchoApps (node,
                              &echoClient,
                              params.appStartTime,
                              startRng, params.appGenerationTime);
            clientApps.Add (appsClass1.first);
        } 
        // Install full buffer BulkSend traffic on only one UE to test the 
        // TCP throughput achieved as the UE moves within the topology
        // This should be installed on one of the fast speed UEs so that it can 
        // cover more distance and visit more cells 
        if(params.traceUlThput)
        {
            if(ueId == 0)
            {
                uint32_t port = 20000;
                BulkSendHelper ulBulkSendHelper ("ns3::TcpSocketFactory",
                                             InetSocketAddress (remoteHostAddr, port));
                ulBulkSendHelper.SetAttribute ("MaxBytes", UintegerValue (0)); // send forever
                ulBulkSendHelper.SetAttribute ("EnableSeqTsSizeHeader", BooleanValue (true));
                clientApps.Add (ulBulkSendHelper.Install (node));
                PacketSinkHelper ulPacketSinkHelper ("ns3::TcpSocketFactory",
                                                   InetSocketAddress (Ipv4Address::GetAny (), port));
                ulPacketSinkHelper.SetAttribute ("EnableSeqTsSizeHeader", BooleanValue (true));   
                serverApps.Add (ulPacketSinkHelper.Install (remoteHost));
                std::cout << " IMSI: " << GetImsi_from_node(node) 
                    << " IP_addr: " << addr 
                    <<" has UL Throughput measurement (BulkSend) app installed" << std::endl;
            }
        }
        if(params.traceDlThput)
        {
            if(ueId == 0)
            {
                uint32_t port = 19000;
                BulkSendHelper dlBulkSendHelper ("ns3::TcpSocketFactory",
                                             InetSocketAddress (addr, port));
                dlBulkSendHelper.SetAttribute ("MaxBytes", UintegerValue (0)); // send forever
                dlBulkSendHelper.SetAttribute ("EnableSeqTsSizeHeader", BooleanValue (true));
                clientApps.Add (dlBulkSendHelper.Install (remoteHost));
                PacketSinkHelper dlPacketSinkHelper ("ns3::TcpSocketFactory",
                                                   InetSocketAddress (Ipv4Address::GetAny (), port));
                dlPacketSinkHelper.SetAttribute ("EnableSeqTsSizeHeader", BooleanValue (true)); 
                serverApps.Add (dlPacketSinkHelper.Install (node));
                std::cout << " IMSI: " << GetImsi_from_node(node) 
                    << " IP_addr: " << addr 
                    <<" has DL Throughput measurement (BulkSend) app installed" << std::endl;
                
            }
        }
        
        
        if (params.traceVr) 
        {
          // Install VR app on the last n UEs which are slow moving UEs 
            if (ueId >= (ueNodesMicro.GetN ()-params.numMicroVrUes) )
            {
                //Install VR and continue so that the regular logic is not encountered
                // Random sample for the start time fo the VR session for each UE  
                double vrStartTime = vrStart->GetValue();
                // The sender of VR traffic to be installed on remoteHost
                BurstyHelper burstyHelper ("ns3::UdpSocketFactory", 
                                           InetSocketAddress (addr, vrPortNum)); 
                burstyHelper.SetAttribute ("FragmentSize", UintegerValue (1200));
                burstyHelper.SetBurstGenerator ("ns3::TraceFileBurstGenerator", 
                                                "TraceFile", StringValue (traceFolder + vrTraceFiles[vrTraceFileIndex]), 
                                                "StartTime", DoubleValue (vrStartTime));
                vrTraceFileIndex = (vrTraceFileIndex + 1)%8;
                serverApps.Add (burstyHelper.Install (remoteHost));
                // The receiver of the VR traffic to be installed on UEs
                clientApps.Add (burstSinkHelper.Install (node));
                // Print the IMSI of the ues that are doing this	
                std::cout << " IMSI: " << GetImsi_from_node(node) 
                    << " Ip_addr: " << addr 
                    << " has VR app installed " << std::endl;
                continue;
            }
        }
        
        // These are the apps that are on a subset of devices 
        if (ueId % 5 == 0)
        {
            // This is on 20% of the micro devices
            // These UEs do web browsing + delay measurement
            if(params.traceHttp)
            {
                
                /*ThreeGppHttpServerHelper httpServer (remoteHostAddr);
                httpServer.SetAttribute("LocalPort", UintegerValue (httpPortNum+100+ueId));
                serverApps.Add (httpServer.Install (remoteHost));
                
                ThreeGppHttpClientHelper httpClient (remoteHostAddr);
                httpClient.SetAttribute("RemoteServerPort", UintegerValue (httpPortNum+100+ueId));*/
                
                //option 1
                Ptr<ThreeGppHttpVariables> httpVariables = CreateObject<ThreeGppHttpVariables> ();
                randomStream += httpVariables->AssignStreams(randomStream);
                httpClient.SetAttribute("Variables",  PointerValue (httpVariables));
                
                auto appsClass6 = InstallHttpApps (node,
                              &httpClient,
                              params.appStartTime,
                              startRng, params.appGenerationTime);
                clientApps.Add (appsClass6.first);
                // Print the IMSI of the ues that are doing this
                std::cout << " IMSI: " << GetImsi_from_node(node) 
                  << " IP_addr: " << addr 
                  << " has HTTP app installed" << std::endl;
            }
        } 
        else if (ueId % 2 == 0)
        {
            // This is on 40% of the micro devices
            // These have only delay probes
            onlyDelayUes.push_back (GetImsi_from_node(node));
        }
        else 
        {
            // This is on 40% of the micro devices
            // These UEs do video + delay measurement
            if(params.traceDash)
            {
                DashServerHelper dashServer ("ns3::TcpSocketFactory", 
                                 InetSocketAddress (Ipv4Address::GetAny (), dashPortNum+100+ueId));
                serverApps.Add (dashServer.Install (remoteHost)); // appId updated on remoteHost
                DashClientHelper dashClient ("ns3::TcpSocketFactory", InetSocketAddress (remoteHostAddr, dashPortNum+100+ueId), params.abr);
                // Configure DASH client application 
                dashClient.SetAttribute ("VideoId", UintegerValue (1)); // VideoId should be positive
                dashClient.SetAttribute ("TargetDt", TimeValue (Seconds (params.targetDt)));
                dashClient.SetAttribute ("window", TimeValue (Seconds (params.window)));
                dashClient.SetAttribute ("bufferSpace", UintegerValue (params.videoBufferSize));

                
                auto appsClass6 = InstallDashApps (node,
                                  &dashClient,
                                  params.appStartTime,
                                  startRng, params.appGenerationTime);
                clientApps.Add (appsClass6.first);
                // Print the IMSI of the ues that are doing this
                std::cout << " IMSI: " << GetImsi_from_node(node)  
                << " IP_addr: " << addr
                <<" has DASH app installed" << std::endl;
            }
        }
    } // end of for over UEs

/*
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
    std::cout << "ueId: " << ueId << " IMSI: " << GetImsi_from_node(node) 
    << " Ip_addr: " << addr 
    << " has UL and DL FLOW app installed " << std::endl;
}*/
    
    
    
    
    
    // Server Start  
    serverApps.Start (params.appStartTime);
    // client apps are started individually using the Install function 

    // Setup X2 interface to enable handover   
    if (lteHelper != nullptr)
    {
        lteHelper->AddX2Interface (allGnbNodes);
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
                       MakeBoundCallback (&NotifyHandoverStartEnb, handoverStream));
        

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
        if(params.traceVr)
        {
            Config::Connect ("/NodeList/*/ApplicationList/*/$ns3::BurstSink/BurstRx", 
                             MakeBoundCallback (&BurstRx, burstRxStream));
            Config::Connect ("/NodeList/*/ApplicationList/*/$ns3::BurstSink/FragmentRx", 
                             MakeBoundCallback (&FragmentRx, fragmentRxStream));
        }
        if(params.traceUlThput || params.traceDlThput)
        {
            Config::Connect ("/NodeList/*/ApplicationList/*/$ns3::PacketSink/RxWithSeqTsSize", 
                             MakeBoundCallback (&ThputMeasurement, thputStream));
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
    Simulator::Schedule (MilliSeconds(400), &ScenarioInfo, scenario);
    
    /*
    // To schedule on/off of apps that I would like to pause and resume randomly 
    Ptr<UniformRandomVariable> appOnTime = CreateObject<UniformRandomVariable> ();
    appOnTime->SetAttribute ("Min", DoubleValue (1));
    appOnTime->SetAttribute ("Max", DoubleValue (300));
    Ptr<UniformRandomVariable> appOffTime = CreateObject<UniformRandomVariable> ();
    appOffTime->SetAttribute ("Min", DoubleValue (1));
    appOffTime->SetAttribute ("Max", DoubleValue (30));
    // sample ON time and pass both on and off random variables  
    Simulator::Schedule (Seconds(appOnTime->GetValue()), &PauseApp, appOnTime, appOffTime);
    */
    /*if ( (params.simulator == "LENA") && (params.useMicroLayer) && (!params.macroMicroSharedSpectrum))
    {
        // This is a nearest cell manual HO that works even between frequencies 
        Simulator::Schedule (MilliSeconds(params.manualHoTriggerTime), &CheckForManualHandovers, lteHelper); 
    }*/
    
   
    /*
    // REM map which is a plot of the DL SINR over the coverage region
    //Ptr<NrRadioEnvironmentMapHelper> remHelper = CreateObject<NrRadioEnvironmentMapHelper> ();
    Ptr<RadioEnvironmentMapHelper> remHelper = CreateObject<RadioEnvironmentMapHelper> ();
    remHelper->SetAttribute("Channel", PointerValue(lteHelper->GetDownlinkSpectrumChannel())); 
    //remHelper->SetAttribute("Channel", PointerValue(lteHelper->GetUplinkSpectrumChannel())); 
    //remHelper->SetAttribute ("ChannelPath", StringValue ("/ChannelList/3"));  
    remHelper->SetAttribute ("Earfcn", UintegerValue(100));//100 default
    remHelper->SetAttribute ("Bandwidth", UintegerValue(50));//PRBs  
    remHelper->SetAttribute ("OutputFile", StringValue ("rem.out"));
    remHelper->SetAttribute ("XMin", DoubleValue (boundingBoxMinX));
    remHelper->SetAttribute ("XMax", DoubleValue (boundingBoxMaxX));
    remHelper->SetAttribute ("YMin", DoubleValue (boundingBoxMinY));
    remHelper->SetAttribute ("YMax", DoubleValue (boundingBoxMaxY));
    remHelper->SetAttribute ("Z", DoubleValue (1.5));
    remHelper->SetAttribute("UseDataChannel", BooleanValue(true));
    remHelper->Install ();
    //remHelper->CreateRem (remNd, remDevice, remPhyIndex);*/
  
    Simulator::Run ();
    std::cout << "\n------------------------------------------------------\n"
            << "End simulation"
            << std::endl;
    Simulator::Destroy ();
}// end of LenaLteComparison
    

} // end of namespace ns3

