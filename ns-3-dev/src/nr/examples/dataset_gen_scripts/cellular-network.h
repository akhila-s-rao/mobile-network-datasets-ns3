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

#ifndef CELLULAR_NETWORK_H
#define CELLULAR_NETWORK_H

namespace ns3 {


struct Parameters
{
    friend
    std::ostream &
    operator << (std::ostream & os, const Parameters & parameters);
    // To locate trace files 
    std::string ns3Dir = "/home/ubuntu/mobile-network-datasets-ns3/ns-3-dev/";

    // Deployment topology parameters
    
    uint16_t numOuterRings = 0;
    uint16_t ueNumPergNb = 3;
    std::string rat = "LTE";
    std::string scenario = "UMi";
    uint16_t numMicroCells = 3;
    uint16_t microCellTxPower = 18;
    //uint16_t numMacroCellsWithMicroLayer = 2;
    bool useMicroLayer = false;
    
    std::string baseStationFile = ""; // path to file of tower/site coordinates
    bool useSiteFile = false; // whether to use baseStationFile parameter,
                            //or to use numOuterRings parameter to create a scenario
    double ueHeight = 1.5;

    // Simulation parameters
    // Don't use double for seconds, use  milliseconds and integers.
    
    bool logging = false; // NS_LOG for debugging
    bool traces = true;
    Time appGenerationTime = Seconds (1000);
    Time appStartTime = MilliSeconds (500);
    Time progressInterval = Seconds (1);
    uint32_t randSeed = 13;
    uint16_t ranSamplePeriodMilli = 40;  
        

    // RAN parameters
    
    std::string operationMode = "FDD";  // TDD or FDD for NR. Only FDD available for LTE
    uint16_t numerologyBwp = 0; // NR specific
    // legend: F->flexible DL->downlink  UL->uplink S->special(LTE DL)
    std::string pattern = "F|F|F|F|F|F|F|F|F|F|"; 
    // Pattern can be e.g. "DL|S|UL|UL|DL|DL|S|UL|UL|DL|" //NR specific
    uint32_t bandwidthMHz = 10; // MHz
    uint32_t microBandwidthMHz = 20; //MHz
    bool enableUlPc = true;
    std::string scheduler = "PF";
    uint32_t freqScenario = 1; // 0->non-overlaping 1->overlapping
    double downtiltAngle = 0;
    std::string handoverAlgo = "A3Rsrp"; // Options are "A3Rsrp" or "A2A4Rsrq"
    uint32_t manualHoTriggerTime = 256 ;// milliSeconds 
    bool macroMicroSharedSpectrum = true;
    // This needs to be adjusted according to data bandwidth, or one will not be able to use the full BW of the RAN 
    uint32_t rlcUmTxBuffSize = 100*1024; //ns3 default is 10240 which is too small and restricts bandwidth 
                                         //especially VR which pushed a very large frame to the buffer all at once  
    
    // network parameters
    
    uint32_t tcpSndRcvBuf = 1000*1024;  // ns3 default is 131072 which is too small and restricts bandwidth 
                                        //especially VR which pushed a very large frame to the buffer all at once
    
    // mobility model 
    
    double slowUeMinSpeed = 0.5; // m/s
    double slowUeMaxSpeed = 1.5; // m/s
    double fastUeMinSpeed = 3; // m/s // used to be 5 
    double fastUeMaxSpeed = 9; // m/s // used to be 15
    double fracFastUes = 0.2; 

    // Application traffic parameters
    
    // delay measurement 
    bool traceDelay = true;
    bool traceRtt = true;
    // traffic apps to make it interesting
    bool traceHttp = false;
    bool traceDash = false;
    bool traceVr = false; 
    // throughput measurement using full buffer traffic 
    bool traceUlThput = false;
    bool traceDlThput = false;
    
    // don't neeed this for now
    bool traceFlow = false;
    
    // VR
    uint16_t numVrUes = 1;
    double vrStartTimeMin = 2; // seconds 
    double vrStartTimeMax = 5; // seconds
    
    // DASH video streaming 
    
    double targetDt = 30.0; // The target time difference between receiving and playing a frame. [s].
    double window = 5.0; // The window for measuring the average throughput. [s].
    uint32_t bufferSpace = 10*(1000000); // The space in bytes that is used for buffering the video
    std::string abr = "ns3::FdashClient";

    // web browsing (http client)
    
    uint32_t httpMainObjMean = 102400; 
    uint32_t httpMainObjStd = 40960;

    // UDP flow
    
    uint32_t flowPacketSize = 1400;
    double trafficLoadFrac = 0.1; // fraction of total BW of the basestation to be used by UDP flow range (0,1)
    
    std::string direction = "both"; // "UL", "DL" or "both"

    // UDP one way delay probes 
    
    uint32_t delayPacketSize = 1400;
    Time delayInterval = Seconds (0.1);

    // UDP echo 
    
    uint32_t echoPacketSize = 1400;
    Time echoInterPacketInterval = Seconds (0.1);

    
    // Throughput measurement using BulkSend
    uint32_t ThputPacketSize = 1400;
    
    
    
    
    /**********************************************************************/
    
  
    // Validate scenario parameter that this setting has   
    bool Validate (void) const
    {
        NS_ABORT_MSG_IF (bandwidthMHz != 20 && bandwidthMHz != 10 && bandwidthMHz != 5,
                       "Valid bandwidth values are 20, 10, 5, you set " << bandwidthMHz);
        NS_ABORT_MSG_IF (trafficLoadFrac < 0 || trafficLoadFrac > 1.0,
                       "Traffic load fraction " << trafficLoadFrac << " not valid. It shoudl be in the range (0,1)");
        NS_ABORT_MSG_IF (numerologyBwp > 4,
                       "At most 4 bandwidth parts supported.");
        NS_ABORT_MSG_IF (direction != "DL" && direction != "UL" && direction != "both",
                       "Flow direction can only be DL, UL or both: " << direction);
        NS_ABORT_MSG_IF (operationMode != "TDD" && operationMode != "FDD",
                       "Operation mode can only be TDD or FDD: " << operationMode);
        NS_ABORT_MSG_IF (rat != "LTE" && rat != "NR",
                       "Unrecognized simulator: " << rat);
        NS_ABORT_MSG_IF (scheduler != "PF" && scheduler != "RR",
                       "Unrecognized scheduler: " << scheduler);
        NS_ABORT_MSG_IF (handoverAlgo != "A3Rsrp" && handoverAlgo != "A2A4Rsrq",
                       "Unrecognized handover algorithm: " << handoverAlgo);
        NS_ABORT_MSG_IF (useMicroLayer && rat != "LTE",
                       "Cannot create micro layer unless using 4G LTE: ");
        return true;
    }   
};      
        
extern void CellularNetwork (const Parameters &params);

} // namespace ns3

#endif // CELLULAR_NETWORK_H
