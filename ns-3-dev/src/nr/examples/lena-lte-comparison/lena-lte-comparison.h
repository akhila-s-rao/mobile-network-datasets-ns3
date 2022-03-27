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

#ifndef LENA_LTE_COMPARISON_H
#define LENA_LTE_COMPARISON_H

namespace ns3 {


struct Parameters
{
  friend
  std::ostream &
  operator << (std::ostream & os, const Parameters & parameters);
  bool Validate (void) const;

  // Deployment topology parameters
  uint16_t numOuterRings = 1;
  uint16_t ueNumPergNb = 30;
  std::string simulator = "LENA";
  std::string scenario = "UMa";
  std::string baseStationFile = ""; // path to file of tower/site coordinates
  bool useSiteFile = false; // whether to use baseStationFile parameter,
                            //or to use numOuterRings parameter to create a scenario

  // Simulation parameters
  // Don't use double for seconds, use  milliseconds and integers.
  bool logging = true;
  bool traces = true;
  Time appGenerationTime = Seconds (1000);
  Time appStartTime = MilliSeconds (500);
  Time progressInterval = Seconds (1);
  uint32_t randSeed = 13;

  // RAN parameters
  std::string operationMode = "FDD";  // TDD or FDD for NR. Only FDD available for LTE
  uint16_t numerologyBwp = 0; // NR specific
  // legend: F->flexible DL->downlink  UL->uplink S->special(LTE DL)
  std::string pattern = "F|F|F|F|F|F|F|F|F|F|"; // Pattern can be e.g. "DL|S|UL|UL|DL|DL|S|UL|UL|DL|" //NR specific
  uint32_t bandwidthMHz = 10;
  bool enableUlPc = false;
  std::string scheduler = "PF";
  uint32_t freqScenario = 0; // 0->non-overlaping 1->overlapping
  double downtiltAngle = 0;

  // mobility model 
  double ueMinSpeed = 1.4; // m/s
  double ueMaxSpeed = 10; // m/s

  // traffic parameters

  // DASH video streaming 
  double targetDt = 20.0; // The target time difference between receiving and playing a frame. [s]. 
  double window = 5.0; // The window for measuring the average throughput. [s].
  uint32_t bufferSpace = 10000000; // 10 MB The space in bytes that is used for buffering the video
  std::string abr = "ns3::FdashClient";
  
  // web browsing (http client)
  uint32_t httpMainObjMean = 102400;
  uint32_t httpMainObjStd = 40960;

  // UDP flow
  //uint32_t flowPacketSize = 1400;
  uint32_t trafficScenario = 0; 
  std::string direction = "both"; // "UL", "DL" or "both"

  // UDP one way delay probes 
  uint32_t delayPacketSize = 1400;
  Time delayInterval = Seconds (0.1);

  // UDP echo 
  uint32_t echoPacketSize = 1400;
  Time echoInterPacketInterval = Seconds (0.1);

};

extern void LenaLteComparison (const Parameters &params);

} // namespace ns3

#endif // LENA_LTE_COMPARISON_H
