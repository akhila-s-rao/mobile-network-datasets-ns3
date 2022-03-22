/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 TEI of Western Macedonia, Greece
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Dimitrios J. Vergados <djvergad@gmail.com>
 */

#ifndef DASH_CLIENT_H
#define DASH_CLIENT_H

//akhila
#include <iostream>
#include <string>
#include <fstream>
#include "ns3/network-module.h"


#include "ns3/application.h"
#include "ns3/ptr.h"
#include "ns3/socket.h"
#include "mpeg-player.h"
#include "ns3/traced-callback.h"
#include "http-parser.h"


namespace ns3 {

/**
   * A structure to pass the parameters I need in the Trace callback function
   */

struct ClientDashSegmentInfo {
  int node_id;  
  uint32_t videoId;
  uint32_t segId;
  uint32_t bitRate;
  uint32_t oldBitRate;
  double thputOverLastSeg_bps;
  double estBitRate;
  int frameQueueSize;
  double interTime_s;
  double playbackTime;
  double realplayTime;
  double Dt;
  double timeToNextReq;
  // equality comparison. doesn't modify object. therefore const.
  bool operator==(const ClientDashSegmentInfo& a) const
  {
      return (node_id == a.node_id && \
	      videoId == a.videoId && \
	      segId == a.segId && \
	      bitRate == a.bitRate && \
	      oldBitRate == a.oldBitRate && \
	      thputOverLastSeg_bps == a.thputOverLastSeg_bps && \
	      estBitRate == a.estBitRate && \
	      frameQueueSize == a.frameQueueSize && \
	      interTime_s == a.interTime_s && \
	      playbackTime == a.playbackTime && \
	      realplayTime == a.realplayTime && \
	      Dt == a.Dt && \
	      timeToNextReq == a.timeToNextReq \
	      );
  }
  // NOT equality comparison. doesn't modify object. therefore const.
  bool operator!=(const ClientDashSegmentInfo& a) const
  {
      return (!(node_id == a.node_id && \
		videoId == a.videoId && \
                segId == a.segId && \
                bitRate == a.bitRate && \
                oldBitRate == a.oldBitRate && \
                thputOverLastSeg_bps == a.thputOverLastSeg_bps && \
                estBitRate == a.estBitRate && \
                frameQueueSize == a.frameQueueSize && \
                interTime_s == a.interTime_s && \
                playbackTime == a.playbackTime && \
                realplayTime == a.realplayTime && \
                Dt == a.Dt && \
                timeToNextReq == a.timeToNextReq \
                ) );
  }
 
};

struct MpegPlayerFrameInfo {
  int node_id; //The NS_LOG below is using m_dashClient->m_id which is different from the older version from which I copied m_dashClient->GetId(), so watch out this ID could be wrong. 
  uint32_t videoId;
  uint32_t segId;
  uint32_t resolution;
  uint32_t frame_id;
  double playback_time;
  char type;
  uint32_t size;
  double interruption_time;
  int queue_size;

  // equality comparison. doesn't modify object. therefore const.
  bool operator==(const MpegPlayerFrameInfo& a) const
  {
      return (node_id == a.node_id && \
              videoId == a.videoId && \
              segId == a.segId && \
              resolution == a.resolution && \
              frame_id == a.frame_id && \
              playback_time == a.playback_time && \
              type == a.type && \
              size == a.size && \
              interruption_time == a.interruption_time && \
              queue_size == a.queue_size \
              );
  }
  // NOT equality comparison. doesn't modify object. therefore const.
  bool operator!=(const MpegPlayerFrameInfo& a) const
  {
      return (node_id == a.node_id && \
              videoId == a.videoId && \
              segId == a.segId && \
              resolution == a.resolution && \
              frame_id == a.frame_id && \
              playback_time == a.playback_time && \
              type == a.type && \
              size == a.size && \
              interruption_time == a.interruption_time && \
              queue_size == a.queue_size \
              );
  }
};

/**
   * \defgroup dash Dash
   * This section documents the API of the ns-3 dash module. For a generic functional
   * description, please refer to the ns-3 manual.
   */

/**
   * \ingroup dash
   *
   * \brief This is the DASH client application, that is
   * used for transmitting and receiving http DASH
   * messages with a DASH server.
   *
   * The client requests one segment at a time. When it
   * receives each tcp segment, it uses a HttpParser object
   * to decode the HTTP messages into MPEG frames.
   *
   * These MPEG frames are then passed on to the MpegPlayer object
   * that buffers and reproduces the frames.
   *
   * When an entire segment has been received, then the next segment's
   * bitrate (resolution) and request time are calculated my the MpegPlayer
   * depending on the buffer level, and the current measured throughput.
   *
   */
class DashClient : public Application
{
  friend class MpegPlayer;
  friend class HttpParser;
  friend class Socket;

public:
  static TypeId GetTypeId (void);

  DashClient ();

  virtual ~DashClient ();

//akhila
  void SetLogStream (Ptr<OutputStreamWrapper> stream);

  /**
     * \return pointer to associated socket
     */
  Ptr<Socket> GetSocket (void) const;

  /**
     * \brief Prints some statistics.
     */
  void GetStats ();

  /**
     * \return The MpegPlayer object that is used for buffering and
     * reproducing the video, and for estimating the next bitrate (resolution)
     * and request time.
     */
  inline MpegPlayer &
  GetPlayer ()
  {
    return m_player;
  }

  void CheckBuffer ();

  //akhila2
  void ReceiveMpegTrace (MpegPlayerFrameInfo & mpegInfo);
  //akhila
//  typedef void (*TracedCallback)(ClientDashSegmentInfo & info);

protected:
  virtual void DoDispose (void);

  double inline GetBitRateEstimate ()
  {
    return m_bitrateEstimate;
  }

  double GetBufferDifferential ();

  void AddBitRate (Time time, double bitrate);

  double GetBufferEstimate ();

  double GetSegmentFetchTime ();

  std::vector<uint32_t> rates = {45000,    89000,    131000,   178000,  221000,  263000,  334000,
                                 396000,   522000,   595000,   791000,  1033000, 1245000, 1547000,
                                 2134000,  2484000,  3079000,  3527000, 3840000, 4220000};
				 //9500000, 15000000, 30000000, 66000000, 85000000};
  //akhila is saving this for if we want to use the dash settings
  // remember to change it in dash-client.cc as well
  //std::vector<uint32_t> rates = {1000000,  2500000, 5000000, 8000000, 16000000, 35000000};

  uint32_t m_bufferSpace;
  MpegPlayer m_player; // The MpegPlayer object

  std::map<Time, Time> m_bufferState;
  uint32_t m_rateChanges;
  Time m_target_dt;
  std::map<Time, double> m_bitrates;
  double m_bitrateEstimate;
  uint32_t m_segmentId; // The id of the current segment
  //akhila
  Ptr<OutputStreamWrapper> m_stream;
  bool logStreamSet = false;

private:
  /**
     * \brief Called the next MPEG segment should be requested from the server.
     *
     * \param The bitrate of the next segment.
     */
  void RequestSegment ();

  void SendBlank ();

  /**
     * \brief Called by the HttpParser when it has received a complete HTTP
     * message containing an MPEG frame.
     *
     * \param the message that was received
     */
  bool MessageReceived (Packet message);

  // inherited from Application base class.
  virtual void StartApplication (void); // Called at time specified by Start
  virtual void StopApplication (void); // Called at time specified by Stop
  void ConnectionSucceeded (Ptr<Socket> socket); // Called when the connections has succeeded
  void ConnectionFailed (Ptr<Socket> socket); // Called when the connection has failed.
  void
  ConnectionNormalClosed (Ptr<Socket> socket); // Called when the connection has closed normally.
  void
  ConnectionErrorClosed (Ptr<Socket> socket); // Called when the connection has closed due to error.
  void DataSend (Ptr<Socket>, uint32_t); // Called when the data has been transmitted
  void HandleRead (Ptr<Socket>); // Called when we receive data from the server
  virtual void CalcNextSegment (uint32_t currRate, uint32_t &nextRate, Time &delay);
  void LogBufferLevel (Time t);
  void inline SetWindow (Time time)
  {
    m_window = time;
  }

  void KeepAliveTimeout ();

  HttpParser m_parser; // An HttpParser object for parsing the incoming stream into http messages
  Ptr<Socket> m_socket; // Associated socket
  Address m_peer; // Peer address
  bool m_connected; // True if connected
  uint32_t m_totBytes; // Total bytes received.

  TypeId m_tid;
  //TracedCallback<Ptr<const Packet>> m_txTrace;
  //akhila
  /// Callbacks for tracing the segment Rx events
  ns3::TracedCallback<ClientDashSegmentInfo &> m_rxTrace;
  ns3::TracedCallback<MpegPlayerFrameInfo &> m_playFrame;
  //ns3::TracedCallback<const uint16_t &> m_rxTrace;

  uint32_t m_videoId; // The Id of the video that is requested
  Time m_started; // Time of application starting
  Time m_sumDt; // Used for calculating the average buffering time
  Time m_lastDt; // The previous buffering time (used for calculating the differential
  static int m_countObjs; // Number of DashClient instances (for generating unique id
  int m_id;
  Time m_requestTime; // Time of sending the last request
  uint32_t m_segment_bytes; // Bytes of the current segment that have been received so far
  uint32_t m_bitRate; // The bitrate of the current segment.
  Time m_window;
  Time m_segmentFetchTime;
  bool m_RequestPending = false; // So that we don't request the same segment repeatedly

  EventId m_keepAliveTimer;

};

} // namespace ns3

#endif /* DASH_CLIENT_H */
